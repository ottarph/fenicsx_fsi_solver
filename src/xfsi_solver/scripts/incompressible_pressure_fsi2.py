# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""A constant outlet pressure in FSI2i, with fsi2_incompressible: the state it gives, the Newton
contraction with the full Jacobian, J_0 and interface ALE, and the pressure geometry terms at the
interface.

Both media of FSI2i are incompressible, so a uniform offset of the fluid pressure p and the solid
pressure p_S by the outlet pressure P0 solves the problem with the outlet traction -P0 n, with the same
velocity and displacements. The geometric derivatives of the two pressure terms cancel at the interface
for a uniform pressure: P0 times the integral of cof(F) : grad(dv) over fluid and solid is a boundary
integral, whose interface parts have opposite normals. fsi2_incompressible's Jacobian, J_0, keeps the
solid's derivative and drops the fluid's, so the cancellation fails there.

``PressureRecorder``, the ``jacobian_callback``, records at Newton iteration ``at_iteration`` of the
steps ``sample_steps`` the Frobenius norms of the pressure geometry derivatives, with the columns of
the interface velocity dofs: K_f S, of the fluid term through the interface mesh displacement; K_s, of
the solid term; and their sum, at the state's pressures and at a unit pressure in both. After the run it
writes the owned and ghost values of v, p, p_S and u_s of the last step, with their dof coordinates, one
file per rank, to compare runs at different outlet pressures dof by dof (on the same number of ranks).

    mpirun -n 4 python -m xfsi_solver.scripts.incompressible_pressure_fsi2 --mode J_0 \\
        --initial-state-dir output/snapshots/fsi2_incompressible/t15.0000 --t0 15 --steps 100 \\
        --outlet-pressure 11200 --outlet-pressure-ramp 0.1 --out output/fsi2i_p11200_J_0.json
"""

import argparse
import json

import dolfinx
import dolfinx.fem.petsc
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.petsc_options.fsi2d import DEFAULT_DIRECT_SOLVER_OPTIONS
from xfsi_solver.solvers.fsi2_incompressible import solve

MODES = {"J_0": None, "interface_ale": "interface", "full": "full"}
DIFFERENT = PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN


def solver_options(mode: str, snes_rtol: float = 1e-10, snes_max_it: int = 40, snes_atol: float = 0.0) -> dict:
    """The PETSc options of the runs: MUMPS LU, the same SNES settings in every mode, and for the full
    Jacobian GMRES to a tight tolerance with the interface-ALE matrix factored as the preconditioner."""
    options = DEFAULT_DIRECT_SOLVER_OPTIONS | {
        "snes_rtol": snes_rtol,
        "snes_atol": snes_atol,
        "snes_max_it": snes_max_it,
    }
    if mode == "full":
        options |= {"ksp_type": "gmres", "ksp_rtol": 1e-12, "ksp_atol": 0.0, "ksp_max_it": 200}
        options |= {"ksp_gmres_restart": 200}
    return options


class PressureRecorder:
    """The ``jacobian_callback`` of the module docstring; ``records`` gets one dict per sample."""

    def __init__(self, records: list, sample_steps, at_iteration: int = 2):
        self.records = records
        self.sample_steps = set(sample_steps)
        self.at_iteration = at_iteration
        self.step = 0
        self.data = None

    def _norms(self, data, suffix: str) -> dict:
        forms = data["pressure_forms"]
        K_f = dolfinx.fem.petsc.create_matrix(forms["fluid" + suffix])
        dolfinx.fem.petsc.assemble_matrix(K_f, forms["fluid" + suffix], bcs=data["bcs"], diag=0.0)
        K_f.assemble()
        K_s = dolfinx.fem.petsc.create_matrix(forms["solid" + suffix])
        dolfinx.fem.petsc.assemble_matrix(K_s, forms["solid" + suffix], bcs=data["bcs"], diag=0.0)
        K_s.assemble()
        K_fS = K_f.matMult(data["S"])
        # The rows of v (K_f's other rows are not of the pressure term), and the columns of the interface
        # velocity dofs only
        rows = K_s.createVecLeft()
        rows.set(0.0)
        rows.array[: data["num_v"]] = 1.0
        mask = K_s.createVecRight()
        mask.set(0.0)
        first = mask.getOwnershipRange()[0]
        mask.array[data["v_int_rows"] - first] = 1.0
        K_s.diagonalScale(L=rows, R=mask)
        K_fS.diagonalScale(L=rows)
        total = K_s.duplicate(copy=True)
        total.axpy(1.0, K_fS, structure=DIFFERENT)
        result = {"fluid": K_fS.norm(), "solid": K_s.norm(), "sum": total.norm()}
        for obj in (K_f, K_s, K_fS, rows, mask, total):
            obj.destroy()
        return result

    def __call__(self, snes: PETSc.SNES, data: dict) -> None:
        self.data = data
        if snes.getIterationNumber() == 0:
            self.step += 1
        if self.step not in self.sample_steps or snes.getIterationNumber() != self.at_iteration:
            return
        record = {
            "step": self.step,
            "iteration": self.at_iteration,
            "state": self._norms(data, ""),
            "unit": self._norms(data, "_unit"),
            "C_S": data["CS"].norm(),
        }
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"pressure terms at step {self.step}: state fluid {record['state']['fluid']:.3e} solid "
                f"{record['state']['solid']:.3e} sum {record['state']['sum']:.3e}; unit fluid "
                f"{record['unit']['fluid']:.3e} sum {record['unit']['sum']:.3e}",
                flush=True,
            )
        self.records.append(record)

    def write_state(self, path: str) -> None:
        """Write v, p, p_S and u_s of the last step, with their dof coordinates, to ``path`` (one file per rank)."""
        arrays = {}
        for name in ("v", "p", "p_S", "u_s"):
            f = self.data["functions"][name]
            arrays[name] = f.x.array.copy()
            arrays[name + "_x"] = f.function_space.tabulate_dof_coordinates()
        np.savez(path, **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="data/meshes/fsi2/mesh_sec_solid2x.xdmf")
    parser.add_argument("--initial-state-dir", required=True)
    parser.add_argument("--mode", choices=list(MODES), default="J_0")
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--t0", type=float, default=15.0)
    parser.add_argument("--sample-steps", type=int, nargs="*", default=[70, 80, 90, 100])
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--outlet-pressure-ramp", type=float, default=None)
    parser.add_argument("--snes-rtol", type=float, default=1e-10)
    parser.add_argument("--snes-atol", type=float, default=0.0)
    parser.add_argument("--snes-max-it", type=int, default=40)
    parser.add_argument("--out", required=True, help="the JSON file of the records")
    args = parser.parse_args()

    stem = args.out.removesuffix(".json")
    records = []
    recorder = PressureRecorder(records, args.sample_steps)
    stats = solve(
        mesh_path=args.mesh,
        T=args.t0 + args.steps * args.dt,
        dt_val=args.dt,
        output_path=f"{stem}.vtkhdf",
        qoi_path=f"{stem}_qoi.txt",
        newton_log_path=f"{stem}_newton.txt",
        initial_state_dir=args.initial_state_dir,
        solver_options=solver_options(args.mode, args.snes_rtol, args.snes_max_it, args.snes_atol),
        gamma_reassemble=None,
        outlet_pressure=args.outlet_pressure,
        outlet_pressure_ramp=args.outlet_pressure_ramp,
        fluid_ale_jacobian=MODES[args.mode],
        jacobian_callback=recorder,
    )
    recorder.write_state(f"{stem}_state_rank{MPI.COMM_WORLD.rank}.npz")
    if MPI.COMM_WORLD.rank == 0:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "samples": records, "stats": stats}, f, indent=1)


if __name__ == "__main__":
    main()
