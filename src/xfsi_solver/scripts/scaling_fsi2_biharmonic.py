# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Newton for fsi2_biharmonic_fulljacobian_initialguess under a diagonal scaling of its equations and
unknowns, F_s(y) = D_r F(D_c y), and the stopping tests on the scaled residual.

Newton on F_s takes the step dy = -(D_r J D_c)^-1 D_r F, which is dx = D_c dy = -J^-1 F in the
unknowns of F: in exact arithmetic the iterates do not depend on the scaling, also with a Jacobian
that drops blocks, as fsi2_simplified does, since dropping a block commutes with a diagonal scaling.
``ScaledLU`` is a PETSc python preconditioner that takes this step: it factors D_r J D_c by MUMPS LU
and applies x -> D_c (D_r J D_c)^-1 D_r x. With ``ksp_type`` preonly, Newton then runs on F_s, seen
in the unknowns of F, so the iterates of a run with and without it differ only by the round-off and
the pivoting of the factorization of the scaled matrix.

The scalings (``scaling_vectors``):

- "physical": each equation divided by the size of its largest term, and each unknown by its
  reference scale, from ``dimensional_analysis_fsi2_biharmonic``: the rows of u by 200 on the fluid
  (mesh motion) and by 100 on the solid and interface (kinematic relation), the rows of v by 5e4 at
  the solid dofs, the interface included, and by 4e3 at the fluid dofs, those of p by 0.1 and those
  of z by 2; the unknowns u, v, p, z by 0.08 m, 1 m/s, 1e3 Pa and 200 1/m. The sizes are of the
  integrals over an area L^2, not of single rows, but the same mesh carries all equations, so they
  set the ratios between the equations.
- "random": every row and every column by its own factor, 10^s with s uniform in [-4, 4], from a
  fixed seed over the global numbering, so the same on the same number of ranks.

``ScalingRecorder``, the ``jacobian_callback``, installs ``ScaledLU`` (unless the scaling is "none"),
makes the Jacobian J - C of fsi2_simplified (``mode`` "J_0") or J - C[:, u_f] ("interface_ale") as
``interface_fsi2_biharmonic`` does, and as a SNES monitor records at every Newton iterate the l2
norm of the residual, unscaled and scaled by the physical D_r, by group of rows. At the end of each
Newton solve, it records the distance of every iterate to the last, unscaled and divided by the
physical D_c, by unknown. Run with a tight tolerance, the last iterate stands for the solution, and
``stopping_tests`` finds where each stopping test would have stopped, and what error that leaves.

    mpirun -n 4 python -m xfsi_solver.scripts.scaling_fsi2_biharmonic --mode J_0 --scaling physical \\
        --initial-state-dir output/snapshots/fsi2_simplified/t16.0000 --steps 40 \\
        --out output/scaling_J_0_physical.json
"""

import argparse
import json

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import solve

MODES = ("full", "interface_ale", "J_0")
SCALINGS = ("none", "physical", "random")

# The rows of the physical D_r: 1 / the size of the largest term of each equation.
ROW_SIZES = {"u_f": 200.0, "u_kin": 100.0, "v_solid": 5.0e4, "v_fluid": 4.0e3, "p": 0.1, "z": 2.0}
# The columns of the physical D_c: the reference scale of each unknown.
COLUMN_SIZES = {"u": 0.08, "v": 1.0, "p": 1.0e3, "z": 0.08 / 0.02**2}
SUBSET = PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN
# The MUMPS options of the solver, for the inner factorization of ``ScaledLU``.
MUMPS_OPTIONS = {
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 80,
    "mat_mumps_cntl_1": 1e-4,
}


class ScaledLU:
    """The python context of a PC that applies x -> D_c (D_r A D_c)^-1 D_r x, with A the operator of
    the PC, D_r = diag(``row``) and D_c = diag(``col``), by an inner KSP that factors D_r A D_c with
    the PETSc options of ``options_prefix``."""

    def __init__(self, row: PETSc.Vec, col: PETSc.Vec, options_prefix: str, options: dict):
        self.row = row
        self.col = col
        self.scaled = None
        self.ksp = PETSc.KSP().create(row.getComm())
        self.ksp.setOptionsPrefix(options_prefix)
        opts = PETSc.Options()
        for key, value in {"ksp_type": "preonly", **options}.items():
            opts[f"{options_prefix}{key}"] = value
        self.ksp.setFromOptions()
        self.work = row.duplicate()
        self.solution = row.duplicate()

    def setUp(self, pc: PETSc.PC) -> None:
        A, _ = pc.getOperators()
        if self.scaled is None:
            self.scaled = A.duplicate(copy=True)
        else:
            A.copy(self.scaled, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        self.scaled.diagonalScale(L=self.row, R=self.col)
        self.ksp.setOperators(self.scaled)
        self.ksp.setUp()

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        self.work.pointwiseMult(self.row, x)
        self.ksp.solve(self.work, self.solution)
        y.pointwiseMult(self.col, self.solution)

    def destroy(self, pc: PETSc.PC = None) -> None:
        # Called by PETSc when the PC is destroyed, and by ``ScalingRecorder.destroy``.
        if self.ksp is not None:
            self.ksp.reset()
            self.ksp.destroy()
            for obj in (self.work, self.solution, self.scaled):
                if obj is not None:
                    obj.destroy()
            self.ksp = None


def scaling_vectors(scaling: str, template: PETSc.Vec, rows: dict, first: int, seed: int = 20261125):
    """D_r and D_c of ``scaling`` ("physical" or "random"), as vectors in the layout of ``template``.
    ``rows`` has the owned rows, in the global numbering, of the row groups of ``ROW_SIZES`` and of the
    unknowns of ``COLUMN_SIZES``."""
    row, col = template.duplicate(), template.duplicate()
    if scaling == "physical":
        for name, size in ROW_SIZES.items():
            row.array[rows[name] - first] = 1.0 / size
        for name, size in COLUMN_SIZES.items():
            col.array[rows[name] - first] = size
    elif scaling == "random":
        n = template.getSize()
        exponents = np.random.default_rng(seed).uniform(-4.0, 4.0, size=(2, n))
        start, end = template.getOwnershipRange()
        row.array[:] = 10.0 ** exponents[0, start:end]
        col.array[:] = 10.0 ** exponents[1, start:end]
    else:
        raise ValueError(f"no scaling vectors for {scaling!r}")
    return row, col


class ScalingRecorder:
    """The ``jacobian_callback`` and SNES monitor of the module docstring. ``records`` gets one list
    per time step, of one dict per Newton iterate."""

    def __init__(self, mode: str, scaling: str, records: list, mumps_options: dict):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if scaling not in SCALINGS:
            raise ValueError(f"scaling must be one of {SCALINGS}, not {scaling!r}")
        self.mode = mode
        self.scaling = scaling
        self.records = records
        self.mumps_options = mumps_options
        self.ready = False
        self.context = None
        self.iterates = []  # copies of the iterates of the current Newton solve

    def _setup(self, snes: PETSc.SNES, nest_view) -> None:
        J = nest_view.mat
        # Its own communicator, which outlives the mesh, for the errors of the last solve after the run.
        self.comm = J.getComm().tompi4py().Dup()
        first = J.getOwnershipRange()[0]
        g = dict(zip(nest_view.names, (s.getIndices() for s in nest_view.index_sets), strict=True))
        v_solid = np.union1d(nest_view.data["v_rows"]["u_s"], nest_view.data["v_rows"]["u_int"])
        rows = {
            "u_f": g["u_f"],
            "u_kin": np.union1d(g["u_s"], g["u_int"]),
            "v_solid": v_solid,
            "v_fluid": np.setdiff1d(g["v"], v_solid),
            "p": g["p"],
            "z": np.union1d(g["z_int"], g["z_f"]),
            "u": np.concatenate([g["u_s"], g["u_int"], g["u_f"]]),
            "v": g["v"],
        }
        self.local = {name: (r - first).astype(np.int32) for name, r in rows.items()}
        template = J.createVecs()[0]
        # The physical scaling, for the scaled norms whatever the scaling of the solve.
        self.row_physical, self.col_physical = scaling_vectors("physical", template, rows, first)
        self.inverse_col_physical = self.col_physical.duplicate()
        self.inverse_col_physical.array[:] = 1.0 / self.col_physical.array
        if self.mode == "interface_ale":
            self.mask_u_f = template.duplicate()
            self.mask_u_f.set(0.0)
            self.mask_u_f.array[self.local["u_f"]] = 1.0
        if self.scaling != "none":
            row, col = (
                (self.row_physical.copy(), self.col_physical.copy())
                if self.scaling == "physical"
                else scaling_vectors(self.scaling, template, rows, first)
            )
            self.context = ScaledLU(row, col, "scaled_lu_", self.mumps_options)
            ksp = snes.getKSP()
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("python")
            pc.setPythonContext(self.context)
        template.destroy()
        # Installed at the first Jacobian, after iterate 0 of the first Newton solve, which is
        # therefore not recorded.
        snes.setMonitor(self.monitor)
        self.ready = True
        self.skip = True

    def _norms(self, vec, groups) -> dict:
        values = vec if isinstance(vec, np.ndarray) else vec.getArray(readonly=True)
        return {
            name: float(np.sqrt(self.comm.allreduce(float(values[self.local[name]] @ values[self.local[name]]))))
            for name in groups
        }

    def __call__(self, snes: PETSc.SNES, nest_view) -> None:
        if not self.ready:
            self._setup(snes, nest_view)
        if self.mode == "full":
            return
        J = nest_view.mat
        C = nest_view.data["fluid_ale"].assemble()
        total = None
        for mat in C.values():
            if total is None:
                total = mat.duplicate(copy=True)
            else:
                total.axpy(1.0, mat, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        if self.mode == "interface_ale":
            total.diagonalScale(R=self.mask_u_f)
        J.axpy(-1.0, total, structure=SUBSET)
        total.destroy()  # the matrices of C belong to the FluidALEDerivative, which reuses them

    def monitor(self, snes: PETSc.SNES, its: int, fnorm: float) -> None:
        if its == 0:
            self.skip = False
            self.finish_step()
            self.records.append([])
        if self.skip:
            return
        F = snes.getFunction()[0]
        scaled = F.duplicate()
        scaled.pointwiseMult(self.row_physical, F)
        rows = tuple(ROW_SIZES)
        self.records[-1].append(
            {"iteration": its, "residual": self._norms(F, rows), "scaled_residual": self._norms(scaled, rows)}
        )
        scaled.destroy()
        self.iterates.append(snes.getSolution().getArray(readonly=True).copy())

    def finish_step(self) -> None:
        """The distances of the iterates of the last Newton solve to its last iterate. Collective."""
        if not self.iterates:
            return
        last = self.iterates[-1]
        inverse_col = self.inverse_col_physical.getArray(readonly=True)
        columns = tuple(COLUMN_SIZES)
        for record, iterate in zip(self.records[-1], self.iterates, strict=True):
            difference = iterate - last
            record["error"] = self._norms(difference, columns)
            record["scaled_error"] = self._norms(inverse_col * difference, columns)
        self.iterates = []

    def destroy(self) -> None:
        self.finish_step()
        if self.ready:
            for obj in (self.row_physical, self.col_physical, self.inverse_col_physical):
                obj.destroy()
            if self.mode == "interface_ale":
                self.mask_u_f.destroy()
            self.comm.Free()


def stopping_tests(steps: list, tests: dict) -> dict:
    """Where each of ``tests`` stops the Newton solves of ``steps`` (the records of a
    ``ScalingRecorder``), and the error it leaves. ``tests`` maps a name to a function of the records
    of one step and an iterate index that says whether to stop there. Returns, per test, the mean
    number of iterations, and per unknown the median and largest scaled error at the stop."""
    result = {}
    for name, stop in tests.items():
        iterations, errors = [], {c: [] for c in COLUMN_SIZES}
        for records in steps:
            k = next((k for k in range(len(records)) if stop(records, k)), len(records) - 1)
            iterations.append(k)
            for c in COLUMN_SIZES:
                errors[c].append(records[k]["scaled_error"][c])
        result[name] = {
            "iterations": float(np.mean(iterations)),
            "max_iterations": int(max(iterations)),
            "error": {c: {"median": float(np.median(e)), "max": float(max(e))} for c, e in errors.items()},
        }
    return result


def total(norms: dict) -> float:
    return float(np.sqrt(sum(value**2 for value in norms.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="data/meshes/fsi2/mesh_sec.xdmf")
    parser.add_argument("--initial-state-dir", required=True)
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--scaling", choices=SCALINGS, default="none")
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--t0", type=float, default=16.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--outlet-pressure-ramp", type=float, default=None)
    parser.add_argument("--snes-atol", type=float, default=0.0)
    parser.add_argument("--snes-rtol", type=float, default=1e-10)
    parser.add_argument("--snes-max-it", type=int, default=40)
    parser.add_argument("--out", required=True, help="the JSON file of the records")
    args = parser.parse_args()

    stem = args.out.removesuffix(".json")
    records = []
    recorder = ScalingRecorder(args.mode, args.scaling, records, MUMPS_OPTIONS)
    solve(
        mesh_path=args.mesh,
        T=args.t0 + args.steps * args.dt,
        dt_val=args.dt,
        output_path=f"{stem}.bp",
        output_path_p=f"{stem}_p.bp",
        qoi_path=f"{stem}_qoi.txt",
        newton_log_path=f"{stem}_newton.txt",
        initial_state_dir=args.initial_state_dir,
        jacobian_callback=recorder,
        outlet_pressure=args.outlet_pressure,
        outlet_pressure_ramp=args.outlet_pressure_ramp,
        snes_atol=args.snes_atol,
        snes_rtol=args.snes_rtol,
        snes_max_it=args.snes_max_it,
    )
    recorder.destroy()
    if MPI.COMM_WORLD.rank == 0:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "steps": records}, f, indent=1)


if __name__ == "__main__":
    main()
