# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""What happens at the solid-fluid interface in the Newton iterations of
fsi2_biharmonic_fulljacobian_initialguess, with the full Jacobian or with the fluid ALE derivative
C dropped as fsi2_simplified drops it.

The ``jacobian_callback`` here, ``InterfaceRecorder``, records at every Newton iteration:

- the residual by row group: "v_int", the momentum rows at the interface dofs, "v_other", the other
  momentum rows, and "p", the mass rows;
- the Frobenius norm of each term of C (``FLUID_ALE_TERMS``: fluid inertia, mesh transport, fluid
  convection, fluid viscous, pressure, mass) in each block of a row group against a column group,
  "u_int", the displacement at the interface, or "u_f", that of the interior fluid mesh;
- the products C_term[rows, cols] du[cols] for the Newton step du the iteration takes, by row and
  column group: the part of the linearised change of the residual that comes from each term of C
  through the interface or the interior mesh, which J_0 leaves out of the Newton step;
- the sizes of the terms of the momentum equation, by row group, at the iterate the step reaches;
- a check that C holds all of the fluid ALE derivative of the Jacobian: the blocks (v_other, u_f),
  (p, u_int) and (p, u_f) of J - C, relative to those of C, which must be round-off.

With ``mode`` "full", Newton solves with the Jacobian. With "J_0", it solves with J - C, the Jacobian
of fsi2_simplified (with the solid stress derivative kept, and the biharmonic mesh motion), and with
"interface_ale", with J - C[:, u_f], which keeps the derivative with respect to the interface
displacement, as ``fsi2_simplified(interface_ale=True)`` does.

    mpirun -n 8 python -m xfsi_solver.scripts.interface_fsi2_biharmonic --mode J_0 \\
        --initial-state-dir output/snapshots/fsi2_simplified/t16.0000 --steps 100 \\
        --outlet-pressure 11200 --outlet-pressure-ramp 0.1 --out output/interface_J_0.json
"""

import argparse
import json

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import solve

MODES = ("full", "interface_ale", "J_0")
ROW_GROUPS = ("v_int", "v_other", "p")
COLUMN_GROUPS = ("u_int", "u_f")
SUBSET = PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN


def _norms(vec: PETSc.Vec, local_rows: np.ndarray, comm) -> dict:
    """The l2 and max norms of the entries of ``vec`` at the owned ``local_rows``. Collective."""
    values = vec.getArray(readonly=True)[local_rows]
    return {
        "l2": float(np.sqrt(comm.allreduce(float(values @ values), op=MPI.SUM))),
        "max": comm.allreduce(float(np.abs(values).max()) if values.size else 0.0, op=MPI.MAX),
    }


class InterfaceRecorder:
    """The ``jacobian_callback`` of the module docstring; ``records`` gets one dict per Newton
    iteration."""

    def __init__(self, mode: str, records: list):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.records = records
        self.ready = False
        self.pending = None  # (record, x, {name: C}) of the last Jacobian, for the product with its step

    def _setup(self, snes: PETSc.SNES, nest_view) -> None:
        J = nest_view.mat
        self.comm = J.getComm().tompi4py()
        self.first = J.getOwnershipRange()[0]
        groups = dict(zip(nest_view.names, (s.getIndices() for s in nest_view.index_sets), strict=True))
        v_int = np.sort(nest_view.data["v_rows"]["u_int"])
        rows = {
            "v_int": v_int,
            "v_other": np.setdiff1d(groups["v"], v_int),
            "p": groups["p"],
            "u_int": groups["u_int"],
            "u_f": groups["u_f"],
        }
        self.local = {name: (r - self.first).astype(np.int32) for name, r in rows.items()}
        self.index_sets = {
            name: PETSc.IS().createGeneral(r.astype(PETSc.IntType), comm=J.getComm()) for name, r in rows.items()
        }
        self.masks = {}
        for name in COLUMN_GROUPS:
            mask = J.createVecs()[0]
            mask.set(0.0)
            mask.array[self.local[name]] = 1.0
            self.masks[name] = mask
        self.term_vectors = nest_view.data["momentum_terms"]
        snes.setMonitor(self.monitor)
        self.ready = True

    def _block_norm(self, mat: PETSc.Mat, row: str, col: str) -> float:
        block = mat.createSubMatrix(self.index_sets[row], self.index_sets[col])
        norm = block.norm()
        block.destroy()
        return norm

    def __call__(self, snes: PETSc.SNES, nest_view) -> None:
        if not self.ready:
            self._setup(snes, nest_view)
        J = nest_view.mat
        C = nest_view.data["fluid_ale"].assemble()
        residual = snes.getFunction()[0]
        record = {
            "iteration": snes.getIterationNumber(),
            "residual": {g: _norms(residual, self.local[g], self.comm) for g in ROW_GROUPS},
            "blocks": {
                name: {f"{r},{c}": self._block_norm(mat, r, c) for r in ROW_GROUPS for c in COLUMN_GROUPS}
                for name, mat in C.items()
            },
        }

        total = None
        for mat in C.values():
            if total is None:
                total = mat.duplicate(copy=True)
            else:
                total.axpy(1.0, mat, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        remainder = J.duplicate(copy=True)
        remainder.axpy(-1.0, total, structure=SUBSET)
        record["check"] = {
            key: self._block_norm(remainder, *key.split(",")) / max(self._block_norm(total, *key.split(",")), 1e-300)
            for key in ("v_other,u_f", "p,u_int", "p,u_f", "v_int,u_f")
        }
        # What remains of the (v_int, u_int) block without C: the solid stress derivative.
        record["solid_v_int_u_int"] = self._block_norm(remainder, "v_int", "u_int")
        remainder.destroy()

        if self.mode == "J_0":
            J.axpy(-1.0, total, structure=SUBSET)
        elif self.mode == "interface_ale":
            masked = total.duplicate(copy=True)
            masked.diagonalScale(R=self.masks["u_f"])
            J.axpy(-1.0, masked, structure=SUBSET)
            masked.destroy()
        total.destroy()
        if self.mode != "full":
            nest_view.update()

        if self.pending is not None:
            for mat in self.pending[2].values():
                mat.destroy()
            self.pending[1].destroy()
        self.pending = (record, snes.getSolution().copy(), {n: m.duplicate(copy=True) for n, m in C.items()})
        self.records.append(record)

    def monitor(self, snes: PETSc.SNES, its: int, fnorm: float) -> None:
        """After the step from the iterate of the last Jacobian: the products of each term of C with
        the step, by row and column group, and the sizes of the momentum terms at the new iterate."""
        if its == 0 or self.pending is None:
            return
        record, x_prev, C = self.pending
        if "step" in record:
            return
        step = snes.getSolution().copy()
        step.axpy(-1.0, x_prev)
        products = {}
        total = {f"{r},{c}": 0.0 for r in ROW_GROUPS for c in COLUMN_GROUPS}
        part = step.duplicate()
        y = step.duplicate()
        summed = {c: step.duplicate() for c in COLUMN_GROUPS}
        for c in COLUMN_GROUPS:
            summed[c].set(0.0)
        for name, mat in C.items():
            products[name] = {}
            for c in COLUMN_GROUPS:
                part.pointwiseMult(step, self.masks[c])
                mat.mult(part, y)
                summed[c].axpy(1.0, y)
                for r in ROW_GROUPS:
                    products[name][f"{r},{c}"] = _norms(y, self.local[r], self.comm)["l2"]
        for c in COLUMN_GROUPS:
            for r in ROW_GROUPS:
                total[f"{r},{c}"] = _norms(summed[c], self.local[r], self.comm)["l2"]
        record["step"] = {
            "u_int": _norms(step, self.local["u_int"], self.comm),
            "u_f": _norms(step, self.local["u_f"], self.comm),
        }
        record["products"] = products
        record["products_total"] = total
        for vec in (step, part, y, *summed.values()):
            vec.destroy()

        # The momentum terms at the new iterate: the functions hold it, since SNES has evaluated its
        # residual before calling the monitors.
        record["terms_after"] = {
            name: {r: _norms(vec, self.local[r], self.comm) for r in ("v_int", "v_other")}
            for name, vec in self.term_vectors.assemble().items()
        }

    def destroy(self) -> None:
        if self.pending is not None:
            for mat in self.pending[2].values():
                mat.destroy()
            self.pending[1].destroy()
        if self.ready:
            for obj in (*self.index_sets.values(), *self.masks.values()):
                obj.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="data/meshes/fsi2/mesh_sec.xdmf")
    parser.add_argument("--initial-state-dir", required=True)
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--steps", type=int, default=100)
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
    recorder = InterfaceRecorder(args.mode, records)
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

    steps = []
    for record in records:
        if record["iteration"] == 0:
            steps.append([])
        steps[-1].append(record)
    if MPI.COMM_WORLD.rank == 0:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "steps": steps}, f, indent=1)


if __name__ == "__main__":
    main()
