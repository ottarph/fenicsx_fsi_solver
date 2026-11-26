# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""The slowest error modes of Newton's method with J_0 = J - C in fsi2_biharmonic_fulljacobian_initialguess,
where C is the fluid ALE derivative, as fsi2_simplified drops it.

Near the solution of a time step, Newton with J_0 iterates the error as e <- -J_0^-1 C e, so its
asymptotic contraction is the spectral radius of J_0^-1 C. ``ModeRecorder``, the ``jacobian_callback``, runs
Newton with the full Jacobian. At the steps ``sample_steps`` (counted from 1 in the run) it computes,
at the iterate of Newton iteration ``at_iteration``, the ``nev`` eigenvalues of largest magnitude of
J_0^-1 C and their eigenvectors x (complex, x = re + i im, of unit 2-norm), with
``xfsi_solver.tools.spectral.dominant_eigenpairs``. For each mode it records:

- the eigenvalue;
- the l2 norm of x on each group of dofs (v, p, u_s, z_int, u_int, z_f, u_f);
- the share of each term of C in C x: ||C_term x|| / ||C x||, by term of ``FLUID_ALE_TERMS``;
- the interface displacement du_I of the mode, along the arc length s of the beam's outline in the
  reference configuration (from the lower junction with the cylinder along the bottom, up the tip, and
  back along the top), as the normal component du_I . n and the magnitude |du_I|;
- its area index, |A| / (||du_I||_L2 sqrt(|Gamma|)), where A = the integral of du_I . n over the
  interface is the change of the beam's area that the mode makes, and |Gamma| is the length of the
  interface. It is 1 for a uniform normal displacement, and near 0 for a mode that moves the interface
  without changing the area;
- where |du_I| peaks: s and the distance to the nearest tip corner.

    mpirun -n 4 python -m xfsi_solver.scripts.modes_fsi2_biharmonic \\
        --initial-state-dir output/snapshots/fsi2_simplified/t16.0000 --steps 100 --sample-steps 70 80 90 100 \\
        --outlet-pressure 11200 --outlet-pressure-ramp 0.1 --out output/modes_p11200.json
"""

import argparse
import json

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import solve
from xfsi_solver.tools.spectral import dominant_eigenpairs

# The beam in the reference configuration: from the cylinder (centre (0.2, 0.2), radius 0.05) to x = 0.6,
# between y = 0.19 and 0.21.
X0, X1, Y0, Y1 = 0.2 + np.sqrt(0.05**2 - 0.01**2), 0.6, 0.19, 0.21
TIP_CORNERS = np.array([[X1, Y0], [X1, Y1]])
DIFFERENT = PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN


def arc_length(x: np.ndarray) -> np.ndarray:
    """The arc length along the beam's outline at the interface points ``x`` (n x 2): along the bottom
    from x = X0, up the tip, and back along the top."""
    tol = 1e-8
    s = np.empty(len(x))
    bottom = np.abs(x[:, 1] - Y0) < tol
    top = np.abs(x[:, 1] - Y1) < tol
    tip = ~bottom & ~top & (np.abs(x[:, 0] - X1) < tol)
    if not np.all(bottom | top | tip):
        raise ValueError("Interface points off the beam outline")
    length = X1 - X0
    s[bottom] = x[bottom, 0] - X0
    s[tip] = length + x[tip, 1] - Y0
    s[top] = length + (Y1 - Y0) + X1 - x[top, 0]
    return s


def interface_measures(points: np.ndarray, du: np.ndarray) -> dict:
    """The measures of the module docstring of a complex interface displacement ``du`` (n x 2) at the
    reference ``points`` (n x 2), all of the interface. The outline is traversed counterclockwise, so the
    outward normal of the solid on each segment is the tangent turned clockwise."""
    s = arc_length(points)
    order = np.argsort(s, kind="stable")
    s, points, du = s[order], points[order], du[order]
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    keep = lengths > 1e-12  # a dof can appear twice only by round-off; skip empty segments
    tangents = segments[keep] / lengths[keep, None]
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    mean = 0.5 * (du[:-1] + du[1:])[keep]
    area = np.sum(lengths[keep] * np.einsum("ij,ij->i", mean, normals))
    magnitude2 = np.sum(np.abs(du) ** 2, axis=1)
    l2 = np.sqrt(np.sum(lengths[keep] * 0.5 * (magnitude2[:-1] + magnitude2[1:])[keep]))
    total = lengths[keep].sum()
    # The normal component at the nodes, with the normal of the side each node lies on
    node_normals = np.zeros_like(points)
    node_normals[np.abs(points[:, 1] - Y0) < 1e-8] = (0.0, -1.0)
    node_normals[np.abs(points[:, 1] - Y1) < 1e-8] = (0.0, 1.0)
    on_tip = (np.abs(points[:, 0] - X1) < 1e-8) & ~np.any(node_normals, axis=1)
    node_normals[on_tip] = (1.0, 0.0)
    peak = int(np.argmax(magnitude2))
    return {
        "area": [float(area.real), float(area.imag)],
        "area_index": float(abs(area) / (l2 * np.sqrt(total))) if l2 > 0 else 0.0,
        "l2": float(l2),
        "peak_s": float(s[peak]),
        "peak_to_tip_corner": float(np.min(np.linalg.norm(TIP_CORNERS - points[peak], axis=1))),
        "peak_over_rms": float(np.sqrt(magnitude2[peak]) / (l2 / np.sqrt(total))) if l2 > 0 else 0.0,
        "s": s.tolist(),
        "normal": np.einsum("ij,ij->i", du, node_normals).real.tolist(),
        "normal_imag": np.einsum("ij,ij->i", du, node_normals).imag.tolist(),
        "magnitude": np.sqrt(magnitude2).tolist(),
    }


class ModeRecorder:
    """The ``jacobian_callback`` of the module docstring; ``records`` gets one dict per sample."""

    def __init__(self, records: list, sample_steps, at_iteration: int = 2, nev: int = 6, tol: float = 1e-6):
        self.records = records
        self.sample_steps = set(sample_steps)
        self.at_iteration = at_iteration
        self.nev = nev
        self.tol = tol
        self.step = 0

    def __call__(self, snes: PETSc.SNES, nest_view) -> None:
        if snes.getIterationNumber() == 0:
            self.step += 1
        if self.step not in self.sample_steps or snes.getIterationNumber() != self.at_iteration:
            return
        J = nest_view.mat
        comm = J.getComm().tompi4py()
        first = J.getOwnershipRange()[0]
        names = list(nest_view.names)
        local = {
            name: (s.getIndices() - first).astype(np.int32) for name, s in zip(names, nest_view.index_sets, strict=True)
        }

        C_terms = nest_view.data["fluid_ale"].assemble()
        C = None
        for mat in C_terms.values():
            if C is None:
                C = mat.duplicate(copy=True)
            else:
                C.axpy(1.0, mat, structure=DIFFERENT)
        J0 = J.duplicate(copy=True)
        J0.axpy(-1.0, C, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        values, vectors = dominant_eigenpairs(J0, C, nev=self.nev, tol=self.tol, options_prefix="modes_")

        points = nest_view.data["u_int_coordinates"][:, :2]
        record = {"step": self.step, "iteration": self.at_iteration, "modes": []}
        y = C.createVecLeft()
        for value, (re, im) in zip(values, vectors, strict=True):
            mode = {"eigenvalue": [float(value.real), float(value.imag)], "abs": float(abs(value))}
            a_re, a_im = re.getArray(readonly=True), im.getArray(readonly=True)
            mode["groups"] = {
                name: float(np.sqrt(comm.allreduce(float(a_re[idx] @ a_re[idx] + a_im[idx] @ a_im[idx]))))
                for name, idx in local.items()
            }

            # The share of each term of C in C x, with the real and imaginary parts summed in squares
            def norm_of(mat, parts=(re, im)):
                total = 0.0
                for part in parts:
                    mat.mult(part, y)
                    total += y.norm() ** 2
                return np.sqrt(total)

            whole = norm_of(C)
            mode["term_shares"] = {
                name: float(norm_of(mat) / whole) if whole > 0 else 0.0 for name, mat in C_terms.items()
            }
            # The interface displacement, gathered on rank 0
            rows = local["u_int"]
            du = (a_re[rows] + 1j * a_im[rows]).reshape(-1, 2)
            all_points = comm.gather(points, root=0)
            all_du = comm.gather(du, root=0)
            if comm.rank == 0:
                mode["interface"] = interface_measures(np.concatenate(all_points), np.concatenate(all_du))
            record["modes"].append(mode)
        for vec in (y, J0, C, *(v for pair in vectors for v in pair)):
            vec.destroy()
        if comm.rank == 0:
            print(f"modes at step {self.step}: " + ", ".join(f"{m['abs']:.3e}" for m in record["modes"]), flush=True)
        self.records.append(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="data/meshes/fsi2/mesh_sec.xdmf")
    parser.add_argument("--initial-state-dir", required=True)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--t0", type=float, default=16.0)
    parser.add_argument("--sample-steps", type=int, nargs="+", required=True)
    parser.add_argument("--nev", type=int, default=6)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--outlet-pressure-ramp", type=float, default=None)
    parser.add_argument("--out", required=True, help="the JSON file of the records")
    args = parser.parse_args()

    stem = args.out.removesuffix(".json")
    records = []
    recorder = ModeRecorder(records, args.sample_steps, nev=args.nev)
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
        snes_atol=0.0,
        snes_rtol=1e-10,
        snes_max_it=40,
    )
    if MPI.COMM_WORLD.rank == 0:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "samples": records}, f, indent=1)


if __name__ == "__main__":
    main()
