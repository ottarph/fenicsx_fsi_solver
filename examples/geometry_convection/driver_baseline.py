"""Baseline: fsi2_simplified from the t = 16 snapshot on mesh_sec, Jacobian on the iterate mesh.
Usage: python driver_baseline.py <gamma_reassemble or none> <output prefix>
Run from the repository root with an output prefix such as output/geometry_convection/<run>."""

import json
import sys

from mpi4py import MPI

from xfsi_solver.solvers import fsi2_simplified as solver

gamma, out = sys.argv[1], sys.argv[2]
atol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-7  # snes_atol = snes_rtol; default 1e-7, rtol 1e-12
dt = 0.0025
res = solver.solve(
    mesh_path="data/meshes/fsi2/mesh_sec.xdmf",
    T=16.0 + 400 * dt,
    dt_val=dt,
    output_path=out + ".bp",
    output_path_p=out + "_p.bp",
    qoi_path=out + "_qoi.txt",
    newton_log_path=out + "_newton.txt",
    initial_state_dir="output/snapshots/fsi2_simplified/t16.0000",
    max_quadrature_degree=5,
    snes_converged_reason_on=False,
    gamma_reassemble=None if gamma == "none" else float(gamma),
    solver_options=solver.DEFAULT_DIRECT_SOLVER_OPTIONS
    | {"snes_max_it": 100, "snes_atol": atol, "snes_rtol": atol if len(sys.argv) > 3 else 1e-12},
)
if MPI.COMM_WORLD.rank == 0:
    json.dump(res, open(out + "_res.json", "w"))
