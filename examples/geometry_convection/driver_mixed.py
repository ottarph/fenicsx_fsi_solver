"""Run fsi2_biharmonic_mixed from the t = 16 snapshot on mesh_sec.
Usage: python driver_mixed.py <ref|new> <order or none> <gamma or none> <prefix> <steps> [kbc | kbc-noext | kbc-hold]

"ref" runs mixed_ref.py next to this script, the solver before these options, which is created by
    git show 3bef5c9:src/xfsi_solver/solvers/fsi2_biharmonic_mixed.py > mixed_ref.py
"new" runs the current solver; with no options it gives the same iterates as "ref".
Run from the repository root with an output prefix such as output/geometry_convection/mixed_new_gnone."""

import importlib.util
import sys
from pathlib import Path

which, order, gamma, out, steps = sys.argv[1:6]
if which == "ref":
    spec = importlib.util.spec_from_file_location("mixed_ref", Path(__file__).parent / "mixed_ref.py")
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
else:
    from xfsi_solver.solvers import fsi2_biharmonic_mixed as solver
dt = 0.0025
kwargs = {} if order == "none" else {"mesh_extrapolation_order": int(order)}
if sys.argv[6:7] in (["kbc"], ["kbc-noext"], ["kbc-hold"]):
    kwargs["kinematic_bc"] = True
    kwargs["kinematic_bc_extension"] = sys.argv[6] != "kbc-noext"
    kwargs["hold_dirichlet_values"] = sys.argv[6] == "kbc-hold"
solver.solve(
    mesh_path="data/meshes/fsi2/mesh_sec.xdmf",
    T=16.0 + int(steps) * dt,
    dt_val=dt,
    output_path=out + ".bp",
    output_path_p=out + "_p.bp",
    qoi_path=out + "_qoi.txt",
    newton_log_path=out + "_newton.txt",
    initial_state_dir="output/snapshots/fsi2_simplified/t16.0000",
    max_quadrature_degree=5,
    snes_converged_reason_on=False,
    gamma_reassemble=None if gamma == "none" else float(gamma),
    direct_k_c_solve=True,
    **kwargs,
)
