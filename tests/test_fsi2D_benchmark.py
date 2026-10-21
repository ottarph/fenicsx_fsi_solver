from restart_helpers import check_start_from_initial_state

from xfsi_solver.component_benchmarks.benchmark_parameters import FSI_parameters
from xfsi_solver.component_benchmarks.fsi2D_benchmark import solve

VECTOR = ("Lagrange", 2, (2,))
# p lives on the fluid submesh but is checkpointed on the full mesh
ELEMENTS = {"u": VECTOR, "v": VECTOR, "p": ("Lagrange", 1), "z": VECTOR}


def test_fsi2D_benchmark_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve,
        "data/meshes/fsi2/mesh_sec_coarse.xdmf",
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        solve_kwargs={"fsi_parameters": FSI_parameters["FSI2"], "solver_prefix": "fsi2_", "direct_k_c_solve": True},
    )
