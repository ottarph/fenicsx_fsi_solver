from restart_helpers import check_start_from_initial_state

from xfsi_solver.solvers.fsi2_biharmonic_mixed_incompressible import solve

VECTOR = ("Lagrange", 2, (2,))
# p and p_S live on the fluid and solid submeshes but are checkpointed on the full mesh
ELEMENTS = {"u": VECTOR, "v": VECTOR, "p": ("Lagrange", 1), "p_S": ("Lagrange", 1), "z": VECTOR}


def test_fsi2_biharmonic_mixed_incompressible_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve,
        "data/meshes/fsi2/mesh_sec_coarse.xdmf",
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        # The solid area change is round-off (about 1e-17) on these few steps, so only its size is compared
        qoi_column_rtol={"solid_area_change": 1.0},
        output_kwargs={"output_path_p_S": "_p_S.bp"},
        solve_kwargs={"direct_k_c_solve": True},
    )
