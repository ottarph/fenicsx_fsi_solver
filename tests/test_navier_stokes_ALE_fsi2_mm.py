import pytest

from xfsi_solver.component_solvers.navier_stokes_ALE_fsi2_mm import solve


@pytest.mark.parametrize(
    "cell_type,mesh_path",
    [("tri", "data/meshes/fsi2/mesh_coarse.xdmf"), ("quad", "data/meshes/fsi2/mesh_quad_coarse.xdmf")],
    ids=["tri", "quad"],
)
def test_navier_stokes_ale_fsi2_mm_solve(output_dirs, cell_type, mesh_path):
    output_path = output_dirs["pv"] / f"navier_stokes_ale_fsi2_mm_{cell_type}.bp"
    solve(
        mesh_path=mesh_path,
        dt_val=0.0025,
        bd_dset_path="data/fsi2_boundary/",
        output_path=str(output_path),
        num_cycles=1,
        t0_val=0.0,
        max_steps=3,
    )

    assert output_path.exists()
