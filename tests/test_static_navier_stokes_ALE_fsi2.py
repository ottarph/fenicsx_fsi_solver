import pytest

from xfsi_solver.component_solvers.static_navier_stokes_ALE_fsi2 import solve


@pytest.mark.parametrize(
    "cell_type,mesh_path",
    [("tri", "data/meshes/fsi2/mesh_coarse.xdmf"), ("quad", "data/meshes/fsi2/mesh_quad_coarse.xdmf")],
    ids=["tri", "quad"],
)
def test_static_navier_stokes_ale_fsi2_solve(output_dirs, cell_type, mesh_path):
    output_path = output_dirs["pv"] / f"static_navier_stokes_ale_fsi2_{cell_type}.bp"
    solve(
        mesh_path=mesh_path,
        t=2.0,
        bd_dset_path="data/fsi2_boundary/",
        output_path=str(output_path),
    )

    assert output_path.exists()
