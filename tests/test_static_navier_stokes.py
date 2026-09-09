import pytest

from xfsi_solver.component_solvers.static_navier_stokes import solve


@pytest.mark.parametrize(
    "cell_type,mesh_path",
    [("tri", "data/meshes/fsi2/mesh_coarse.xdmf"), ("quad", "data/meshes/fsi2/mesh_quad_coarse.xdmf")],
    ids=["tri", "quad"],
)
def test_static_navier_stokes_solve(output_dirs, cell_type, mesh_path):
    output_path = output_dirs["pv"] / f"static_navier_stokes_{cell_type}.bp"
    solve(
        mesh_path=mesh_path,
        output_path=str(output_path),
        t=2.0,
    )

    assert output_path.exists()
