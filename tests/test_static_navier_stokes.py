import pytest

from xfsi_solver.component_solvers.static_navier_stokes import solve


@pytest.mark.parametrize(
    "mesh_path",
    ["data/meshes/fsi2/mesh_coarse.xdmf", "data/meshes/fsi2/mesh_quad_coarse.xdmf"],
    ids=["tri", "quad"],
)
def test_static_navier_stokes_solve(output_dirs, mesh_path):
    output_path = output_dirs["pv"] / "static_navier_stokes.bp"
    solve(
        mesh_path=mesh_path,
        output_path=str(output_path),
        t=2.0,
    )

    assert output_path.exists()
