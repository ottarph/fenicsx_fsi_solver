import pytest

from xfsi_solver.component_solvers.static_solid_elasticity import solve


@pytest.mark.parametrize(
    "mesh_path",
    ["data/meshes/fsi2/mesh_coarse.xdmf", "data/meshes/fsi2/mesh_quad_coarse.xdmf"],
    ids=["tri", "quad"],
)
def test_static_solid_elasticity_solve(output_dirs, mesh_path):
    output_path = output_dirs["pv"] / "static_solid_elasticity.bp"
    solve(
        mesh_path=mesh_path,
        output_path=str(output_path),
    )

    assert output_path.exists()
