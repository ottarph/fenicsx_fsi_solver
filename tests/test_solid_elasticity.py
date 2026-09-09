import pytest

from xfsi_solver.component_solvers.solid_elasticity import solve


@pytest.mark.parametrize(
    "mesh_path",
    ["data/meshes/fsi2/mesh_coarse.xdmf", "data/meshes/fsi2/mesh_quad_coarse.xdmf"],
    ids=["tri", "quad"],
)
def test_solid_elasticity_solve(output_dirs, mesh_path):
    dt_val = 0.0025
    output_path = output_dirs["pv"] / "solid_elasticity.bp"
    solve(
        mesh_path=mesh_path,
        T=3 * dt_val,
        dt_val=dt_val,
        output_path=str(output_path),
    )

    assert output_path.exists()
