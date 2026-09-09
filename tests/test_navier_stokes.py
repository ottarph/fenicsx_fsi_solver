import pytest

from xfsi_solver.component_solvers.navier_stokes import solve


@pytest.mark.parametrize(
    "cell_type,mesh_path",
    [("tri", "data/meshes/fsi2/mesh_coarse.xdmf"), ("quad", "data/meshes/fsi2/mesh_quad_coarse.xdmf")],
    ids=["tri", "quad"],
)
def test_navier_stokes_solve(output_dirs, cell_type, mesh_path):
    dt_val = 0.02
    output_path = output_dirs["pv"] / f"navier_stokes_{cell_type}.bp"
    solve(
        mesh_path=mesh_path,
        T=3 * dt_val,
        dt_val=dt_val,
        output_path=str(output_path),
    )

    assert output_path.exists()
