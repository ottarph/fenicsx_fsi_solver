import pytest

from xfsi_solver.component_solvers.navier_stokes_ALE_fsi2 import solve


@pytest.mark.parametrize(
    "mesh_path",
    ["data/meshes/fsi2/mesh_coarse.xdmf", "data/meshes/fsi2/mesh_quad_coarse.xdmf"],
    ids=["tri", "quad"],
)
def test_navier_stokes_ale_fsi2_solve(output_dirs, mesh_path):
    output_path = output_dirs["pv"] / "navier_stokes_ale_fsi2.bp"
    solve(
        mesh_path=mesh_path,
        dt_val=0.0025,
        bd_dset_path="data/fsi2_boundary/",
        output_path=str(output_path),
        num_cycles=1,
        max_steps=3,
    )

    assert output_path.exists()
