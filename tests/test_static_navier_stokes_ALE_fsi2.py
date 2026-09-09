from xfsi_solver.component_solvers.static_navier_stokes_ALE_fsi2 import solve


def test_static_navier_stokes_ale_fsi2_solve(output_dirs):
    output_path = output_dirs["pv"] / "static_navier_stokes_ale_fsi2.bp"
    solve(
        mesh_path="data/meshes/fsi2/mesh_coarse.xdmf",
        t=2.0,
        bd_dset_path="data/fsi2_boundary/",
        output_path=str(output_path),
    )

    assert output_path.exists()
