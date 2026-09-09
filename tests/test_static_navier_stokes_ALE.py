from xfsi_solver.component_solvers.static_navier_stokes_ALE import solve


def test_static_navier_stokes_ale_solve(output_dirs):
    output_path = output_dirs["pv"] / "static_navier_stokes_ale.bp"
    solve(
        mesh_path="data/meshes/fsi2/mesh_coarse.xdmf",
        output_path=str(output_path),
        t=2.0,
    )

    assert output_path.exists()
