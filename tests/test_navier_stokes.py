from xfsi_solver.component_solvers.navier_stokes import solve


def test_navier_stokes_solve(output_dirs):
    dt_val = 0.02
    output_path = output_dirs["pv"] / "navier_stokes.bp"
    solve(
        mesh_path="data/meshes/fsi2/mesh.xdmf",
        T=3 * dt_val,
        dt_val=dt_val,
        output_path=str(output_path),
    )

    assert output_path.exists()
