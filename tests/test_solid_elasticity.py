from xfsi_solver.component_solvers.solid_elasticity import solve


def test_solid_elasticity_solve(output_dirs):
    dt_val = 0.0025
    output_path = output_dirs["pv"] / "solid_elasticity.bp"
    solve(
        mesh_path="data/meshes/fsi2/mesh_coarse.xdmf",
        T=3 * dt_val,
        dt_val=dt_val,
        output_path=str(output_path),
    )

    assert output_path.exists()
