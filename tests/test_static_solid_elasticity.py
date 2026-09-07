from xfsi_solver.component_solvers.static_solid_elasticity import solve


def test_static_solid_elasticity_solve(output_dirs):
    output_path = output_dirs["pv"] / "static_solid_elasticity.bp"
    solve(
        mesh_path="data/meshes/fsi2/mesh.xdmf",
        output_path=str(output_path),
    )

    assert output_path.exists()
