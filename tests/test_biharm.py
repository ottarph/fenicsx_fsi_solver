from xfsi_solver.component_solvers.biharm import solve


def test_biharm_solve(output_dirs):
    output_path = output_dirs["pv"] / "biharm.bp"
    solve(
        N=8,
        output_path=str(output_path),
    )

    assert output_path.exists()
