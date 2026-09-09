from xfsi_solver.solvers.fsi2_harmonic import solve


def test_fsi2_harmonic_solve(output_dirs):
    dt_val = 0.0025
    solve(
        mesh_path="data/meshes/fsi2/mesh.xdmf",
        # save_every=4 in the solver, so run enough steps to save at least 2 snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_harm.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harm_p.bp"),
        disp_path=str(output_dirs["qoi"] / "fsi2_harm_Adisp.txt"),
    )

    assert (output_dirs["qoi"] / "fsi2_harm_Adisp.txt").exists()
