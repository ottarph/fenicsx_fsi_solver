from xfsi_solver.solvers.fsi2_harmonic_diffmesh import solve


def test_fsi2_harmonic_diffmesh_solve(output_dirs):
    dt_val = 0.0025
    solve(
        # smaller stand-in for the default "mesh_quad_fine_sec.xdmf" mesh
        mesh_path="data/meshes/fsi2/mesh_sec_coarse.xdmf",
        # save_every=4 in the solver, so run enough steps to save at least 2 snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_harm_dm.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harm_p_dm.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_harm_dm_qoi.txt"),
    )

    assert (output_dirs["qoi"] / "fsi2_harm_dm_qoi.txt").exists()
