from restart_helpers import check_restart_rejects_different_dt, check_restart_reproduces_continuous_run

from xfsi_solver.solvers.fsi2_harmonic_lagrange import solve


def test_fsi2_harmonic_lagrange_solve(output_dirs):
    dt_val = 0.0025
    solve(
        mesh_path="data/meshes/fsi2/mesh_sec_coarse.xdmf",
        # save_every=8 in the solver, so run enough steps to save at least 2 snapshots
        T=10 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_harm_lg.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harm_lg_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_harm_lg_qoi.txt"),
    )

    assert (output_dirs["qoi"] / "fsi2_harm_lg_qoi.txt").exists()


def test_fsi2_harmonic_lagrange_restart_reproduces_continuous_run(output_dirs, tmp_path):
    # All fields live on the fluid or solid submesh but are checkpointed on the full mesh. The
    # interface Lagrange multipliers are not checkpointed (see the solver), and restarting them from
    # zero changes u_f by ~1e-6 and the other fields by < 1e-9 relative. u_f only needs to give a
    # non-degenerate ALE map in the interior, and its interface change is far below the u_f - u_s gap
    vector = ("Lagrange", 2, (2, ))
    elements = {"u_f": vector, "v_f": vector, "u_s": vector, "v_s": vector, "p": ("Lagrange", 1)}
    check_restart_reproduces_continuous_run(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", elements, output_dirs,
                                            tmp_path, rtol=1e-8, field_rtol={"u_f": 1e-5})


def test_fsi2_harmonic_lagrange_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)
