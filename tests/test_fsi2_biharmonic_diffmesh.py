from restart_helpers import check_restart_rejects_different_dt, check_restart_reproduces_continuous_run

from xfsi_solver.solvers.fsi2_biharmonic_diffmesh import solve


def test_fsi2_biharmonic_diffmesh_solve(output_dirs):
    dt_val = 0.0025
    solve(
        mesh_path="data/meshes/fsi2/mesh_sec_coarse.xdmf",
        # save_every=4 in the solver, so run enough steps to save at least 2 snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_biharm_dm.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_biharm_p_dm.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_biharm_qoi.txt"),
    )

    assert (output_dirs["qoi"] / "fsi2_biharm_qoi.txt").exists()


def test_fsi2_biharmonic_diffmesh_restart_reproduces_continuous_run(output_dirs, tmp_path):
    # p and z live on the fluid submesh but are checkpointed on the full mesh
    elements = {
        "u": ("Lagrange", 2, (2,)),
        "v": ("Lagrange", 2, (2,)),
        "p": ("Lagrange", 1),
        "z": ("Lagrange", 2, (2,)),
    }
    check_restart_reproduces_continuous_run(
        solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", elements, output_dirs, tmp_path
    )


def test_fsi2_biharmonic_diffmesh_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)
