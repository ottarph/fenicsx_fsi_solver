import numpy as np
import pytest

from xfsi_solver.solvers.fsi2_harmonic import solve


@pytest.mark.parametrize(
    "cell_type,mesh_path",
    [("tri", "data/meshes/fsi2/mesh_coarse.xdmf"), ("quad", "data/meshes/fsi2/mesh_quad_coarse.xdmf")],
    ids=["tri", "quad"],
)
def test_fsi2_harmonic_solve(output_dirs, cell_type, mesh_path):
    dt_val = 0.0025
    solve(
        mesh_path=mesh_path,
        # save_every=4 in the solver, so run enough steps to save at least 2 snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"fsi2_harm_{cell_type}.bp"),
        output_path_p=str(output_dirs["pv"] / f"fsi2_harm_p_{cell_type}.bp"),
        qoi_path=str(output_dirs["qoi"] / f"fsi2_harm_qoi_{cell_type}.txt"),
    )

    assert (output_dirs["qoi"] / f"fsi2_harm_qoi_{cell_type}.txt").exists()


def _read_state(mesh_path, checkpoint_dir, dt_val):
    import dolfinx as dfx
    from mpi4py import MPI

    from xfsi_solver.tools.checkpoint import Checkpointer

    with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
    U = dfx.fem.functionspace(mesh, ("CG", 2, (2, )))
    P = dfx.fem.functionspace(mesh, ("CG", 1))
    u, v, p = dfx.fem.Function(U, name="u"), dfx.fem.Function(U, name="v"), dfx.fem.Function(P, name="p")
    t, step = Checkpointer(checkpoint_dir, mesh).read([u, v, p], dt_val)
    return t, step, u, v, p


def test_fsi2_harmonic_restart_reproduces_continuous_run(output_dirs, tmp_path):
    mesh_path = "data/meshes/fsi2/mesh_coarse.xdmf"
    dt_val = 0.0025

    def run(name, n_steps, checkpoint_dir, restart=False):
        solve(
            mesh_path=mesh_path,
            T=n_steps * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(output_dirs["qoi"] / f"{name}_qoi.txt"),
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=4,
            restart=restart,
        )

    run("continuous", 8, tmp_path / "continuous")
    # Stop two steps after the checkpoint at step 4, so the restart has QoI rows to drop
    run("restarted", 6, tmp_path / "restarted")
    run("restarted", 8, tmp_path / "restarted", restart=True)

    qoi_continuous = np.loadtxt(output_dirs["qoi"] / "continuous_qoi.txt")
    qoi_restarted = np.loadtxt(output_dirs["qoi"] / "restarted_qoi.txt")
    assert qoi_restarted.shape == (8, 5)
    np.testing.assert_allclose(qoi_restarted, qoi_continuous, rtol=1e-10, atol=1e-14)
    assert (output_dirs["pv"] / "restarted_from_t0.0100.bp").exists()

    t_c, step_c, *state_c = _read_state(mesh_path, tmp_path / "continuous", dt_val)
    t_r, step_r, *state_r = _read_state(mesh_path, tmp_path / "restarted", dt_val)
    assert t_r == t_c
    assert step_r == step_c == 7
    for f_r, f_c in zip(state_r, state_c, strict=True):
        np.testing.assert_allclose(f_r.x.array, f_c.x.array, rtol=1e-10, atol=1e-12)


def test_fsi2_harmonic_restart_rejects_different_dt(output_dirs, tmp_path):
    dt_val = 0.0025
    kwargs = dict(
        mesh_path="data/meshes/fsi2/mesh_coarse.xdmf",
        output_path=str(output_dirs["pv"] / "dt.bp"),
        output_path_p=str(output_dirs["pv"] / "dt_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "dt_qoi.txt"),
        checkpoint_dir=tmp_path / "dt",
    )
    solve(T=dt_val, dt_val=dt_val, checkpoint_every=1, **kwargs)
    with pytest.raises(ValueError, match="dt"):
        solve(T=4 * dt_val, dt_val=2 * dt_val, restart=True, **kwargs)
