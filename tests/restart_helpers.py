"""Shared check that a solver restarted from a checkpoint reproduces a continuous run."""

import dolfinx as dfx
import numpy as np
import pytest
from mpi4py import MPI

from xfsi_solver.tools.checkpoint import Checkpointer


def read_checkpoint_state(mesh_path, checkpoint_dir, dt_val, elements):
    """Read the latest checkpoint as ``{name: Function}`` on the full mesh.

    ``elements`` maps each checkpointed function name to its element, e.g.
    ``("Lagrange", 1)``. Functions on submeshes are stored on the full mesh,
    so they are read as full-mesh functions too.
    """
    with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
    functions = {name: dfx.fem.Function(dfx.fem.functionspace(mesh, element), name=name)
                 for name, element in elements.items()}
    t, step = Checkpointer(checkpoint_dir, mesh).read(list(functions.values()), dt_val)
    return t, step, functions


def check_restart_reproduces_continuous_run(solve, mesh_path, elements, output_dirs, tmp_path, dt_val=0.0025):
    """Compare a continuous 8-step run with a run stopped after 6 steps and restarted from step 4."""

    def run(name, n_steps, restart=False):
        solve(
            mesh_path=mesh_path,
            T=n_steps * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(output_dirs["qoi"] / f"{name}_qoi.txt"),
            checkpoint_dir=tmp_path / name,
            checkpoint_every=4,
            restart=restart,
        )

    run("continuous", 8)
    # Stop two steps after the checkpoint at step 4, so the restart has QoI rows to drop
    run("restarted", 6)
    run("restarted", 8, restart=True)

    qoi_continuous = np.loadtxt(output_dirs["qoi"] / "continuous_qoi.txt")
    qoi_restarted = np.loadtxt(output_dirs["qoi"] / "restarted_qoi.txt")
    assert qoi_continuous.shape[0] >= 8
    np.testing.assert_allclose(qoi_restarted, qoi_continuous, rtol=1e-10, atol=1e-14)
    assert (output_dirs["pv"] / "restarted_from_t0.0100.bp").exists()

    t_c, step_c, state_c = read_checkpoint_state(mesh_path, tmp_path / "continuous", dt_val, elements)
    t_r, step_r, state_r = read_checkpoint_state(mesh_path, tmp_path / "restarted", dt_val, elements)
    assert (t_r, step_r) == (t_c, step_c)
    for name in elements:
        np.testing.assert_allclose(state_r[name].x.array, state_c[name].x.array, rtol=1e-10, atol=1e-12)


def check_restart_rejects_different_dt(solve, mesh_path, output_dirs, tmp_path, dt_val=0.0025):
    kwargs = dict(
        mesh_path=mesh_path,
        output_path=str(output_dirs["pv"] / "dt.bp"),
        output_path_p=str(output_dirs["pv"] / "dt_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "dt_qoi.txt"),
        checkpoint_dir=tmp_path / "dt",
    )
    solve(T=dt_val, dt_val=dt_val, checkpoint_every=1, **kwargs)
    with pytest.raises(ValueError, match="dt"):
        solve(T=4 * dt_val, dt_val=2 * dt_val, restart=True, **kwargs)
