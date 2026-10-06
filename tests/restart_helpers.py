"""Shared check that a solver restarted from a checkpoint reproduces a continuous run."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import dolfinx
import numpy as np
import pytest
from mpi4py import MPI

import xfsi_solver
from xfsi_solver.tools.checkpoint import Checkpointer

_MPI_SOLVE_SCRIPT = """
import importlib
importlib.import_module({module!r}).solve(**{kwargs!r})
"""


def run_mpi_python(script, ranks, timeout=None):
    """Run the Python source ``script`` on ``ranks`` MPI ranks with ``mpiexec``, in the current directory.

    Skips the calling test if ``mpiexec`` is missing or the test process
    itself runs on more than one rank. With ``timeout`` (in seconds), a run
    that takes longer, e.g. because it hangs, fails the test.
    """
    if MPI.COMM_WORLD.size > 1 or shutil.which("mpiexec") is None:
        pytest.skip("spawns its own MPI run, so needs mpiexec and a serial test process")
    # Run the same xfsi_solver source as this process, which may not be the installed one
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(Path(xfsi_solver.__file__).resolve().parents[1]), env.get("PYTHONPATH")] if p
    )
    subprocess.run(["mpiexec", "-n", str(ranks), sys.executable, "-c", script], check=True, env=env, timeout=timeout)


def _run(solve, ranks, **kwargs):
    """Call ``solve(**kwargs)`` in this process, or on ``ranks`` MPI ranks in an ``mpiexec`` subprocess."""
    if ranks == 1:
        solve(**kwargs)
        return
    kwargs = {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()}
    run_mpi_python(_MPI_SOLVE_SCRIPT.format(module=solve.__module__, kwargs=kwargs), ranks)


def _read_qoi(path):
    """QoI file as ``{column name: values}``, with the names from its header."""
    with open(path) as f:
        names = f.readline().lstrip("#").split()
    return dict(zip(names, np.loadtxt(path, ndmin=2).T, strict=True))


def read_checkpoint_state(mesh_path, checkpoint_dir, dt_val, elements):
    """Read the latest checkpoint as ``{name: Function}`` on the full mesh.

    ``elements`` maps each checkpointed function name to its element, e.g.
    ``("Lagrange", 1)``. Functions on submeshes are stored on the full mesh,
    so they are read as full-mesh functions too.
    """
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
    functions = {
        name: dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, element), name=name)
        for name, element in elements.items()
    }
    t, step = Checkpointer(checkpoint_dir, mesh).read(list(functions.values()), dt_val)
    return t, step, functions


def check_restart_reproduces_continuous_run(
    solve,
    mesh_path,
    elements,
    output_dirs,
    tmp_path,
    dt_val=0.0025,
    rtol=1e-10,
    field_rtol=None,
    qoi_rtol=1e-10,
    qoi_column_rtol=None,
    first_ranks=1,
    restart_ranks=1,
):
    """Compare a continuous 8-step run with a run stopped after 6 steps and restarted from step 4.

    The continuous run is serial; the stopped run and the restart run on
    ``first_ranks`` and ``restart_ranks`` MPI ranks. Each final field must
    agree to ``rtol`` (or ``field_rtol[name]``), and each QoI column to
    ``qoi_rtol`` (or ``qoi_column_rtol[name]``), relative to its largest
    absolute value in the continuous run.
    """

    def run(name, n_steps, restart=False, ranks=1):
        _run(
            solve,
            ranks,
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
    run("restarted", 6, ranks=first_ranks)
    run("restarted", 8, restart=True, ranks=restart_ranks)

    qoi_continuous = _read_qoi(output_dirs["qoi"] / "continuous_qoi.txt")
    qoi_restarted = _read_qoi(output_dirs["qoi"] / "restarted_qoi.txt")
    assert len(qoi_continuous["t"]) >= 8
    assert qoi_restarted.keys() == qoi_continuous.keys()
    for name, values in qoi_continuous.items():
        tol = (qoi_column_rtol or {}).get(name, qoi_rtol) * np.max(np.abs(values))
        np.testing.assert_allclose(qoi_restarted[name], values, rtol=0, atol=tol, err_msg=f"QoI {name}")
    assert (output_dirs["pv"] / "restarted_from_t0.0100.bp").exists()

    t_c, step_c, state_c = read_checkpoint_state(mesh_path, tmp_path / "continuous", dt_val, elements)
    t_r, step_r, state_r = read_checkpoint_state(mesh_path, tmp_path / "restarted", dt_val, elements)
    assert (t_r, step_r) == (t_c, step_c)
    for name in elements:
        tol = (field_rtol or {}).get(name, rtol) * np.max(np.abs(state_c[name].x.array))
        np.testing.assert_allclose(state_r[name].x.array, state_c[name].x.array, rtol=0, atol=tol, err_msg=name)


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
