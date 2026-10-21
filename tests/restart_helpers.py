"""Shared checks of solver restarts from checkpoints, and of runs started from another run's checkpoint."""

import hashlib
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
    solve_kwargs=None,
):
    """Compare a continuous 8-step run with a run stopped after 6 steps and restarted from step 4.

    The continuous run is serial; the stopped run and the restart run on
    ``first_ranks`` and ``restart_ranks`` MPI ranks. Each final field must
    agree to ``rtol`` (or ``field_rtol[name]``), and each QoI column to
    ``qoi_rtol`` (or ``qoi_column_rtol[name]``), relative to its largest
    absolute value in the continuous run. ``solve_kwargs`` are passed on to
    every ``solve`` call.
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
            vtx_save_every=4,
            restart=restart,
            **(solve_kwargs or {}),
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


def check_restart_rejects_T_not_after_checkpoint(solve, mesh_path, output_dirs, tmp_path, dt_val=0.0025):
    kwargs = dict(
        mesh_path=mesh_path,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "T.bp"),
        output_path_p=str(output_dirs["pv"] / "T_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "T_qoi.txt"),
        checkpoint_dir=tmp_path / "T",
    )
    solve(T=2 * dt_val, checkpoint_every=1, **kwargs)
    for T in (dt_val, 2 * dt_val):
        with pytest.raises(ValueError, match="no time step is left"):
            solve(T=T, restart=True, **kwargs)


def _directory_digest(directory):
    """``{relative path: sha256}`` of every file under ``directory``."""
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def check_start_from_initial_state(
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
    output_kwargs=None,
    solve_kwargs=None,
):
    """Start runs with time step ``dt_val`` from the checkpoint of a run with time step ``2 * dt_val``.

    A source run takes 2 steps of ``2 * dt_val``, to t0 = 4 dt_val. From its checkpoint:

    - a new run takes 2 steps of ``dt_val``: its QoI rows are at t0, t0 + dt_val and
      t0 + 2 dt_val, the first one equals the source's last, its checkpoints store t0, and
      the source's checkpoint directory is unchanged;
    - another new run stops after 1 step and is restarted: it reproduces the first new run,
      to ``rtol`` (or ``field_rtol[name]``) in the fields and ``qoi_rtol`` (or
      ``qoi_column_rtol[name]``) in the QoIs, relative to their largest absolute values.

    ``output_kwargs`` maps the names of further output path arguments of ``solve`` (e.g.
    ``output_path_p_S``) to file suffixes. ``solve_kwargs`` are passed on to every call.
    """

    def run(name, T, dt, **kwargs):
        solve(
            mesh_path=mesh_path,
            T=T,
            dt_val=dt,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(output_dirs["qoi"] / f"{name}_qoi.txt"),
            checkpoint_dir=tmp_path / name,
            checkpoint_every=1,
            **{key: str(output_dirs["pv"] / f"{name}{suffix}") for key, suffix in (output_kwargs or {}).items()},
            **kwargs,
            **(solve_kwargs or {}),
        )

    t0 = 4 * dt_val
    run("source", t0, 2 * dt_val)
    source_digest = _directory_digest(tmp_path / "source")

    run("started", t0 + 2 * dt_val, dt_val, initial_state_dir=tmp_path / "source")
    run("resumed", t0 + dt_val, dt_val, initial_state_dir=tmp_path / "source")
    run("resumed", t0 + 2 * dt_val, dt_val, restart=True)

    assert _directory_digest(tmp_path / "source") == source_digest

    qoi_source = _read_qoi(output_dirs["qoi"] / "source_qoi.txt")
    qoi_started = _read_qoi(output_dirs["qoi"] / "started_qoi.txt")
    qoi_resumed = _read_qoi(output_dirs["qoi"] / "resumed_qoi.txt")
    np.testing.assert_allclose(qoi_started["t"], t0 + dt_val * np.arange(3), rtol=1e-12)
    assert qoi_resumed.keys() == qoi_started.keys() == qoi_source.keys()
    for name, values in qoi_started.items():
        tol = (qoi_column_rtol or {}).get(name, qoi_rtol) * np.max(np.abs(values))
        np.testing.assert_allclose(values[0], qoi_source[name][-1], rtol=0, atol=tol, err_msg=f"first QoI {name}")
        np.testing.assert_allclose(qoi_resumed[name], values, rtol=0, atol=tol, err_msg=f"QoI {name}")

    t_s, step_s, state_s = read_checkpoint_state(mesh_path, tmp_path / "started", dt_val, elements)
    t_r, step_r, state_r = read_checkpoint_state(mesh_path, tmp_path / "resumed", dt_val, elements)
    assert step_s == step_r == 2
    np.testing.assert_allclose([t_s, t_r], t0 + 2 * dt_val, rtol=1e-12)
    for name in elements:
        tol = (field_rtol or {}).get(name, rtol) * np.max(np.abs(state_s[name].x.array))
        np.testing.assert_allclose(state_r[name].x.array, state_s[name].x.array, rtol=0, atol=tol, err_msg=name)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
    for name in ("started", "resumed"):
        np.testing.assert_allclose(Checkpointer(tmp_path / name, mesh).latest_times()[1], t0, rtol=1e-12)


def check_initial_state_rejects_T_not_after_it(solve, mesh_path, output_dirs, tmp_path, dt_val=0.0025):
    kwargs = dict(
        mesh_path=mesh_path,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "T.bp"),
        output_path_p=str(output_dirs["pv"] / "T_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "T_qoi.txt"),
    )
    solve(T=2 * dt_val, checkpoint_dir=tmp_path / "source", checkpoint_every=1, **kwargs)
    with pytest.raises(ValueError, match="no time step is left"):
        solve(T=2 * dt_val, initial_state_dir=tmp_path / "source", **kwargs)
