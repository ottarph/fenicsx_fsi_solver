import shutil
import subprocess
import sys
from pathlib import Path

import dolfinx as dfx
import io4dolfinx
import numpy as np
import pytest
from mpi4py import MPI

from xfsi_solver.tools.checkpoint import Checkpointer

DT = 0.1


def _setup(comm=MPI.COMM_WORLD):
    mesh = dfx.mesh.create_unit_square(comm, 6, 5, dfx.mesh.CellType.triangle)
    u = dfx.fem.Function(dfx.fem.functionspace(mesh, ("Lagrange", 2, (2, ))), name="u")
    p = dfx.fem.Function(dfx.fem.functionspace(mesh, ("Lagrange", 1)), name="p")
    return mesh, u, p


def _fill(u, p, t):
    u.interpolate(lambda x: np.vstack([np.sin(3 * x[0] + t) * x[1], np.cos(2 * x[1] - t)]))
    p.interpolate(lambda x: np.exp(x[0] * x[1]) + t)


def _assert_state(u, p, t):
    u_ex, p_ex = dfx.fem.Function(u.function_space), dfx.fem.Function(p.function_space)
    _fill(u_ex, p_ex, t)
    np.testing.assert_allclose(u.x.array, u_ex.x.array, atol=1e-14)
    np.testing.assert_allclose(p.x.array, p_ex.x.array, atol=1e-14)


def _write_steps(directory, steps, clear=True):
    mesh, u, p = _setup()
    checkpointer = Checkpointer(directory, mesh)
    if clear:
        checkpointer.clear()
    for step in steps:
        _fill(u, p, (step + 1) * DT)
        checkpointer.write([u, p], (step + 1) * DT, step, DT)


def _read(directory):
    mesh, u, p = _setup()
    t, step = Checkpointer(directory, mesh).read([u, p], DT)
    return t, step, u, p


def test_reads_latest_checkpoint(tmp_path):
    _write_steps(tmp_path, [0, 1, 2])
    t, step, u, p = _read(tmp_path)
    assert step == 2
    _assert_state(u, p, t)


def test_skips_incomplete_checkpoint(tmp_path):
    _write_steps(tmp_path, [0, 1])
    # A run killed while writing step 2: functions written into the next file, but no attributes
    mesh, u, p = _setup()
    _fill(u, p, 3 * DT)
    io4dolfinx.write_mesh_input_order(tmp_path / "checkpoint_0.bp", mesh)
    io4dolfinx.write_function_on_input_mesh(tmp_path / "checkpoint_0.bp", u, time=3 * DT)
    t, step, u, p = _read(tmp_path)
    assert step == 1
    _assert_state(u, p, t)


def test_clear_removes_checkpoints_of_earlier_run(tmp_path):
    _write_steps(tmp_path, [0, 1, 2, 3])
    _write_steps(tmp_path, [0])
    assert _read(tmp_path)[1] == 0


def test_rejects_different_dt(tmp_path):
    _write_steps(tmp_path, [0])
    mesh, u, p = _setup()
    with pytest.raises(ValueError, match="dt"):
        Checkpointer(tmp_path, mesh).read([u, p], 2 * DT)


def test_missing_checkpoint(tmp_path):
    mesh, u, p = _setup()
    with pytest.raises(FileNotFoundError):
        Checkpointer(tmp_path, mesh).read([u, p], DT)


_MPI_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
from test_checkpoint import Checkpointer, DT, _fill, _setup
mesh, u, p = _setup()
checkpointer = Checkpointer({directory!r}, mesh)
checkpointer.read([u, p], DT)
for step in {steps!r}:
    _fill(u, p, (step + 1) * DT)
    checkpointer.write([u, p], (step + 1) * DT, step, DT)
"""


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1 or shutil.which("mpiexec") is None, reason="spawns its own MPI run")
def test_restart_on_different_number_of_ranks(tmp_path):
    """Restart a serial run on 2 ranks, then read its checkpoints back in serial."""
    _write_steps(tmp_path, [0, 1])
    script = _MPI_SCRIPT.format(tests=str(Path(__file__).parent), directory=str(tmp_path),
                                steps=[2, 3])
    subprocess.run(["mpiexec", "-n", "2", sys.executable, "-c", script], check=True)
    t, step, u, p = _read(tmp_path)
    assert step == 3
    _assert_state(u, p, t)
