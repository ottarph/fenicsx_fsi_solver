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
    """Fields u, p on a mesh and q on a cell submesh of it."""
    mesh = dfx.mesh.create_unit_square(comm, 6, 5, dfx.mesh.CellType.triangle)
    submesh, entity_map, _, _ = dfx.mesh.create_submesh(
        mesh, 2, dfx.mesh.locate_entities(mesh, 2, lambda x: x[0] <= 0.5 + 1e-12)
    )
    u = dfx.fem.Function(dfx.fem.functionspace(mesh, ("Lagrange", 2, (2,))), name="u")
    p = dfx.fem.Function(dfx.fem.functionspace(mesh, ("Lagrange", 1)), name="p")
    q = dfx.fem.Function(dfx.fem.functionspace(submesh, ("Lagrange", 2)), name="q")
    return mesh, [(submesh, entity_map)], [u, p, q]


def _fill(functions, t):
    u, p, q = functions
    u.interpolate(lambda x: np.vstack([np.sin(3 * x[0] + t) * x[1], np.cos(2 * x[1] - t)]))
    p.interpolate(lambda x: np.exp(x[0] * x[1]) + t)
    q.interpolate(lambda x: np.sin(5 * x[0] * x[1] - t))


def _assert_state(functions, t):
    expected = [dfx.fem.Function(f.function_space) for f in functions]
    _fill(expected, t)
    for f, f_ex in zip(functions, expected, strict=True):
        np.testing.assert_allclose(f.x.array, f_ex.x.array, atol=1e-14)


def _write_steps(directory, steps, clear=True):
    mesh, submeshes, functions = _setup()
    checkpointer = Checkpointer(directory, mesh, submeshes)
    if clear:
        checkpointer.clear()
    for step in steps:
        _fill(functions, (step + 1) * DT)
        checkpointer.write(functions, (step + 1) * DT, step, DT)


def _read(directory):
    mesh, submeshes, functions = _setup()
    t, step = Checkpointer(directory, mesh, submeshes).read(functions, DT)
    return t, step, functions


def test_reads_latest_checkpoint(tmp_path):
    _write_steps(tmp_path, [0, 1, 2])
    t, step, functions = _read(tmp_path)
    assert step == 2
    _assert_state(functions, t)


def test_skips_incomplete_checkpoint(tmp_path):
    """A run killed while writing a checkpoint restarts from the previous, complete one.

    Checkpointer alternates between checkpoint_0.bp and checkpoint_1.bp, and
    writes the attributes (t, step, dt, num_cells) last, as the marker that a
    file is complete. Here a crash halfway through a write is simulated by
    writing a file by hand without that marker.
    """
    # Complete checkpoints of step 0 (in checkpoint_0.bp, t = DT) and step 1 (in checkpoint_1.bp, t = 2 DT)
    _write_steps(tmp_path, [0, 1])

    # The next write, of step 2 at t = 3 DT, goes to checkpoint_0.bp. Simulate a run killed during it:
    # writing the mesh recreates the file (so the step-0 checkpoint is gone), then only the first
    # function (u) is written before the "crash", and the attributes never are. The values are
    # those of t = 3 DT, so the state check below would fail if they were read back.
    mesh, _, functions = _setup()
    _fill(functions, 3 * DT)
    io4dolfinx.write_mesh_input_order(tmp_path / "checkpoint_0.bp", mesh)
    io4dolfinx.write_function_on_input_mesh(tmp_path / "checkpoint_0.bp", functions[0], time=3 * DT)

    # A restart must ignore checkpoint_0.bp, which has no attributes, and read step 1 from
    # checkpoint_1.bp: every function (also those on the submesh) with its values at t = 2 DT
    t, step, functions = _read(tmp_path)
    assert step == 1
    _assert_state(functions, t)


def test_clear_removes_checkpoints_of_earlier_run(tmp_path):
    _write_steps(tmp_path, [0, 1, 2, 3])
    _write_steps(tmp_path, [0])
    assert _read(tmp_path)[1] == 0


def test_rejects_different_dt(tmp_path):
    _write_steps(tmp_path, [0])
    mesh, submeshes, functions = _setup()
    with pytest.raises(ValueError, match="dt"):
        Checkpointer(tmp_path, mesh, submeshes).read(functions, 2 * DT)


def test_missing_checkpoint(tmp_path):
    mesh, submeshes, functions = _setup()
    with pytest.raises(FileNotFoundError):
        Checkpointer(tmp_path, mesh, submeshes).read(functions, DT)


def test_rejects_function_on_unlisted_submesh(tmp_path):
    mesh, _, functions = _setup()
    with pytest.raises(ValueError, match="submeshes"):
        Checkpointer(tmp_path, mesh).write(functions, DT, 0, DT)


def test_rejects_facet_submesh(tmp_path):
    mesh, _, _ = _setup()
    mesh.topology.create_entities(1)
    facet_submesh, entity_map, _, _ = dfx.mesh.create_submesh(mesh, 1, np.arange(3, dtype=np.int32))
    with pytest.raises(NotImplementedError):
        Checkpointer(tmp_path, mesh, [(facet_submesh, entity_map)])


_MPI_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
from test_checkpoint import Checkpointer, DT, _fill, _setup
mesh, submeshes, functions = _setup()
checkpointer = Checkpointer({directory!r}, mesh, submeshes)
checkpointer.read(functions, DT)
for step in {steps!r}:
    _fill(functions, (step + 1) * DT)
    checkpointer.write(functions, (step + 1) * DT, step, DT)
"""


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1 or shutil.which("mpiexec") is None, reason="spawns its own MPI run")
def test_restart_on_different_number_of_ranks(tmp_path):
    """Restart a serial run on 2 ranks, then read its checkpoints back in serial."""
    _write_steps(tmp_path, [0, 1])
    script = _MPI_SCRIPT.format(tests=str(Path(__file__).parent), directory=str(tmp_path), steps=[2, 3])
    subprocess.run(["mpiexec", "-n", "2", sys.executable, "-c", script], check=True)
    t, step, functions = _read(tmp_path)
    assert step == 3
    _assert_state(functions, t)
