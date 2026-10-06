from pathlib import Path

import dolfinx
import io4dolfinx
import numpy as np
import pytest
from mpi4py import MPI
from restart_helpers import run_mpi_python

from xfsi_solver.tools.checkpoint import Checkpointer

DT = 0.1

# Meshes created by dolfinx and read from XDMF (as in the solvers) can be ordered and partitioned
# differently, so every test runs on both, and on triangles and quadrilaterals
FSI_MESHES = {"fsi_tri": "data/meshes/fsi2/mesh_sec_coarse.xdmf", "fsi_quad": "data/meshes/fsi2/mesh_quad_coarse.xdmf"}
FLUID, SOLID = 2, 1  # cell tags in the FSI meshes


@pytest.fixture(params=["unit_square", *FSI_MESHES])
def mesh_kind(request):
    return request.param


def _setup(mesh_kind, comm=MPI.COMM_WORLD):
    """A mesh with fields on it and on cell submeshes of it, as ``(mesh, submeshes, functions)``.

    The unit square has u, p on the mesh and q on its left half. An FSI mesh
    has fields like the solvers': u on the mesh, p on the fluid submesh and
    v_s on the solid submesh.
    """
    if mesh_kind == "unit_square":
        mesh = dolfinx.mesh.create_unit_square(comm, 6, 5, dolfinx.mesh.CellType.triangle)
        submesh, entity_map, _, _ = dolfinx.mesh.create_submesh(
            mesh, 2, dolfinx.mesh.locate_entities(mesh, 2, lambda x: x[0] <= 0.5 + 1e-12)
        )
        u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (2,))), name="u")
        p = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)), name="p")
        q = dolfinx.fem.Function(dolfinx.fem.functionspace(submesh, ("Lagrange", 2)), name="q")
        return mesh, [(submesh, entity_map)], [u, p, q]

    with dolfinx.io.XDMFFile(comm, FSI_MESHES[mesh_kind], "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name="Cell tags")
    fluid_mesh, fluid_map, _, _ = dolfinx.mesh.create_submesh(mesh, 2, cell_tags.find(FLUID))
    solid_mesh, solid_map, _, _ = dolfinx.mesh.create_submesh(mesh, 2, cell_tags.find(SOLID))
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (2,))), name="u")
    p = dolfinx.fem.Function(dolfinx.fem.functionspace(fluid_mesh, ("Lagrange", 1)), name="p")
    v_s = dolfinx.fem.Function(dolfinx.fem.functionspace(solid_mesh, ("Lagrange", 2, (2,))), name="v_s")
    return mesh, [(fluid_mesh, fluid_map), (solid_mesh, solid_map)], [u, p, v_s]


def _fill(functions, t):
    """Interpolate distinct, time-dependent, non-polynomial values into each function."""
    for i, f in enumerate(functions):
        if f.ufl_shape == (2,):
            f.interpolate(lambda x, i=i: np.vstack([np.sin(3 * x[0] + t + i) * x[1], np.cos(2 * x[1] - t + i)]))
        else:
            f.interpolate(lambda x, i=i: np.exp(x[0] * x[1]) * np.sin(5 * x[1] - t + i) + t)


def _assert_state(functions, t):
    expected = [dolfinx.fem.Function(f.function_space) for f in functions]
    _fill(expected, t)
    for f, f_ex in zip(functions, expected, strict=True):
        np.testing.assert_allclose(f.x.array, f_ex.x.array, atol=1e-14)


def _write_steps(mesh_kind, directory, steps, clear=True):
    mesh, submeshes, functions = _setup(mesh_kind)
    checkpointer = Checkpointer(directory, mesh, submeshes)
    if clear:
        checkpointer.clear()
    for step in steps:
        _fill(functions, (step + 1) * DT)
        checkpointer.write(functions, (step + 1) * DT, step, DT)


def _read(mesh_kind, directory):
    mesh, submeshes, functions = _setup(mesh_kind)
    t, step = Checkpointer(directory, mesh, submeshes).read(functions, DT)
    return t, step, functions


def test_reads_latest_checkpoint(mesh_kind, tmp_path):
    _write_steps(mesh_kind, tmp_path, [0, 1, 2])
    t, step, functions = _read(mesh_kind, tmp_path)
    assert step == 2
    _assert_state(functions, t)


def test_skips_incomplete_checkpoint(mesh_kind, tmp_path):
    """A run killed while writing a checkpoint restarts from the previous, complete one.

    Checkpointer alternates between checkpoint_0.bp and checkpoint_1.bp, and
    writes the attributes (t, step, dt, num_cells) last, as the marker that a
    file is complete. Here a crash halfway through a write is simulated by
    writing a file by hand without that marker.
    """
    # Complete checkpoints of step 0 (in checkpoint_0.bp, t = DT) and step 1 (in checkpoint_1.bp, t = 2 DT)
    _write_steps(mesh_kind, tmp_path, [0, 1])

    # The next write, of step 2 at t = 3 DT, goes to checkpoint_0.bp. Simulate a run killed during it:
    # writing the mesh recreates the file (so the step-0 checkpoint is gone), then only the first
    # function (u) is written before the "crash", and the attributes never are. The values are
    # those of t = 3 DT, so the state check below would fail if they were read back.
    mesh, _, functions = _setup(mesh_kind)
    _fill(functions, 3 * DT)
    io4dolfinx.write_mesh_input_order(tmp_path / "checkpoint_0.bp", mesh)
    io4dolfinx.write_function_on_input_mesh(tmp_path / "checkpoint_0.bp", functions[0], time=3 * DT)

    # A restart must ignore checkpoint_0.bp, which has no attributes, and read step 1 from
    # checkpoint_1.bp: every function (also those on the submesh) with its values at t = 2 DT
    t, step, functions = _read(mesh_kind, tmp_path)
    assert step == 1
    _assert_state(functions, t)


def test_clear_removes_checkpoints_of_earlier_run(mesh_kind, tmp_path):
    _write_steps(mesh_kind, tmp_path, [0, 1, 2, 3])
    _write_steps(mesh_kind, tmp_path, [0])
    assert _read(mesh_kind, tmp_path)[1] == 0


def test_rejects_different_dt(mesh_kind, tmp_path):
    _write_steps(mesh_kind, tmp_path, [0])
    mesh, submeshes, functions = _setup(mesh_kind)
    with pytest.raises(ValueError, match="dt"):
        Checkpointer(tmp_path, mesh, submeshes).read(functions, 2 * DT)


def test_missing_checkpoint(mesh_kind, tmp_path):
    mesh, submeshes, functions = _setup(mesh_kind)
    with pytest.raises(FileNotFoundError):
        Checkpointer(tmp_path, mesh, submeshes).read(functions, DT)


def test_rejects_function_on_unlisted_submesh(mesh_kind, tmp_path):
    mesh, _, functions = _setup(mesh_kind)
    with pytest.raises(ValueError, match="submeshes"):
        Checkpointer(tmp_path, mesh).write(functions, DT, 0, DT)


def test_rejects_facet_submesh(mesh_kind, tmp_path):
    mesh, _, _ = _setup(mesh_kind)
    mesh.topology.create_entities(1)
    facet_submesh, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, 1, np.arange(3, dtype=np.int32))
    with pytest.raises(NotImplementedError):
        Checkpointer(tmp_path, mesh, [(facet_submesh, entity_map)])


_MPI_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
from test_checkpoint import Checkpointer, DT, _assert_state, _fill, _setup
mesh, submeshes, functions = _setup({mesh_kind!r})
checkpointer = Checkpointer({directory!r}, mesh, submeshes)
t, step = checkpointer.read(functions, DT)
assert step == {expected_step}, step
_assert_state(functions, t)
for step in {steps!r}:
    _fill(functions, (step + 1) * DT)
    checkpointer.write(functions, (step + 1) * DT, step, DT)
"""


def test_restart_on_different_number_of_ranks(mesh_kind, tmp_path):
    """Restart a run repeatedly on other numbers of ranks: 1 -> 2 -> 3 -> 1.

    Each run reads the latest checkpoint, checks every field exactly, and
    writes two more steps; the 2- and 3-rank runs are ``mpiexec`` subprocesses.
    After the 2-rank run, both checkpoint files were written on 2 ranks, so
    the 3-rank run also reads a checkpoint partitioned differently from its own.
    """
    _write_steps(mesh_kind, tmp_path, [0, 1])
    last_step = 1
    for ranks in [2, 3]:
        steps = [last_step + 1, last_step + 2]
        script = _MPI_SCRIPT.format(
            tests=str(Path(__file__).parent),
            mesh_kind=mesh_kind,
            directory=str(tmp_path),
            expected_step=last_step,
            steps=steps,
        )
        run_mpi_python(script, ranks)
        last_step = steps[-1]
    t, step, functions = _read(mesh_kind, tmp_path)
    assert step == last_step == 5
    _assert_state(functions, t)
