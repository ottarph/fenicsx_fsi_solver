from pathlib import Path

import dolfinx
import io4dolfinx
import numpy as np
import pytest
from mpi4py import MPI
from restart_helpers import run_mpi_python

from xfsi_solver.tools.checkpoint import Checkpointer, check_start_options, load_start_state, restart_output_path

DT = 0.1

# Meshes created by dolfinx and read from XDMF (as in the solvers) can be ordered and partitioned
# differently, so every test runs on both, and on triangles and quadrilaterals
FSI_MESHES = {"fsi_tri": "data/meshes/fsi2/mesh_sec_coarse.xdmf", "fsi_quad": "data/meshes/fsi2/mesh_quad_coarse.xdmf"}
FSI3D_MESH = "data/meshes/fsi3d/mesh_sec_coarse.xdmf"
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


def test_reads_any_dt_without_dt(mesh_kind, tmp_path):
    """With dt=None, as for the initial state of a new run, a checkpoint with any time step is read."""
    _write_steps(mesh_kind, tmp_path, [0, 1])
    mesh, submeshes, functions = _setup(mesh_kind)
    t, step = Checkpointer(tmp_path, mesh, submeshes).read(functions, None)
    assert step == 1
    _assert_state(functions, t)


def test_latest_times(mesh_kind, tmp_path):
    mesh, submeshes, functions = _setup(mesh_kind)
    checkpointer = Checkpointer(tmp_path, mesh, submeshes)
    _fill(functions, 0.5 + DT)
    checkpointer.write(functions, 0.5 + DT, 0, DT, 0.5)
    _fill(functions, 0.5 + 2 * DT)
    checkpointer.write(functions, 0.5 + 2 * DT, 1, DT, 0.5)
    assert Checkpointer(tmp_path, mesh, submeshes).latest_times() == (0.5 + 2 * DT, 0.5)


def test_latest_times_without_stored_t0(mesh_kind, tmp_path):
    """A checkpoint written before t0 was stored is from a run that started at t0 = 0."""
    mesh, _, functions = _setup(mesh_kind)
    u = functions[0]
    _fill([u], DT)
    path = tmp_path / "checkpoint_0.bp"
    io4dolfinx.write_mesh_input_order(path, mesh)
    io4dolfinx.write_function_on_input_mesh(path, u, time=DT)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    attrs = {
        "t": np.array([DT]),
        "step": np.array([0], dtype=np.int64),
        "dt": np.array([DT]),
        "num_cells": np.array([num_cells], dtype=np.int64),
    }
    io4dolfinx.write_attributes(path, mesh.comm, "checkpoint", attrs)
    assert Checkpointer(tmp_path, mesh).latest_times() == (DT, 0.0)


@pytest.mark.parametrize(
    "restart,checkpoint_dir,checkpoint_every,initial_state_dir,match",
    [
        (True, None, None, None, "requires checkpoint_dir"),
        (False, None, 1, None, "requires checkpoint_dir"),
        (True, "run", None, "other", "initial_state_dir"),
        (False, "run", 1, "run", "must differ"),
    ],
    ids=["restart_without_dir", "every_without_dir", "restart_and_initial_state", "initial_state_is_checkpoint_dir"],
)
def test_check_start_options_rejects(tmp_path, restart, checkpoint_dir, checkpoint_every, initial_state_dir, match):
    checkpoint_dir = None if checkpoint_dir is None else tmp_path / checkpoint_dir
    initial_state_dir = None if initial_state_dir is None else tmp_path / initial_state_dir
    with pytest.raises(ValueError, match=match):
        check_start_options(restart, checkpoint_dir, checkpoint_every, initial_state_dir)


def test_check_start_options_accepts(tmp_path):
    check_start_options(False, None, None, None)
    check_start_options(True, tmp_path / "run", 1, None)
    check_start_options(False, tmp_path / "run", 1, tmp_path / "other")
    check_start_options(False, None, None, tmp_path / "other")


def _load_start_state(
    directory=None, restart=False, checkpoint_every=None, initial_state_dir=None, t0=0.0, num_steps=5
):
    """load_start_state on the unit square, as a solver with output paths a.bp and b.bp calls it."""
    mesh, submeshes, functions = _setup("unit_square")
    state = load_start_state(
        mesh,
        functions,
        t0=t0,
        T=t0 + num_steps * DT,
        dt=DT,
        num_steps=num_steps,
        restart=restart,
        checkpoint_dir=directory,
        checkpoint_every=checkpoint_every,
        initial_state_dir=initial_state_dir,
        submeshes=submeshes,
        output_paths=["out/a.bp", "out/b.bp"],
        write_vtx=True,
    )
    return state, functions


def test_load_start_state_from_rest_clears_earlier_checkpoints(tmp_path):
    _write_steps("unit_square", tmp_path, [0, 1])
    (t, step, checkpointer, output_paths), functions = _load_start_state(tmp_path, checkpoint_every=1)
    assert (t, step, output_paths) == (0.0, 0, ["out/a.bp", "out/b.bp"])
    assert checkpointer.directory == tmp_path
    assert not any(tmp_path.iterdir())
    for f in functions:
        assert not f.x.array.any()


def test_load_start_state_restart_reads_latest_checkpoint(tmp_path):
    _write_steps("unit_square", tmp_path, [0, 1])
    (t, step, checkpointer, output_paths), functions = _load_start_state(tmp_path, restart=True, checkpoint_every=1)
    assert (t, step) == (2 * DT, 1)
    assert output_paths == [restart_output_path("out/a.bp", t), restart_output_path("out/b.bp", t)]
    _assert_state(functions, t)
    # The checkpoint read is kept: the next write goes to the other file
    checkpointer.write(functions, 3 * DT, 2, DT)
    assert len(list(tmp_path.iterdir())) == 2


def test_load_start_state_restart_rejects_no_step_left(tmp_path):
    _write_steps("unit_square", tmp_path, [0, 1])
    with pytest.raises(ValueError, match="no time step is left"):
        _load_start_state(tmp_path, restart=True, num_steps=1)


def test_load_start_state_restart_message_lists_output_paths(tmp_path, capsys):
    _write_steps("unit_square", tmp_path, [0, 1])
    mesh, submeshes, functions = _setup("unit_square")
    t, *_ = load_start_state(
        mesh,
        functions,
        t0=0.0,
        T=5 * DT,
        dt=DT,
        num_steps=5,
        restart=True,
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
        initial_state_dir=None,
        submeshes=submeshes,
        output_paths=["out/a.bp", "out/b.bp", "out/c.bp"],
        write_vtx=True,
    )
    a, b, c = (restart_output_path(f"out/{name}.bp", t) for name in "abc")
    if mesh.comm.rank == 0:
        assert capsys.readouterr().out == (
            f"Restarting from {tmp_path} at t = {t:.4f} (step = 1), writing VTX output to {a}, {b} and {c}\n"
        )


def test_load_start_state_from_initial_state(tmp_path):
    _write_steps("unit_square", tmp_path / "other", [0, 1])
    (t, step, checkpointer, output_paths), functions = _load_start_state(
        initial_state_dir=tmp_path / "other", t0=2 * DT
    )
    assert (t, step, checkpointer, output_paths) == (2 * DT, 0, None, ["out/a.bp", "out/b.bp"])
    _assert_state(functions, t)


def test_load_start_state_rejects_changed_initial_state(tmp_path):
    _write_steps("unit_square", tmp_path, [0, 1])
    with pytest.raises(RuntimeError, match="changed"):
        _load_start_state(initial_state_dir=tmp_path, t0=DT)


def test_load_start_state_rejects_initial_state_with_no_step_left(tmp_path):
    _write_steps("unit_square", tmp_path, [0, 1])
    with pytest.raises(ValueError, match="no time step is left"):
        _load_start_state(initial_state_dir=tmp_path, t0=2 * DT, num_steps=0)


def test_read_leaves_missing_optional_function(mesh_kind, tmp_path):
    """A function named in optional that the checkpoint does not hold is left as it is, and one that it
    holds is read; a missing function that is not optional still fails the read."""
    mesh, submeshes, functions = _setup(mesh_kind)
    _fill(functions, DT)
    Checkpointer(tmp_path, mesh, submeshes).write(functions[:-1], DT, 0, DT)  # without the last, on a submesh

    mesh, submeshes, functions = _setup(mesh_kind)
    functions[-1].x.array[:] = 7.0
    t, _ = Checkpointer(tmp_path, mesh, submeshes).read(functions, DT, optional=[functions[0].name, functions[-1].name])
    _assert_state(functions[:-1], t)
    np.testing.assert_array_equal(functions[-1].x.array, 7.0)

    with pytest.raises(KeyError):
        Checkpointer(tmp_path, mesh, submeshes).read(functions, DT)


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


_MPI_SUBMESH_OWNERS_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
from test_checkpoint import check_submesh_round_trip
check_submesh_round_trip({directory!r})
"""


def check_submesh_round_trip(directory):
    """Write p (P1, fluid) and v_s (P2 vector, solid) on the 3d FSI mesh twice and read them back exactly.

    Runs with both the solvers' ghost mode and without ghost cells, where more owners of interface dofs hold
    no cell of the submesh. The second write has other values, so a stale value of the first fails too.
    """
    for ghost_mode in [dolfinx.mesh.GhostMode.shared_facet, dolfinx.mesh.GhostMode.none]:
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, FSI3D_MESH, "r") as infile:
            mesh = infile.read_mesh(ghost_mode=ghost_mode)
            cell_tags = infile.read_meshtags(mesh, name="Cell tags")
        fluid_mesh, fluid_map, _, _ = dolfinx.mesh.create_submesh(mesh, 3, cell_tags.find(FLUID))
        solid_mesh, solid_map, _, _ = dolfinx.mesh.create_submesh(mesh, 3, cell_tags.find(SOLID))
        submeshes = [(fluid_mesh, fluid_map), (solid_mesh, solid_map)]

        spaces = {
            "p": dolfinx.fem.functionspace(fluid_mesh, ("Lagrange", 1)),
            "v_s": dolfinx.fem.functionspace(solid_mesh, ("Lagrange", 2, (3,))),
        }
        written = [dolfinx.fem.Function(V, name=name) for name, V in spaces.items()]
        checkpointer = Checkpointer(directory, mesh, submeshes)
        checkpointer.clear()
        for step, c in enumerate([1.0, 5.0]):
            written[0].interpolate(lambda x, c=c: c + x[0] + 2 * x[1] + 3 * x[2])
            written[1].interpolate(lambda x, c=c: np.vstack([c + x[0] * x[1], x[1] * x[2], c * x[2] ** 2]))
            checkpointer.write(written, step * DT, step, DT)

        read = [dolfinx.fem.Function(V, name=name) for name, V in spaces.items()]
        Checkpointer(directory, mesh, submeshes).read(read, DT)
        for f, f_written in zip(read, written, strict=True):
            np.testing.assert_allclose(f.x.array, f_written.x.array, atol=1e-13, err_msg=f"{f.name}, {ghost_mode}")


@pytest.mark.parametrize("ranks", [1, 2, 3])
def test_submesh_dofs_owned_by_ranks_without_submesh_cells(ranks, tmp_path):
    """Functions on submeshes round-trip exactly also where the owner of a dof on the boundary of the
    submesh holds no submesh cell with that dof, so only other ranks can give its value.

    Before this was handled, the 3d FSI mesh lost a few values of v_s on 2 and 3 ranks with the
    shared_facet ghost mode, and of p and v_s on 2 ranks without ghost cells.
    """
    if ranks == 1:
        check_submesh_round_trip(tmp_path)
        return
    script = _MPI_SUBMESH_OWNERS_SCRIPT.format(tests=str(Path(__file__).parent), directory=str(tmp_path))
    run_mpi_python(script, ranks, timeout=300)
