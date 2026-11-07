# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Restart checkpoints for time-stepping solvers, built on io4dolfinx.

Functions are stored in the cell ordering of the input mesh file
(``write_mesh_input_order`` / ``write_function_on_input_mesh``), so a restart
reads the mesh from the same XDMF file as a fresh run and the number of MPI
ranks may differ between the run that wrote the checkpoint and the restart
(N-to-M). This only works for functions on meshes read from file: submeshes
have no ``original_cell_index``, so io4dolfinx cannot reorder their dofs.
Functions on cell submeshes (from ``dolfinx.mesh.create_submesh``) are
therefore stored as functions with the same element on the parent mesh,
copied cell by cell through the submesh's entity map (exact, since the
elements match), and copied back to the submesh after reading. Only the
owned values of a function are written, and the owner of a parent dof on the
boundary of a submesh need not hold a submesh cell with it, so the ranks that
do add their values into the owner, which divides by their number.

Each checkpoint is a self-contained file, and the checkpoint directory
alternates between two of them. io4dolfinx stores a function's dofmap (in the
global dof numbering of the run that wrote it) only the first time the
function is written to a file, so appending to a file from a run on a
different number of ranks would silently pair new values with the old
dofmap. The step counter and time are written as attributes *after* the
functions, so they act as a commit marker: a run killed while writing one
file still restarts from the complete checkpoint in the other.

A checkpoint serves two purposes. A run that stopped is resumed from the
latest checkpoint in its own directory (``restart``), with the same time
step. A new run can also start from the state in a checkpoint of another run
(``initial_state_dir``), with any time step: it starts at the time of that
checkpoint, counts its steps from there, and only reads from the other run's
directory. The start time ``t0`` of the run is stored with each checkpoint,
so that such a run can be resumed too.

Snapshots (:class:`SnapshotWriter`) are checkpoints of the state at chosen
times, kept for the whole run, unlike the restart checkpoints, which are
overwritten. Each snapshot is a directory with a single checkpoint file in it,
so a run can start from it with ``initial_state_dir``, as from any other
checkpoint directory.
"""

import shutil
import warnings
from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple

import dolfinx
import io4dolfinx
import numpy as np

from xfsi_solver.tools.submesh import all_cells

_ATTRIBUTES = "checkpoint"
# Checkpoints written before t0 was stored have no t0; they were all written by runs starting at t0 = 0
_ATTRIBUTE_KEYS = {"t", "step", "dt", "num_cells"}
_NUM_FILES = 2


def _num_global_cells(mesh: dolfinx.mesh.Mesh) -> int:
    return mesh.topology.index_map(mesh.topology.dim).size_global


def _checkpoint_file(directory: Path, i: int) -> Path:
    return directory / f"checkpoint_{i}.bp"


def _read_complete_attributes(path: Path, comm) -> dict[str, np.ndarray] | None:
    """Attributes of the checkpoint file ``path``, or None if it is missing or incomplete."""
    if not path.exists():
        return None
    # An incomplete file, from a run killed while writing it, has no attributes
    try:
        attrs = io4dolfinx.read_attributes(path, comm, _ATTRIBUTES)
    except Exception:  # noqa: BLE001
        return None
    return attrs if _ATTRIBUTE_KEYS <= attrs.keys() else None


def _submesh_rank_count(V: dolfinx.fem.FunctionSpace, parent_cells: np.ndarray) -> np.ndarray:
    """The number of ranks that hold one of ``parent_cells`` with each owned dof of ``V``, or 1 if none does."""
    bs = V.dofmap.index_map_bs
    count = dolfinx.fem.Function(V)
    count.x.array.reshape(-1, bs)[V.dofmap.list[parent_cells], :] = 1.0
    count.x.scatter_reverse(dolfinx.la.InsertMode.add)
    return np.maximum(count.x.array[: V.dofmap.index_map.size_local * bs], 1.0)


class Checkpointer:
    """Write and read restart checkpoints of a list of functions in ``directory``.

    The functions live on ``mesh``, read from file, or on one of the cell
    submeshes of it listed in ``submeshes`` as ``(submesh, entity_map)``
    pairs, with the entity map returned by ``dolfinx.mesh.create_submesh``.
    """

    def __init__(
        self,
        directory: str | Path,
        mesh: dolfinx.mesh.Mesh,
        submeshes: list[tuple[dolfinx.mesh.Mesh, dolfinx.mesh.EntityMap]] = (),
    ):
        self.directory = Path(directory)
        self.mesh = mesh
        self._next_file = 0
        self._submesh_cells = []
        for submesh, entity_map in submeshes:
            if submesh.topology.dim != mesh.topology.dim:
                raise NotImplementedError("Only functions on cell submeshes (codimension 0) can be checkpointed")
            sub_cells = all_cells(submesh)
            parent_cells = entity_map.sub_topology_to_topology(sub_cells, inverse=False)
            self._submesh_cells.append((submesh, (sub_cells, parent_cells)))
        self._parent_functions = {}
        self._rank_counts = {}

    def _on_parent(self, f: dolfinx.fem.Function):
        """The function stored for ``f``, and the submesh and parent cells to copy between (None on ``mesh``)."""
        f_mesh = f.function_space.mesh
        if f_mesh is self.mesh:
            return f, None
        cells = next((cells for submesh, cells in self._submesh_cells if submesh is f_mesh), None)
        if cells is None:
            raise ValueError(f"Function {f.name!r} is neither on the checkpoint mesh nor on one of its submeshes")
        if f.name not in self._parent_functions:
            V = dolfinx.fem.functionspace(self.mesh, f.function_space.ufl_element())
            self._parent_functions[f.name] = dolfinx.fem.Function(V, name=f.name)
            self._rank_counts[f.name] = _submesh_rank_count(V, cells[1])
        return self._parent_functions[f.name], cells

    def _copy_to_parent(self, g: dolfinx.fem.Function, f: dolfinx.fem.Function, cells) -> None:
        """``g`` <- ``f`` on a submesh, with ``(sub_cells, parent_cells)`` the cells to copy between.

        The owner of a dof on the boundary of the submesh need not hold a submesh cell with it, as its
        cells with that dof may touch the submesh only at a vertex or an edge, and then interpolation
        leaves the owned value unset. So the ranks that do hold one add their values into the owner by
        a reverse scatter, and the owner divides the sum by the number of such ranks.
        """
        sub_cells, parent_cells = cells
        rank_count = self._rank_counts[g.name]
        g.x.array[:] = 0.0
        g.interpolate(f, cells0=sub_cells, cells1=parent_cells)
        g.x.scatter_reverse(dolfinx.la.InsertMode.add)
        g.x.array[: len(rank_count)] /= rank_count
        g.x.scatter_forward()

    def _file(self, i: int) -> Path:
        return _checkpoint_file(self.directory, i)

    def clear(self) -> None:
        """Remove checkpoints from an earlier run, which a restart could otherwise pick up as the latest."""
        if self.mesh.comm.rank == 0:
            for i in range(_NUM_FILES):
                shutil.rmtree(self._file(i), ignore_errors=True)
        self.mesh.comm.Barrier()
        self._next_file = 0

    def write(self, functions: list[dolfinx.fem.Function], t: float, step: int, dt: float, t0: float = 0.0) -> None:
        """Save ``functions`` at time ``t``, after completing time step ``step`` of a run starting at ``t0``."""
        self._write_file(self._file(self._next_file), functions, t, step, dt, t0)
        self._next_file = (self._next_file + 1) % _NUM_FILES

    def _write_file(
        self, path: Path, functions: list[dolfinx.fem.Function], t: float, step: int, dt: float, t0: float
    ) -> None:
        """Write ``functions`` and the attributes of time ``t`` to the checkpoint file ``path``, replacing it."""
        if self.mesh.comm.rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.mesh.comm.Barrier()
        io4dolfinx.write_mesh_input_order(path, self.mesh)
        for f in functions:
            g, cells = self._on_parent(f)
            if cells is not None:
                self._copy_to_parent(g, f, cells)
            io4dolfinx.write_function_on_input_mesh(path, g, time=t)
        attrs = {
            "t": np.array([t], dtype=np.float64),
            "step": np.array([step], dtype=np.int64),
            "dt": np.array([dt], dtype=np.float64),
            "num_cells": np.array([_num_global_cells(self.mesh)], dtype=np.int64),
            "t0": np.array([t0], dtype=np.float64),
        }
        assert _ATTRIBUTE_KEYS <= attrs.keys()
        io4dolfinx.write_attributes(path, self.mesh.comm, _ATTRIBUTES, attrs)

    def _read_attributes(self, i: int) -> dict[str, np.ndarray] | None:
        return _read_complete_attributes(self._file(i), self.mesh.comm)

    def _latest(self) -> tuple[dict[str, np.ndarray], int]:
        """Attributes and file index of the latest complete checkpoint."""
        candidates = [(attrs, i) for i in range(_NUM_FILES) if (attrs := self._read_attributes(i)) is not None]
        if not candidates:
            raise FileNotFoundError(f"No complete checkpoint in {self.directory}")
        return max(candidates, key=lambda c: int(c[0]["step"][0]))

    def latest_times(self) -> tuple[float, float]:
        """Time ``t`` of the latest complete checkpoint and start time ``t0`` of the run that wrote it."""
        attrs, _ = self._latest()
        return float(attrs["t"][0]), float(attrs.get("t0", [0.0])[0])

    def read(
        self, functions: list[dolfinx.fem.Function], dt: float | None, optional: Collection[str] = ()
    ) -> tuple[float, int]:
        """Read ``functions`` from the latest complete checkpoint.

        The functions are matched by name, so they must have the names they
        were written with; other functions in the checkpoint are ignored.
        Functions named in ``optional`` that the checkpoint does not hold are
        left as they are, for a field that a solver with other unknowns does
        not write, such as z of a mixed mesh motion. Returns the time ``t``
        and step counter ``step`` of the checkpoint, and makes the next
        :meth:`write` go to the other file. Raises
        ``FileNotFoundError`` if there is no complete checkpoint and
        ``ValueError`` if it was written on a different mesh, or, unless ``dt``
        is None, with a time step other than ``dt``.
        """
        attrs, i = self._latest()
        t, step = float(attrs["t"][0]), int(attrs["step"][0])
        if dt is not None and not np.isclose(attrs["dt"][0], dt, rtol=1e-12, atol=0.0):
            raise ValueError(f"Checkpoint {self._file(i)} was written with dt = {attrs['dt'][0]}, not {dt}")
        if int(attrs["num_cells"][0]) != _num_global_cells(self.mesh):
            raise ValueError(
                f"Checkpoint {self._file(i)} was written on a mesh with {int(attrs['num_cells'][0])} cells, "
                f"not {_num_global_cells(self.mesh)}"
            )
        for f in functions:
            if f.name in optional and len(io4dolfinx.read_timestamps(self._file(i), self.mesh.comm, f.name)) == 0:
                continue
            g, cells = self._on_parent(f)
            io4dolfinx.read_function(self._file(i), g, time=t, name=f.name)
            # Every dof of f, owned or ghost, is in a local submesh cell, and g has its ghost values from
            # read_function, so unlike in write, interpolating cell by cell sets all of f
            if cells is not None:
                sub_cells, parent_cells = cells
                f.interpolate(g, cells0=parent_cells, cells1=sub_cells)
        self._next_file = (i + 1) % _NUM_FILES
        return t, step


def check_start_options(
    restart, checkpoint_dir, checkpoint_every, initial_state_dir, snapshot_dir=None, snapshot_times=()
) -> None:
    """Raise ``ValueError`` for combinations of the checkpoint arguments of a solver that do not fit together."""
    if len(snapshot_times) > 0 and snapshot_dir is None:
        raise ValueError("snapshot_times requires snapshot_dir")
    if (
        snapshot_dir is not None
        and initial_state_dir is not None
        and Path(initial_state_dir).resolve().is_relative_to(Path(snapshot_dir).resolve())
    ):
        raise ValueError(
            "initial_state_dir must not be in snapshot_dir, whose snapshots a new run clears; "
            "start from a snapshot of another run, or copy it out of snapshot_dir first"
        )
    if restart and checkpoint_dir is None:
        raise ValueError("restart=True requires checkpoint_dir")
    if checkpoint_every is not None and checkpoint_dir is None:
        raise ValueError("checkpoint_every requires checkpoint_dir")
    if restart and initial_state_dir is not None:
        raise ValueError(
            "restart=True resumes the run in checkpoint_dir, which already has its initial state; "
            "do not also pass initial_state_dir"
        )
    if (
        initial_state_dir is not None
        and checkpoint_dir is not None
        and Path(initial_state_dir).resolve() == Path(checkpoint_dir).resolve()
    ):
        raise ValueError("initial_state_dir must differ from checkpoint_dir, which a new run clears")


def start_time(mesh: dolfinx.mesh.Mesh, restart: bool, checkpoint_dir, initial_state_dir) -> float:
    """Start time ``t0`` of a solver run.

    0 for a new run from rest; the time of the latest checkpoint in
    ``initial_state_dir`` for a new run from that state; and on a restart, the
    start time of the run being resumed, stored with its checkpoints.
    """
    if restart:
        return Checkpointer(checkpoint_dir, mesh).latest_times()[1]
    if initial_state_dir is not None:
        return Checkpointer(initial_state_dir, mesh).latest_times()[0]
    return 0.0


def snapshot_name(t: float) -> str:
    """Name of the directory of the snapshot at time ``t`` in a snapshot directory, ``t<t:.4f>``."""
    return f"t{t:.4f}"


def snapshot_steps(times: Collection[float], t0: float, dt: float, num_steps: int) -> dict[int, float]:
    """The time steps closest to ``times``, as ``{step: requested time}``.

    Step ``k`` of a run starting at ``t0`` ends at ``t0 + k * dt``, and the closest one to a time
    ``tau`` is ``k = round((tau - t0) / dt)``, rounded as the solvers round the number of time steps
    ``num_steps`` up to ``T``, so it is at most ``dt / 2`` from ``tau``. Step 0 is the initial state.
    A time closer to a step after ``num_steps`` raises ``ValueError``, as does a time whose step, or
    whose snapshot name, is that of another time. A time closer to a step before the start is
    skipped with a warning: a run started from the state of another run at ``t0`` holds no earlier
    state.
    """
    steps = {}
    for tau in sorted(times):
        k = round((tau - t0) / dt)
        if k > num_steps:
            raise ValueError(
                f"Snapshot time {tau} is after the end of the run at T = {t0 + num_steps * dt:.4f} ({num_steps = })"
            )
        if k < 0:
            warnings.warn(f"Snapshot time {tau} is before the start of the run at {t0 = }; skipping it", stacklevel=3)
            continue
        if k in steps:
            raise ValueError(
                f"Snapshot times {steps[k]} and {tau} are both closest to step {k}, at t = {t0 + k * dt:.4f}"
            )
        steps[k] = tau
    names = [snapshot_name(t0 + k * dt) for k in steps]
    if len(set(names)) < len(names):
        raise ValueError(f"Snapshot times {sorted(times)} give steps with the same snapshot name, in {names}")
    return steps


class SnapshotWriter:
    """Write snapshots, checkpoints of the state at the time steps closest to chosen times.

    The snapshot at time ``t`` goes to ``directory / snapshot_name(t)``, a checkpoint directory with
    one complete checkpoint, from which a run starts with ``initial_state_dir``. ``t`` is the time of
    the step closest to the requested time, by :func:`snapshot_steps`; the requested times ``times``
    are those of a run starting at ``t0`` with ``num_steps`` steps of ``dt``, and ``mesh`` and
    ``submeshes`` are those of :class:`Checkpointer`.

    A new run (``restart=False``) removes the snapshots of an earlier run in ``directory``, which
    would otherwise mix with its own, and nothing else there. A run restarted after step ``step``
    keeps the snapshots up to that step, written by the run it resumes, and warns about the ones
    missing among them, which it cannot write; the later ones are written again as the run reaches
    them. Solvers write a snapshot before the restart checkpoint of the same step, so every snapshot
    up to the step of a restart checkpoint is complete. With ``directory=None`` no snapshot is
    written.

    For following a run in its log, rank 0 prints the snapshots still to write when it is created,
    and a line for each snapshot as it is written, both flushed at once.
    """

    def __init__(
        self,
        directory: str | Path | None,
        mesh: dolfinx.mesh.Mesh,
        submeshes: list[tuple[dolfinx.mesh.Mesh, dolfinx.mesh.EntityMap]] = (),
        times: Collection[float] = (),
        *,
        t0: float,
        dt: float,
        num_steps: int,
        step: int = 0,
        restart: bool = False,
    ):
        self.directory = None if directory is None else Path(directory)
        self._steps = {}
        if self.directory is None:
            return
        self._comm = mesh.comm
        self._checkpointer = Checkpointer(self.directory, mesh, submeshes)
        self._steps = snapshot_steps(times, t0, dt, num_steps)
        if restart:
            done = [k for k in self._steps if k <= step]
            missing = [
                self._steps[k] for k in done if _read_complete_attributes(self.path(t0 + k * dt), self._comm) is None
            ]
            if missing:
                warnings.warn(
                    f"The snapshots for t = {missing} in {self.directory} are missing or incomplete, and are before "
                    f"the restart at step {step}, so they cannot be written",
                    stacklevel=2,
                )
            for k in done:
                del self._steps[k]
        else:
            self.clear()
        if self._steps and self._comm.rank == 0:
            times = ", ".join(f"{t0 + k * dt:.4f}" for k in self._steps)
            steps = ", ".join(str(k) for k in self._steps)
            print(f"Writing snapshots at t = {times} (steps {steps}) to {self.directory}", flush=True)

    def path(self, t: float) -> Path:
        """The checkpoint file of the snapshot at time ``t``."""
        return _checkpoint_file(self.directory / snapshot_name(t), 0)

    def clear(self) -> None:
        """Remove the snapshots of an earlier run, and nothing else in the directory."""
        if self._comm.rank == 0 and self.directory.is_dir():
            for path in self.directory.glob(f"t*/{_checkpoint_file(Path(), 0)}"):
                shutil.rmtree(path, ignore_errors=True)
                if not any(path.parent.iterdir()):
                    path.parent.rmdir()
        self._comm.Barrier()

    def due(self, step: int) -> bool:
        """Whether the state after time step ``step`` is a snapshot."""
        return step in self._steps

    def write(self, functions: list[dolfinx.fem.Function], t: float, step: int, dt: float, t0: float) -> None:
        """Write ``functions``, the state at time ``t`` after time step ``step``, as its snapshot."""
        path = self.path(t)
        self._checkpointer._write_file(path, functions, t, step, dt, t0)
        if self._comm.rank == 0:
            print(
                f"Wrote the snapshot for t = {self._steps[step]} at {t = :.4f} ({step = }) to {path.parent}", flush=True
            )


class StartState(NamedTuple):
    """The state a solver run starts from, returned by :func:`load_start_state`."""

    t: float  # time of the state
    step: int  # number of time steps of the run completed at t
    checkpointer: Checkpointer | None  # writes the checkpoints of the run, None without checkpoint_dir
    output_paths: list[str]  # the VTX output paths, renamed on a restart so earlier output is kept


def load_start_state(
    mesh: dolfinx.mesh.Mesh,
    functions: list[dolfinx.fem.Function],
    *,
    t0: float,
    T: float,
    dt: float,
    num_steps: int,
    restart: bool,
    checkpoint_dir,
    checkpoint_every,
    initial_state_dir,
    submeshes: list[tuple[dolfinx.mesh.Mesh, dolfinx.mesh.EntityMap]] = (),
    output_paths=(),
    write_vtx: bool = False,
    optional: Collection[str] = (),
) -> StartState:
    """Read the state a solver run starts from into ``functions``, and set up the run's checkpointer.

    The arguments are those of the solver: ``t0`` is from :func:`start_time`, ``num_steps`` the number
    of time steps from ``t0`` to ``T``, and ``mesh`` and ``submeshes`` those of :class:`Checkpointer`.
    On a restart, ``functions`` are read from the latest checkpoint in ``checkpoint_dir`` and the
    ``output_paths`` renamed by :func:`restart_output_path`; on a new run with ``checkpoint_every``,
    the checkpoints of an earlier run in ``checkpoint_dir`` are removed. With ``initial_state_dir``,
    ``functions`` are read from the latest checkpoint there, at ``t0``, which counts as step 0. On a
    new run from rest, ``functions`` are left as they are. ``write_vtx`` only adds the output paths
    to the restart message. Functions named in ``optional`` are read only if the checkpoint holds them,
    as :meth:`Checkpointer.read` does.

    Raises ``ValueError`` if no time step is left to solve up to ``T``, and ``RuntimeError`` if the
    checkpoint in ``initial_state_dir`` is no longer the one at ``t0``.
    """
    output_paths = [str(path) for path in output_paths]
    t, step = t0, 0
    checkpointer = None
    if checkpoint_dir is not None:
        checkpointer = Checkpointer(checkpoint_dir, mesh, submeshes)
    if restart:
        t, step = checkpointer.read(functions, dt, optional)
        if step >= num_steps:
            raise ValueError(
                f"Cannot restart from the checkpoint at {t = :.4f} ({step = }): no time step is left to solve "
                f"up to {T = } ({num_steps = })"
            )
        output_paths = [restart_output_path(path, t) for path in output_paths]
        vtx_message = f", writing VTX output to {_join_and(output_paths)}" if write_vtx else ""
        if mesh.comm.rank == 0:
            print(f"Restarting from {checkpoint_dir} at {t = :.4f} ({step = }){vtx_message}")
    elif checkpoint_every is not None:
        checkpointer.clear()
    if initial_state_dir is not None:
        t, _ = Checkpointer(initial_state_dir, mesh, submeshes).read(functions, None, optional)
        if t != t0:
            raise RuntimeError(
                f"The latest checkpoint in {initial_state_dir} changed from t = {t0} to {t} while it was read; "
                "start from a copy of a run that is still writing checkpoints"
            )
        if num_steps <= 0:
            raise ValueError(
                f"Cannot start from the state at {t = :.4f} in {initial_state_dir}: "
                f"no time step is left to solve up to {T = }"
            )
        if mesh.comm.rank == 0:
            print(f"Starting from the state in {initial_state_dir} at {t = :.4f}")
    return StartState(t, step, checkpointer, output_paths)


def _join_and(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``, ..."""
    return items[-1] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def restart_output_path(path: str | Path, t: float) -> str:
    """``out/name.bp`` -> ``out/name_from_t<t>.bp``, so a restart at time ``t`` keeps earlier VTX output."""
    path = Path(path)
    return str(path.with_name(f"{path.stem}_from_t{t:.4f}{path.suffix}"))


def truncate_qoi_file(qoi_path: str | Path, t_last: float, dt: float) -> None:
    """Drop rows after time ``t_last`` from a QoI text file whose first column is time.

    ``t_last`` is the time of the last QoI row of the checkpointed state; rows
    after it were written between the checkpoint and the end of the run.
    """
    with open(qoi_path) as f:
        lines = f.readlines()
    kept = [line for line in lines if line.startswith("#") or float(line.split()[0]) <= t_last + 0.5 * dt]
    with open(qoi_path, "w") as f:
        f.writelines(kept)
