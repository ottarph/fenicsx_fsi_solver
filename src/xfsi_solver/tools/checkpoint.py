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
elements match), and copied back to the submesh after reading.

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
"""

import shutil
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
        return self._parent_functions[f.name], cells

    def _file(self, i: int) -> Path:
        return self.directory / f"checkpoint_{i}.bp"

    def clear(self) -> None:
        """Remove checkpoints from an earlier run, which a restart could otherwise pick up as the latest."""
        if self.mesh.comm.rank == 0:
            for i in range(_NUM_FILES):
                shutil.rmtree(self._file(i), ignore_errors=True)
        self.mesh.comm.Barrier()
        self._next_file = 0

    def write(self, functions: list[dolfinx.fem.Function], t: float, step: int, dt: float, t0: float = 0.0) -> None:
        """Save ``functions`` at time ``t``, after completing time step ``step`` of a run starting at ``t0``."""
        path = self._file(self._next_file)
        if self.mesh.comm.rank == 0:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.mesh.comm.Barrier()
        io4dolfinx.write_mesh_input_order(path, self.mesh)
        for f in functions:
            g, cells = self._on_parent(f)
            if cells is not None:
                sub_cells, parent_cells = cells
                g.interpolate(f, cells0=sub_cells, cells1=parent_cells)
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
        self._next_file = (self._next_file + 1) % _NUM_FILES

    def _read_attributes(self, i: int) -> dict[str, np.ndarray] | None:
        path = self._file(i)
        if not path.exists():
            return None
        # An incomplete file, from a run killed while writing it, has no attributes
        try:
            attrs = io4dolfinx.read_attributes(path, self.mesh.comm, _ATTRIBUTES)
        except Exception:  # noqa: BLE001
            return None
        return attrs if _ATTRIBUTE_KEYS <= attrs.keys() else None

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

    def read(self, functions: list[dolfinx.fem.Function], dt: float | None) -> tuple[float, int]:
        """Read ``functions`` from the latest complete checkpoint.

        The functions are matched by name, so they must have the names they
        were written with; other functions in the checkpoint are ignored.
        Returns the time ``t`` and step counter ``step`` of the checkpoint, and
        makes the next :meth:`write` go to the other file. Raises
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
            g, cells = self._on_parent(f)
            io4dolfinx.read_function(self._file(i), g, time=t, name=f.name)
            if cells is not None:
                sub_cells, parent_cells = cells
                f.interpolate(g, cells0=parent_cells, cells1=sub_cells)
        self._next_file = (i + 1) % _NUM_FILES
        return t, step


def check_start_options(restart, checkpoint_dir, checkpoint_every, initial_state_dir) -> None:
    """Raise ``ValueError`` for combinations of the checkpoint arguments of a solver that do not fit together."""
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
) -> StartState:
    """Read the state a solver run starts from into ``functions``, and set up the run's checkpointer.

    The arguments are those of the solver: ``t0`` is from :func:`start_time`, ``num_steps`` the number
    of time steps from ``t0`` to ``T``, and ``mesh`` and ``submeshes`` those of :class:`Checkpointer`.
    On a restart, ``functions`` are read from the latest checkpoint in ``checkpoint_dir`` and the
    ``output_paths`` renamed by :func:`restart_output_path`; on a new run with ``checkpoint_every``,
    the checkpoints of an earlier run in ``checkpoint_dir`` are removed. With ``initial_state_dir``,
    ``functions`` are read from the latest checkpoint there, at ``t0``, which counts as step 0. On a
    new run from rest, ``functions`` are left as they are. ``write_vtx`` only adds the output paths
    to the restart message.

    Raises ``ValueError`` if no time step is left to solve up to ``T``, and ``RuntimeError`` if the
    checkpoint in ``initial_state_dir`` is no longer the one at ``t0``.
    """
    output_paths = [str(path) for path in output_paths]
    t, step = t0, 0
    checkpointer = None
    if checkpoint_dir is not None:
        checkpointer = Checkpointer(checkpoint_dir, mesh, submeshes)
    if restart:
        t, step = checkpointer.read(functions, dt)
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
        t, _ = Checkpointer(initial_state_dir, mesh, submeshes).read(functions, None)
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
