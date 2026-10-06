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
"""

import shutil
from pathlib import Path

import dolfinx
import io4dolfinx
import numpy as np

_ATTRIBUTES = "checkpoint"
_ATTRIBUTE_KEYS = {"t", "step", "dt", "num_cells"}
_NUM_FILES = 2


def _num_global_cells(mesh: dolfinx.mesh.Mesh) -> int:
    return mesh.topology.index_map(mesh.topology.dim).size_global


def _all_cells(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    index_map = mesh.topology.index_map(mesh.topology.dim)
    return np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)


class Checkpointer:
    """Write and read restart checkpoints of a list of functions in ``directory``.

    The functions live on ``mesh``, read from file, or on one of the cell
    submeshes of it listed in ``submeshes`` as ``(submesh, entity_map)``
    pairs, with the entity map returned by ``dolfinx.mesh.create_submesh``.
    """

    def __init__(
        self, directory: str | Path, mesh: dolfinx.mesh.Mesh, submeshes: list[tuple[dolfinx.mesh.Mesh, dolfinx.mesh.EntityMap]] = ()
    ):
        self.directory = Path(directory)
        self.mesh = mesh
        self._next_file = 0
        self._submesh_cells = []
        for submesh, entity_map in submeshes:
            if submesh.topology.dim != mesh.topology.dim:
                raise NotImplementedError("Only functions on cell submeshes (codimension 0) can be checkpointed")
            sub_cells = _all_cells(submesh)
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

    def write(self, functions: list[dolfinx.fem.Function], t: float, step: int, dt: float) -> None:
        """Save ``functions`` at time ``t``, after completing time step ``step``."""
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
        }
        assert attrs.keys() == _ATTRIBUTE_KEYS
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

    def read(self, functions: list[dolfinx.fem.Function], dt: float) -> tuple[float, int]:
        """Read ``functions`` from the latest complete checkpoint.

        The functions are matched by name, so they must have the names they
        were written with. Returns the time ``t`` and step counter ``step`` of
        the checkpoint, and makes the next :meth:`write` go to the other file.
        Raises ``FileNotFoundError`` if there is no complete checkpoint and
        ``ValueError`` if it was written with a different time step or on a
        different mesh.
        """
        candidates = [(attrs, i) for i in range(_NUM_FILES) if (attrs := self._read_attributes(i)) is not None]
        if not candidates:
            raise FileNotFoundError(f"No complete checkpoint in {self.directory}")
        attrs, i = max(candidates, key=lambda c: int(c[0]["step"][0]))
        t, step = float(attrs["t"][0]), int(attrs["step"][0])
        if not np.isclose(attrs["dt"][0], dt, rtol=1e-12, atol=0.0):
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
