# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Output and restart files in VTKHDF, with functions as point data at the nodes of the mesh.

A VTKHDF file holds a mesh and data at its nodes and cells, and opens in ParaView. A function is
stored by its values at the nodes of the mesh with a Lagrange coordinate element of degree
``degree`` (2 by default), which holds the continuous Lagrange functions up to that degree exactly.
Each cell and node is stored once, with no ghosts, so ParaView counts no cell twice, as it does in
the per-rank blocks of ``dolfinx.io.VTXWriter``.

The mesh is written in the cell order of the input mesh file (``on_input_mesh=True`` in io4dolfinx).
Functions are read back onto the mesh read from the input mesh file, or onto cell submeshes of it,
not onto the mesh in the VTKHDF file: each cell gets its values from the stored cell with the same
input cell index (``io4dolfinx.transfer_submesh_function``), so the number of MPI ranks may differ
between writing and reading.

Functions on cell submeshes are stored on the whole mesh, with zero outside the submesh, copied cell
by cell. The owner of a node on the boundary of a submesh need not hold a submesh cell with it (its
cells may touch the submesh only at a vertex or an edge), so the ranks that do add their values into
the owner, which divides by their number.

Vectors of 2 components, on 2D meshes, are written by io4dolfinx with a third, zero component, as VTK
takes vectors with 3 components (otherwise ParaView's Warp By Vector offers none), and read back without it.

With the mesh go the cell tags (0 where a cell has none) and the rank that owns each cell, as cell
data. A function need not be written at every time: at a time where it is not, the file, and so
ParaView and the reading here, gives its value from the last time it was written, without warning;
before its first time, the value of its first time.

Restart files carry attributes of the state (time, step, ...), which the VTKHDF backend of io4dolfinx
cannot write, in a top-level group of their own, which the VTKHDF specification leaves to other uses.
"""

from collections.abc import Sequence
from pathlib import Path

import basix
import dolfinx
import h5py
import io4dolfinx
import io4dolfinx.comm_helpers
import io4dolfinx.readers
import io4dolfinx.structures
import numpy as np
from mpi4py import MPI

from xfsi_solver.tools.submesh import all_cells

BACKEND = "vtkhdf"
SUFFIX = ".vtkhdf"
# Lossless compression of the mesh and data. The shuffle filter matters for floating-point data: on the
# 201-step output of fsi2_simplified on mesh_sec, gzip alone shrinks u, v and p by 6%, with shuffle by
# 13%; a level above 4 gains nothing.
_BACKEND_ARGS = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
_MESH_GROUP = f"VTKHDF/{io4dolfinx.backends.DEFAULT_MESH_NAME}"

Submeshes = Sequence[tuple[dolfinx.mesh.Mesh, dolfinx.mesh.EntityMap]]


def _check_element(V: dolfinx.fem.FunctionSpace, degree: int) -> None:
    """Point data at the nodes of a degree ``degree`` geometry holds the continuous Lagrange elements up to it."""
    element = V.ufl_element()
    if (
        element.basix_element.family != basix.ElementFamily.P
        or element.discontinuous
        or element.embedded_superdegree > degree
    ):
        raise NotImplementedError(
            f"Only continuous Lagrange functions of degree <= {degree} are stored exactly as point data, not {element}"
        )


class PointDataLayout:
    """The mesh of the VTKHDF files of functions on ``mesh`` and on its cell submeshes ``submeshes``.

    ``submeshes`` lists ``(submesh, entity_map)`` pairs, with the entity map returned by
    ``dolfinx.mesh.create_submesh``. ``mesh`` must be read from a mesh file, so that its cells have
    their input order. The layout keeps the output mesh, of degree ``degree``, and the functions on it
    that the functions are copied to, so that it is set up once for all the files of a run.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, submeshes: Submeshes = (), degree: int = 2):
        self.mesh = mesh
        self.degree = degree
        # A new geometry even if mesh already has this degree: its nodes are then numbered by their
        # current global index, the order the point data is written in. The topology, and so the cell
        # numbering and original_cell_index, is that of mesh.
        cmap = dolfinx.fem.coordinate_element(mesh.topology.cell_type, degree, dtype=mesh.geometry.x.dtype)
        self.out_mesh = dolfinx.fem.interpolate_geometry(mesh, cmap)
        imap = self.out_mesh.geometry.index_map()
        nodes = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
        if not np.array_equal(self.out_mesh.geometry.input_global_indices, imap.local_to_global(nodes)):
            raise RuntimeError(
                "The nodes of the output mesh are not numbered in the order the point data is written in"
            )

        self._cells = {}  # submesh -> (submesh cells, mesh cells)
        for submesh, entity_map in submeshes:
            if submesh.topology.dim != mesh.topology.dim:
                raise NotImplementedError("Only functions on cell submeshes (codimension 0) can be stored")
            sub_cells = all_cells(submesh)
            self._cells[submesh] = (sub_cells, entity_map.sub_topology_to_topology(sub_cells, inverse=False))
        self._out_functions = {}
        self._rank_counts = {}

    def write_mesh(self, path: Path, t: float, cell_tags: dolfinx.mesh.MeshTags | None = None) -> None:
        """Create the file ``path`` with the mesh at time ``t``, and the cell tags and owning ranks as cell data."""
        comm = self.mesh.comm
        if comm.rank == 0:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        io4dolfinx.write_mesh(
            path,
            self.out_mesh,
            mode=io4dolfinx.FileMode.write,
            time=t,
            backend_args=_BACKEND_ARGS,
            backend=BACKEND,
            on_input_mesh=True,
        )
        tdim = self.mesh.topology.dim
        num_owned = self.mesh.topology.index_map(tdim).size_local
        if cell_tags is not None:
            if cell_tags.dim != tdim:
                raise ValueError(f"The cell tags tag entities of dimension {cell_tags.dim}, not cells ({tdim})")
            tags = np.zeros(num_owned, dtype=np.int32)
            owned = cell_tags.indices < num_owned
            tags[cell_tags.indices[owned]] = cell_tags.values[owned]
            self.write_cell_data(path, "cell_tags", tags, t)
        self.write_cell_data(path, "owner", np.full(num_owned, comm.rank, dtype=np.int32), t)

    def write_cell_data(self, path: Path, name: str, values: np.ndarray, t: float) -> None:
        """Write ``values``, a row for each owned cell of ``mesh``, as cell data at time ``t``.

        The file holds the cells in input order, so each value is sent to the rank that holds its
        cell's input index, as ``io4dolfinx.write_cell_data`` does for a function.
        """
        comm = self.mesh.comm
        cell_map = self.mesh.topology.index_map(self.mesh.topology.dim)
        num_owned, num_global = cell_map.size_local, cell_map.size_global
        input_index = np.asarray(self.mesh.topology.original_cell_index[:num_owned], dtype=np.int64)
        values = np.asarray(values).reshape(num_owned, -1)
        (ordered,), local_range = io4dolfinx.comm_helpers.send_to_input_order(comm, input_index, [values], num_global)
        data = io4dolfinx.structures.ArrayData(
            name=name,
            values=ordered,
            global_shape=(num_global, values.shape[1]),
            local_range=(int(local_range[0]), int(local_range[1])),
            type="Cell",
        )
        io4dolfinx.get_backend(BACKEND).write_data(
            path, data, comm, float(t), io4dolfinx.FileMode.append, _BACKEND_ARGS
        )

    def _out_function(self, f: dolfinx.fem.Function) -> dolfinx.fem.Function:
        """The function at the nodes of the output mesh that ``f`` is written as."""
        if f.name not in self._out_functions:
            _check_element(f.function_space, self.degree)
            f_mesh = f.function_space.mesh
            if f_mesh is not self.mesh and f_mesh not in self._cells:
                raise ValueError(f"Function {f.name!r} is neither on the mesh nor on one of its listed submeshes")
            num_components = int(np.prod(f.ufl_shape))
            V = io4dolfinx.readers.create_geometry_function_space(self.out_mesh, num_components, f.ufl_shape)
            self._out_functions[f.name] = dolfinx.fem.Function(V, name=f.name)
            if f_mesh is not self.mesh:
                self._rank_counts[f.name] = self._rank_count(V, self._cells[f_mesh][1])
        return self._out_functions[f.name]

    @staticmethod
    def _rank_count(V: dolfinx.fem.FunctionSpace, cells: np.ndarray) -> np.ndarray:
        """The number of ranks that hold one of ``cells`` with each owned dof of ``V``, or 1 if none does."""
        bs = V.dofmap.index_map_bs
        count = dolfinx.fem.Function(V)
        count.x.array.reshape(-1, bs)[V.dofmap.list[cells], :] = 1.0
        count.x.scatter_reverse(dolfinx.la.InsertMode.add)
        return np.maximum(count.x.array[: V.dofmap.index_map.size_local * bs], 1.0)

    def write_functions(self, path: Path, functions: Sequence[dolfinx.fem.Function], t: float) -> None:
        """Append ``functions`` at time ``t`` to the file ``path``, which ``write_mesh`` created."""
        for f in functions:
            g = self._out_function(f)
            f_mesh = f.function_space.mesh
            if f_mesh is self.mesh:
                cells = all_cells(self.mesh)
                g.interpolate(f, cells0=cells, cells1=cells)
            else:
                sub_cells, cells = self._cells[f_mesh]
                rank_count = self._rank_counts[f.name]
                g.x.array[:] = 0.0
                g.interpolate(f, cells0=sub_cells, cells1=cells)
                g.x.scatter_reverse(dolfinx.la.InsertMode.add)
                g.x.array[: len(rank_count)] /= rank_count
            g.x.scatter_forward()
            self._write_point_data(path, g, t)

    def _write_point_data(self, path: Path, g: dolfinx.fem.Function, t: float) -> None:
        """Write the owned values of ``g``, on the output mesh's nodes, as point data at time ``t``.

        As ``io4dolfinx.write_point_data``, without interpolating ``g``, which is already at the nodes.
        """
        imap = g.function_space.dofmap.index_map
        bs = g.function_space.dofmap.index_map_bs
        data = io4dolfinx.structures.ArrayData(
            name=g.name,
            values=g.x.array.reshape(-1, bs)[: imap.size_local],
            global_shape=(imap.size_global, bs),
            local_range=(int(imap.local_range[0]), int(imap.local_range[1])),
            type="Point",
            value_shape=tuple(g.ufl_shape),
        )
        io4dolfinx.get_backend(BACKEND).write_data(
            path, data, self.mesh.comm, float(t), io4dolfinx.FileMode.append, _BACKEND_ARGS
        )


class VTKHDFWriter:
    """Write ``functions`` to the VTKHDF file ``path`` at each time :meth:`write` is called, for ParaView.

    The counterpart of ``dolfinx.io.VTXWriter`` for functions on ``mesh``, read from a mesh file, and
    on its cell submeshes ``submeshes``, all in one file, with ``cell_tags`` as cell data. The file is
    created, replacing any file at ``path``, at the first :meth:`write`, so that its first time is the
    first time written.
    """

    def __init__(
        self,
        path: str | Path,
        mesh: dolfinx.mesh.Mesh,
        functions: Sequence[dolfinx.fem.Function],
        submeshes: Submeshes = (),
        cell_tags: dolfinx.mesh.MeshTags | None = None,
        degree: int = 2,
    ):
        self.path = Path(path)
        self.functions = list(functions)
        self._layout = PointDataLayout(mesh, submeshes, degree)
        self._cell_tags = cell_tags
        self._created = False

    def write(self, t: float) -> None:
        if not self._created:
            self._layout.write_mesh(self.path, t, self._cell_tags)
            self._created = True
        self._layout.write_functions(self.path, self.functions, t)

    def close(self) -> None:
        """Nothing to do, as every write closes the file; for the interface of ``dolfinx.io.VTXWriter``."""


def _on_rank_0(comm: MPI.Comm, read):
    """``read()`` on rank 0, sent to every rank; an exception on rank 0 is raised on every rank."""
    result, error = None, None
    if comm.rank == 0:
        try:
            result = read()
        except Exception as e:  # noqa: BLE001
            error = e
    result, error = comm.bcast((result, error), root=0)
    if error is not None:
        raise error
    return result


def output_times(path: str | Path, comm: MPI.Comm) -> np.ndarray:
    """The times in the VTKHDF file ``path``.

    These are the times of the mesh, at which every function in the file has values (see the module
    docstring), so no function name is needed, unlike for ``io4dolfinx.read_timestamps``.
    """

    def read():
        with h5py.File(path, "r") as h5file:
            return h5file[f"{_MESH_GROUP}/Steps/Values"][:]

    return _on_rank_0(comm, read)


def point_data_names(path: str | Path, comm: MPI.Comm) -> list[str]:
    """The names of the point data in the VTKHDF file ``path``.

    ``io4dolfinx.read_function_names`` looks for them at the top of the file, but io4dolfinx writes a
    multiblock file with the point data in the block of the mesh, so they are read here directly.
    """

    def read():
        with h5py.File(path, "r") as h5file:
            point_data = h5file[_MESH_GROUP].get("PointData")
            return [] if point_data is None else list(point_data.keys())

    return _on_rank_0(comm, read)


def read_functions(
    path: str | Path,
    functions: Sequence[dolfinx.fem.Function],
    t: float,
    mesh: dolfinx.mesh.Mesh,
    submeshes: Submeshes = (),
) -> None:
    """Read ``functions``, on ``mesh`` or on its cell submeshes ``submeshes``, at time ``t`` from ``path``.

    ``mesh`` is read from the input mesh file the VTKHDF file was written from; the functions are
    matched by name. The mesh in the VTKHDF file is read only as the carrier of the point data. Raises
    ``KeyError`` for a function the file does not hold, and ``ValueError`` if the file holds a mesh
    with another number of cells.
    """
    comm = mesh.comm
    names = point_data_names(path, comm)
    missing = [f.name for f in functions if f.name not in names]
    if missing:
        raise KeyError(f"{path} holds no point data named {missing}, only {names}")
    file_mesh = io4dolfinx.read_mesh(path, comm, time=None, backend=BACKEND)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    num_file_cells = file_mesh.topology.index_map(file_mesh.topology.dim).size_global
    if num_file_cells != num_cells:
        raise ValueError(f"{path} holds a mesh with {num_file_cells} cells, not {num_cells}")
    original_cell_index = np.asarray(mesh.topology.original_cell_index, dtype=np.int64)
    cell_maps = dict(submeshes)
    for f in functions:
        f_mesh = f.function_space.mesh
        cells = all_cells(f_mesh)
        if f_mesh is not mesh:
            if f_mesh not in cell_maps:
                raise ValueError(f"Function {f.name!r} is neither on the mesh nor on one of its listed submeshes")
            cells = cell_maps[f_mesh].sub_topology_to_topology(cells, inverse=False)
        g = io4dolfinx.read_point_data(path, f.name, file_mesh, time=float(t), backend=BACKEND)
        if g.ufl_shape != f.ufl_shape:
            raise ValueError(f"{path} holds {f.name!r} with value shape {g.ufl_shape}, not {f.ufl_shape}")
        io4dolfinx.transfer_submesh_function(g, f, original_cell_index[cells])


def write_attributes(path: str | Path, comm: MPI.Comm, name: str, attributes: dict[str, np.ndarray]) -> None:
    """Write ``attributes`` to the group ``name`` at the top of the file ``path``, after all ranks are done with it."""
    comm.Barrier()
    if comm.rank == 0:
        with h5py.File(path, "a") as h5file:
            group = h5file.require_group(name)
            for key, value in attributes.items():
                group.attrs[key] = value
    comm.Barrier()


def read_attributes(path: str | Path, comm: MPI.Comm, name: str) -> dict[str, np.ndarray]:
    """The attributes of the group ``name`` at the top of the file ``path``; ``KeyError`` if there is none."""

    def read():
        with h5py.File(path, "r") as h5file:
            return {key: np.asarray(value) for key, value in h5file[name].attrs.items()}

    return _on_rank_0(comm, read)
