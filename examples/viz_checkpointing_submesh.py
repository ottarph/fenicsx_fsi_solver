"""Checkpointing and visualization output in VTKHDF, read back onto the original mesh and its submesh.

A VTKHDF file holds a mesh and point data at its nodes, so a function is stored by its values at
the nodes of a mesh with a coordinate element of the same degree (P2 here). This is exact for
continuous Lagrange functions up to that degree, and the same file opens in ParaView.

The checkpoint mesh is written in the cell order of the input mesh file
(``io4dolfinx.write_mesh_input_order``). On reading, the functions are defined on the *original*
mesh, read from the input mesh file as in a fresh run, and on a cell submesh of it, not on the mesh
read from the checkpoint. Each cell of the original mesh (or of the submesh) gets its values from
the checkpoint cell with the same input cell index (``io4dolfinx.transfer_submesh_function``), so
the number of MPI ranks may differ between writing and reading (N-to-M).

Functions on the fluid submesh are stored on the whole mesh, copied cell by cell. The owner of a
mesh node on the submesh boundary need not hold a submesh cell with it, so the ranks that do add
their values into the owner, which divides by their number.

With the mesh go two fields of cell data, for viewing: the cell tags (0 where a cell has none) and
the rank that owns each cell in the run that wrote the file. They are written once, at the time of
the mesh. Each cell and node is stored once, with no ghosts, so ParaView counts no cell twice.

A field need not be written at every time. At a time where it is not, the file, and so ParaView
and ``io4dolfinx.read_point_data``, gives its value from the last time it was written, without
warning; before its first time, the value of its first time. So the fields a restart needs must be
written at every time a run may restart from.

    mpirun -n 1 python examples/viz_checkpointing_submesh.py mesh    # input mesh file, as from gmsh
    mpirun -n 3 python examples/viz_checkpointing_submesh.py write   # a "run" writing output at a few times
    mpirun -n 2 python examples/viz_checkpointing_submesh.py read    # a restart, on another number of ranks

With no phase given, all three run in turn.
"""

import argparse
from pathlib import Path

import basix
import dolfinx
import dolfinx.fem.petsc  # noqa: F401
import h5py
import io4dolfinx
import numpy as np
from mpi4py import MPI

FLUID, SOLID = 1, 2
TIMES = (0.0, 0.5, 1.0)  # output times of the "run"
READ_TIME = 0.5  # the time the "restart" reads, not the last one, to check that the time is picked right
BACKEND = "vtkhdf"
BACKEND_ARGS = {"compression": "gzip"}

# The fields of the 3D solvers: u and v on the whole mesh, p and z on the fluid submesh
ELEMENTS = {"u": ("P", 2, (3,)), "v": ("P", 2, (3,)), "p": ("P", 1), "z": ("P", 2, (3,))}
ON_FLUID = ("p", "z")


def exact(name: str, t: float):
    """The field ``name`` at time ``t``, polynomials that the elements hold exactly."""
    fields = {
        "u": lambda x: (t * (x[0] ** 2 + x[1] ** 2), t * x[0] * x[1], t * (x[0] + x[2])),
        "v": lambda x: (1 + t * x[2] ** 2, x[1] * x[2], -t * x[0]),
        "p": lambda x: t + x[0] - 2 * x[1] + 3 * x[2],
        "z": lambda x: (t * x[1] * x[2], x[0] ** 2 - t, x[2]),
    }
    return fields[name]


def all_cells(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    """The local cells of ``mesh``, owned and ghost."""
    index_map = mesh.topology.index_map(mesh.topology.dim)
    return np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)


# --- the original mesh, read from the input mesh file as in a fresh run --------------------------


def write_input_mesh(path: Path, geometry_degree: int) -> None:
    """A unit cube with a fluid (x < 0.5) and a solid part, written to XDMF as the gmsh scripts do."""
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 6, 6, 6)
    if geometry_degree != 1:
        cmap = dolfinx.fem.coordinate_element(mesh.topology.cell_type, geometry_degree, dtype=mesh.geometry.x.dtype)
        mesh = dolfinx.fem.interpolate_geometry(mesh, cmap)
    cells = all_cells(mesh)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, cells)
    values = np.where(midpoints[:, 0] < 0.5, FLUID, SOLID).astype(np.int32)
    cell_tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, values)
    cell_tags.name = "Cell tags"
    path.parent.mkdir(parents=True, exist_ok=True)
    with dolfinx.io.XDMFFile(comm, str(path), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(cell_tags, mesh.geometry)


class State:
    """The original mesh, its fluid submesh, and the fields on them, as a solver sets them up."""

    def __init__(self, mesh_path: Path, comm: MPI.Comm):
        with dolfinx.io.XDMFFile(comm, str(mesh_path), "r") as infile:
            self.mesh = infile.read_mesh()
            self.cell_tags = infile.read_meshtags(self.mesh, name="Cell tags")
        tdim = self.mesh.topology.dim
        self.fluid_mesh, self.fluid_cell_map, _, _ = dolfinx.mesh.create_submesh(
            self.mesh, tdim, self.cell_tags.find(FLUID)
        )
        self.functions = [
            dolfinx.fem.Function(
                dolfinx.fem.functionspace(self.fluid_mesh if name in ON_FLUID else self.mesh, element), name=name
            )
            for name, element in ELEMENTS.items()
        ]
        self.submeshes = [(self.fluid_mesh, self.fluid_cell_map)]


# --- writing ---------------------------------------------------------------------------------------


def _check_element(V: dolfinx.fem.FunctionSpace, degree: int) -> None:
    """Point data at the nodes of a degree ``degree`` mesh holds exactly the continuous Lagrange elements up to it."""
    element = V.ufl_element()
    if (
        element.basix_element.family != basix.ElementFamily.P
        or element.discontinuous
        or element.embedded_superdegree > degree
    ):
        raise NotImplementedError(
            f"Only continuous Lagrange functions of degree <= {degree} are stored exactly as point data, not {element}"
        )


class PointDataCheckpointer:
    """Write functions on ``mesh`` and on its cell submeshes as point data of a VTKHDF file.

    ``submeshes`` lists ``(submesh, entity_map)`` pairs, with the entity map returned by
    ``dolfinx.mesh.create_submesh``. The file holds ``mesh`` with a coordinate element of degree
    ``degree``, in the cell order of the input mesh file, and the functions at its nodes at each
    time written, so it serves as visualization output and as a checkpoint. The mesh is written at
    time ``t0``, the first time of the run, with ``cell_tags`` and the owning rank of each cell as
    cell data.
    """

    def __init__(
        self,
        path: Path,
        mesh: dolfinx.mesh.Mesh,
        submeshes: list[tuple[dolfinx.mesh.Mesh, dolfinx.mesh.EntityMap]] = (),
        cell_tags: dolfinx.mesh.MeshTags | None = None,
        t0: float = 0.0,
        degree: int = 2,
    ):
        self.path = Path(path)
        self.degree = degree
        self.mesh = mesh
        # A new geometry even if mesh already has this degree: its nodes are then numbered by
        # their current global index, the order write_point_data writes the values in. The
        # topology, and so the cell numbering and original_cell_index, is shared with mesh.
        cmap = dolfinx.fem.coordinate_element(mesh.topology.cell_type, degree, dtype=mesh.geometry.x.dtype)
        self.out_mesh = dolfinx.fem.interpolate_geometry(mesh, cmap)
        imap = self.out_mesh.geometry.index_map()
        nodes = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
        if not np.array_equal(self.out_mesh.geometry.input_global_indices, imap.local_to_global(nodes)):
            raise RuntimeError("The output mesh nodes are not numbered in the order the point data is written in")

        self._cells = {}  # submesh -> (submesh cells, mesh cells)
        for submesh, entity_map in submeshes:
            if submesh.topology.dim != mesh.topology.dim:
                raise NotImplementedError("Only functions on cell submeshes (codimension 0) can be stored")
            sub_cells = all_cells(submesh)
            self._cells[submesh] = (sub_cells, entity_map.sub_topology_to_topology(sub_cells, inverse=False))
        self._out_functions = {}
        self._rank_counts = {}

        if mesh.comm.rank == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        mesh.comm.Barrier()
        io4dolfinx.write_mesh_input_order(self.path, self.out_mesh, time=t0, backend=BACKEND, backend_args=BACKEND_ARGS)
        self._write_partition(cell_tags, t0)

    def _write_partition(self, cell_tags: dolfinx.mesh.MeshTags | None, t: float) -> None:
        """Write the cell tags and the owning rank of each cell at time ``t``."""
        tdim = self.mesh.topology.dim
        cell_map = self.mesh.topology.index_map(tdim)
        num_owned = cell_map.size_local
        if cell_tags is not None:
            if cell_tags.dim != tdim:
                raise ValueError(
                    f"Cell tags must tag cells (dimension {tdim}), not entities of dimension {cell_tags.dim}"
                )
            tags = np.zeros(num_owned, dtype=np.int32)
            owned = cell_tags.indices < num_owned
            tags[cell_tags.indices[owned]] = cell_tags.values[owned]
            self.write_cell_data("cell_tags", tags, t)
        self.write_cell_data("owner", np.full(num_owned, self.mesh.comm.rank, dtype=np.int32), t)

    def write_cell_data(self, name: str, values: np.ndarray, t: float) -> None:
        """Write ``values``, one row per owned cell of ``mesh``, as cell data at time ``t``.

        ``io4dolfinx.write_cell_data`` writes in the current cell order, but the file holds the cells
        in input order, so each value is sent to the rank that holds its cell's input index there.
        """
        comm = self.mesh.comm
        cell_map = self.mesh.topology.index_map(self.mesh.topology.dim)
        num_owned, num_global = cell_map.size_local, cell_map.size_global
        input_index = np.asarray(self.mesh.topology.original_cell_index[:num_owned], dtype=np.int64)
        values = np.asarray(values).reshape(num_owned, -1)
        owners = io4dolfinx.utils.index_owner(comm, input_index, num_global)
        (recv_index, recv_values), *_ = io4dolfinx.comm_helpers.exchange_to_owners(comm, owners, [input_index, values])
        local_range = io4dolfinx.utils.compute_local_range(comm, num_global)
        ordered = np.empty((local_range[1] - local_range[0], values.shape[1]), dtype=values.dtype)
        ordered[recv_index - local_range[0]] = recv_values
        data = io4dolfinx.structures.ArrayData(
            name=name,
            values=ordered,
            global_shape=(num_global, values.shape[1]),
            local_range=(int(local_range[0]), int(local_range[1])),
            type="Cell",
        )
        io4dolfinx.get_backend(BACKEND).write_data(
            self.path, data, comm, float(t), io4dolfinx.FileMode.append, BACKEND_ARGS
        )

    def _out_function(self, f: dolfinx.fem.Function) -> dolfinx.fem.Function:
        """The function at the nodes of ``out_mesh`` that ``f`` is written as."""
        if f.name not in self._out_functions:
            _check_element(f.function_space, self.degree)
            num_components = int(np.prod(f.ufl_shape))
            V = io4dolfinx.readers.create_geometry_function_space(self.out_mesh, num_components)
            self._out_functions[f.name] = dolfinx.fem.Function(V, name=f.name)
            f_mesh = f.function_space.mesh
            if f_mesh is not self.mesh:
                if f_mesh not in self._cells:
                    raise ValueError(f"Function {f.name!r} is neither on the mesh nor on one of its submeshes")
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

    def write(self, functions: list[dolfinx.fem.Function], t: float) -> None:
        """Write ``functions`` at time ``t``."""
        for f in functions:
            g = self._out_function(f)
            f_mesh = f.function_space.mesh
            if f_mesh is self.mesh:
                cells = all_cells(self.mesh)
                g.interpolate(f, cells0=cells, cells1=cells)
            else:
                # The values outside the submesh are written as zero
                sub_cells, cells = self._cells[f_mesh]
                rank_count = self._rank_counts[f.name]
                g.x.array[:] = 0.0
                g.interpolate(f, cells0=sub_cells, cells1=cells)
                g.x.scatter_reverse(dolfinx.la.InsertMode.add)
                g.x.array[: len(rank_count)] /= rank_count
            g.x.scatter_forward()
            io4dolfinx.write_point_data(
                self.path, g, time=t, mode=io4dolfinx.FileMode.append, backend=BACKEND, backend_args=BACKEND_ARGS
            )


# --- reading ---------------------------------------------------------------------------------------


def read_point_data_checkpoint(
    path: Path,
    functions: list[dolfinx.fem.Function],
    t: float,
    mesh: dolfinx.mesh.Mesh,
    submeshes: list[tuple[dolfinx.mesh.Mesh, dolfinx.mesh.EntityMap]] = (),
) -> None:
    """Read ``functions``, on ``mesh`` read from the input mesh file or on its cell submeshes, at time ``t``.

    The mesh in the checkpoint is read only as the carrier of the point data. Each cell of the
    mesh of a function gets its values from the checkpoint cell with the same input cell index.
    """
    comm = mesh.comm
    file_mesh = io4dolfinx.read_mesh(path, comm, time=None, backend=BACKEND)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    if file_mesh.topology.index_map(file_mesh.topology.dim).size_global != num_cells:
        raise ValueError(f"The checkpoint {path} was written on another mesh than the one with {num_cells} cells")
    original_cell_index = np.asarray(mesh.topology.original_cell_index, dtype=np.int64)
    cell_maps = dict(submeshes)
    for f in functions:
        f_mesh = f.function_space.mesh
        cells = all_cells(f_mesh)
        if f_mesh is not mesh:
            cells = cell_maps[f_mesh].sub_topology_to_topology(cells, inverse=False)
        g = io4dolfinx.read_point_data(path, f.name, file_mesh, time=t, backend=BACKEND)
        io4dolfinx.transfer_submesh_function(g, f, original_cell_index[cells])


def output_times(path: Path, comm: MPI.Comm) -> np.ndarray:
    """The times stored in the VTKHDF file ``path``, for picking the one to restart from.

    ``io4dolfinx.read_timestamps`` looks for the steps at the top of the file, but io4dolfinx writes
    a multiblock file with the steps under the mesh block, so they are read here directly.
    """
    times = None
    if comm.rank == 0:
        with h5py.File(path, "r") as h5file:
            times = h5file[f"VTKHDF/{io4dolfinx.backends.DEFAULT_MESH_NAME}/Steps/Values"][:]
    return comm.bcast(times, root=0)


# --- the sample ------------------------------------------------------------------------------------


def max_error(f: dolfinx.fem.Function, t: float) -> float:
    """Largest difference between the dofs of ``f`` and those of the exact field at ``t``, over all ranks."""
    f_ex = dolfinx.fem.Function(f.function_space)
    f_ex.interpolate(exact(f.name, t))
    local = np.max(np.abs(f.x.array - f_ex.x.array), initial=0.0)
    return f.function_space.mesh.comm.allreduce(local, op=MPI.MAX)


def read_owned_cell_data(path: Path, name: str, file_mesh: dolfinx.mesh.Mesh, t: float) -> np.ndarray:
    """Cell data ``name`` at time ``t`` on the owned cells of ``file_mesh``, whose nodes are numbered as in the file."""
    f = io4dolfinx.read_cell_data(path, name, file_mesh, time=t, backend=BACKEND)
    num_owned = file_mesh.topology.index_map(file_mesh.topology.dim).size_local
    return f.x.array[f.function_space.dofmap.list[:num_owned, 0]]


def wrong_cell_tags(path: Path, file_mesh: dolfinx.mesh.Mesh, t: float) -> int:
    """The number of cells whose stored tag is not that of ``write_input_mesh``, over all ranks."""
    tdim = file_mesh.topology.dim
    cells = np.arange(file_mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    expected = np.where(dolfinx.mesh.compute_midpoints(file_mesh, tdim, cells)[:, 0] < 0.5, FLUID, SOLID)
    tags = read_owned_cell_data(path, "cell_tags", file_mesh, t)
    return file_mesh.comm.allreduce(int(np.count_nonzero(tags != expected)), op=MPI.SUM)


def run_write(mesh_path: Path, out_path: Path) -> None:
    comm = MPI.COMM_WORLD
    state = State(mesh_path, comm)
    checkpointer = PointDataCheckpointer(out_path, state.mesh, state.submeshes, state.cell_tags, t0=TIMES[0])
    for t in TIMES:
        for f in state.functions:
            f.interpolate(exact(f.name, t))
            f.x.scatter_forward()
        checkpointer.write(state.functions, t)

    # The cell data, written with the mesh at the first time, read at the last: the output mesh has
    # its nodes numbered as in the file, so it reads the cell data onto the cells of this run
    out_mesh, t = checkpointer.out_mesh, TIMES[-1]
    num_wrong_tags = wrong_cell_tags(out_path, out_mesh, t)
    owner = read_owned_cell_data(out_path, "owner", out_mesh, t)
    num_wrong_owner = comm.allreduce(int(np.count_nonzero(owner != comm.rank)), op=MPI.SUM)
    if comm.rank == 0:
        print(f"Wrote {', '.join(ELEMENTS)} at t = {TIMES} to {out_path} on {comm.size} ranks")
        print(f"  cell data at t = {t}: {num_wrong_tags} wrong cell tags, {num_wrong_owner} wrong owners")
    assert num_wrong_tags == 0 and num_wrong_owner == 0


def run_read(mesh_path: Path, out_path: Path) -> None:
    comm = MPI.COMM_WORLD
    state = State(mesh_path, comm)
    timestamps = output_times(out_path, comm)
    read_point_data_checkpoint(out_path, state.functions, READ_TIME, state.mesh, state.submeshes)
    errors = {f.name: max_error(f, READ_TIME) for f in state.functions}
    # The cell tags in the file, on the cells of the file mesh as read on this number of ranks
    file_mesh = io4dolfinx.read_mesh(out_path, comm, time=None, backend=BACKEND)
    num_wrong_tags = wrong_cell_tags(out_path, file_mesh, READ_TIME)
    if comm.rank == 0:
        print(f"Read t = {READ_TIME} of {timestamps} from {out_path} on {comm.size} ranks; largest dof error:")
        for name, error in errors.items():
            where = "fluid submesh" if name in ON_FLUID else "mesh"
            print(f"  {name} ({where}): {error:.2e}")
        print(f"  {num_wrong_tags} wrong cell tags")
    assert all(error < 1e-10 for error in errors.values()), errors
    assert num_wrong_tags == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phases", nargs="*", choices=["mesh", "write", "read"], default=["mesh", "write", "read"])
    parser.add_argument("--output-dir", type=Path, default=Path("output/viz_checkpointing_submesh"))
    parser.add_argument("--geometry-degree", type=int, default=2, help="degree of the input mesh geometry")
    args = parser.parse_args()
    mesh_path = args.output_dir / "mesh.xdmf"
    out_path = args.output_dir / "solution.vtkhdf"
    for phase in args.phases:
        if phase == "mesh":
            write_input_mesh(mesh_path, args.geometry_degree)
        elif phase == "write":
            run_write(mesh_path, out_path)
        else:
            run_read(mesh_path, out_path)


if __name__ == "__main__":
    main()
