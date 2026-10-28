# Copyright (C) 2026 Jørgen S. Dokken, Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Mesh partitioning with weighted dual-graph edges at the solid, for the iterative K_c solves.

GAMG on the velocity block smooths with SOR, which in parallel is Gauss-Seidel within each rank and Jacobi
between ranks. The thin, stiff beam couples strongly across rank boundaries, where the Jacobi part converges
poorly or diverges (notes/fsi3d/implementation-log.md). Weighting the edges of the dual graph at the solid cells
makes the graph partitioner avoid cutting through the solid and along the interface, so that these couplings
stay within a rank. The functions here are those of ``mwe_partition_solid_edge_weights.py`` in the repository
root, which compares the partitions without solving.

The edge weights need a DOLFINx newer than 0.11, the first with edge weights in the graph partitioners.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import basix.ufl
import dolfinx
import dolfinx.common
import dolfinx.graph
import dolfinx.io
import dolfinx.la
import dolfinx.mesh
import numpy as np
import ufl
from mpi4py import MPI


def read_tag_values(comm: MPI.Comm, path: Path, name: str) -> np.ndarray:
    """The values of the meshtags ``name`` in the XDMF file ``path``, in the block of rows that
    ``read_topology_data(name)`` reads on this process.

    The XDMF API cannot read tag values without a mesh, so they are read from the HDF5 dataset that the XDMF
    file names."""
    import h5py

    path = Path(path)
    data_item = ET.parse(path).getroot().find(f".//Grid[@Name='{name}']/Attribute/DataItem")
    h5_file, dataset = data_item.text.strip().split(":", 1)
    with h5py.File(path.parent / h5_file, "r") as h5:
        values = h5[dataset]
        r0, r1 = dolfinx.common.local_range(comm.rank, values.shape[0], comm.size)
        return values[r0:r1].reshape(-1)


def _send(comm: MPI.Comm, rows: np.ndarray, dest: np.ndarray) -> np.ndarray:
    """Send row ``i`` of ``rows`` to process ``dest[i]``, and return the rows received."""
    received, _, _, _ = dolfinx.graph.distribute(comm, rows, dolfinx.graph.adjacencylist(dest.astype(np.int32)))
    return received.reshape(-1, rows.shape[1])


def input_cell_tags(
    comm: MPI.Comm, cell_type: dolfinx.mesh.CellType, cells: np.ndarray, tag_cells: np.ndarray, tag_values: np.ndarray
) -> np.ndarray:
    """The tag of each of the input ``cells`` on this process, -1 if untagged.

    ``tag_cells`` and ``tag_values`` are the rows of the cell tags on this process, which are distributed
    independently of ``cells``. Both are in global node indices. The tags are matched to the cells by their
    sorted vertices on the process given by their smallest vertex.
    """
    num_vertices = dolfinx.mesh.cell_num_entities(cell_type, 0)
    num_cells = len(cells)
    offsets = np.concatenate([[0], np.cumsum(comm.allgather(num_cells))])
    tag_keys = np.sort(tag_cells[:, :num_vertices], axis=1)
    cell_keys = np.sort(cells[:, :num_vertices], axis=1)

    # The last column tells a tag row (0), with its value, from a cell row (1), with its global index
    tag_rows = np.column_stack([tag_keys, tag_values, np.zeros(len(tag_keys), dtype=np.int64)])
    cell_rows = np.column_stack([cell_keys, offsets[comm.rank] + np.arange(num_cells), np.ones(num_cells, np.int64)])
    rows = np.vstack(
        [_send(comm, tag_rows, tag_keys[:, 0] % comm.size), _send(comm, cell_rows, cell_keys[:, 0] % comm.size)]
    )

    # After sorting by vertices and then kind, the row of a cell follows the row of its tag
    rows = rows[np.lexsort(np.column_stack([rows[:, :num_vertices], rows[:, -1]]).T[::-1])]
    match = (rows[1:, :num_vertices] == rows[:-1, :num_vertices]).all(axis=1) & (rows[:-1, -1] < rows[1:, -1])
    cell_tag = np.column_stack([rows[1:, -2][match], rows[:-1, -2][match]])

    # Send each (global cell index, tag) to the process with the cell
    cell_tag = _send(comm, cell_tag, np.searchsorted(offsets, cell_tag[:, 0], side="right") - 1)
    tags = np.full(num_cells, -1, dtype=np.int64)
    tags[cell_tag[:, 0] - offsets[comm.rank]] = cell_tag[:, 1]
    return tags


def marked_edge_partitioner(partitioner, marked: np.ndarray, weight: int):
    """A graph partitioner for ``dolfinx.mesh.create_mesh`` that calls ``partitioner`` with the weight
    ``weight`` on the edges of the dual graph with a marked cell at either end, and 1 on all others.

    ``marked`` tells, for each cell passed to ``create_mesh`` on this process, whether it is marked.
    """

    def partition(comm, nparts, graph, node_weights, edge_weights, ghosting):
        # The nodes of the dual graph are the input cells, numbered consecutively over the processes, and the
        # links are the global indices of the cells sharing a facet with them. create_mesh passes no edge weights.
        num_cells = graph.num_nodes
        assert len(marked) == num_cells
        offsets = np.concatenate([[0], np.cumsum(comm.allgather(num_cells))])
        offset = offsets[comm.rank]
        neighbours = graph.array
        off_process = (neighbours < offset) | (neighbours >= offset + num_cells)

        # Fetch whether the neighbours on other processes are marked
        ghosts = np.unique(neighbours[off_process])
        owners = (np.searchsorted(offsets, ghosts, side="right") - 1).astype(np.int32)
        index_map = dolfinx.common.index_map(comm, num_cells, (ghosts, owners))
        is_marked = dolfinx.la.vector(index_map, dtype=np.int32)
        is_marked.array[:num_cells] = marked
        is_marked.scatter_forward()

        neighbour = neighbours - offset
        neighbour[off_process] = num_cells + np.searchsorted(ghosts, neighbours[off_process])
        cell = np.repeat(np.arange(num_cells), np.diff(graph.offsets))
        at_marked = (is_marked.array[cell] > 0) | (is_marked.array[neighbour] > 0)
        weights = np.where(at_marked, weight, 1).astype(np.int32)
        return partitioner(comm, nparts, graph, node_weights, weights, ghosting)

    return partition


def read_mesh(
    comm: MPI.Comm,
    mesh_path,
    solid_edge_weight: int | None = None,
    solid_tags=(1,),
    graph_partitioner: str = "parmetis",
    mesh_name: str = "mesh",
    cell_tags_name: str = "Cell tags",
    facet_tags_name: str = "Facet tags",
) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
    """The mesh in the XDMF file ``mesh_path`` with its cell and facet tags, ghosted by shared facets.

    With ``solid_edge_weight=None``, the mesh is partitioned as ``XDMFFile.read_mesh`` does. Otherwise, by
    ``graph_partitioner`` (``"scotch"``, ``"parmetis"`` or ``"kahip"``) with the weight ``solid_edge_weight`` on
    the edges of the dual graph with a cell tagged one of ``solid_tags`` at either end, and 1 on all others;
    a weight of 1 is the unweighted partition by the same partitioner. The weights need a DOLFINx newer than
    0.11, and raise ``NotImplementedError`` on 0.11.

    ParMETIS is the default, as in ``XDMFFile.read_mesh`` with the DOLFINx builds here, so that a weight of 1
    gives its partition exactly, and it partitions the same way in every run. PT-SCOTCH does not: on the
    coarse 3d mesh on 26 ranks, its partitions differed from run to run, with or without weights.
    """
    with dolfinx.io.XDMFFile(comm, mesh_path, "r") as xdmf:
        if solid_edge_weight is None:
            mesh = xdmf.read_mesh(name=mesh_name)
        else:
            if tuple(int(n) for n in dolfinx.__version__.split(".")[:2]) <= (0, 11):
                raise NotImplementedError(
                    f"Edge weights in the graph partitioners need a DOLFINx newer than 0.11, not {dolfinx.__version__}"
                )
            cell_type, degree = xdmf.read_cell_type(mesh_name)
            cells = xdmf.read_topology_data(mesh_name)
            x = xdmf.read_geometry_data(mesh_name)
            tag_values = read_tag_values(comm, mesh_path, cell_tags_name)
            tag_cells = xdmf.read_topology_data(cell_tags_name)
            assert len(tag_cells) == len(tag_values)
            tags = input_cell_tags(comm, cell_type, cells, tag_cells, tag_values)
            if comm.allreduce(np.count_nonzero(tags < 0)) > 0:
                raise RuntimeError(f"Cells of {mesh_path} without a tag in {cell_tags_name!r}")
            partitioner = getattr(dolfinx.graph, f"partitioner_{graph_partitioner}")()
            # The coordinate element of XDMFFile.read_mesh, as a UFL mesh, which create_mesh needs to give
            # the mesh a UFL domain
            domain = ufl.Mesh(
                basix.ufl.element("Lagrange", cell_type.name, degree, basix.LagrangeVariant.unset, shape=(x.shape[1],))
            )
            mesh = dolfinx.mesh.create_mesh(
                comm,
                cells,
                domain,
                x,
                partitioner=marked_edge_partitioner(partitioner, np.isin(tags, solid_tags), solid_edge_weight),
                ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            )
            mesh.name = mesh_name
        cell_tags = xdmf.read_meshtags(mesh, name=cell_tags_name)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        facet_tags = xdmf.read_meshtags(mesh, name=facet_tags_name)
    return mesh, cell_tags, facet_tags


def partition_quality(mesh: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, solid_tags=(1,)) -> dict:
    """How ``mesh`` is split over the processes: the cell imbalance (largest over average per process), the
    number of processes with solid cells (tagged one of ``solid_tags``), and the numbers of facets on process
    boundaries in all, between two solid cells, and between a solid and a fluid cell. Needs ghosted cells,
    so that both cells of a facet on a process boundary are local."""
    comm = mesh.comm
    tdim = mesh.topology.dim
    cell_map = mesh.topology.index_map(tdim)
    num_owned = cell_map.size_local
    solid = np.zeros(num_owned + cell_map.num_ghosts, dtype=bool)
    solid[cell_tags.indices[np.isin(cell_tags.values, solid_tags)]] = True

    mesh.topology.create_connectivity(tdim - 1, tdim)
    interprocess = mesh.topology.interprocess_facets()
    interprocess = interprocess[interprocess < mesh.topology.index_map(tdim - 1).size_local]
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
    num_solid = np.array([np.count_nonzero(solid[facet_to_cell.links(f)]) for f in interprocess], dtype=np.int32)
    return {
        "cell_imbalance": comm.allreduce(num_owned, MPI.MAX) * comm.size / cell_map.size_global,
        "ranks_with_solid": comm.allreduce(int(solid[:num_owned].any())),
        "cut_facets": comm.allreduce(len(interprocess)),
        "cut_solid_facets": comm.allreduce(np.count_nonzero(num_solid == 2)),
        "cut_interface_facets": comm.allreduce(np.count_nonzero(num_solid == 1)),
    }


def partition_description(quality: dict, num_ranks: int) -> str:
    """One line with the ``quality`` from :func:`partition_quality` of a mesh on ``num_ranks`` ranks."""
    return (
        f"cell imbalance {quality['cell_imbalance']:.3f}, solid on {quality['ranks_with_solid']} of {num_ranks} "
        f"ranks, facets on rank boundaries: {quality['cut_facets']} in all, {quality['cut_solid_facets']} "
        f"solid-solid, {quality['cut_interface_facets']} solid-fluid"
    )
