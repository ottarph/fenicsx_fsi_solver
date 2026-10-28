# Copyright (C) 2026 Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT

"""MWE: partition a mesh read from XDMF with weighted dual-graph edges at the cells of given tags.

The cells and nodes are read as raw arrays with ``XDMFFile.read_topology_data`` and ``read_geometry_data``,
and passed to ``dolfinx.mesh.create_mesh`` with a graph partitioner that gives each edge of the dual graph with
a cell of one of the given tags (by default the solid) at either end a larger weight, and all other edges the
weight 1. The partitioner then avoids cutting through the solid and along the solid-fluid interface.

The rows of the stiff, thin beam are dominated by the solid stiffness, and PETSc's parallel SOR, Gauss-Seidel
within each process and Jacobi between processes, converges poorly or diverges where such strong couplings
cross a process boundary (notes/fsi3d/implementation-log.md). Keeping the beam on few processes, cut only
across it, is meant to keep these couplings within a process.

Requires a DOLFINx newer than 0.11, the first with edge weights in the graph partitioners.

    mpirun -n 6 python3 mwe_partition_solid_edge_weights.py data/meshes/fsi3d/mesh_sec_coarse.xdmf --weights 1 10 100
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import dolfinx
import dolfinx.common
import dolfinx.fem
import dolfinx.graph
import dolfinx.io
import dolfinx.la
import dolfinx.mesh
import h5py
import numpy as np
from mpi4py import MPI


def read_tag_values(comm: MPI.Comm, path: Path, name: str) -> np.ndarray:
    """The values of the meshtags ``name`` in the XDMF file ``path``, in the block of rows that
    ``read_topology_data(name)`` reads on this process."""
    data_item = ET.parse(path).getroot().find(f".//Grid[@Name='{name}']/Attribute/DataItem")
    h5_file, dataset = data_item.text.strip().split(":", 1)
    with h5py.File(path.parent / h5_file, "r") as h5:
        values = h5[dataset]
        r0, r1 = dolfinx.common.local_range(comm.rank, values.shape[0], comm.size)
        return values[r0:r1].reshape(-1)


def send(comm: MPI.Comm, rows: np.ndarray, dest: np.ndarray) -> np.ndarray:
    """Send row ``i`` of ``rows`` to process ``dest[i]``, and return the rows received."""
    received, _, _, _ = dolfinx.graph.distribute(comm, rows, dolfinx.graph.adjacencylist(dest.astype(np.int32)))
    return received.reshape(-1, rows.shape[1])


def input_cell_tags(
    comm: MPI.Comm, cell_type: dolfinx.mesh.CellType, cells: np.ndarray, tag_cells: np.ndarray, tag_values: np.ndarray
) -> np.ndarray:
    """The tag of each of the input ``cells`` on this process, -1 if untagged.

    ``tag_cells`` and ``tag_values`` are the rows of the cell tags on this process, which are distributed
    independently of ``cells``. Both are in global node indices.
    """
    # Cells by their sorted vertices, which are the first nodes of a cell
    num_vertices = dolfinx.mesh.cell_num_entities(cell_type, 0)
    num_cells = len(cells)
    offsets = np.concatenate([[0], np.cumsum(comm.allgather(num_cells))])
    tag_keys = np.sort(tag_cells[:, :num_vertices], axis=1)
    cell_keys = np.sort(cells[:, :num_vertices], axis=1)

    # Match the tags with the cells on the process given by their smallest vertex. The last column tells a tag
    # row (0), with its value, from a cell row (1), with its global index.
    tag_rows = np.column_stack([tag_keys, tag_values, np.zeros(len(tag_keys), dtype=np.int64)])
    cell_rows = np.column_stack([cell_keys, offsets[comm.rank] + np.arange(num_cells), np.ones(num_cells, np.int64)])
    rows = np.vstack(
        [send(comm, tag_rows, tag_keys[:, 0] % comm.size), send(comm, cell_rows, cell_keys[:, 0] % comm.size)]
    )

    # After sorting by vertices and then kind, the row of a cell follows the row of its tag
    rows = rows[np.lexsort(np.column_stack([rows[:, :num_vertices], rows[:, -1]]).T[::-1])]
    match = (rows[1:, :num_vertices] == rows[:-1, :num_vertices]).all(axis=1) & (rows[:-1, -1] < rows[1:, -1])
    cell_tag = np.column_stack([rows[1:, -2][match], rows[:-1, -2][match]])

    # Send each (global cell index, tag) to the process with the cell
    cell_tag = send(comm, cell_tag, np.searchsorted(offsets, cell_tag[:, 0], side="right") - 1)
    tags = np.full(num_cells, -1, dtype=np.int64)
    tags[cell_tag[:, 0] - offsets[comm.rank]] = cell_tag[:, 1]
    return tags


def marked_edge_partitioner(partitioner, marked: np.ndarray, weight: int):
    """A graph partitioner for ``create_mesh`` that calls ``partitioner`` with the weight ``weight`` on the edges
    of the dual graph with a marked cell at either end, and 1 on all others.

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


def partition_quality(mesh: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, tags: list[int]):
    """The cell imbalance (largest over average per process), the number of processes with marked cells (with a
    tag in ``tags``), and the numbers of facets on process boundaries: in all, between two marked cells, and
    between a marked and an unmarked cell."""
    comm = mesh.comm
    tdim = mesh.topology.dim
    cell_map = mesh.topology.index_map(tdim)
    num_owned = cell_map.size_local
    imbalance = comm.allreduce(num_owned, MPI.MAX) * comm.size / cell_map.size_global

    # Owned and ghost cells: with ghosted facets, both cells of a facet on a process boundary are local
    marked = np.zeros(num_owned + cell_map.num_ghosts, dtype=bool)
    marked[cell_tags.indices[np.isin(cell_tags.values, tags)]] = True
    num_ranks_marked = comm.allreduce(int(marked[:num_owned].any()))

    mesh.topology.create_connectivity(tdim - 1, tdim)
    interprocess = mesh.topology.interprocess_facets()
    interprocess = interprocess[interprocess < mesh.topology.index_map(tdim - 1).size_local]
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
    num_marked = np.array([np.count_nonzero(marked[facet_to_cell.links(f)]) for f in interprocess], dtype=np.int32)
    cut = comm.allreduce(len(interprocess))
    cut_marked = comm.allreduce(np.count_nonzero(num_marked == 2))
    cut_interface = comm.allreduce(np.count_nonzero(num_marked == 1))
    return imbalance, num_ranks_marked, cut, cut_marked, cut_interface


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", type=Path, help="XDMF file with the mesh and its cell tags")
    parser.add_argument("--tags", type=int, nargs="+", default=[1], help="Cell tags to mark, by default the solid")
    parser.add_argument(
        "--weights", type=int, nargs="+", default=[1, 10, 100], help="Edge weights to compare, 1 is unweighted"
    )
    parser.add_argument("--partitioner", choices=["scotch", "kahip", "parmetis"], default="scotch")
    parser.add_argument("--mesh-name", default="mesh")
    parser.add_argument("--cell-tags-name", default="Cell tags")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    graph_partitioner = getattr(dolfinx.graph, f"partitioner_{args.partitioner}")()
    tag_values = read_tag_values(comm, args.mesh, args.cell_tags_name)
    with dolfinx.io.XDMFFile(comm, args.mesh, "r") as xdmf:
        cell_type, degree = xdmf.read_cell_type(args.mesh_name)
        cells = xdmf.read_topology_data(args.mesh_name)
        x = xdmf.read_geometry_data(args.mesh_name)
        tag_cells = xdmf.read_topology_data(args.cell_tags_name)
        assert len(tag_cells) == len(tag_values)
        tags = input_cell_tags(comm, cell_type, cells, tag_cells, tag_values)
        assert comm.allreduce(np.count_nonzero(tags < 0)) == 0, "Untagged cells"
        marked = np.isin(tags, args.tags)
        num_marked = comm.allreduce(np.count_nonzero(marked))
        num_cells = comm.allreduce(len(cells))

        if comm.rank == 0:
            print(f"{num_marked} of {num_cells} cells marked (tags {args.tags})")
            print("Facets on process boundaries: in all, between marked cells, between marked and unmarked cells")
            print(f"{'weight':>8}{'imbalance':>11}{'ranks marked':>14}{'cut':>8}{'marked':>8}{'interface':>11}")
        for weight in args.weights:
            mesh = dolfinx.mesh.create_mesh(
                comm,
                cells,
                dolfinx.fem.coordinate_element(cell_type, degree),
                x,
                partitioner=marked_edge_partitioner(graph_partitioner, marked, weight),
                ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            )
            mesh.name = args.mesh_name
            cell_tags = xdmf.read_meshtags(mesh, args.cell_tags_name)
            imbalance, ranks_marked, cut, cut_marked, cut_interface = partition_quality(mesh, cell_tags, args.tags)
            if comm.rank == 0:
                print(
                    f"{weight:>8}{imbalance:>11.3f}{f'{ranks_marked}/{comm.size}':>14}{cut:>8}{cut_marked:>8}"
                    f"{cut_interface:>11}"
                )


if __name__ == "__main__":
    main()
