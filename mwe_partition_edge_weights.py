# Copyright (C) 2026 Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT

"""MWE: partition a mesh read from XDMF with weighted dual-graph edges across tagged facets.

The cells and nodes are read as raw arrays with ``XDMFFile.read_topology_data`` and ``read_geometry_data``,
and passed to ``dolfinx.mesh.create_mesh`` with a graph partitioner that gives the edges of the dual graph
across facets with one of the given tags the weight ``--weight``, and all other edges the weight 1. The
partitioner then prefers to keep the two cells of such a facet on the same process.

Requires a DOLFINx newer than 0.11, the first with edge weights in the graph partitioners.

    mpirun -n 4 python3 mwe_partition_edge_weights.py data/meshes/fsi3d/mesh_sec_coarse.xdmf --tags 11
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import basix
import dolfinx
import dolfinx.common
import dolfinx.fem
import dolfinx.graph
import dolfinx.io
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


def tagged_cell_pairs(
    comm: MPI.Comm, cell_type: dolfinx.mesh.CellType, cells: np.ndarray, facets: np.ndarray, tagged: np.ndarray
) -> np.ndarray:
    """The pairs (local index, global index) of the cells sharing a facet ``facets[tagged]``.

    ``cells`` are the input cells on this process and ``facets`` the input facets, in global node indices. The
    global index of input cell ``i`` is ``i`` plus the number of input cells on the lower ranks, as in the dual
    graph that ``create_mesh`` builds.
    """
    # Facets of the reference cell, by their vertices, which are the first nodes of a cell or facet
    tdim = dolfinx.mesh.cell_dim(cell_type)
    reference_facets = np.array(basix.topology(basix.CellType[cell_type.name])[tdim - 1])
    num_facet_vertices = reference_facets.shape[1]
    num_cells = len(cells)
    offsets = np.concatenate([[0], np.cumsum(comm.allgather(num_cells))])

    # The facets as sorted vertices, of the tagged facets and of the facets of every cell, with the cell
    tagged_facets = np.sort(facets[tagged, :num_facet_vertices], axis=1)
    cell_facets = np.sort(cells[:, reference_facets].reshape(-1, num_facet_vertices), axis=1)
    cell_index = np.repeat(offsets[comm.rank] + np.arange(num_cells), len(reference_facets))

    # Match the cell facets with the tagged facets on the process given by their smallest vertex
    tagged_facets = send(comm, tagged_facets, tagged_facets[:, 0] % comm.size)
    cell_facets = send(comm, np.column_stack([cell_facets, cell_index]), cell_facets[:, 0] % comm.size)
    key = np.dtype((np.void, num_facet_vertices * np.dtype(np.int64).itemsize))
    matched = cell_facets[
        np.isin(
            np.ascontiguousarray(cell_facets[:, :num_facet_vertices]).view(key).ravel(),
            np.ascontiguousarray(tagged_facets).view(key).ravel(),
        )
    ]

    # Consecutive rows of the same facet, after sorting, are the cells on either side of it
    matched = matched[np.lexsort(matched[:, :num_facet_vertices].T[::-1])]
    same_facet = (matched[1:, :num_facet_vertices] == matched[:-1, :num_facet_vertices]).all(axis=1)
    a, b = matched[:-1, -1][same_facet], matched[1:, -1][same_facet]
    pairs = np.vstack([np.column_stack([a, b]), np.column_stack([b, a])])

    # Send each pair to the process with its first cell
    pairs = send(comm, pairs, np.searchsorted(offsets, pairs[:, 0], side="right") - 1)
    pairs[:, 0] -= offsets[comm.rank]
    return pairs


def edge_weighted_partitioner(partitioner, pairs: np.ndarray, weight: int):
    """A graph partitioner for ``create_mesh`` that calls ``partitioner`` with the weight ``weight`` on the edges
    of the dual graph from local cell ``pairs[:, 0]`` to global cell ``pairs[:, 1]``, and 1 on all others."""

    def partition(comm, nparts, graph, node_weights, edge_weights, ghosting):
        # create_mesh passes no edge weights. Number the edges (local cell, global cell) uniquely.
        num_cells_global = comm.allreduce(graph.num_nodes)
        cells = np.repeat(np.arange(graph.num_nodes, dtype=np.int64), np.diff(graph.offsets))
        edges = cells * num_cells_global + graph.array
        weighted = np.isin(edges, pairs[:, 0] * num_cells_global + pairs[:, 1])
        num_weighted = comm.allreduce(np.count_nonzero(weighted))
        num_edges = comm.allreduce(len(edges))
        if comm.rank == 0:
            print(f"Weighted {num_weighted // 2} of {num_edges // 2} dual graph edges")
        weights = np.where(weighted, weight, 1).astype(np.int32)
        return partitioner(comm, nparts, graph, node_weights, weights, ghosting)

    return partition


def partition_quality(mesh: dolfinx.mesh.Mesh, facet_tags: dolfinx.mesh.MeshTags, tags: list[int]):
    """The cell imbalance, and the number of facets and of tagged facets on process boundaries."""
    comm = mesh.comm
    tdim = mesh.topology.dim
    cell_map = mesh.topology.index_map(tdim)
    imbalance = comm.allreduce(cell_map.size_local, MPI.MAX) * comm.size / cell_map.size_global

    interprocess = mesh.topology.interprocess_facets()
    interprocess = interprocess[interprocess < mesh.topology.index_map(tdim - 1).size_local]
    tagged = facet_tags.indices[np.isin(facet_tags.values, tags)]
    cut = comm.allreduce(len(interprocess))
    cut_tagged = comm.allreduce(len(np.intersect1d(interprocess, tagged)))
    return imbalance, cut, cut_tagged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", type=Path, help="XDMF file with the mesh and its facet tags")
    parser.add_argument("--tags", type=int, nargs="+", default=[11], help="Facet tags to weight the edges across")
    parser.add_argument("--weight", type=int, default=10, help="Weight of the edges across the tagged facets")
    parser.add_argument("--partitioner", choices=["scotch", "kahip", "parmetis"], default="scotch")
    parser.add_argument("--mesh-name", default="mesh")
    parser.add_argument("--facet-tags-name", default="Facet tags")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    graph_partitioner = getattr(dolfinx.graph, f"partitioner_{args.partitioner}")()
    facet_values = read_tag_values(comm, args.mesh, args.facet_tags_name)
    with dolfinx.io.XDMFFile(comm, args.mesh, "r") as xdmf:
        cell_type, degree = xdmf.read_cell_type(args.mesh_name)
        cells = xdmf.read_topology_data(args.mesh_name)
        x = xdmf.read_geometry_data(args.mesh_name)
        facets = xdmf.read_topology_data(args.facet_tags_name)
        assert len(facets) == len(facet_values)
        pairs = tagged_cell_pairs(comm, cell_type, cells, facets, np.isin(facet_values, args.tags))

        if comm.rank == 0:
            print(f"{'weight':>8}{'imbalance':>12}{'cut facets':>12}{'cut tagged':>12}")
        # Weight 1 is the unweighted partition, for comparison
        for weight in [1, args.weight]:
            mesh = dolfinx.mesh.create_mesh(
                comm,
                cells,
                dolfinx.fem.coordinate_element(cell_type, degree),
                x,
                partitioner=edge_weighted_partitioner(graph_partitioner, pairs, weight),
                ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            )
            mesh.name = args.mesh_name
            mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
            facet_tags = xdmf.read_meshtags(mesh, args.facet_tags_name)
            imbalance, cut, cut_tagged = partition_quality(mesh, facet_tags, args.tags)
            if comm.rank == 0:
                print(f"{weight:>8}{imbalance:>12.3f}{cut:>12}{cut_tagged:>12}")


if __name__ == "__main__":
    main()
