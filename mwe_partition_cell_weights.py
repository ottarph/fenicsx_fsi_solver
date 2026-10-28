# Copyright (C) 2026 Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT

"""MWE: partition a mesh read from XDMF with cell weights given by the cell tags.

The cells and nodes are read as raw arrays with ``XDMFFile.read_topology_data`` and ``read_geometry_data``,
and passed to ``dolfinx.mesh.create_mesh`` with one weight per cell, from its tag. Giving the fluid cells a
larger weight than the solid cells makes the partitioner balance the work per process, rather than the number
of cells, when a fluid cell costs more than a solid cell: it carries the pressure and the mesh motion, and its
forms have more terms.

KaHIP ignores cell weights, so only SCOTCH and ParMETIS are offered.

    mpirun -n 4 python3 mwe_partition_cell_weights.py data/meshes/fsi3d/mesh_sec_coarse.xdmf --weights 2:2
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

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


def cell_weights(tags: np.ndarray, weights: dict[int, int]) -> np.ndarray:
    """The weight of each cell with tag ``tags``: ``weights[tag]``, or 1 for the tags not in ``weights``."""
    w = np.ones(len(tags), dtype=np.int32)
    for tag, weight in weights.items():
        w[tags == tag] = weight
    return w


def partition_quality(
    mesh: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, weights: dict[int, int], all_tags: list[int]
):
    """The imbalance (largest over average per process) of the number of cells, of the weighted number of cells,
    and of the number of cells of each tag in ``all_tags``, and the number of facets on process boundaries."""
    comm = mesh.comm
    tdim = mesh.topology.dim
    num_owned = mesh.topology.index_map(tdim).size_local
    tags = np.full(num_owned, -1, dtype=np.int64)
    owned = cell_tags.indices < num_owned
    tags[cell_tags.indices[owned]] = cell_tags.values[owned]

    def imbalance(local: float) -> float:
        total = comm.allreduce(local)
        return comm.allreduce(local, MPI.MAX) * comm.size / total if total > 0 else 1.0

    tag_imbalance = {tag: imbalance(np.count_nonzero(tags == tag)) for tag in all_tags}

    mesh.topology.create_connectivity(tdim - 1, tdim)
    interprocess = mesh.topology.interprocess_facets()
    cut = comm.allreduce(np.count_nonzero(interprocess < mesh.topology.index_map(tdim - 1).size_local))
    return imbalance(num_owned), imbalance(int(cell_weights(tags, weights).sum())), tag_imbalance, cut


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", type=Path, help="XDMF file with the mesh and its cell tags")
    parser.add_argument(
        "--weights",
        nargs="+",
        default=["2:2"],
        help="Cell weights as TAG:WEIGHT, 1 for the other tags. The default weights the fluid (2) twice the solid",
    )
    parser.add_argument("--partitioner", choices=["scotch", "parmetis"], default="scotch")
    parser.add_argument("--mesh-name", default="mesh")
    parser.add_argument("--cell-tags-name", default="Cell tags")
    args = parser.parse_args()
    weights = dict(tuple(map(int, w.split(":"))) for w in args.weights)

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
        all_tags = sorted(set().union(*comm.allgather(set(np.unique(tags).tolist()))))

        if comm.rank == 0:
            tag_header = "".join(f"{f'tag {tag}':>10}" for tag in all_tags)
            print("Imbalance (largest over average per process) and facets on process boundaries")
            print(f"{'':>12}{'cells':>10}{'weighted':>10}{tag_header}{'cut':>8}")
        # Without cell weights, for comparison
        for label, partitioner in [
            ("unweighted", graph_partitioner),
            ("weighted", (graph_partitioner, cell_weights(tags, weights))),
        ]:
            mesh = dolfinx.mesh.create_mesh(
                comm,
                cells,
                dolfinx.fem.coordinate_element(cell_type, degree),
                x,
                partitioner=partitioner,
                ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            )
            mesh.name = args.mesh_name
            cell_tags = xdmf.read_meshtags(mesh, args.cell_tags_name)
            num_cells, weighted, tag_imbalance, cut = partition_quality(mesh, cell_tags, weights, all_tags)
            if comm.rank == 0:
                tag_columns = "".join(f"{tag_imbalance[tag]:>10.3f}" for tag in all_tags)
                print(f"{label:>12}{num_cells:>10.3f}{weighted:>10.3f}{tag_columns}{cut:>8}")


if __name__ == "__main__":
    main()
