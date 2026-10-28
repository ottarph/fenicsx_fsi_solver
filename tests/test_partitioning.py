import dolfinx
import pytest
from mpi4py import MPI
from restart_helpers import run_mpi_python

from xfsi_solver.tools.partitioning import partition_quality, read_mesh

MESHES = ["data/meshes/fsi3d/mesh_sec_coarse.xdmf", "data/meshes/fsi3d/mesh_hex_sec_coarse_tetnodes.xdmf"]
SOLID = 1

EDGE_WEIGHTS = tuple(int(n) for n in dolfinx.__version__.split(".")[:2]) > (0, 11)
needs_edge_weights = pytest.mark.skipif(not EDGE_WEIGHTS, reason="edge weights need a DOLFINx newer than 0.11")

# On every rank: the tags that input_cell_tags matches to the input cells are those that read_meshtags gives
# the same cells of the mesh, found by their original (input) index; a weight of 1 gives the default partition;
# and a weighted partition keeps the cells and their tags.
_PARALLEL_SCRIPT = """
import dolfinx
import dolfinx.io
import numpy as np
from mpi4py import MPI

from xfsi_solver.tools.partitioning import input_cell_tags, partition_quality, read_mesh, read_tag_values

comm = MPI.COMM_WORLD
path = {path!r}
with dolfinx.io.XDMFFile(comm, path, "r") as xdmf:
    cell_type, _ = xdmf.read_cell_type("mesh")
    cells = xdmf.read_topology_data("mesh")
    tags = input_cell_tags(
        comm, cell_type, cells, xdmf.read_topology_data("Cell tags"), read_tag_values(comm, path, "Cell tags")
    )
offset = comm.exscan(len(cells)) or 0
input_tags = np.concatenate(comm.allgather(tags))
assert np.all(input_tags >= 0)

def owned_tags(mesh, cell_tags):
    num_owned = mesh.topology.index_map(mesh.topology.dim).size_local
    original = np.asarray(mesh.topology.original_cell_index)[:num_owned]
    values = np.full(num_owned, -1)
    owned = cell_tags.indices < num_owned
    values[cell_tags.indices[owned]] = cell_tags.values[owned]
    return original, values

default = read_mesh(comm, path)
original, values = owned_tags(default[0], default[1])
assert np.array_equal(input_tags[original], values)

unit = read_mesh(comm, path, solid_edge_weight=1)
assert np.array_equal(
    np.asarray(unit[0].topology.original_cell_index), np.asarray(default[0].topology.original_cell_index)
)

weighted = read_mesh(comm, path, solid_edge_weight=10)
original, values = owned_tags(weighted[0], weighted[1])
assert np.array_equal(input_tags[original], values)
assert comm.allreduce(len(original)) == len(input_tags)
for mesh, cell_tags, _ in [default, weighted]:
    quality = partition_quality(mesh, cell_tags, ({solid},))
    assert 1 <= quality["ranks_with_solid"] <= comm.size
"""


@needs_edge_weights
@pytest.mark.parametrize("path", MESHES)
def test_parallel_partitions(path):
    run_mpi_python(_PARALLEL_SCRIPT.format(path=path, solid=SOLID), 3, timeout=120)


@pytest.mark.parametrize("path", MESHES)
def test_serial_partition_quality(path):
    mesh, cell_tags, facet_tags = read_mesh(MPI.COMM_WORLD, path)
    quality = partition_quality(mesh, cell_tags, (SOLID,))
    if MPI.COMM_WORLD.size == 1:
        assert quality["ranks_with_solid"] == 1
        assert quality["cut_facets"] == quality["cut_solid_facets"] == quality["cut_interface_facets"] == 0
        assert quality["cell_imbalance"] == pytest.approx(1.0)
    assert len(facet_tags.values) > 0


@pytest.mark.skipif(EDGE_WEIGHTS, reason="DOLFINx newer than 0.11 has edge weights")
def test_edge_weights_need_newer_dolfinx():
    with pytest.raises(NotImplementedError):
        read_mesh(MPI.COMM_WORLD, MESHES[0], solid_edge_weight=10)
