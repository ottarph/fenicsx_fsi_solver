import dolfinx
import h5py
import io4dolfinx
import numpy as np
import pytest
from mpi4py import MPI
from test_checkpoint import ATOL, FLUID, FSI_MESHES, SOLID, _assert_state, _fill, _setup

from xfsi_solver.tools import vtkhdf


def _cell_tags(mesh_kind, mesh):
    """The cell tags of the FSI mesh, read onto ``mesh``, which was read from the same file."""
    with dolfinx.io.XDMFFile(mesh.comm, FSI_MESHES[mesh_kind], "r") as infile:
        return infile.read_meshtags(mesh, name="Cell tags")


def _owned_cell_data(path, name, out_mesh, t):
    """Cell data ``name`` at time ``t`` on the owned cells of ``out_mesh``, whose nodes are numbered as in the file."""
    f = io4dolfinx.read_cell_data(path, name, out_mesh, time=t, backend=vtkhdf.BACKEND)
    num_owned = out_mesh.topology.index_map(out_mesh.topology.dim).size_local
    return f.x.array[f.function_space.dofmap.list[:num_owned, 0]]


@pytest.mark.parametrize("mesh_kind", ["unit_square", *FSI_MESHES])
def test_writer_writes_a_time_series_to_read_back(mesh_kind, tmp_path):
    """The writer creates the file at its first write, holds every function at every time written, and
    the state at any of them reads back onto the mesh and submeshes of a new setup."""
    path = tmp_path / "out.vtkhdf"
    path.write_text("an earlier file, replaced at the first write")
    mesh, submeshes, functions = _setup(mesh_kind)
    writer = vtkhdf.VTKHDFWriter(path, mesh, functions, submeshes)
    times = [0.25, 0.5, 0.75]
    for t in times:
        _fill(functions, t)
        writer.write(t)
    writer.close()

    np.testing.assert_allclose(vtkhdf.output_times(path, mesh.comm), times)
    assert sorted(vtkhdf.point_data_names(path, mesh.comm)) == sorted(f.name for f in functions)
    mesh, submeshes, functions = _setup(mesh_kind)
    vtkhdf.read_functions(path, functions, 0.5, mesh, submeshes)
    _assert_state(functions, 0.5, ATOL["vtkhdf"])


def test_writer_stores_each_cell_and_node_once(tmp_path):
    """Unlike the per-rank blocks of VTXWriter, which hold the ghost cells of each rank, the file holds one
    grid with each cell and node once, so ParaView counts no cell twice."""
    path = tmp_path / "out.vtkhdf"
    mesh, submeshes, functions = _setup("fsi_tri")
    writer = vtkhdf.VTKHDFWriter(path, mesh, functions, submeshes)
    _fill(functions, 0.0)
    writer.write(0.0)
    num_cells = mesh.topology.index_map(2).size_global
    num_nodes = writer._layout.out_mesh.geometry.index_map().size_global
    if mesh.comm.rank == 0:
        with h5py.File(path, "r") as h5file:
            grid = h5file["VTKHDF/mesh"]
            assert grid["NumberOfCells"][0] == len(grid["Types"]) == num_cells
            assert grid["NumberOfPoints"][0] == len(grid["Points"]) == num_nodes
            points = grid["Points"][:]
            assert len(np.unique(points.round(12), axis=0)) == num_nodes
            assert grid["PointData/u"].shape == (num_nodes, 3)  # padded for VTK


def test_2d_vectors_are_written_with_3_components_for_vtk(tmp_path):
    """VTK takes vectors with 3 components, so a vector on a 2D mesh is written with a third, zero one,
    and VTK's vtkWarpVector, which ParaView's Warp By Vector runs, moves each node by it."""
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    path = tmp_path / "out.vtkhdf"
    mesh, submeshes, functions = _setup("fsi_tri")
    writer = vtkhdf.VTKHDFWriter(path, mesh, functions, submeshes)
    _fill(functions, 0.5)
    writer.write(0.5)
    if mesh.comm.rank != 0:
        return
    with h5py.File(path, "r") as h5file:
        point_data = h5file["VTKHDF/mesh/PointData"]
        assert point_data["u"].shape[1] == point_data["v_s"].shape[1] == 3
        assert point_data["p"].shape[1] == 1
        assert not point_data["u"][:, 2].any()

    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(path))
    reader.Update()
    blocks = reader.GetOutputDataObject(0).NewIterator()
    blocks.InitTraversal()
    grid = blocks.GetCurrentDataObject()
    warp = vtk.vtkWarpVector()
    warp.SetInputData(grid)
    warp.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "u")
    warp.SetScaleFactor(1.0)
    warp.Update()
    points = vtk_to_numpy(grid.GetPoints().GetData())
    warped = vtk_to_numpy(warp.GetOutput().GetPoints().GetData())
    u = vtk_to_numpy(grid.GetPointData().GetArray("u"))
    assert np.abs(u).max() > 0.1
    np.testing.assert_allclose(warped, points + u, atol=1e-12)


def test_reads_vectors_written_with_2_components(tmp_path):
    """Files written before vectors were padded to 3 components hold 2D vectors with 2, as
    io4dolfinx.write_point_data writes them; they read back as well."""
    path = tmp_path / "out.vtkhdf"
    mesh, submeshes, functions = _setup("fsi_tri")
    _fill(functions, 0.5)
    layout = vtkhdf.PointDataLayout(mesh, submeshes)
    layout.write_mesh(path, 0.5)
    layout.write_functions(path, functions[1:], 0.5)
    u = functions[0]
    g = layout._out_function(u)
    cells = np.arange(mesh.topology.index_map(2).size_local + mesh.topology.index_map(2).num_ghosts, dtype=np.int32)
    g.interpolate(u, cells0=cells, cells1=cells)
    io4dolfinx.write_point_data(
        path, g, time=0.5, mode=io4dolfinx.FileMode.append, backend_args=None, backend=vtkhdf.BACKEND
    )

    mesh, submeshes, functions = _setup("fsi_tri")
    vtkhdf.read_functions(path, functions, 0.5, mesh, submeshes)
    _assert_state(functions, 0.5, ATOL["vtkhdf"])


@pytest.mark.parametrize("mesh_kind", FSI_MESHES)
def test_writer_writes_cell_tags_and_owners(mesh_kind, tmp_path):
    """The cell tags and the owning rank of each cell, written with the mesh, hold at every later time too."""
    path = tmp_path / "out.vtkhdf"
    mesh, submeshes, functions = _setup(mesh_kind)
    cell_tags = _cell_tags(mesh_kind, mesh)
    writer = vtkhdf.VTKHDFWriter(path, mesh, functions, submeshes, cell_tags=cell_tags)
    for t in [0.0, 0.5]:
        _fill(functions, t)
        writer.write(t)

    out_mesh = writer._layout.out_mesh
    num_owned = mesh.topology.index_map(2).size_local
    expected = np.zeros(num_owned)
    owned = cell_tags.indices < num_owned
    expected[cell_tags.indices[owned]] = cell_tags.values[owned]
    assert set(np.unique(expected)) == {FLUID, SOLID}
    np.testing.assert_array_equal(_owned_cell_data(path, "cell_tags", out_mesh, 0.5), expected)
    np.testing.assert_array_equal(_owned_cell_data(path, "owner", out_mesh, 0.5), mesh.comm.rank)


def test_reading_at_a_time_a_function_was_not_written_gives_its_last_values(tmp_path):
    """A function need not be written at every time; reading it at a time where it was not gives the
    values of the last time it was."""
    path = tmp_path / "out.vtkhdf"
    mesh, submeshes, (u, p, q) = _setup("unit_square")
    layout = vtkhdf.PointDataLayout(mesh, submeshes)
    layout.write_mesh(path, 0.0)
    _fill([u, p], 0.0)
    layout.write_functions(path, [u, p], 0.0)
    _fill([u, p], 1.0)
    layout.write_functions(path, [u], 1.0)

    mesh, submeshes, (u, p, q) = _setup("unit_square")
    vtkhdf.read_functions(path, [u, p], 1.0, mesh, submeshes)
    # _fill gives each function values of its own, by its place in the list
    expected = [dolfinx.fem.Function(f.function_space) for f in (u, p)]
    _fill(expected, 1.0)
    np.testing.assert_allclose(u.x.array, expected[0].x.array, atol=ATOL["vtkhdf"])
    _fill(expected, 0.0)
    np.testing.assert_allclose(p.x.array, expected[1].x.array, atol=ATOL["vtkhdf"])


def test_read_rejects_missing_function_and_other_mesh(tmp_path):
    path = tmp_path / "out.vtkhdf"
    mesh, submeshes, functions = _setup("unit_square")
    writer = vtkhdf.VTKHDFWriter(path, mesh, functions[:2], submeshes)
    writer.write(0.0)

    mesh, submeshes, functions = _setup("unit_square")
    with pytest.raises(KeyError, match="q"):
        vtkhdf.read_functions(path, functions, 0.0, mesh, submeshes)
    other = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(other, ("Lagrange", 2, (2,))), name="u")
    with pytest.raises(ValueError, match="cells"):
        vtkhdf.read_functions(path, [u], 0.0, other)


@pytest.mark.parametrize("element", [("DG", 1), ("Lagrange", 3)], ids=["DG1", "P3"])
def test_writer_rejects_elements_not_held_by_the_nodes(element, tmp_path):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    f = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, element), name="f")
    with pytest.raises(NotImplementedError, match="continuous Lagrange"):
        vtkhdf.VTKHDFWriter(tmp_path / "out.vtkhdf", mesh, [f]).write(0.0)
