from pathlib import Path

import adios2
import dolfinx
import h5py
import io4dolfinx
import numpy as np
import pytest
from restart_helpers import run_mpi_python
from test_checkpoint import ATOL, DT, FSI_MESHES, _assert_state, _fill, _setup

from xfsi_solver.scripts.convert_to_vtkhdf import convert, group_outputs
from xfsi_solver.tools import vtkhdf
from xfsi_solver.tools.checkpoint import BP_BACKEND, Checkpointer

TIMES = [0.1, 0.2, 0.3]


def write_old_files(mesh_kind, directory):
    """The files of a run of an old solver: VTX output of the functions on the mesh, the fluid submesh and
    the solid submesh, each in its own file, and .bp checkpoints. The solid file holds the mesh in its
    first step only, as fsi2_harmonic_lagrange wrote it."""
    directory = Path(directory)
    mesh, submeshes, functions = _setup(mesh_kind)
    u, p, v_s = functions
    comm = mesh.comm
    writers = [
        dolfinx.io.VTXWriter(comm, directory / "pv" / "run.bp", [u]),
        dolfinx.io.VTXWriter(comm, directory / "pv" / "run_p.bp", [p]),
        dolfinx.io.VTXWriter(
            comm, directory / "pv" / "run_solid.bp", [v_s], mesh_policy=dolfinx.io.VTXMeshPolicy.reuse
        ),
    ]
    checkpointer = Checkpointer(directory / "checkpoints", mesh, submeshes)
    for step, t in enumerate(TIMES):
        _fill(functions, t)
        for writer in writers:
            writer.write(t)
        checkpointer.write(functions, t, step, DT)
    for writer in writers:
        writer.close()


def check_converted(mesh_kind, directory):
    """The converted output holds every function at every time, and the converted checkpoint the last state."""
    directory = Path(directory)
    mesh, submeshes, functions = _setup(mesh_kind)
    path = directory / "pv" / "run.vtkhdf"
    np.testing.assert_allclose(vtkhdf.output_times(path, mesh.comm), TIMES)
    assert sorted(vtkhdf.point_data_names(path, mesh.comm)) == ["p", "u", "v_s"]
    for t in TIMES:
        vtkhdf.read_functions(path, functions, t, mesh, submeshes)
        _assert_state(functions, t, ATOL["vtkhdf"])

    assert sorted(p.name for p in (directory / "checkpoints").iterdir()) == [
        "checkpoint_0.vtkhdf",
        "checkpoint_1.vtkhdf",
    ]
    t, step = Checkpointer(directory / "checkpoints", mesh, submeshes, "vtkhdf").read(functions, DT)
    assert (t, step) == (TIMES[-1], len(TIMES) - 1)
    _assert_state(functions, t, ATOL["vtkhdf"])


@pytest.mark.parametrize("mesh_kind", FSI_MESHES)
def test_converts_a_run_into_files_as_the_solvers_write_them(mesh_kind, tmp_path):
    """The VTX files of a run are merged into one VTKHDF file, the same as VTKHDFWriter writes, with the cell
    tags of the mesh file, and its checkpoints become VTKHDF checkpoints that a run restarts from."""
    write_old_files(mesh_kind, tmp_path / "old")
    written = convert(FSI_MESHES[mesh_kind], [tmp_path / "old"], output_dir=tmp_path / "new")
    assert sorted(written) == [
        tmp_path / "new" / "checkpoints" / "checkpoint_0.vtkhdf",
        tmp_path / "new" / "checkpoints" / "checkpoint_1.vtkhdf",
        tmp_path / "new" / "pv" / "run.vtkhdf",
    ]
    check_converted(mesh_kind, tmp_path / "new")

    # The same file as the writer's, on the same number of ranks
    mesh, submeshes, functions = _setup(mesh_kind)
    with dolfinx.io.XDMFFile(mesh.comm, FSI_MESHES[mesh_kind], "r") as infile:
        cell_tags = infile.read_meshtags(mesh, name="Cell tags")
    reference = tmp_path / "reference.vtkhdf"
    writer = vtkhdf.VTKHDFWriter(reference, mesh, functions, submeshes, cell_tags)
    for t in TIMES:
        _fill(functions, t)
        writer.write(t)
    if mesh.comm.rank == 0:
        with h5py.File(reference, "r") as expected, h5py.File(tmp_path / "new" / "pv" / "run.vtkhdf", "r") as actual:
            grid, grid_expected = actual["VTKHDF/mesh"], expected["VTKHDF/mesh"]
            np.testing.assert_array_equal(grid["CellData/cell_tags"][:], grid_expected["CellData/cell_tags"][:])
            assert set(grid["CellData/cell_tags"][:].ravel()) == {1, 2}
            np.testing.assert_array_equal(grid["Points"][:], grid_expected["Points"][:])
            for name in ["u", "p", "v_s"]:
                np.testing.assert_allclose(
                    grid[f"PointData/{name}"][:], grid_expected[f"PointData/{name}"][:], rtol=0, atol=1e-14
                )


_MPI_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
from test_convert_to_vtkhdf import write_old_files
from xfsi_solver.scripts.convert_to_vtkhdf import convert
{call}
"""


def test_converts_on_another_number_of_ranks_than_the_run(tmp_path):
    """Files written on 5 ranks, converted on 3 and read on 1.

    On 5 ranks, some ranks hold no solid cell, and write empty blocks of the solid file.
    """
    tests = str(Path(__file__).parent)
    run_mpi_python(_MPI_SCRIPT.format(tests=tests, call=f"write_old_files('fsi_tri', {str(tmp_path)!r})"), 5)
    with adios2.FileReader(str(tmp_path / "pv" / "run_solid.bp")) as f:
        assert 0 in f.read("NumberOfNodes", step_selection=[0, 1])
    call = f"convert({FSI_MESHES['fsi_tri']!r}, [{str(tmp_path)!r}])"
    run_mpi_python(_MPI_SCRIPT.format(tests=tests, call=call), 3, timeout=300)
    # Written next to the old files, which are kept
    assert (tmp_path / "pv" / "run_p.bp").exists()
    for name in ["checkpoint_0.bp", "checkpoint_1.bp"]:
        (tmp_path / "checkpoints" / name).rename(tmp_path / name)
    check_converted("fsi_tri", tmp_path)


def test_groups_the_files_of_a_run(tmp_path):
    names = ["a", "a_p", "a_solid", "a_from_t0.0100", "a_p_from_t0.0100", "b_p", "c", "c_solid_from_t1.0000"]
    groups = group_outputs([tmp_path / f"{name}.bp" for name in names])
    assert sorted([path.stem for path in group] for group in groups) == [
        ["a", "a_p", "a_solid"],
        ["a_from_t0.0100", "a_p_from_t0.0100"],
        ["b_p"],
        ["c"],
        ["c_solid_from_t1.0000"],
    ]


def test_skips_incomplete_checkpoints_and_converted_files(tmp_path):
    write_old_files("fsi_tri", tmp_path)
    # A checkpoint a run killed while writing it left without its attributes
    mesh, _, (u, _, _) = _setup("fsi_tri")
    incomplete = tmp_path / "killed" / "checkpoint_0.bp"
    io4dolfinx.write_mesh(incomplete, mesh, backend=BP_BACKEND, on_input_mesh=True)
    io4dolfinx.write_function(incomplete, u, time=0.0, backend=BP_BACKEND, on_input_mesh=True)

    written = convert(FSI_MESHES["fsi_tri"], [tmp_path])
    assert len(written) == 3
    assert not (tmp_path / "killed" / "checkpoint_0.vtkhdf").exists()
    assert convert(FSI_MESHES["fsi_tri"], [tmp_path]) == []
    assert sorted(convert(FSI_MESHES["fsi_tri"], [tmp_path / "pv" / "run.bp"], overwrite=True)) == [
        tmp_path / "pv" / "run.vtkhdf"
    ]


def test_rejects_files_of_another_mesh(tmp_path):
    write_old_files("fsi_tri", tmp_path)
    with pytest.raises(RuntimeError, match="(?s)Could not convert.*(another mesh|cells)"):
        convert(FSI_MESHES["fsi_quad"], [tmp_path], output_dir=tmp_path / "new")
    assert not list((tmp_path / "new").rglob("*.vtkhdf"))
