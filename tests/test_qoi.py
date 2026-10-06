import dolfinx
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from xfsi_solver.tools.qoi import (
    LAGRANGE_QOI_HEADER,
    QOI_HEADER,
    append_qoi_row,
    assemble_force,
    assemble_gap_norm,
    find_point_dof,
    init_qoi_file,
    point_value,
)

COMM = MPI.COMM_WORLD


@pytest.fixture
def displacement():
    mesh = dolfinx.mesh.create_unit_square(COMM, 4, 4)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (2,))))
    u.interpolate(lambda x: np.stack([x[0] + 2 * x[1], 3 * x[0] - x[1]]))
    return u


def test_point_value_is_the_value_at_the_point(displacement):
    node = find_point_dof(displacement.function_space, np.array([0.5, 0.25, 0.0]))
    value = point_value(displacement, node)
    if COMM.rank == 0:
        np.testing.assert_allclose(value, [1.0, 1.25])


def test_find_point_dof_rejects_a_point_outside_the_mesh(displacement):
    with pytest.raises(AssertionError, match="None or multiple dofs"):
        find_point_dof(displacement.function_space, np.array([2.0, 2.0, 0.0]))


def test_assemble_force_sums_the_forms():
    mesh = dolfinx.mesh.create_unit_square(COMM, 3, 3)
    left, right = (dolfinx.fem.form(ufl.as_ufl(c) * ufl.dx(domain=mesh)) for c in (1.0, 2.0))
    total = assemble_force([left, right], COMM)
    if COMM.rank == 0:
        assert total == pytest.approx(3.0)


def test_qoi_file_is_created_with_header_and_rows(tmp_path):
    path = tmp_path / "sub" / "qoi.txt"
    init_qoi_file(path, COMM, restart=False, t=0.0, dt_val=0.1)
    append_qoi_row(path, COMM, 0.0, 1.0, 2.0, np.array([3.0, 4.0]))
    lines = path.read_text().splitlines()
    assert lines[0] == "# " + QOI_HEADER
    assert [float(x) for x in lines[1].split()] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_restart_drops_rows_after_the_checkpoint(tmp_path):
    path = tmp_path / "qoi.txt"
    init_qoi_file(path, COMM, restart=False, t=0.0, dt_val=0.1)
    for k in range(4):
        append_qoi_row(path, COMM, 0.1 * k, 0.0, 0.0, np.zeros(2))
    init_qoi_file(path, COMM, restart=True, t=0.2, dt_val=0.1)
    rows = np.loadtxt(path, ndmin=2)
    np.testing.assert_allclose(rows[:, 0], [0.0, 0.1, 0.2])


def test_assemble_gap_norm_is_the_root_of_the_squared_norm():
    mesh = dolfinx.mesh.create_unit_square(COMM, 3, 3)
    squared = dolfinx.fem.form(ufl.as_ufl(4.0) * ufl.dx(domain=mesh))
    norm = assemble_gap_norm(squared, COMM)
    if COMM.rank == 0:
        assert norm == pytest.approx(2.0)


def test_assemble_gap_norm_clips_a_negative_square_to_zero():
    mesh = dolfinx.mesh.create_unit_square(COMM, 3, 3)
    negative = dolfinx.fem.form(ufl.as_ufl(-1e-20) * ufl.dx(domain=mesh))
    norm = assemble_gap_norm(negative, COMM)
    if COMM.rank == 0:
        assert norm == 0.0


def test_qoi_file_with_the_interface_gap_columns(tmp_path):
    path = tmp_path / "qoi.txt"
    init_qoi_file(path, COMM, restart=False, t=0.0, dt_val=0.1, header=LAGRANGE_QOI_HEADER)
    append_qoi_row(path, COMM, 0.0, 1.0, 2.0, np.array([3.0, 4.0]), 5.0, 6.0)
    lines = path.read_text().splitlines()
    assert lines[0].split()[1:] == ["t", "drag", "lift", "A_x", "A_y", "interface_u_gap", "interface_v_gap"]
    assert [float(x) for x in lines[1].split()] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
