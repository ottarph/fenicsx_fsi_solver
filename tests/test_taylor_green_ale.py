"""The moving-domain Taylor-Green vortex with the FSIForms residual, on quadrilaterals."""

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
from mpi4py import MPI

from xfsi_solver.component_benchmarks.taylor_green_ale import (
    ELEMENT_PAIRS,
    _pin_pressure_dof,
    convergence_rates,
    solve,
)
from xfsi_solver.tools.vtkhdf import output_times, point_data_names

# A few steps of a fast mesh motion: the mesh velocity is about 0.5, but the time error of the few steps stays
# below the spatial error of the meshes, whose rates are then those of the elements.
SHORT = {"T": 0.004, "dt_val": 0.001, "A": 0.08, "T_G": 1.0, "theta_value": 0.5, "comm": MPI.COMM_SELF}


@pytest.mark.parametrize("pair", ["Q2Q1", "Q2DPC1"])
def test_spatial_convergence_rates(pair):
    """The L2 errors on the moving domain converge at rate 3 for the Q2 velocity and 2 for the pressure, for a
    continuous and a discontinuous pressure."""
    velocity_degree, pressure_family, pressure_degree = ELEMENT_PAIRS[pair]
    ns = [4, 8, 16]
    errors = []
    for n in ns:
        history = solve(
            n,
            velocity_degree=velocity_degree,
            pressure_family=pressure_family,
            pressure_degree=pressure_degree,
            **SHORT,
        )
        errors.append((history["error_v"][-1], history["error_p"][-1]))
        assert np.all(history["newton_iterations"][1:] <= 3)
    errors = np.array(errors)
    h = 1.0 / np.array(ns)
    assert convergence_rates(h, errors[:, 0])[-1] == pytest.approx(3.0, abs=0.3)
    assert convergence_rates(h, errors[:, 1])[-1] == pytest.approx(2.0, abs=0.3)


def test_mesh_motion_changes_the_discrete_solution():
    """The mesh motion enters the forms: the discrete velocity on the reference square differs from that of a
    run without motion, while both approximate the same exact solution."""
    moving = solve(8, **SHORT)
    fixed = solve(8, **(SHORT | {"A": 0.0}))
    difference = np.linalg.norm(moving["v"] - fixed["v"]) / np.linalg.norm(fixed["v"])
    assert difference > 1e-4
    assert moving["error_v"][-1] == pytest.approx(fixed["error_v"][-1], rel=0.1)


def test_vtkhdf_output_of_a_dpc_pressure(tmp_path):
    """The output holds v, p and u_f at every ``output_every`` steps, with the DPC pressure interpolated into a
    continuous Lagrange function."""
    path = tmp_path / "taylor_green.vtkhdf"
    solve(4, pressure_family="DPC", output_path=path, output_every=2, **SHORT)
    np.testing.assert_allclose(output_times(path, MPI.COMM_SELF), [0.0, 0.002, 0.004])
    assert set(point_data_names(path, MPI.COMM_SELF)) >= {"v", "p", "u_f"}


def test_qoi_file_has_a_row_per_step(tmp_path):
    path = tmp_path / "qoi.txt"
    history = solve(4, qoi_path=path, **SHORT)
    table = np.loadtxt(path, skiprows=2)
    np.testing.assert_allclose(table[:, 0], history["t"])
    np.testing.assert_allclose(table[:, 1], history["error_v"], rtol=1e-8)
    np.testing.assert_allclose(table[:, 3], history["error_p_theta"], rtol=1e-8)


def test_crank_nicolson_pressure_is_the_midpoint_pressure():
    """On a fixed domain, the pressure of Crank-Nicolson approximates the exact pressure at t^{n+1/2} better
    than at t^{n+1}, the time step error of the latter being first order."""
    histories = [
        solve(8, velocity_degree=3, pressure_degree=2, **(SHORT | {"A": 0.0, "T": 0.2, "dt_val": dt_val}))
        for dt_val in (0.05, 0.025)
    ]
    end = [h["error_p"][-1] for h in histories]
    mid = [h["error_p_theta"][-1] for h in histories]
    assert mid[1] < 0.5 * end[1]
    assert end[0] / end[1] == pytest.approx(2.0, rel=0.25)


@pytest.mark.parametrize("family", ["Lagrange", "DPC"])
def test_pinning_one_dof_removes_the_constant_mode(family):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 2, 2, dolfinx.mesh.CellType.quadrilateral)
    P = dolfinx.fem.functionspace(mesh, basix.ufl.element(family, mesh.basix_cell(), 1))
    bc = _pin_pressure_dof(P)
    assert len(bc.dof_indices()[0]) == 1


def test_plot_of_a_study(tmp_path):
    from xfsi_solver.component_benchmarks.plot_taylor_green_convergence import plot_study

    for name, pair in (("spatial", "Q2Q1"), ("spatial", "Q2DPC1")):
        table = np.array([[8, 1e-3, 1e-3, 1e-2, 9e-3], [16, 1e-3, 1.25e-4, 2.5e-3, 2.2e-3]])
        np.savetxt(tmp_path / f"{name}_{pair}.txt", table, header="n\tdt\terror_v\terror_p\terror_p_theta")
    for pressure_at_theta in (False, True):
        out = tmp_path / f"spatial_{pressure_at_theta}.png"
        plot_study(sorted(tmp_path.glob("spatial_*.txt")), out, {"Q2Q1": (3.0, 2.0)}, pressure_at_theta)
        assert out.stat().st_size > 0
