import dolfinx
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from xfsi_solver.tools.quadrature import cap_quadrature_degree, estimated_degree

COMM = MPI.COMM_WORLD


@pytest.fixture
def space():
    mesh = dolfinx.mesh.create_unit_cube(COMM, 2, 2, 2)
    return dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))


def nonlinear_form(space):
    """A residual with inv(F) and det(F) of a P2 displacement, as in the ALE terms."""
    u = dolfinx.fem.Function(space)
    F = ufl.Identity(3) + ufl.grad(u)
    return ufl.inner(ufl.det(F) * ufl.inv(F).T, ufl.grad(ufl.TestFunction(space))) * ufl.dx, u


def test_high_degree_integrals_are_capped(space):
    form, _ = nonlinear_form(space)
    assert estimated_degree(form) > 8
    capped = cap_quadrature_degree(form, 8)
    assert [integral.metadata()["quadrature_degree"] for integral in capped.integrals()] == [8]


def test_low_degree_integrals_are_unchanged(space):
    mass = ufl.inner(ufl.TrialFunction(space), ufl.TestFunction(space)) * ufl.dx
    assert estimated_degree(mass) == 4
    capped = cap_quadrature_degree(mass, 8)
    assert "quadrature_degree" not in capped.integrals()[0].metadata()


def test_derivative_keeps_the_cap(space):
    form, u = nonlinear_form(space)
    jacobian = ufl.derivative(cap_quadrature_degree(form, 8), u)
    assert [integral.metadata()["quadrature_degree"] for integral in jacobian.integrals()] == [8]


def test_capped_form_assembles_close_to_the_exact_one(space):
    form, u = nonlinear_form(space)
    u.interpolate(lambda x: 0.1 * np.stack([x[1] ** 2, x[2] * x[0], x[0] ** 2]))
    exact = dolfinx.fem.assemble_vector(dolfinx.fem.form(form)).array
    capped = dolfinx.fem.assemble_vector(dolfinx.fem.form(cap_quadrature_degree(form, 8))).array
    np.testing.assert_allclose(capped, exact, atol=1e-6 * np.abs(exact).max())
