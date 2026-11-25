import dolfinx
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from xfsi_solver.tools.quadrature import (
    cap_quadrature_degree,
    estimated_degree,
    fix_quadrature_degree,
    quadrature_description,
)

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


def test_no_cap_returns_the_form_itself(space):
    form, _ = nonlinear_form(space)
    assert cap_quadrature_degree(form, None) is form


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


def test_fix_sets_the_estimate_or_the_cap(space):
    form, _ = nonlinear_form(space)
    mass = ufl.inner(ufl.TrialFunction(space), ufl.TestFunction(space)) * ufl.dx
    for f in (form, mass):
        degrees = [integral.metadata()["quadrature_degree"] for integral in fix_quadrature_degree(f, None).integrals()]
        assert degrees == [estimated_degree(f)]
    assert [integral.metadata()["quadrature_degree"] for integral in fix_quadrature_degree(form, 8).integrals()] == [8]
    assert [integral.metadata()["quadrature_degree"] for integral in fix_quadrature_degree(mass, 8).integrals()] == [4]


def test_fix_gives_the_derivative_the_degree_of_the_form(space):
    # UFL estimates a higher degree for the derivative than for the form itself, so without a set
    # degree the derivative would be integrated with another quadrature than the form.
    form, u = nonlinear_form(space)
    assert estimated_degree(ufl.derivative(form, u)) > estimated_degree(form)
    jacobian = ufl.derivative(fix_quadrature_degree(form, None), u)
    assert [integral.metadata()["quadrature_degree"] for integral in jacobian.integrals()] == [estimated_degree(form)]


def test_fix_without_a_cap_integrates_the_form_as_before(space):
    form, u = nonlinear_form(space)
    u.interpolate(lambda x: 0.1 * np.stack([x[1] ** 2, x[2] * x[0], x[0] ** 2]))
    exact = dolfinx.fem.assemble_vector(dolfinx.fem.form(form)).array
    fixed = dolfinx.fem.assemble_vector(dolfinx.fem.form(fix_quadrature_degree(form, None))).array
    np.testing.assert_array_equal(fixed, exact)


def test_quadrature_description():
    assert quadrature_description(None) == "as estimated by UFL"
    assert quadrature_description(5) == "at most 5"
