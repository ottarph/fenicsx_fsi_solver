import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from xfsi_solver.fsi.materials import Solid


@pytest.fixture(params=[2, 3], ids=["2d-plane-strain", "3d"])
def material(request):
    dim = request.param
    if dim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    else:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("CG", 2, (dim,))))
    p = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("CG", 1)))
    c1, c2 = dolfinx.fem.Constant(mesh, 3.0), dolfinx.fem.Constant(mesh, 1.0)
    return mesh, u, p, c1, c2


def set_deformation(u, F):
    dim = F.shape[0]
    u.interpolate(lambda x: (F - np.eye(dim)) @ x[:dim])


def evaluate(stress, mesh):
    points = np.full((1, mesh.topology.dim), 0.25)
    expression = dolfinx.fem.Expression(stress, points)
    return expression.eval(mesh, np.array([0], dtype=np.int32)).reshape(stress.ufl_shape)


def test_imr_stress_free_reference_and_rigid_rotation(material):
    mesh, u, p, c1, c2 = material
    p.x.array[:] = 2.0 * (c1.value - c2.value)
    angle = 0.6
    rotation = np.eye(u.ufl_shape[0])
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    for F in (np.eye(u.ufl_shape[0]), rotation):
        set_deformation(u, F)
        np.testing.assert_allclose(evaluate(Solid.IMR(u, p, c1, c2), mesh), 0.0, atol=1e-12)


def test_imr_simple_shear_modulus_and_normal_stress(material):
    mesh, u, p, c1, c2 = material
    shear = 0.7
    F = np.eye(u.ufl_shape[0])
    F[0, 1] = shear
    set_deformation(u, F)
    stress = evaluate(Solid.IMR(u, p, c1, c2), mesh)
    mu = 2.0 * (c1.value + c2.value)
    assert stress[0, 1] == pytest.approx(mu * shear)
    assert stress[0, 0] - stress[1, 1] == pytest.approx(mu * shear**2)
    np.testing.assert_allclose(stress, stress.T, atol=1e-12)


def test_imr_pressure_shift_changes_only_normal_stress(material):
    mesh, u, p, c1, c2 = material
    dim = u.ufl_shape[0]
    F = np.eye(dim)
    F[0, 1] = 0.4
    set_deformation(u, F)
    elastic = evaluate(Solid.IMR_elastic(u, c1, c2), mesh)
    p.x.array[:] = 5.0
    stress = evaluate(Solid.IMR(u, p, c1, c2), mesh)
    np.testing.assert_allclose(stress - elastic, -5.0 * np.eye(dim), atol=1e-12)


def test_imr_neo_hookean_limit(material):
    mesh, u, p, c1, c2 = material
    dim = u.ufl_shape[0]
    F = np.eye(dim)
    F[0, 0], F[1, 1] = 1.4, 1.0 / 1.4
    set_deformation(u, F)
    c2.value = 0.0
    p.x.array[:] = 2.0 * c1.value / 1.4**2
    expected = 2.0 * c1.value * (F @ F.T) - p.x.array[0] * np.eye(dim)
    np.testing.assert_allclose(evaluate(Solid.IMR(u, p, c1, c2), mesh), expected, atol=1e-12)


def test_imr_uniaxial_tension_has_analytic_axial_stress():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("CG", 2, (3,))))
    p = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("CG", 1)))
    c1, c2 = dolfinx.fem.Constant(mesh, 3.0), dolfinx.fem.Constant(mesh, 1.0)
    stretch = 1.5
    set_deformation(u, np.diag([stretch, stretch**-0.5, stretch**-0.5]))
    p.x.array[:] = 2.0 * c1.value / stretch - 2.0 * c2.value * stretch
    stress = evaluate(Solid.IMR(u, p, c1, c2), mesh)
    axial = 2.0 * (c1.value + c2.value / stretch) * (stretch**2 - 1.0 / stretch)
    np.testing.assert_allclose(stress, np.diag([axial, 0.0, 0.0]), atol=1e-12)


def test_imr_piola_tangent_matches_directional_finite_difference(material):
    mesh, u, p, c1, c2 = material
    dim = u.ufl_shape[0]
    F = np.eye(dim)
    F[0, 1] = 0.4
    set_deformation(u, F)
    p.x.array[:] = 2.0
    direction = np.zeros((dim, dim))
    direction[0, 0], direction[0, 1], direction[1, 1] = 0.2, 0.3, -0.1
    increment = dolfinx.fem.Function(u.function_space)
    increment.interpolate(lambda x: direction @ x[:dim])
    deformation = ufl.Identity(dim) + ufl.grad(u)
    piola = ufl.det(deformation) * Solid.IMR(u, p, c1, c2) * ufl.inv(deformation).T
    tangent = evaluate(ufl.derivative(piola, u, increment), mesh)
    epsilon = 1e-6
    set_deformation(u, F + epsilon * direction)
    plus = evaluate(piola, mesh)
    set_deformation(u, F - epsilon * direction)
    minus = evaluate(piola, mesh)
    np.testing.assert_allclose(tangent, (plus - minus) / (2.0 * epsilon), rtol=1e-7, atol=1e-8)
