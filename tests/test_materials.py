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


# The compressible three-term Mooney-Rivlin material (MR3). The coefficients are those of the FSI2 test
# preset (C10 + C01 = mu_s / 2 of FSI2, D1 = 2 / K with K = lambda_s + 2 mu_s / 3), with C10 made nonzero
# so that both distortional invariants enter.
MR3 = {"C10": 1.0e5, "C01": 1.5e5, "C11": 1.8e6, "D1": 6.0 / 7.0e6}


def mr3_energy_numpy(F3, C10, C01, C11, D1):
    """The MR3 energy of a 3 x 3 deformation gradient, written independently of the UFL one, in invariants of
    C = F^T F: I1bar = J^(-2/3) I1, I2bar = J^(-4/3) I2."""
    C = F3.T @ F3
    J = np.linalg.det(F3)
    I1 = np.trace(C)
    I2 = 0.5 * (I1**2 - np.trace(C @ C))
    I1bar, I2bar = J ** (-2.0 / 3.0) * I1, J ** (-4.0 / 3.0) * I2
    return C10 * (I1bar - 3) + C01 * (I2bar - 3) + C11 * (I1bar - 3) * (I2bar - 3) + (J - 1) ** 2 / D1


def embed(F):
    """The 3 x 3 plane-strain deformation gradient of a 2 x 2 one."""
    F3 = np.eye(3)
    F3[: F.shape[0], : F.shape[1]] = F
    return F3


@pytest.fixture(params=[2, 3], ids=["2d-plane-strain", "3d"])
def mr3_material(request):
    dim = request.param
    if dim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    else:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (dim,))))
    constants = {name: dolfinx.fem.Constant(mesh, value) for name, value in MR3.items()}
    return mesh, u, constants


def mr3_energy_ufl(u, constants):
    """The UFL energy at F = I + grad(u), embedded in 3D as in MR3_piola."""
    dim = u.ufl_shape[0]
    F = ufl.Identity(dim) + ufl.grad(u)
    if dim == 2:
        F = ufl.as_matrix([[F[0, 0], F[0, 1], 0.0], [F[1, 0], F[1, 1], 0.0], [0.0, 0.0, 1.0]])
    return Solid.MR3_energy(F, **constants)


def rotation_matrix(dim, angle=0.6):
    rotation = np.eye(dim)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return rotation


def deformed(dim):
    """A nontrivial admissible deformation gradient, with stretch, shear and volume change."""
    F = np.eye(dim) + 0.1 * np.arange(1, dim * dim + 1).reshape(dim, dim) / (dim * dim)
    F[0, 1] += 0.25
    F[1, 0] -= 0.1
    assert np.linalg.det(F) > 0
    return F


def test_mr3_zero_at_identity_and_under_rigid_rotation(mr3_material):
    mesh, u, constants = mr3_material
    dim = u.ufl_shape[0]
    for F in (np.eye(dim), rotation_matrix(dim)):
        set_deformation(u, F)
        # The energies are sums of terms of size C ~ 1e5 that cancel
        assert abs(evaluate(mr3_energy_ufl(u, constants), mesh)) < 1e-9
        np.testing.assert_allclose(evaluate(Solid.MR3_piola(u, **constants), mesh), 0.0, atol=1e-9)


def test_mr3_is_objective(mr3_material):
    """W(R F) = W(F) and P(R F) = R P(F) for a rotation R of a deformed state."""
    mesh, u, constants = mr3_material
    dim = u.ufl_shape[0]
    F, R = deformed(dim), rotation_matrix(dim)
    set_deformation(u, F)
    energy, piola = evaluate(mr3_energy_ufl(u, constants), mesh), evaluate(Solid.MR3_piola(u, **constants), mesh)
    set_deformation(u, R @ F)
    assert evaluate(mr3_energy_ufl(u, constants), mesh) == pytest.approx(energy, rel=1e-12)
    np.testing.assert_allclose(evaluate(Solid.MR3_piola(u, **constants), mesh), R @ piola, rtol=1e-12, atol=1e-6)


def test_mr3_energy_and_piola_match_an_independent_energy(mr3_material):
    """The UFL energy equals one written in other invariants, and P is its derivative, by central
    differences in each component of F (for 2D, of the in-plane components of the embedded F3)."""
    mesh, u, constants = mr3_material
    dim = u.ufl_shape[0]
    F = deformed(dim)
    set_deformation(u, F)
    assert evaluate(mr3_energy_ufl(u, constants), mesh) == pytest.approx(mr3_energy_numpy(embed(F), **MR3), rel=1e-12)
    piola = evaluate(Solid.MR3_piola(u, **constants), mesh)
    assert piola.shape == (dim, dim)
    epsilon = 1e-6
    expected = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            H = np.zeros((dim, dim))
            H[i, j] = epsilon
            plus, minus = mr3_energy_numpy(embed(F + H), **MR3), mr3_energy_numpy(embed(F - H), **MR3)
            expected[i, j] = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(piola, expected, rtol=1e-7, atol=1e-7 * np.abs(expected).max())


def test_mr3_piola_tangent_matches_directional_finite_difference(mr3_material):
    """ufl.derivative of P, the tangent the Jacobian is built from, is the derivative of P in u, so P is not
    cut off from u by the UFL variable it is differentiated with respect to."""
    mesh, u, constants = mr3_material
    dim = u.ufl_shape[0]
    F = deformed(dim)
    set_deformation(u, F)
    direction = np.zeros((dim, dim))
    direction[0, 0], direction[0, 1], direction[1, 0], direction[1, 1] = 0.2, 0.3, -0.15, -0.1
    increment = dolfinx.fem.Function(u.function_space)
    increment.interpolate(lambda x: direction @ x[:dim])
    piola = Solid.MR3_piola(u, **constants)
    tangent = evaluate(ufl.derivative(piola, u, increment), mesh)
    assert np.abs(tangent).max() > 1e4
    epsilon = 1e-6
    set_deformation(u, F + epsilon * direction)
    plus = evaluate(piola, mesh)
    set_deformation(u, F - epsilon * direction)
    minus = evaluate(piola, mesh)
    np.testing.assert_allclose(tangent, (plus - minus) / (2.0 * epsilon), rtol=1e-6, atol=1e-6 * np.abs(tangent).max())


def test_mr3_small_strain_moduli(mr3_material):
    """For a small strain eps, P = 2 mu0 dev(eps) + K0 tr(eps) I, to first order, with mu0 = 2 (C10 + C01),
    K0 = 2 / D1, the deviator taken in 3D also in plane strain; this is the small-strain stress of the SVK
    solid with mu = mu0 and lambda = K0 - 2 mu0 / 3."""
    mesh, u, constants = mr3_material
    dim = u.ufl_shape[0]
    mu0, K0 = 2.0 * (MR3["C10"] + MR3["C01"]), 2.0 / MR3["D1"]
    gradient = np.array([[0.3, 0.5, -0.2], [0.1, -0.4, 0.25], [0.2, 0.15, 0.6]])[:dim, :dim]
    scale = 1e-7
    set_deformation(u, np.eye(dim) + scale * gradient)
    eps = 0.5 * (gradient + gradient.T)
    expected = 2.0 * mu0 * (eps - np.trace(eps) / 3.0 * np.eye(dim)) + K0 * np.trace(eps) * np.eye(dim)
    piola = evaluate(Solid.MR3_piola(u, **constants), mesh) / scale
    np.testing.assert_allclose(piola, expected, rtol=1e-5, atol=1e-5 * np.abs(expected).max())

    lambda_ = dolfinx.fem.Constant(mesh, K0 - 2.0 * mu0 / 3.0)
    svk = evaluate(Solid.STVK(u, lambda_, dolfinx.fem.Constant(mesh, mu0)), mesh) / scale
    np.testing.assert_allclose(piola, svk, rtol=1e-5, atol=1e-5 * np.abs(expected).max())


@pytest.mark.parametrize("gamma", [0.1, 0.4])
def test_mr3_finite_simple_shear(mr3_material, gamma):
    """For F = I + gamma e1 e2, J = 1, I1bar = I2bar = 3 + gamma^2, so W = (C10 + C01) gamma^2 + C11 gamma^4
    and P12 = 2 (C10 + C01) gamma + 4 C11 gamma^3."""
    mesh, u, constants = mr3_material
    dim = u.ufl_shape[0]
    F = np.eye(dim)
    F[0, 1] = gamma
    set_deformation(u, F)
    c = MR3["C10"] + MR3["C01"]
    assert evaluate(mr3_energy_ufl(u, constants), mesh) == pytest.approx(
        c * gamma**2 + MR3["C11"] * gamma**4, rel=1e-12
    )
    piola = evaluate(Solid.MR3_piola(u, **constants), mesh)
    assert piola[0, 1] == pytest.approx(2.0 * c * gamma + 4.0 * MR3["C11"] * gamma**3, rel=1e-12)
    # C11 matters: at gamma = 0.4 it more than doubles the shear stress of the two-term material
    if gamma == 0.4:
        assert 4.0 * MR3["C11"] * gamma**3 > 2.0 * c * gamma


def test_mr3_pure_dilation_3d():
    """For F = s I, the distortional energy vanishes, W = (J - 1)^2 / D1, and the Cauchy stress is
    2 (J - 1) / D1 I."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,))))
    constants = {name: dolfinx.fem.Constant(mesh, value) for name, value in MR3.items()}
    for s in (0.9, 1.1):
        set_deformation(u, s * np.eye(3))
        J = s**3
        assert evaluate(mr3_energy_ufl(u, constants), mesh) == pytest.approx((J - 1) ** 2 / MR3["D1"], rel=1e-12)
        F = ufl.Identity(3) + ufl.grad(u)
        cauchy = evaluate(Solid.MR3_piola(u, **constants) * F.T / ufl.det(F), mesh)
        np.testing.assert_allclose(cauchy, 2.0 * (J - 1) / MR3["D1"] * np.eye(3), rtol=1e-12, atol=1e-12 * 1e6)


def test_mr3_plane_strain_is_the_embedded_3d_state():
    """The 2D P is the in-plane block of the 3D P of the embedded state F3 = diag(F, 1), evaluated by the 3D
    code path; the out-of-plane stress of that state does not vanish (plane strain, not plane stress)."""
    mesh2 = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    mesh3 = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    u2 = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh2, ("Lagrange", 1, (2,))))
    u3 = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh3, ("Lagrange", 1, (3,))))
    for F in (deformed(2), 1.1 * np.eye(2)):  # a general state, and an in-plane dilation
        set_deformation(u2, F)
        set_deformation(u3, embed(F))
        piola2 = evaluate(Solid.MR3_piola(u2, **{k: dolfinx.fem.Constant(mesh2, v) for k, v in MR3.items()}), mesh2)
        piola3 = evaluate(Solid.MR3_piola(u3, **{k: dolfinx.fem.Constant(mesh3, v) for k, v in MR3.items()}), mesh3)
        np.testing.assert_allclose(piola2, piola3[:2, :2], rtol=1e-12, atol=1e-12 * np.abs(piola3).max())
        np.testing.assert_allclose(piola3[:2, 2], 0.0, atol=1e-9)
        assert abs(piola3[2, 2]) > 1e3


def test_mr3_energy_rejects_a_2d_deformation_gradient():
    with pytest.raises(ValueError):
        Solid.MR3_energy(ufl.Identity(2), 1.0, 1.0, 1.0, 1.0)
