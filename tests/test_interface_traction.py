import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.fsi.forms import FSIForms
from xfsi_solver.tools.interface_traction import (
    STRESSES,
    check_interface_traction_jacobian,
    cofactor,
    interface_traction_description,
    interface_traction_form,
    jacobian_blocks,
)

DT, THETA = 0.0025, 0.505


@pytest.mark.parametrize("stress", [None, *STRESSES])
def test_stress_accepted(stress):
    check_interface_traction_jacobian(stress, interface_ale=False)


@pytest.mark.parametrize("stress", ["viscous", "Pressure", ("pressure",)])
def test_stress_rejected(stress):
    with pytest.raises(ValueError, match="interface_traction_jacobian must be"):
        check_interface_traction_jacobian(stress, interface_ale=False)


def test_rejected_with_interface_ale():
    check_interface_traction_jacobian(None, interface_ale=True)
    with pytest.raises(ValueError, match="does not go with interface_ale"):
        check_interface_traction_jacobian("pressure", interface_ale=True)


def test_description():
    assert interface_traction_description(None) == "off"
    assert "pressure's" in interface_traction_description("pressure")
    assert "viscous" in interface_traction_description("stress")


@pytest.fixture
def state():
    """Fields on the unit square, whose whole boundary stands in for the interface, with u_f set by the
    kinematic relation from u_f_old, v and v_old, as the solvers set it on the interface."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (2,)))
    P = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    v, v_old, u_f, u_f_old = (dolfinx.fem.Function(V) for _ in range(4))
    v.interpolate(lambda x: np.stack([np.sin(3 * x[1]), x[0] * x[1]]))
    v_old.interpolate(lambda x: np.stack([x[1] ** 2, np.cos(2 * x[0])]))
    u_f_old.interpolate(lambda x: np.stack([0.05 * x[0] * x[1], 0.08 * np.sin(np.pi * x[0])]))
    u_f.x.array[:] = u_f_old.x.array + DT * (THETA * v.x.array + (1.0 - THETA) * v_old.x.array)
    p = dolfinx.fem.Function(P)
    p.interpolate(lambda x: 1.0e4 + 300.0 * x[0] - 200.0 * x[1] ** 2)

    def constant(value):
        return dolfinx.fem.Constant(mesh, value)

    forms = FSIForms(
        dv=ufl.TestFunction(V),
        dp=ufl.TestFunction(P),
        dx_fluid=ufl.dx,
        dx_solid=ufl.dx,
        rho_f=constant(1.0e3),
        nu_f=constant(1.0e-3),
        rho_s=constant(1.0e4),
        mu_s=constant(5.0e5),
        lambda_s=constant(2.0e6),
        dt=constant(DT),
        theta=constant(THETA),
    )
    ds = ufl.Measure("ds", domain=mesh)
    return {"forms": forms, "v": v, "v_old": v_old, "p": p, "u_f": u_f, "u_f_old": u_f_old, "ds": ds}


def traction(stress, state):
    s = state
    return interface_traction_form(stress, s["forms"], s["v"], s["v_old"], s["p"], s["u_f"], s["u_f_old"], s["ds"])


def assemble_vector(form):
    vector = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(form))
    vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vector


def test_no_terms_without_a_stress(state):
    assert traction(None, state) is None


@pytest.mark.parametrize("stress", STRESSES)
def test_terms_are_zero_where_u_f_holds_the_kinematic_relation(stress, state):
    # The pressure traction of the new geometry alone, for the scale
    s = state
    forms, n = s["forms"], ufl.FacetNormal(s["u_f"].function_space.mesh)
    one_side = DT * THETA * ufl.inner(-s["p"] * cofactor(s["u_f"]) * n, forms.dv) * s["ds"]
    scale = assemble_vector(one_side).norm()
    assert scale > 1.0
    assert assemble_vector(traction(stress, state)).norm() < 1e-13 * scale


def test_jacobian_holds_the_derivative_of_the_pressure_traction_through_the_interface_motion(state):
    # The terms' derivative with respect to v, in the direction w, is that of the pressure's traction
    # dt * theta * (-p) cof(F(u_f)) N . dv as u_f moves by theta * dt * w. In 2d, cof(F) is linear in grad(u_f),
    # so a central difference gives the derivative to round-off.
    s = state
    forms, v = s["forms"], s["v"]
    residual = ufl.inner(v, forms.dv) * ufl.dx
    [[with_terms]] = jacobian_blocks([residual], [v], traction("pressure", state))
    [[without]] = jacobian_blocks([residual], [v])
    matrices = []
    for form in (with_terms, without):
        matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(form))
        matrix.assemble()
        matrices.append(matrix)
    w = dolfinx.fem.Function(v.function_space)
    w.interpolate(lambda x: np.stack([np.cos(x[0] + 2 * x[1]), x[0] ** 2 - x[1]]))
    derivative = matrices[0].createVecLeft()
    matrices[0].axpy(-1.0, matrices[1])
    matrices[0].mult(w.x.petsc_vec, derivative)

    u = dolfinx.fem.Function(v.function_space)
    n = ufl.FacetNormal(v.function_space.mesh)
    force = DT * THETA * ufl.inner(-s["p"] * cofactor(u) * n, forms.dv) * s["ds"]
    eps = 1.0e-3
    forces = []
    for sign in (1.0, -1.0):
        u.x.array[:] = s["u_f"].x.array + sign * eps * THETA * DT * w.x.array
        forces.append(assemble_vector(force))
    difference = (forces[0] - forces[1]) / (2 * eps)

    assert derivative.norm() > 0.0
    assert (derivative - difference).norm() < 1e-10 * derivative.norm()
