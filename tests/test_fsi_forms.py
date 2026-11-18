"""The residual of FSIForms with the solid materials, on a unit square split into a fluid and a solid half."""

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from xfsi_solver.fsi.forms import FSIForms
from xfsi_solver.fsi.materials import MR3_FSI2_MATCHED

FLUID, SOLID = 1, 2
DT, THETA = 0.0025, 0.5025
LAMBDA_S, MU_S = 2.0e6, 5.0e5
MR3 = MR3_FSI2_MATCHED | {"D1": 2.0 / (LAMBDA_S + 2.0 * MU_S / 3.0)}


@pytest.fixture
def setup():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, np.arange(32, dtype=np.int32))
    cell_tags = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim, np.arange(32, dtype=np.int32), np.where(midpoints[:, 0] < 0.5, FLUID, SOLID)
    )
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (2,)))
    P = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    dv, dp = ufl.TestFunctions(ufl.MixedFunctionSpace(V, P))
    v, v_old, u_f, u_f_old, u_s_old = (dolfinx.fem.Function(V) for _ in range(5))
    p = dolfinx.fem.Function(P)

    # A bending-like finite deformation with J in about [0.85, 1.15], the velocity of a rotation, and a pressure
    u_s_old.interpolate(lambda x: np.stack([0.15 * x[0] * x[1] - 0.05 * x[1] ** 2, 0.2 * x[0] ** 2 + 0.1 * x[1]]))
    u_f_old.interpolate(lambda x: np.stack([0.02 * x[0] * x[1], -0.01 * x[0] ** 2]))
    u_f.interpolate(lambda x: np.stack([0.021 * x[0] * x[1], -0.011 * x[0] ** 2]))
    v_old.interpolate(lambda x: np.stack([-(x[1] - 0.5), x[0] - 0.5]))
    v.interpolate(lambda x: np.stack([-1.1 * (x[1] - 0.5) + 0.3 * x[0] ** 2, (x[0] - 0.5) - 0.2 * x[1]]))
    p.interpolate(lambda x: 100.0 * x[0])

    def forms(rho_f=1e3, rho_s=1e4, **material):
        values = dict(rho_f=rho_f, nu_f=1e-3, rho_s=rho_s, mu_s=MU_S, lambda_s=LAMBDA_S, dt=DT, theta=THETA)
        constants = {name: dolfinx.fem.Constant(mesh, value) for name, value in values.items()}
        if "mooney_rivlin_3" in material:
            material["mooney_rivlin_3"] = {
                k: dolfinx.fem.Constant(mesh, val) for k, val in material["mooney_rivlin_3"].items()
            }
        return FSIForms(dv=dv, dp=dp, dx_fluid=dx(FLUID), dx_solid=dx(SOLID), **constants, **material)

    return forms, v, v_old, p, u_f, u_f_old, u_s_old


def assemble_momentum(residual):
    """The momentum block of the residual, as an array."""
    return dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.extract_blocks(residual)[0])).array.copy()


@pytest.mark.parametrize("C11", [MR3["C11"], 0.0], ids=["C11", "C11_0"])
def test_mooney_rivlin_3_residual_jacobian_in_v_through_the_condensed_displacement(setup, C11):
    """The momentum residual, with u_s = u_s_old + dt ((1 - theta) v_old + theta v), is differentiated in v,
    so the solid stress enters the Jacobian through the condensed displacement. With rho_f = rho_s = 0, the
    momentum residual depends on v only through it, and the Taylor remainder ||R(v + h dv) - R(v) - h J dv||
    is O(h^2); with the stress cut off from v, J dv would vanish and the remainder be O(h)."""
    forms, v, v_old, p, u_f, u_f_old, u_s_old = setup
    fsi_forms = forms(solid_material="mooney_rivlin_3", mooney_rivlin_3=MR3 | {"C11": C11}, rho_f=0.0, rho_s=0.0)
    residual = fsi_forms.build_residual(v, v_old, p, u_f, u_f_old, u_s_old)
    direction = dolfinx.fem.Function(v.function_space)
    direction.interpolate(lambda x: np.stack([np.sin(3 * x[1]) * x[0], np.cos(2 * x[0]) * x[1]]))
    tangent = dolfinx.fem.form(ufl.derivative(ufl.extract_blocks(residual)[0], v, direction))
    J_direction = dolfinx.fem.assemble_vector(tangent).array.copy()

    R0 = assemble_momentum(residual)
    v0 = v.x.array.copy()
    remainders, first_order = [], []
    for h in (0.4, 0.2, 0.1):
        v.x.array[:] = v0 + h * direction.x.array
        difference = assemble_momentum(residual) - R0
        first_order.append(np.linalg.norm(difference))
        remainders.append(np.linalg.norm(difference - h * J_direction))
    v.x.array[:] = v0
    # The tangent carries the change of the residual, which is nonzero and first order in h
    assert np.linalg.norm(J_direction) > 0
    assert remainders[-1] < 1e-2 * first_order[-1]
    rates = np.log2(np.array(remainders[:-1]) / np.array(remainders[1:]))
    np.testing.assert_allclose(rates, 2.0, atol=0.1)


def test_mooney_rivlin_3_c11_changes_the_solid_residual_at_finite_strain(setup):
    """At the finite deformation of the fixture, the third term changes the solid stress noticeably."""
    forms, v, v_old, p, u_f, u_f_old, u_s_old = setup
    with_c11 = assemble_momentum(forms(solid_material="mooney_rivlin_3", mooney_rivlin_3=MR3).A_E_solid(u_s_old))
    without = assemble_momentum(
        forms(solid_material="mooney_rivlin_3", mooney_rivlin_3=MR3 | {"C11": 0.0}).A_E_solid(u_s_old)
    )
    assert np.linalg.norm(with_c11 - without) > 0.01 * np.linalg.norm(without)


def test_stvk_is_the_default_solid(setup):
    forms, v, v_old, p, u_f, u_f_old, u_s_old = setup
    default = assemble_momentum(forms().build_residual(v, v_old, p, u_f, u_f_old, u_s_old))
    stvk = assemble_momentum(forms(solid_material="stvk").build_residual(v, v_old, p, u_f, u_f_old, u_s_old))
    np.testing.assert_array_equal(default, stvk)
    mr3 = assemble_momentum(
        forms(solid_material="mooney_rivlin_3", mooney_rivlin_3=MR3).build_residual(v, v_old, p, u_f, u_f_old, u_s_old)
    )
    assert not np.allclose(mr3, default)


@pytest.mark.parametrize(
    "material",
    [
        {"solid_material": "neo_hookean"},
        {"solid_material": "mooney_rivlin_3"},
        {"solid_material": "mooney_rivlin_3", "mooney_rivlin_3": {"C10": 0.0, "C01": 1.0, "C11": 0.0}},
        {"mooney_rivlin_3": MR3},
    ],
    ids=["unknown", "no_coefficients", "no_D1", "coefficients_without_material"],
)
def test_fsi_forms_reject_an_inconsistent_material(setup, material):
    forms = setup[0]
    with pytest.raises(ValueError):
        forms(**material)
