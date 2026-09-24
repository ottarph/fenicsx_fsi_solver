"""Checks of the full and no-ALE Jacobians of the shared-space FSI solver.

The Jacobians are evaluated at a small, nonzero, admissible state (positive
deformation Jacobian everywhere) and compared with central finite differences
of the assembled residual. The no-ALE Jacobian is not the derivative of the
residual; its displacement column is compared with a linearization in which
the geometry of the fluid momentum and incompressibility terms is frozen.
"""

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.solvers.fsi2_harmonic_diffmesh import (
    PHYSICAL_MARKERS,
    build_problem,
    jacobian_forms,
    restrict_to_cells,
)

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"
DT = 0.0025


def set_admissible_state(problem):
    """Smooth nonzero state vanishing on the channel walls, with det(F) > 0."""
    L, H = 2.5, 0.41

    def bump(x):
        return np.sin(np.pi * x[0] / L) * np.sin(np.pi * x[1] / H)

    problem.u.interpolate(lambda x: np.vstack((0.004 * bump(x), 0.006 * bump(x) * np.cos(3 * x[0]))))
    problem.u_old.interpolate(lambda x: np.vstack((0.003 * bump(x), 0.005 * bump(x) * np.cos(3 * x[0]))))
    problem.v.interpolate(lambda x: np.vstack((0.8 * bump(x) + 0.1 * x[1], 0.3 * bump(x) * np.sin(4 * x[0]))))
    problem.v_old.interpolate(lambda x: np.vstack((0.7 * bump(x), 0.2 * bump(x))))
    problem.p.interpolate(lambda x: 50.0 * (2.5 - x[0]) + 10.0 * np.sin(5 * x[1]))
    for f in (*problem.solution, problem.u_old, problem.v_old):
        f.x.scatter_forward()

    Q = dfx.fem.functionspace(problem.mesh, ("DG", 0))
    detF = dfx.fem.Function(Q)
    detF.interpolate(dfx.fem.Expression(ufl.det(ufl.Identity(2) + ufl.grad(problem.u)), Q.element.interpolation_points))
    min_det = problem.mesh.comm.allreduce(detF.x.array.min(), op=MPI.MIN)
    assert min_det > 0.9


def random_direction(space, seed):
    d = dfx.fem.Function(space)
    rng = np.random.default_rng(seed + space.mesh.comm.rank)
    d.x.array[:] = rng.standard_normal(d.x.array.size)
    d.x.scatter_forward()
    return d


def assemble_vector(form, entity_maps):
    b = dfx.fem.petsc.assemble_vector(dfx.fem.form(form, entity_maps=entity_maps))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b


def assemble_matrix(form, entity_maps):
    A = dfx.fem.petsc.assemble_matrix(dfx.fem.form(form, entity_maps=entity_maps))
    A.assemble()
    return A


def central_difference(residual_i, w_j, d_j, eps, entity_maps):
    w0 = w_j.x.array.copy()
    w_j.x.array[:] = w0 + eps * d_j.x.array
    r_plus = assemble_vector(residual_i, entity_maps)
    w_j.x.array[:] = w0 - eps * d_j.x.array
    r_minus = assemble_vector(residual_i, entity_maps)
    w_j.x.array[:] = w0
    r_plus.axpy(-1.0, r_minus)
    r_plus.scale(0.5 / eps)
    r_minus.destroy()
    return r_plus


def jacobian_action(form, d_j, entity_maps):
    A = assemble_matrix(form, entity_maps)
    y = A.createVecLeft()
    A.mult(d_j.x.petsc_vec, y)
    A.destroy()
    return y


# finite-difference step per field, relative to the field scale
FD_EPS = {0: 1e-7, 1: 1e-5, 2: 1e-2}


@pytest.fixture(scope="module")
def problem():
    problem = build_problem(MESH, DT)
    set_admissible_state(problem)
    return problem


def frozen_fluid_geometry_residual(problem):
    """Residual with the fluid momentum/incompressibility geometry frozen at the current ``u``."""
    u_frozen = dfx.fem.Function(problem.U)
    u_frozen.x.array[:] = problem.u.x.array
    F = list(problem.residual)
    fluid = PHYSICAL_MARKERS["ALE_fluid"]
    solid = PHYSICAL_MARKERS["solid"]
    F[1] = restrict_to_cells(F[1], solid) + ufl.replace(restrict_to_cells(F[1], fluid), {problem.u: u_frozen})
    F[2] = ufl.replace(F[2], {problem.u: u_frozen})
    return F


@pytest.mark.parametrize("mode", ["full", "no_ale"])
def test_jacobian_matches_finite_differences(problem, mode):
    J = jacobian_forms(problem, mode)
    F = problem.residual if mode == "full" else frozen_fluid_geometry_residual(problem)
    w = problem.solution
    for j, w_j in enumerate(w):
        d_j = random_direction(w_j.function_space, seed=10 + j)
        for i in range(3):
            fd = central_difference(F[i], w_j, d_j, FD_EPS[j], problem.entity_maps)
            fd_norm = fd.norm()
            if J[i][j] is None:
                assert fd_norm == pytest.approx(0.0, abs=1e-8), f"block ({i},{j}) should vanish"
                continue
            Jd = jacobian_action(J[i][j], d_j, problem.entity_maps)
            Jd.axpy(-1.0, fd)
            assert fd_norm > 0.0
            assert Jd.norm() <= 1e-6 * fd_norm, f"block ({i},{j}) differs from finite differences"


def test_no_ale_omits_nontrivial_fluid_derivatives(problem):
    J_full = jacobian_forms(problem, "full")
    J_no_ale = jacobian_forms(problem, "no_ale")

    assert J_no_ale[2][0] is None
    for i in range(3):
        for j in range(1, 3):
            assert (J_full[i][j] is None) == (J_no_ale[i][j] is None)
    assert (J_full[0][0] is None) == (J_no_ale[0][0] is None)

    d_u = random_direction(problem.U, seed=3)
    full_vu = jacobian_action(J_full[1][0], d_u, problem.entity_maps)
    solid_vu = jacobian_action(J_no_ale[1][0], d_u, problem.entity_maps)
    full_pu = jacobian_action(J_full[2][0], d_u, problem.entity_maps)

    # the solid tangent survives, and the omitted fluid derivatives are far above
    # rounding (with a random direction the solid stiffness dominates the norm)
    assert solid_vu.norm() > 0.0
    diff = full_vu.copy()
    diff.axpy(-1.0, solid_vu)
    assert diff.norm() > 1e-5 * full_vu.norm()
    assert full_pu.norm() > 0.0

    # the omitted part is exactly the fluid-domain contribution
    fluid_vu = jacobian_action(
        ufl.derivative(restrict_to_cells(problem.residual[1], PHYSICAL_MARKERS["ALE_fluid"]), problem.u,
                       ufl.TrialFunction(problem.U)),
        d_u, problem.entity_maps)
    diff.axpy(-1.0, fluid_vu)
    assert diff.norm() <= 1e-12 * full_vu.norm()


def test_restrict_to_cells_rejects_facet_integrals(problem):
    with pytest.raises(ValueError):
        restrict_to_cells(problem.residual[0], PHYSICAL_MARKERS["solid"])
