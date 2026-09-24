"""Field-split index sets and block factorization of the shared-space FSI Jacobian.

These tests are MPI-aware; ``test_fieldsplit_on_two_ranks`` reruns this module
on two ranks, because a serial run cannot establish correct distributed or
nested-split numbering.
"""

import os
import subprocess
import sys
from pathlib import Path

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc
from test_fsi2_harmonic_diffmesh_jacobian import set_admissible_state

from xfsi_solver.linalg.fieldsplit import (
    check_partition,
    field_index_sets,
    nested_index_sets,
    union_index_set,
)
from xfsi_solver.solvers.fsi2_harmonic_diffmesh import build_problem, jacobian_forms
from xfsi_solver.solvers.fsi2_harmonic_diffmesh_fieldsplit import configure_schur_fieldsplit, setup_with_options

ROOT = Path(__file__).resolve().parent.parent
MESH = str(ROOT / "data/meshes/fsi2/mesh_sec_coarse.xdmf")
DT = 0.0025


@pytest.fixture(scope="module")
def problem():
    problem = build_problem(MESH, DT)
    set_admissible_state(problem)
    return problem


def assemble_jacobian(problem, mode):
    forms = dfx.fem.form(jacobian_forms(problem, mode), entity_maps=problem.entity_maps)
    A = dfx.fem.petsc.assemble_matrix(forms, bcs=problem.bcs, diag=1.0)
    A.assemble()
    return A


@pytest.fixture(scope="module")
def jacobian(problem):
    return assemble_jacobian(problem, "no_ale")


def random_vector(A, seed):
    b = A.createVecLeft()
    rng = np.random.default_rng(seed + MPI.COMM_WORLD.rank)
    b.array[:] = rng.standard_normal(b.getLocalSize())
    return b


def lu_solve(A, b):
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    x = b.duplicate()
    ksp.solve(b, x)
    ksp.destroy()
    return x


def relative_difference(x, y):
    d = x.copy()
    d.axpy(-1.0, y)
    return d.norm() / y.norm()


def test_field_index_sets_partition_owned_rows(problem, jacobian):
    spaces = [problem.U, problem.V, problem.P]
    sets = field_index_sets(jacobian, spaces)
    check_partition(sets, *jacobian.getOwnershipRange())
    for V, index_set in zip(spaces, sets, strict=True):
        assert index_set.getSize() == V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    # the index sets select exactly the entries DOLFINx assigns to each field
    x = dfx.fem.petsc.create_vector(spaces)
    for f in problem.solution:
        f.x.array[:] = np.arange(f.x.array.size) + 1000.0 * MPI.COMM_WORLD.rank
    dfx.fem.petsc.assign(problem.solution, x)
    for f, index_set in zip(problem.solution, sets, strict=True):
        n_owned = f.function_space.dofmap.index_map.size_local * f.function_space.dofmap.index_map_bs
        sub = x.getSubVector(index_set)
        np.testing.assert_array_equal(sub.array, f.x.array[:n_owned])
        x.restoreSubVector(index_set, sub)
    set_admissible_state(problem)

    with pytest.raises(ValueError):
        check_partition([sets[0], sets[0], sets[2]], *jacobian.getOwnershipRange())


def test_nested_index_sets_use_submatrix_numbering(problem, jacobian):
    is_u, is_v, is_p = field_index_sets(jacobian, [problem.U, problem.V, problem.P])
    is_vp = union_index_set([is_v, is_p])
    vp_v, vp_p = nested_index_sets(is_vp, [is_v, is_p])

    A_vp = jacobian.createSubMatrix(is_vp, is_vp)
    check_partition([vp_v, vp_p], *A_vp.getOwnershipRange())
    for (row_parent, col_parent), (row_nested, col_nested) in [
        ((is_v, is_v), (vp_v, vp_v)),
        ((is_v, is_p), (vp_v, vp_p)),
        ((is_p, is_v), (vp_p, vp_v)),
    ]:
        direct = jacobian.createSubMatrix(row_parent, col_parent)
        nested = A_vp.createSubMatrix(row_nested, col_nested)
        assert direct.norm() > 0.0
        nested.axpy(-1.0, direct, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        assert nested.norm() <= 1e-14 * direct.norm()

    with pytest.raises(ValueError):
        nested_index_sets(is_vp, [is_u])


def test_exact_block_factorization_matches_lu(problem, jacobian):
    A = jacobian
    b = random_vector(A, seed=1)
    x_lu = lu_solve(A, b)

    is_u, is_v, is_p = field_index_sets(A, [problem.U, problem.V, problem.P])
    is_vp = union_index_set([is_v, is_p])

    # PCFIELDSPLIT full Schur factorization with exact subsolves
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOptionsPrefix("test_exact_fieldsplit_")
    ksp.setOperators(A)
    ksp.setType("preonly")
    options = configure_schur_fieldsplit(ksp, is_u, is_vp, "exact")
    setup_with_options(ksp, options)
    x_fs = b.duplicate()
    ksp.solve(b, x_fs)
    assert relative_difference(x_fs, x_lu) < 1e-9

    # the same correction from the reduced right-hand side and displacement recovery
    A00 = A.createSubMatrix(is_u, is_u)
    A01 = A.createSubMatrix(is_u, is_vp)
    A10 = A.createSubMatrix(is_vp, is_u)
    A11 = A.createSubMatrix(is_vp, is_vp)
    b_u, b_vp = b.getSubVector(is_u), b.getSubVector(is_vp)
    x_u_lu, x_vp_lu = x_lu.getSubVector(is_u), x_lu.getSubVector(is_vp)

    ksp_A = PETSc.KSP().create(A.comm)
    ksp_A.setOperators(A00)
    ksp_A.setType("preonly")
    ksp_A.getPC().setType("lu")
    ksp_A.getPC().setFactorSolverType("mumps")

    def solve_A00(r):
        y = A00.createVecLeft()
        ksp_A.solve(r, y)
        return y

    # reduced right-hand side r_vp - A10 A00^{-1} r_u
    reduced_rhs = b_vp.copy()
    tmp = A10.createVecLeft()
    A10.mult(solve_A00(b_u), tmp)
    reduced_rhs.axpy(-1.0, tmp)

    # S x_vp = A11 x_vp - A10 A00^{-1} A01 x_vp must reproduce it
    S_x = A11.createVecLeft()
    A11.mult(x_vp_lu, S_x)
    coupling = A01.createVecLeft()
    A01.mult(x_vp_lu, coupling)
    A10.mult(solve_A00(coupling), tmp)
    S_x.axpy(-1.0, tmp)
    assert relative_difference(S_x, reduced_rhs) < 1e-9

    # displacement recovery A00^{-1} (r_u - A01 x_vp)
    recovery_rhs = b_u.copy()
    recovery_rhs.axpy(-1.0, coupling)
    assert relative_difference(solve_A00(recovery_rhs), x_u_lu) < 1e-9

    ksp.destroy()
    ksp_A.destroy()


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1 or os.environ.get("XFSI_SKIP_MPI_TESTS"), reason="launcher only")
def test_fieldsplit_on_two_ranks():
    env = dict(os.environ, XFSI_SKIP_MPI_TESTS="1")
    cmd = ["mpiexec", "-n", "2", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
