"""The fieldsplit solver with the stiffened elastic mesh extension.

The production displacement preconditioner extracts the fluid-interior
block from the preconditioning matrix; these checks establish that it is the
elastic operator, that the iterative alternatives solve it, and that the
hierarchy and block factorization are unchanged. MPI-aware;
``test_elastic_fieldsplit_on_two_ranks`` reruns this module on two ranks.
"""

import os
import subprocess
import sys
from pathlib import Path

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from test_fsi2_harmonic_diffmesh_jacobian import set_admissible_state

from xfsi_solver.linalg.fieldsplit import check_partition, field_dof_rows, field_index_sets, union_index_set
from xfsi_solver.solvers import fsi2_harmonic_diffmesh as harmonic
from xfsi_solver.solvers.fsi2_harmonic_diffmesh import SolverConfig, jacobian_forms
from xfsi_solver.solvers.fsi2_harmonic_diffmesh_fieldsplit import (
    DisplacementPC,
    FieldSplitConfig,
    configure_schur_fieldsplit,
    setup_with_options,
)
from xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh import OPTIONS_PREFIX, build_problem

ROOT = Path(__file__).resolve().parent.parent
MESH = str(ROOT / "data/meshes/fsi2/mesh_sec_coarse.xdmf")
DT = 0.0025
comm = MPI.COMM_WORLD


@pytest.fixture(scope="module")
def problem():
    problem = build_problem(MESH, DT)
    set_admissible_state(problem)
    return problem


def assemble(forms, problem, bcs):
    A = dfx.fem.petsc.assemble_matrix(dfx.fem.form(forms, entity_maps=problem.entity_maps), bcs=bcs, diag=1.0)
    A.assemble()
    return A


def relative_difference(x, y):
    d = x.copy()
    d.axpy(-1.0, y)
    return d.norm() / y.norm()


@pytest.fixture(scope="module")
def jacobian(problem):
    return assemble(jacobian_forms(problem, "no_ale"), problem, problem.bcs)


def test_exact_block_factorization_matches_lu(problem, jacobian):
    A = jacobian
    is_u, is_v, is_p = field_index_sets(A, [problem.U, problem.V, problem.P])
    check_partition([is_u, is_v, is_p], *A.getOwnershipRange())
    b = A.createVecLeft()
    b.array[:] = np.random.default_rng(1 + comm.rank).standard_normal(b.getLocalSize())

    lu = PETSc.KSP().create(A.comm)
    lu.setOperators(A)
    lu.setType("preonly")
    lu.getPC().setType("lu")
    lu.getPC().setFactorSolverType("mumps")
    x_lu = b.duplicate()
    lu.solve(b, x_lu)

    ksp = PETSc.KSP().create(A.comm)
    ksp.setOptionsPrefix("test_elastic_exact_fieldsplit_")
    ksp.setOperators(A)
    ksp.setType("preonly")
    setup_with_options(ksp, configure_schur_fieldsplit(ksp, is_u, union_index_set([is_v, is_p]), "exact"))
    x_fs = b.duplicate()
    ksp.solve(b, x_fs)
    assert relative_difference(x_fs, x_lu) < 1e-9
    for obj in (ksp, lu):
        obj.destroy()


@pytest.fixture
def fieldsplit_solver(problem):
    created = []

    def make(fs_config):
        config = SolverConfig(jacobian_mode="no_ale", linear_solver="fieldsplit", fieldsplit=fs_config,
                              snes_monitor=False)
        nonlinear_problem, solver = harmonic.create_nonlinear_problem(problem, config, OPTIONS_PREFIX)
        created.append(solver)
        dfx.fem.petsc.assign(problem.solution, nonlinear_problem.x)
        nonlinear_problem.solver.computeJacobian(nonlinear_problem.x, nonlinear_problem.A, nonlinear_problem.A)
        b = nonlinear_problem.A.createVecLeft()
        b.array[:] = np.random.default_rng(5 + comm.rank).standard_normal(b.getLocalSize())
        y = b.duplicate()
        solver.ksp.getPC().apply(b, y)
        return nonlinear_problem, solver

    yield make
    for solver in created:
        solver.destroy()


def fluid_interior_block(problem, ctx, A00):
    """The alpha-scaled elastic stiffness, assembled from the mesh operator, on the ``f`` DOFs of ``ctx``."""
    op = problem.mesh_operator
    U = problem.U
    stiffness = problem.constants["alpha_u"] * op.volume_form(ufl.TrialFunction(U), ufl.TestFunction(U))
    blocks = []
    for bcs in (problem.bcs_u, []):
        K = assemble(stiffness, problem, bcs)
        rows_K = field_dof_rows(K, [U])[0]
        is_f_K = PETSc.IS().createGeneral(rows_K[ctx.fluid_dofs].astype(PETSc.IntType), comm=K.comm)
        blocks.append(K.createSubMatrix(is_f_K, is_f_K))
    return blocks


def test_displacement_pc_extracts_elastic_block(fieldsplit_solver, problem):
    """DisplacementPC's A_ff is the elastic mesh operator, found algebraically in the preconditioning matrix."""
    _, solver = fieldsplit_solver(FieldSplitConfig())
    ctx = next(c for c in solver.contexts if isinstance(c, DisplacementPC))
    A00 = solver.ksp.getPC().getFieldSplitSubKSP()[0].getOperators()[1]
    check_partition([ctx.is_I, ctx.is_f], *A00.getOwnershipRange())

    K_ff, K_ff_without_bcs = fluid_interior_block(problem, ctx, A00)
    # (the unit Dirichlet diagonal dominates the norm of A_ff, the mesh entries carry alpha)
    scale = K_ff_without_bcs.norm()
    D = ctx.A_ff.copy()
    D.axpy(-1.0, K_ff, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    assert D.norm() <= 1e-12 * scale
    assert ctx.A_ff.isSymmetric(1e-12 * scale)

    # the harmonic block is different
    harmonic_problem = harmonic.build_problem(MESH, DT)
    set_admissible_state(harmonic_problem)
    A_h = assemble(jacobian_forms(harmonic_problem, "no_ale")[0][0], harmonic_problem, harmonic_problem.bcs_u)
    A_h_ff = A_h.createSubMatrix(ctx.is_f, ctx.is_f)
    A_h_ff.axpy(-1.0, ctx.A_ff, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    assert A_h_ff.norm() > 0.1 * scale

    # the dropped mesh terms in the solid/interface block stay small relative to the solid mass
    A_II = A00.createSubMatrix(ctx.is_I, ctx.is_I)
    A_II.axpy(-1.0, ctx.M_II, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    assert A_II.norm() <= 1e-6 * ctx.M_II.norm()


@pytest.mark.parametrize("displacement_fluid", ["amg", "gamg"])
def test_iterative_fluid_displacement_solvers(fieldsplit_solver, displacement_fluid):
    """BoomerAMG and GAMG (rigid body near-nullspace) CG solve the elastic A_ff like Cholesky."""
    _, solver = fieldsplit_solver(FieldSplitConfig(displacement_fluid=displacement_fluid))
    ctx = next(c for c in solver.contexts if isinstance(c, DisplacementPC))
    A_ff = ctx.A_ff
    if displacement_fluid == "gamg":
        nullspace = A_ff.getNearNullSpace()
        assert len(nullspace.getVecs()) == 3
        # rigid modes are only near-null for the Dirichlet-restricted operator
        y = A_ff.createVecLeft()
        A_ff.mult(nullspace.getVecs()[2], y)
        assert y.norm() > 0.0
    assert ctx.ksp_fluid.getPC().getType() == {"amg": "hypre", "gamg": "gamg"}[displacement_fluid]
    b = A_ff.createVecLeft()
    b.array[:] = np.random.default_rng(9 + comm.rank).standard_normal(b.getLocalSize())
    x, x_ref = b.duplicate(), b.duplicate()
    ctx.ksp_fluid.solve(b, x)
    iterations = ctx.ksp_fluid.getIterationNumber()
    assert ctx.ksp_fluid.getConvergedReason() > 0 and iterations < 100
    chol = PETSc.KSP().create(A_ff.comm)
    chol.setOperators(A_ff)
    chol.setType("preonly")
    chol.getPC().setType("lu")
    chol.getPC().setFactorSolverType("mumps")
    chol.solve(b, x_ref)
    assert relative_difference(x, x_ref) < 1e-5
    chol.destroy()


def test_production_hierarchy_views(fieldsplit_solver, tmp_path):
    _, solver = fieldsplit_solver(FieldSplitConfig())
    ksp = solver.ksp
    assert ksp.getOptionsPrefix() == OPTIONS_PREFIX
    assert solver.aux.P_vp.getOptionsPrefix() == f"{OPTIONS_PREFIX}P_vp_"
    ksp_u, ksp_vp = ksp.getPC().getFieldSplitSubKSP()
    # Amat/Pmat: the outer operator is the Newton Jacobian, the (v,p) Schur preconditioner P_vp
    assert ksp_vp.getOperators()[1].handle == solver.aux.P_vp.handle
    path = tmp_path / f"view_{comm.rank}.txt"
    viewer = PETSc.Viewer().createASCII(str(path), comm=ksp.comm)
    ksp.view(viewer)
    viewer.destroy()
    comm.barrier()
    if comm.rank == 0:
        text = path.read_text()
        assert "Block triangular displacement PC" in text
        assert "Inner v|p Schur fieldsplit on the assembled P_vp" in text
        assert "type: lu" not in text
        assert f"({OPTIONS_PREFIX}fieldsplit_u_fluid_)" in text


@pytest.mark.skipif(comm.size > 1 or os.environ.get("XFSI_SKIP_MPI_TESTS"), reason="launcher only")
def test_elastic_fieldsplit_on_two_ranks():
    env = dict(os.environ, XFSI_SKIP_MPI_TESTS="1")
    cmd = ["mpiexec", "-n", "2", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
