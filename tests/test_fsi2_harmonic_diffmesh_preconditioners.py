"""Auxiliary operators and approximate subsolvers of the shared-space FSI fieldsplit solver.

MPI-aware; ``test_preconditioners_on_two_ranks`` reruns this module on two ranks.
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

from xfsi_solver.fsi.forms import nonzero, restrict_to_cells
from xfsi_solver.linalg.fieldsplit import check_partition, field_index_sets, union_index_set
from xfsi_solver.solvers.fsi2_harmonic_diffmesh import (
    SolverConfig,
    build_problem,
    create_nonlinear_problem,
    jacobian_forms,
    solve,
)
from xfsi_solver.solvers.fsi2_harmonic_diffmesh_fieldsplit import (
    AuxiliaryOperators,
    DisplacementPC,
    FieldSplitConfig,
    MomentumPressurePC,
)

ROOT = Path(__file__).resolve().parent.parent
MESH = str(ROOT / "data/meshes/fsi2/mesh_sec_coarse.xdmf")
DT = 0.0025


@pytest.fixture(scope="module")
def problem():
    problem = build_problem(MESH, DT)
    set_admissible_state(problem)
    return problem


def assemble(forms, problem, bcs):
    A = dfx.fem.petsc.assemble_matrix(dfx.fem.form(forms, entity_maps=problem.entity_maps), bcs=bcs, diag=1.0)
    A.assemble()
    return A


def lu(A):
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()
    return ksp


def random_vector(A, seed):
    b = A.createVecLeft()
    b.array[:] = np.random.default_rng(seed + MPI.COMM_WORLD.rank).standard_normal(b.getLocalSize())
    return b


def relative_difference(x, y):
    d = x.copy()
    d.axpy(-1.0, y)
    return d.norm() / y.norm()


@pytest.fixture
def fieldsplit_solver(problem):
    """A configured FieldSplitSolver at the admissible state, with its PC applied once."""
    created = []

    def make(fs_config, jacobian_mode="no_ale", preconditioner_mode=None):
        config = SolverConfig(jacobian_mode=jacobian_mode, preconditioner_mode=preconditioner_mode,
                              linear_solver="fieldsplit", fieldsplit=fs_config, snes_monitor=False)
        nonlinear_problem, solver = create_nonlinear_problem(problem, config)
        created.append((nonlinear_problem, solver))
        dfx.fem.petsc.assign(problem.solution, nonlinear_problem.x)
        P_mat = nonlinear_problem.P_mat if nonlinear_problem.P_mat is not None else nonlinear_problem.A
        nonlinear_problem.solver.computeJacobian(nonlinear_problem.x, nonlinear_problem.A, P_mat)
        # the nested Python PCs are set up lazily at the first application
        b = random_vector(nonlinear_problem.A, seed=5)
        y = b.duplicate()
        solver.ksp.getPC().apply(b, y)
        return nonlinear_problem, solver

    yield make
    for _, solver in created:
        solver.destroy()


def test_algebraic_P_vp_matches_forms(problem):
    J = jacobian_forms(problem, "no_ale")
    A = assemble(J, problem, problem.bcs)
    _, is_v, is_p = field_index_sets(A, [problem.U, problem.V, problem.P])
    is_vp = union_index_set([is_v, is_p])
    aux = AuxiliaryOperators(problem, FieldSplitConfig(convection=True), J, A, is_vp)
    aux.assemble()

    c = problem.constants
    w = J[2][1].arguments()[1]
    T_s = c["theta"] * c["dt"] * nonzero(ufl.derivative(restrict_to_cells(problem.residual[1], 1), problem.u, w))
    reference = assemble([[J[1][1] + T_s, J[1][2]], [J[2][1], None]], problem, problem.bcs_v)
    d = aux.P_vp.copy()
    d.axpy(-1.0, reference, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    assert d.norm() <= 1e-13 * reference.norm()

    # the solid elastic contribution is present and not negligible
    J_vp = A.createSubMatrix(is_vp, is_vp)
    d = aux.P_vp.copy()
    d.axpy(-1.0, J_vp, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    assert d.norm() > 1e-2 * J_vp.norm()
    aux.destroy()


def test_form_based_velocity_operator_is_symmetric(problem):
    J = jacobian_forms(problem, "no_ale")
    aux = AuxiliaryOperators(problem, FieldSplitConfig(convection=False), J)
    aux.assemble()
    is_v, _ = field_index_sets(aux.P_vp, [problem.V, problem.P])
    H = aux.P_vp.createSubMatrix(is_v, is_v)
    assert H.isSymmetric(1e-10 * H.norm())
    aux.destroy()


def test_displacement_pc_is_block_triangular_solve(fieldsplit_solver, problem):
    nonlinear_problem, solver = fieldsplit_solver(FieldSplitConfig())
    ctx = next(c for c in solver.contexts if isinstance(c, DisplacementPC))
    A00 = solver.ksp.getPC().getFieldSplitSubKSP()[0].getOperators()[1]

    # one copy of every displacement DOF, interface DOFs in the solid/interface set
    check_partition([ctx.is_I, ctx.is_f], *A00.getOwnershipRange())
    n_interface_solid = MPI.COMM_WORLD.allreduce(int(problem.solid_displacement_dofs().sum()), op=MPI.SUM)
    assert ctx.is_I.getSize() == n_interface_solid

    # y_I = M_II^{-1} x_I, y_f = A_ff^{-1} (x_f - A_fI y_I)
    x = random_vector(A00, seed=2)
    y = x.duplicate()
    ctx.apply(None, x, y)
    x_I, x_f = x.getSubVector(ctx.is_I), x.getSubVector(ctx.is_f)
    y_I = x_I.duplicate()
    lu(ctx.M_II).solve(x_I, y_I)
    A_ff = A00.createSubMatrix(ctx.is_f, ctx.is_f)
    A_fI = A00.createSubMatrix(ctx.is_f, ctx.is_I)
    r_f = x_f.copy()
    t = x_f.duplicate()
    A_fI.mult(y_I, t)
    r_f.axpy(-1.0, t)
    y_f = x_f.duplicate()
    lu(A_ff).solve(r_f, y_f)
    assert relative_difference(y.getSubVector(ctx.is_I), y_I) < 1e-10
    assert relative_difference(y.getSubVector(ctx.is_f), y_f) < 1e-8

    # the solid mass differs from the true A_II only by the alpha-scaled mesh terms
    A_II = A00.createSubMatrix(ctx.is_I, ctx.is_I)
    A_II.axpy(-1.0, ctx.M_II, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    assert A_II.norm() <= 1e-8 * ctx.M_II.norm()


def test_cahouet_chabard_is_sum_of_inverses(fieldsplit_solver, problem):
    _, solver = fieldsplit_solver(FieldSplitConfig(pressure="cahouet_chabard"))
    mp = next(c for c in solver.contexts if isinstance(c, MomentumPressurePC))
    pc, aux = mp.pressure_pc, solver.aux
    assert len(aux.pressure_bcs[0].dof_indices()[0]) > 0 or MPI.COMM_WORLD.size > 1

    x = aux.M_p.createVecLeft()
    x.array[:] = np.random.default_rng(3 + MPI.COMM_WORLD.rank).standard_normal(x.getLocalSize())
    # exact inner solves, to check the formula and the numbering map
    for ksp in (pc.ksp_K, pc.ksp_M):
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
    pc.setUp(None)
    x_split = x.duplicate()
    x_split.array[:] = x.array[pc.order]  # split entry j is pressure DOF order[j]
    y_split = x.duplicate()
    pc.apply(None, x_split, y_split)

    a, b = x.duplicate(), x.duplicate()
    lu(aux.K_p).solve(x, a)
    lu(aux.M_p).solve(x, b)
    c_K, c_M = aux.pressure_coefficients
    expected = x.duplicate()
    expected.array[:] = c_K * a.array + c_M * b.array
    y = x.duplicate()
    y.array[pc.order] = y_split.array
    assert relative_difference(y, expected) < 1e-10
    assert c_K == pytest.approx(1.0e3 / DT)
    assert c_M == pytest.approx((0.5 + DT) * 1.0e3 * 1.0e-3)


def test_production_hierarchy(fieldsplit_solver, tmp_path):
    _, solver = fieldsplit_solver(FieldSplitConfig())
    ksp = solver.ksp
    pc = ksp.getPC()
    assert (ksp.getType(), pc.getType()) == ("fgmres", "fieldsplit")
    ksp_u, ksp_vp = pc.getFieldSplitSubKSP()
    assert ksp_u.getPC().getType() == ksp_vp.getPC().getType() == "python"
    assert ksp_vp.getOperators()[1].handle == solver.aux.P_vp.handle
    mp = ksp_vp.getPC().getPythonContext()
    ksp_v, ksp_p = mp.inner.getFieldSplitSubKSP()
    assert ksp_v.getPC().getType() == ksp_p.getPC().getType() == "hypre"

    path = tmp_path / f"ksp_view_{MPI.COMM_WORLD.rank}.txt"
    viewer = PETSc.Viewer().createASCII(str(path), comm=ksp.comm)
    ksp.view(viewer)
    viewer.destroy()
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        text = path.read_text()
        assert "Block triangular displacement PC" in text
        assert "Inner v|p Schur fieldsplit on the assembled P_vp" in text
        # no monolithic LU and no dense Schur complement in production
        assert "type: lu" not in text
        assert "dense" not in text


@pytest.mark.parametrize("jacobian_mode,preconditioner_mode", [("no_ale", None), ("full", "no_ale")])
def test_production_fieldsplit_matches_full_direct(tmp_path, jacobian_mode, preconditioner_mode):
    def run(name, config):
        out = tmp_path / name
        return solve(MESH, T=12 * DT, dt_val=DT, output_path=str(out / "uv.bp"), output_path_p=str(out / "p.bp"),
                     qoi_path=str(out / "qoi.txt"), config=config)

    tight = dict(snes_atol=1e-10, snes_rtol=1e-14, snes_monitor=False)
    reference = run("direct", SolverConfig(**tight))
    result = run("fieldsplit", SolverConfig(jacobian_mode=jacobian_mode, preconditioner_mode=preconditioner_mode,
                                            linear_solver="fieldsplit", ksp_rtol=1e-8, **tight))
    for f, g in zip(result.problem.solution, reference.problem.solution, strict=True):
        diff = MPI.COMM_WORLD.allreduce(np.sum((f.x.array - g.x.array) ** 2), op=MPI.SUM)
        norm = MPI.COMM_WORLD.allreduce(np.sum(g.x.array ** 2), op=MPI.SUM)
        assert np.sqrt(diff / norm) < 1e-7, f.name
    for s, r in zip(result.steps, reference.steps, strict=True):
        assert s.drag == pytest.approx(r.drag, rel=1e-6, abs=1e-9)
        assert s.lift == pytest.approx(r.lift, rel=1e-6, abs=1e-9)
        np.testing.assert_allclose(s.tip_displacement, r.tip_displacement, rtol=1e-6, atol=1e-12)
        # the mesh rows are resolved although they are invisible in the total norm
        # (after the first Newton iteration of a step the linear mesh equation is
        # satisfied to rounding and there is nothing left to resolve)
        for solve_info in s.linear_solves:
            u_fluid_residual, u_fluid_rhs = solve_info["field_true_residuals"][1], solve_info["field_rhs"][1]
            if u_fluid_rhs > 1e-20:
                assert u_fluid_residual <= 1e-6 * u_fluid_rhs
    iterations = [d["iterations"] for s in result.steps for d in s.linear_solves]
    assert max(iterations) < 40


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1 or os.environ.get("XFSI_SKIP_MPI_TESTS"), reason="launcher only")
def test_preconditioners_on_two_ranks():
    env = dict(os.environ, XFSI_SKIP_MPI_TESTS="1")
    cmd = ["mpiexec", "-n", "2", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
