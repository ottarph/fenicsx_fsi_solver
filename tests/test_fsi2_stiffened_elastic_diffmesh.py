"""The shared-space FSI2 solver with stiffened elastic mesh motion.

MPI-aware; ``test_stiffened_elastic_on_two_ranks`` reruns the focused
numerical checks of this module on two ranks.
"""

import json
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
from test_fsi2_harmonic_diffmesh_jacobian import (
    FD_EPS,
    central_difference,
    frozen_fluid_geometry_residual,
    jacobian_action,
    random_direction,
    set_admissible_state,
)

from xfsi_solver.fsi.forms import restrict_to_cells
from xfsi_solver.solvers import fsi2_harmonic_diffmesh as harmonic
from xfsi_solver.solvers.fsi2_harmonic_diffmesh import PHYSICAL_MARKERS, SolverConfig, jacobian_forms
from xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh import (
    OPTIONS_PREFIX,
    GeometryMonitor,
    InvalidStateError,
    MeshMotionConfig,
    build_problem,
    deformation_measures,
    solve,
)

ROOT = Path(__file__).resolve().parent.parent
MESH = str(ROOT / "data/meshes/fsi2/mesh_sec_coarse.xdmf")
QUAD_MESH = str(ROOT / "data/meshes/fsi2/mesh_quad_ssq_sec.xdmf")
DT = 0.0025
comm = MPI.COMM_WORLD


@pytest.fixture
def shared_path(tmp_path):
    """Rank 0's ``tmp_path`` on every rank: solver output paths must agree across ranks."""
    return Path(comm.bcast(str(tmp_path), root=0))


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


def global_relative(a, b, floor=1e-300):
    diff = comm.allreduce(np.sum((a - b) ** 2), op=MPI.SUM)
    norm = comm.allreduce(np.sum(b ** 2), op=MPI.SUM)
    return np.sqrt(diff / max(norm, floor))


def test_mesh_motion_config_validation():
    for bad in (dict(mesh_equation_scale=0.0), dict(mesh_equation_scale=float("inf")),
                dict(mesh_stiffening_exponent=-0.5), dict(mesh_poisson_ratio=0.5)):
        with pytest.raises(ValueError):
            MeshMotionConfig(**bad)


def test_elastic_problem_replaces_harmonic_mesh_terms(problem):
    """The displacement residual holds the elastic volume term and interface flux, no harmonic terms."""
    op = problem.mesh_operator
    assert problem.mesh_extension.name == "stiffened_elastic"
    assert op.info["j_star"] > 0 and op.info["weight_max"] > op.info["weight_min"] > 0
    du = problem.residual[0].arguments()[0]  # the test function of the mixed space
    alpha = problem.constants["alpha_u"]
    kinematic = problem.constants["rho_s"] * ufl.inner((problem.u - problem.u_old) / problem.constants["dt"], du)
    kinematic -= problem.constants["theta"] * problem.constants["rho_s"] * ufl.inner(problem.v, du)
    kinematic -= (1 - problem.constants["theta"]) * problem.constants["rho_s"] * ufl.inner(problem.v_old, du)
    expected = kinematic * problem.dx_solid + op.residual(problem.u, du, alpha)
    a = dfx.fem.petsc.assemble_vector(dfx.fem.form(problem.residual[0], entity_maps=problem.entity_maps))
    b = dfx.fem.petsc.assemble_vector(dfx.fem.form(expected, entity_maps=problem.entity_maps))
    for vec in (a, b):
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert relative_difference(a, b) < 1e-12

    # and differs from the harmonic residual only in the fluid-mesh rows
    harmonic_residual = problem.residual[0] - op.residual(problem.u, du, alpha) + \
        ufl.inner(alpha * ufl.grad(problem.u), ufl.grad(du)) * problem.dx_fluid - \
        ufl.inner(alpha * ufl.grad(problem.u) * ufl.FacetNormal(problem.mesh), du) * problem.ds_interface_fluid
    c = dfx.fem.petsc.assemble_vector(dfx.fem.form(harmonic_residual, entity_maps=problem.entity_maps))
    c.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert relative_difference(c, a) > 1e-12


@pytest.mark.parametrize("mode", ["full", "no_ale"])
def test_jacobian_matches_finite_differences(problem, mode):
    """Including the elastic mesh tangent and its fluid-side interface term (block (u, u))."""
    J = jacobian_forms(problem, mode)
    F = problem.residual if mode == "full" else frozen_fluid_geometry_residual(problem)
    for j, w_j in enumerate(problem.solution):
        d_j = random_direction(w_j.function_space, seed=20 + j)
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

    # the mesh part alone (volume and interface), directionally
    op = problem.mesh_operator
    du = ufl.TestFunction(problem.U)
    d_u = random_direction(problem.U, seed=7)
    for form in (op.volume_form(problem.u, du), op.interface_form(problem.u, du)):
        fd = central_difference(form, problem.u, d_u, 1e-4, problem.entity_maps)
        Jd = jacobian_action(ufl.derivative(form, problem.u, ufl.TrialFunction(problem.U)), d_u,
                             problem.entity_maps)
        assert fd.norm() > 0.0
        Jd.axpy(-1.0, fd)
        assert Jd.norm() <= 1e-8 * fd.norm()


def test_no_ale_keeps_mesh_and_solid_tangents(problem):
    J_full, J_no_ale = jacobian_forms(problem, "full"), jacobian_forms(problem, "no_ale")
    A_full = assemble(J_full[0][0], problem, problem.bcs_u)
    A_no_ale = assemble(J_no_ale[0][0], problem, problem.bcs_u)
    A_no_ale.axpy(-1.0, A_full)
    assert A_no_ale.norm() == 0.0
    assert J_no_ale[2][0] is None
    d_u = random_direction(problem.U, seed=3)
    solid_vu = jacobian_action(J_no_ale[1][0], d_u, problem.entity_maps)
    expected = jacobian_action(ufl.derivative(restrict_to_cells(problem.residual[1], PHYSICAL_MARKERS["solid"]),
                                              problem.u, ufl.TrialFunction(problem.U)), d_u, problem.entity_maps)
    assert solid_vu.norm() > 0.0 and relative_difference(solid_vu, expected) < 1e-12


def test_deformation_measures():
    theta = 0.3
    rotation = np.array([[np.cos(theta) - 1, -np.sin(theta)], [np.sin(theta), np.cos(theta) - 1]])
    stretch = np.array([[1.0, 0.0], [0.0, -0.5]])  # F = diag(2, 0.5)
    inverted = np.array([[-2.0, 0.0], [0.0, 0.0]])  # F = diag(-1, 1)
    det, cond = deformation_measures(np.array([rotation, stretch, inverted]))
    np.testing.assert_allclose(det, [1.0, 1.0, -1.0])
    np.testing.assert_allclose(cond, [1.0, 4.0, 1.0])


def test_invalid_accepted_state_aborts(problem, shared_path):
    monitor = GeometryMonitor(shared_path / "invalid")
    nonlinear_problem, solver = harmonic.create_nonlinear_problem(problem, SolverConfig(snes_monitor=False),
                                                                  OPTIONS_PREFIX)
    try:
        u0 = problem.u.x.array.copy()
        problem.u.x.array[:] = 0.0
        monitor.setup(problem, nonlinear_problem, solver)
        assert monitor.check_iterate(problem)
        # fold the fluid: u_x = -2 x
        problem.u.interpolate(lambda x: np.vstack((-2.0 * x[0], 0.0 * x[1])))
        assert not monitor.check_iterate(problem)
        geometry = monitor.geometry()
        assert geometry["fluid"]["J_min"] < 0 and not geometry["valid"]
        step = harmonic.StepInfo(t=0.5, snes_iterations=1, linear_iterations=1, converged_reason=2,
                                 residual_history=np.ones(2), time=0.0, field_residuals=np.ones((2, 4)),
                                 linear_solves=[], timings={}, preconditioner_statistics={}, drag=0.0, lift=0.0,
                                 tip_displacement=np.zeros(2), dt=DT)
        with pytest.raises(InvalidStateError):
            monitor.accepted(problem, step)
        problem.u.x.array[:] = u0
    finally:
        solver.destroy()


def _run(path, name, config, n_steps=8, **kwargs):
    result = solve(MESH, T=n_steps * DT, dt_val=DT, output_dir=path / name, config=config, **kwargs)
    comm.barrier()  # the output files are written by rank 0
    return result


def check_boundary_conditions(problem, t):
    u, v = problem.u, problem.v
    for bc in problem.bcs_u:
        dofs = bc.dof_indices()[0]
        assert np.abs(u.x.array[dofs]).max(initial=0.0) == 0.0
    inflow = dfx.fem.Function(problem.V)
    inflow.interpolate(harmonic.InflowFunc(t))
    inflow_bc, noslip_bc = problem.bcs_v
    dofs = inflow_bc.dof_indices()[0]
    np.testing.assert_allclose(v.x.array[dofs], inflow.x.array[dofs], atol=1e-14)
    assert np.abs(v.x.array[noslip_bc.dof_indices()[0]]).max(initial=0.0) == 0.0


def test_coupled_startup_direct_and_fieldsplit(shared_path):
    tight = dict(snes_atol=1e-10, snes_rtol=1e-14, snes_monitor=False)
    runs = {
        "full_direct": _run(shared_path, "full_direct", SolverConfig(**tight)),
        "no_ale_direct": _run(shared_path, "no_ale_direct", SolverConfig(jacobian_mode="no_ale", **tight)),
        "no_ale_fieldsplit": _run(shared_path, "no_ale_fieldsplit",
                                  SolverConfig(jacobian_mode="no_ale", linear_solver="fieldsplit", ksp_rtol=1e-8,
                                               **tight)),
    }
    reference = runs["full_direct"]
    for name, result in runs.items():
        assert [s.t for s in result.steps] == [n * DT for n in range(1, 9)], name
        check_boundary_conditions(result.problem, 8 * DT)
        qoi = np.loadtxt(shared_path / name / "qoi.txt")
        assert qoi.shape == (9, 5) and np.all(np.isfinite(qoi))
        np.testing.assert_allclose(qoi[1:, 0], [n * DT for n in range(1, 9)], rtol=1e-6)
        for step in result.steps:
            geometry = step.diagnostics["geometry"]
            assert geometry["valid"] and geometry["fluid"]["J_min"] > 0.99 and geometry["solid"]["J_min"] > 0.99
            assert step.converged_reason > 0 and step.residual_history[-1] < 1e-10
        records = [json.loads(line) for line in (shared_path / name / "diagnostics.jsonl").read_text().splitlines()]
        assert [r["t"] for r in records] == [n * DT for n in range(1, 9)]
        run = json.loads((shared_path / name / "run.json").read_text())
        assert run["status"] == "completed" and run["t_final"] == 8 * DT
        assert run["mesh_extension"]["mesh_extension"] == "stiffened_elastic"

        for f, g in zip(result.problem.solution, reference.problem.solution, strict=True):
            assert global_relative(f.x.array, g.x.array) < 1e-7, (name, f.name)
        for s, r in zip(result.steps, reference.steps, strict=True):
            assert s.drag == pytest.approx(r.drag, rel=1e-6, abs=1e-9)
            assert s.lift == pytest.approx(r.lift, rel=1e-6, abs=1e-9)
            np.testing.assert_allclose(s.tip_displacement, r.tip_displacement, rtol=1e-6, atol=1e-12)

    fieldsplit = runs["no_ale_fieldsplit"]
    solves = [d for s in fieldsplit.steps for d in s.linear_solves]
    assert solves and all(d["true_relative_residual"] < 1e-7 for d in solves)
    assert 1 < max(d["iterations"] for d in solves) < 40
    # the mesh rows are resolved although scaled by alpha
    for d in solves:
        if d["field_rhs"][1] > 1e-20:
            assert d["field_true_residuals"][1] <= 1e-6 * d["field_rhs"][1]

    # the elastic trajectory differs from the harmonic one
    harmonic_result = harmonic.solve(MESH, T=8 * DT, dt_val=DT, output_path=str(shared_path / "h.bp"),
                                     output_path_p=str(shared_path / "hp.bp"), qoi_path=str(shared_path / "h.txt"),
                                     config=SolverConfig(**tight), time_semantics="accepted")
    assert global_relative(harmonic_result.problem.u.x.array, reference.problem.u.x.array) > 1e-6


def test_restart_and_metadata(shared_path):
    config = SolverConfig(jacobian_mode="no_ale", snes_monitor=False)
    continuous = _run(shared_path, "continuous", config, n_steps=6)
    _run(shared_path, "first", config, n_steps=3, checkpoint_every=3 * DT)
    state = shared_path / "first" / "checkpoints" / f"state_t{3 * DT:.4f}.npz"
    second = solve(MESH, T=6 * DT, dt_val=DT, output_dir=shared_path / "second", config=config, initial_state=state)
    assert [s.t for s in second.steps] == pytest.approx([s.t for s in continuous.steps[3:]], abs=1e-15)
    for f, g in zip(second.problem.solution, continuous.problem.solution, strict=True):
        assert global_relative(f.x.array, g.x.array, floor=1e-24) < 1e-12, f.name
    with pytest.raises(ValueError, match="does not match"):
        solve(MESH, T=6 * DT, dt_val=DT, output_dir=shared_path / "bad", config=config, initial_state=state,
              mesh_config=MeshMotionConfig(mesh_stiffening_exponent=2.0))


def test_quadrilateral_newton_step():
    """Assembly and one production fieldsplit Newton solve on quadratic quadrilaterals.

    (The quadrilateral meshes have no node at the tip point, so the QoI driver is not used.)
    """
    problem = build_problem(QUAD_MESH, DT)
    assert problem.mesh.topology.cell_name() == "quadrilateral"
    assert problem.mesh_operator.quadrature_degree == 10
    problem.set_inflow(0.5)
    config = SolverConfig(jacobian_mode="no_ale", linear_solver="fieldsplit", snes_monitor=False,
                          snes_atol=1e-10)
    nonlinear_problem, solver = harmonic.create_nonlinear_problem(problem, config, OPTIONS_PREFIX)
    try:
        nonlinear_problem.solve()
        snes = nonlinear_problem.solver
        assert snes.getConvergedReason() > 0
        assert all(d["true_relative_residual"] < 1e-5 for d in solver.linear_solves)
        detF = problem.mesh_operator.geometry.sample(ufl.det(ufl.Identity(2) + ufl.grad(problem.u)))
        assert comm.allreduce(detF.min(initial=np.inf), op=MPI.MIN) > 0.99
        assert comm.allreduce(np.abs(problem.v.x.array).max(), op=MPI.MAX) > 0.1
    finally:
        solver.destroy()


@pytest.mark.skipif(comm.size > 1, reason="serial only")
def test_command_line(tmp_path):
    out = tmp_path / "cli"
    cmd = [sys.executable, "-m", "xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh", "--mesh", MESH,
           "--T", str(3 * DT), "--dt", str(DT), "--linear-solver", "fieldsplit", "--jacobian-mode", "no_ale",
           "--mesh-stiffening-exponent", "2.5", "--mesh-poisson-ratio", "0.3", "--output-dir", str(out)]
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "completed" and run["steps"] == 3
    assert run["solver_config"]["linear_solver"] == "fieldsplit"
    assert run["mesh"]["sha256"] and run["dofs"]["total"] > 0
    assert (out / "qoi.txt").exists() and (out / "diagnostics.jsonl").exists()


@pytest.mark.skipif(comm.size > 1 or os.environ.get("XFSI_SKIP_MPI_TESTS"), reason="launcher only")
def test_stiffened_elastic_on_two_ranks():
    env = dict(os.environ, XFSI_SKIP_MPI_TESTS="1")
    cmd = ["mpiexec", "-n", "2", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__,
           "-k", "not command_line"]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


@pytest.mark.skipif(comm.size > 1, reason="serial only")
def test_benchmark_script(tmp_path):
    out = tmp_path / "bench"
    cmd = [sys.executable, "-m", "xfsi_solver.scripts.fsi2_harmonic_diffmesh_benchmark", "--solver",
           "stiffened_elastic", "--mesh", MESH, "--dt", str(DT), "--steps", "3", "--modes",
           "full/direct,no_ale/fieldsplit,no_ale/fieldsplit(u+vp lu)", "--out", str(out)]
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    rows = {row["mode"]: row for row in json.loads((out / "results.json").read_text())["rows"]}
    assert set(rows) == {"full/direct", "no_ale/fieldsplit", "no_ale/fieldsplit(u+vp lu)"}
    for mode in ("no_ale/fieldsplit", "no_ale/fieldsplit(u+vp lu)"):
        assert rows[mode]["rel err u"] < 1e-6
    raw = json.loads((out / "results.json").read_text())["raw"]
    assert raw["full/direct"]["qoi"][-1][0] == pytest.approx(3 * DT)
    assert raw["no_ale/fieldsplit"]["min_J_fluid"] > 0.99
