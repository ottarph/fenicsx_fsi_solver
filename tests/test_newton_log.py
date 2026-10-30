from pathlib import Path

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from restart_helpers import run_mpi_python

from xfsi_solver.tools.newton_log import NewtonLog, field_blocks, read_newton_log


def _make_problem(kind=None):
    """Newton for the blocked system u^3 = 1, w^3 + u = 3 (in the L2 sense) from u = 0.9, w = 1.2, with
    the unknowns stored as a blocked vector, or with ``kind="nest"`` as a nest vector."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    Q = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    u, w = dolfinx.fem.Function(V), dolfinx.fem.Function(Q)
    u.x.array[:] = 0.9
    w.x.array[:] = 1.2
    dv, dq = ufl.TestFunction(V), ufl.TestFunction(Q)
    F = [(u**3 - 1) * dv * ufl.dx, (w**3 + u - 3) * dq * ufl.dx]
    if kind == "nest":
        lu = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
        solve_options = {
            "ksp_type": "gmres",
            "ksp_rtol": 1e-12,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative",
            **{f"fieldsplit_{field}_{key}": value for field in ("0", "1") for key, value in lu.items()},
        }
    else:
        solve_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F,
        [u, w],
        kind=kind,
        petsc_options_prefix=f"newton_log_test_{kind}_",
        petsc_options={"snes_linesearch_type": "none", "snes_rtol": 1e-10, **solve_options},
    )
    if kind == "nest":
        problem.solver.getKSP().getPC().setFieldSplitIS(*zip(("0", "1"), problem.A.getNestISs()[0], strict=True))
    return problem


@pytest.mark.parametrize("kind", [None, "nest"], ids=["blocked", "nest"])
def test_newton_log_records_every_iterate(tmp_path, kind):
    problem = _make_problem(kind)
    snes = problem.solver
    snes.setConvergenceHistory()
    path = tmp_path / "newton.txt"
    log = NewtonLog(path, field_blocks(problem, ["u", "w"]), MPI.COMM_WORLD)
    snes.setMonitor(log)
    log.start_step(0.25)
    problem.solve()
    assert snes.getConvergedReason() > 0
    # The residual vector holds the residual at the last iterate.
    if kind == "nest":
        final_block_norms = [sub.norm() for sub in problem.b.getNestSubVecs()]

    if MPI.COMM_WORLD.rank != 0:
        return
    rows = read_newton_log(path)
    assert list(rows.dtype.names) == log.columns
    if kind == "nest":
        assert np.allclose([rows["residual_u"][-1], rows["residual_w"][-1]], final_block_norms, rtol=1e-5)
    num_iterations = snes.getIterationNumber()
    assert list(rows["it"]) == list(range(num_iterations + 1))
    assert np.all(rows["t"] == 0.25)
    # The residual is the one the SNES evaluates, and its blocks split it, to the 7 digits written.
    history, _ = snes.getConvergenceHistory()
    assert np.allclose(rows["residual"], history, rtol=1e-5)
    assert np.allclose(rows["residual"] ** 2, rows["residual_u"] ** 2 + rows["residual_w"] ** 2, rtol=1e-5)
    # No step leads to the initial guess; each step after it assembles the Jacobian anew.
    assert np.isnan(rows["update"][0]) and np.isnan(rows["update_u"][0])
    assert np.allclose(rows["update"][1:] ** 2, rows["update_u"][1:] ** 2 + rows["update_w"][1:] ** 2, rtol=1e-5)
    assert list(rows["assembled"]) == [0] + [1] * num_iterations
    # Newton converges quadratically, so the updates shrink too.
    assert np.all(np.diff(rows["update"][1:]) < 0)
    assert np.all(np.diff(rows["time"]) >= 0)


def test_newton_log_marks_reused_jacobians(tmp_path):
    problem = _make_problem()
    snes = problem.solver
    J, P, (fem_jacobian, args, kwargs) = snes.getJacobian()
    num_assemblies = 0

    def jacobian_once(snes, x, J, P):
        # Assemble only in the first iteration of the first solve, and reuse that Jacobian afterwards.
        nonlocal num_assemblies
        if num_assemblies == 0:
            fem_jacobian(snes, x, J, P, *args, **kwargs)
            num_assemblies += 1

    snes.setJacobian(jacobian_once, J, P)
    path = tmp_path / "newton.txt"
    log = NewtonLog(path, field_blocks(problem, ["u", "w"]), MPI.COMM_WORLD)
    snes.setMonitor(log)
    iterations = []
    for t in (0.1, 0.2):
        log.start_step(t)
        problem.solve()
        assert snes.getConvergedReason() > 0
        iterations.append(snes.getIterationNumber())

    if MPI.COMM_WORLD.rank != 0:
        return
    rows = read_newton_log(path)
    first, second = rows[rows["t"] == 0.1], rows[rows["t"] == 0.2]
    assert len(first) == iterations[0] + 1 and len(second) == iterations[1] + 1
    assert list(first["assembled"]) == [0, 1] + [0] * (iterations[0] - 1)
    assert np.all(second["assembled"] == 0)


def test_newton_log_restart_drops_later_rows(tmp_path):
    path = tmp_path / "newton.txt"
    dt = 0.1
    problem = _make_problem()
    log = NewtonLog(path, field_blocks(problem, ["u", "w"]), MPI.COMM_WORLD)
    problem.solver.setMonitor(log)
    for t in (dt, 2 * dt):
        log.start_step(t)
        problem.solve()

    # A restart from the state at t = dt keeps the rows up to it and appends after them.
    problem = _make_problem()
    log = NewtonLog(path, field_blocks(problem, ["u", "w"]), MPI.COMM_WORLD, restart=True, t=dt, dt=dt)
    problem.solver.setMonitor(log)
    log.start_step(3 * dt)
    problem.solve()

    if MPI.COMM_WORLD.rank != 0:
        return
    t = read_newton_log(path)["t"]
    assert np.all(t[: np.count_nonzero(np.isclose(t, dt))] == dt)
    assert set(np.round(t / dt).astype(int)) == {1, 3}


def test_field_blocks_rejects_wrong_number_of_names():
    with pytest.raises(ValueError, match="1 names for 2 blocks"):
        field_blocks(_make_problem(), ["u"])


def test_newton_log_norms_do_not_depend_on_the_number_of_ranks(tmp_path):
    script = f"""
import sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
from mpi4py import MPI
from test_newton_log import _make_problem
from xfsi_solver.tools.newton_log import NewtonLog, field_blocks

problem = _make_problem()
log = NewtonLog({str(tmp_path / "parallel.txt")!r}, field_blocks(problem, ["u", "w"]), MPI.COMM_WORLD)
problem.solver.setMonitor(log)
problem.solve()
"""
    run_mpi_python(script, ranks=3, timeout=120)
    problem = _make_problem()
    problem.solver.setMonitor(NewtonLog(tmp_path / "serial.txt", field_blocks(problem, ["u", "w"]), MPI.COMM_WORLD))
    problem.solve()

    parallel, serial = read_newton_log(tmp_path / "parallel.txt"), read_newton_log(tmp_path / "serial.txt")
    assert len(parallel) == len(serial)
    for name in ["residual", "residual_u", "residual_w", "update", "update_u", "update_w"]:
        assert np.allclose(parallel[name], serial[name], rtol=1e-8, equal_nan=True), name
