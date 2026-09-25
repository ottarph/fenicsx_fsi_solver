import numpy as np
import pytest

from xfsi_solver.solvers.fsi2_harmonic_diffmesh import DIAGNOSTIC_FIELDS, SolverConfig, solve

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"
DT = 0.0025


def _solve(output_dirs, name, config=None, n_steps=6):
    return solve(
        # smaller stand-in for the default "mesh_quad_fine_sec.xdmf" mesh
        mesh_path=MESH,
        # save_every=4 in the solver, so run enough steps to save at least 2 snapshots
        T=n_steps * DT,
        dt_val=DT,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}_qoi.txt"),
        config=config,
    )


def test_fsi2_harmonic_diffmesh_solve(output_dirs):
    result = _solve(output_dirs, "fsi2_harm_dm")

    assert (output_dirs["qoi"] / "fsi2_harm_dm_qoi.txt").exists()
    assert len(result.steps) == 6
    assert all(step.converged_reason > 0 for step in result.steps)


def test_nonlinear_problem_leaves_no_options(output_dirs):
    """Options passed to one solver must not configure the next one with the same prefix."""
    from petsc4py import PETSc

    _solve(output_dirs, "direct", SolverConfig(snes_monitor=False), n_steps=1)
    leftover = [key for key in PETSc.Options().getAll() if key.startswith("fsi2_harmonic_diffmesh_")]
    assert leftover == []


def relative_error(a, b, floor):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), floor)


def test_no_ale_direct_matches_full_direct(output_dirs):
    tight = dict(snes_atol=1e-10, snes_rtol=1e-14)
    reference = _solve(output_dirs, "full_direct", SolverConfig(jacobian_mode="full", **tight), n_steps=12)
    no_ale = _solve(output_dirs, "no_ale_direct", SolverConfig(jacobian_mode="no_ale", **tight), n_steps=12)

    for f, g in zip(no_ale.problem.solution, reference.problem.solution, strict=True):
        assert relative_error(f.x.array, g.x.array, floor=1e-12) < 1e-8, f.name

    for s, r in zip(no_ale.steps, reference.steps, strict=True):
        assert s.t == pytest.approx(r.t)
        assert s.drag == pytest.approx(r.drag, rel=1e-7, abs=1e-9)
        assert s.lift == pytest.approx(r.lift, rel=1e-7, abs=1e-9)
        np.testing.assert_allclose(s.tip_displacement, r.tip_displacement, rtol=1e-7, atol=1e-13)


@pytest.mark.parametrize("preconditioner_mode", [None, "no_ale"])
def test_exact_fieldsplit_matches_full_direct(output_dirs, preconditioner_mode):
    tight = dict(snes_atol=1e-10, snes_rtol=1e-14)
    reference = _solve(output_dirs, "full_direct", SolverConfig(**tight))
    config = SolverConfig(linear_solver="fieldsplit", fieldsplit="exact", preconditioner_mode=preconditioner_mode,
                          ksp_rtol=1e-10, **tight)
    result = _solve(output_dirs, "full_fieldsplit", config)

    for f, g in zip(result.problem.solution, reference.problem.solution, strict=True):
        assert relative_error(f.x.array, g.x.array, floor=1e-12) < 1e-8, f.name

    solves = [solve for step in result.steps for solve in step.linear_solves]
    assert solves
    assert all(solve["true_relative_residual"] < 1e-9 for solve in solves)
    if preconditioner_mode is None:
        # exact factorization of the Newton operator itself
        assert all(solve["iterations"] == 1 for solve in solves)
    assert all(step.field_residuals.shape == (step.snes_iterations + 1, len(DIAGNOSTIC_FIELDS))
               for step in result.steps)


def test_restart_reproduces_continuous_run(output_dirs, tmp_path):
    config = SolverConfig(jacobian_mode="no_ale", snes_monitor=False)
    continuous = _solve(output_dirs, "continuous", config, n_steps=8)

    checkpoints = tmp_path / "checkpoints"
    first = solve(MESH, T=3.5 * DT, dt_val=DT, output_path=str(output_dirs["pv"] / "first.bp"),
                  output_path_p=str(output_dirs["pv"] / "first_p.bp"), qoi_path=str(output_dirs["qoi"] / "first.txt"),
                  config=config, checkpoint_dir=checkpoints, checkpoint_every=4 * DT)
    assert len(first.steps) == 4
    state = next(checkpoints.glob("state_t*.npz"))
    second = solve(MESH, T=7.5 * DT, dt_val=DT, output_path=str(output_dirs["pv"] / "second.bp"),
                   output_path_p=str(output_dirs["pv"] / "second_p.bp"),
                   qoi_path=str(output_dirs["qoi"] / "second.txt"), config=config, initial_state=state)

    assert [s.t for s in second.steps] == pytest.approx([s.t for s in continuous.steps[4:]])
    for f, g in zip(second.problem.solution, continuous.problem.solution, strict=True):
        assert relative_error(f.x.array, g.x.array, floor=1e-12) < 1e-12, f.name


def test_benchmark_script(tmp_path):
    import json
    import subprocess
    import sys

    out = tmp_path / "bench"
    cmd = [sys.executable, "-m", "xfsi_solver.scripts.fsi2_harmonic_diffmesh_benchmark", "--mesh", MESH,
           "--dt", str(DT), "--steps", "3", "--modes", "full/direct,no_ale/fieldsplit", "--out", str(out)]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    rows = {row["mode"]: row for row in json.loads((out / "results.json").read_text())["rows"]}
    assert rows["no_ale/fieldsplit"]["rel err u"] < 1e-6
    assert rows["no_ale/fieldsplit"]["krylov/solve"] > 1
    assert (out / "results.md").exists()


def _solve_accepted(output_dirs, name, n_steps, **kwargs):
    return solve(MESH, T=n_steps * DT, dt_val=DT, output_path=str(output_dirs["pv"] / f"{name}.bp"),
                 output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
                 qoi_path=str(output_dirs["qoi"] / f"{name}_qoi.txt"), time_semantics="accepted", **kwargs)


def test_accepted_time_semantics(output_dirs):
    """Accepted states are labelled t0 + n dt through T; the legacy loop computes the same states.

    The legacy loop spends its first step from rest with zero inflow, so its
    step labelled ``t`` coincides with the accepted state at ``t``.
    """
    config = SolverConfig(jacobian_mode="no_ale", snes_monitor=False)
    accepted = _solve_accepted(output_dirs, "accepted", 6, config=config)
    assert [s.t for s in accepted.steps] == [n * DT for n in range(1, 7)]
    assert accepted.steps[-1].t == 6 * DT
    assert all(s.dt == DT for s in accepted.steps)
    qoi = np.loadtxt(output_dirs["qoi"] / "accepted_qoi.txt")
    assert qoi.shape == (7, 5)
    np.testing.assert_array_equal(qoi[0], 0.0)
    np.testing.assert_allclose(qoi[1:, 0], [n * DT for n in range(1, 7)], rtol=1e-6)

    legacy = _solve(output_dirs, "legacy", config, n_steps=6.5)
    assert [s.t for s in legacy.steps] == pytest.approx([n * DT for n in range(7)])
    for s, r in zip(accepted.steps, legacy.steps[1:], strict=True):
        assert s.lift == pytest.approx(r.lift, rel=1e-10, abs=1e-14)
        np.testing.assert_allclose(s.tip_displacement, r.tip_displacement, rtol=1e-10, atol=1e-16)
    for f, g in zip(accepted.problem.solution, legacy.problem.solution, strict=True):
        assert relative_error(f.x.array, g.x.array, floor=1e-12) < 1e-12, f.name

    with pytest.raises(ValueError, match="multiple of dt"):
        _solve_accepted(output_dirs, "bad", 6.5, config=config)


def test_accepted_restart_checks_metadata(output_dirs, tmp_path):
    config = SolverConfig(jacobian_mode="no_ale", snes_monitor=False)
    metadata = {"solver": "test", "mesh": MESH}
    continuous = _solve_accepted(output_dirs, "continuous", 6, config=config)
    checkpoints = tmp_path / "checkpoints"
    first = _solve_accepted(output_dirs, "first", 3, config=config, checkpoint_dir=checkpoints,
                            checkpoint_every=3 * DT, state_metadata=metadata)
    assert first.metadata["time_semantics"] == "accepted"
    state = checkpoints / f"state_t{3 * DT:.4f}.npz"
    second = solve(MESH, T=6 * DT, dt_val=DT, output_path=str(output_dirs["pv"] / "second.bp"),
                   output_path_p=str(output_dirs["pv"] / "second_p.bp"),
                   qoi_path=str(output_dirs["qoi"] / "second.txt"), config=config, initial_state=state,
                   time_semantics="accepted", state_metadata=metadata)
    assert [s.t for s in second.steps] == pytest.approx([s.t for s in continuous.steps[3:]], abs=1e-15)
    for f, g in zip(second.problem.solution, continuous.problem.solution, strict=True):
        assert relative_error(f.x.array, g.x.array, floor=1e-12) < 1e-12, f.name

    for kwargs, match in ((dict(time_semantics="accepted", state_metadata={**metadata, "solver": "other"}),
                           "does not match"),
                          (dict(time_semantics="legacy"), "time semantics")):
        with pytest.raises(ValueError, match=match):
            solve(MESH, T=6 * DT, dt_val=DT, output_path=str(output_dirs["pv"] / "bad.bp"),
                  output_path_p=str(output_dirs["pv"] / "bad_p.bp"), qoi_path=str(output_dirs["qoi"] / "bad.txt"),
                  config=config, initial_state=state, **kwargs)


def test_options_prefix_and_problem_builder(output_dirs):
    from petsc4py import PETSc

    from xfsi_solver.solvers.fsi2_harmonic_diffmesh import StepMonitor, build_problem

    built, seen = [], {}

    def builder(mesh_path, dt_val):
        built.append(mesh_path)
        return build_problem(mesh_path, dt_val)

    class Monitor(StepMonitor):
        def setup(self, problem, nonlinear_problem, linear_solver):
            seen["prefix"] = linear_solver.ksp.getOptionsPrefix()
            seen["P_vp"] = linear_solver.aux.P_vp.getOptionsPrefix()
            return {"monitor": True}

        def accepted(self, problem, step):
            return {"t": step.t}

    config = SolverConfig(jacobian_mode="no_ale", linear_solver="fieldsplit", snes_monitor=False)
    result = _solve_accepted(output_dirs, "prefix", 2, config=config, problem_builder=builder,
                             options_prefix="custom_fsi_", monitor=Monitor())
    assert built == [MESH]
    assert seen == {"prefix": "custom_fsi_", "P_vp": "custom_fsi_P_vp_"}
    assert result.metadata["monitor"] and result.metadata["options_prefix"] == "custom_fsi_"
    assert [s.diagnostics for s in result.steps] == [{"t": DT}, {"t": 2 * DT}]
    leftover = [key for key in PETSc.Options().getAll() if key.startswith(("custom_fsi_", "fsi2_harmonic_diffmesh_"))]
    assert leftover == []


def test_rejected_iterate_stops_newton(output_dirs):
    """A monitor rejecting an iterate stops the step with an error instead of accepting it."""
    from petsc4py import PETSc

    from xfsi_solver.solvers.fsi2_harmonic_diffmesh import StepMonitor

    class Reject(StepMonitor):
        calls = 0
        failures = []

        def check_iterate(self, problem):
            Reject.calls += 1
            return Reject.calls < 3  # reject the second Newton iterate of the first step

        def failed(self, problem, error):
            Reject.failures.append(error)

    with pytest.raises(PETSc.Error):
        _solve_accepted(output_dirs, "reject", 3, config=SolverConfig(snes_monitor=False), monitor=Reject())
    assert len(Reject.failures) == 1
