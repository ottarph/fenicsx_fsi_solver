import numpy as np
import pytest

from xfsi_solver.solvers.fsi2_harmonic_diffmesh import SolverConfig, solve

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
    assert all(step.field_residuals.shape == (step.snes_iterations + 1, 3) for step in result.steps)
