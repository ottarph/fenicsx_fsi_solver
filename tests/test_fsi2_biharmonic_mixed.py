import numpy as np
import pytest
from newton_log_helpers import check_solver_newton_log
from restart_helpers import (
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
    check_start_from_initial_state,
)

from xfsi_solver.solvers.fsi2_biharmonic_mixed import solve
from xfsi_solver.tools.newton_log import read_newton_log

# p and z live on the fluid submesh but are checkpointed on the full mesh
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1), "z": ("Lagrange", 2, (2,))}

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "iterative"])
def test_fsi2_biharmonic_mixed_solve(output_dirs, direct):
    dt_val = 0.0025
    qoi_path = output_dirs["qoi"] / "fsi2_biharmonic_mixed.txt"
    solve(
        mesh_path=MESH,
        # Run enough steps to save at least 2 VTX snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_biharmonic_mixed.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_biharmonic_mixed_p.bp"),
        qoi_path=str(qoi_path),
        vtx_save_every=4,
        direct_k_c_solve=direct,
        newton_log_path=str(output_dirs["qoi"] / "fsi2_biharmonic_mixed_newton.txt"),
    )

    # The initial state and six time steps, with t, drag, lift and the tip displacement.
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (7, 5)
    # The inflow drives the fluid in the x-direction, so the drag grows.
    assert 0.0 < rows[1, 1] < rows[-1, 1]
    assert (output_dirs["pv"] / "fsi2_biharmonic_mixed.bp").exists()
    check_solver_newton_log(output_dirs["qoi"] / "fsi2_biharmonic_mixed_newton.txt", qoi_path, ["u", "v", "p", "z"])


def test_fsi2_biharmonic_mixed_restart_reproduces_continuous_run(output_dirs, tmp_path):
    # The restarted run assembles the approximate Jacobian at its first Newton iteration, where the
    # continuous run reuses the one from the time step before, so the two converge to states that
    # differ within the Newton tolerance (about 1e-8 relative).
    check_restart_reproduces_continuous_run(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        solve_kwargs={"direct_k_c_solve": True},
    )


def test_fsi2_biharmonic_mixed_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, MESH, output_dirs, tmp_path)


def test_fsi2_biharmonic_mixed_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, MESH, output_dirs, tmp_path)


def test_fsi2_biharmonic_mixed_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve, MESH, ELEMENTS, output_dirs, tmp_path, rtol=1e-6, qoi_rtol=1e-6, solve_kwargs={"direct_k_c_solve": True}
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"absolute_theta_shift": 0.01, "relative_theta_shift": 1.0}, "theta_shift"),
        ({"nu_s": 0.4, "lambda_s": 2.0e6}, "nu_s and lambda_s"),
    ],
    ids=["theta_shift", "nu_s_lambda_s"],
)
def test_fsi2_biharmonic_mixed_rejects_conflicting_options(output_dirs, kwargs, match):
    # The error is raised before the mesh is read
    with pytest.raises(ValueError, match=match):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "fsi2_biharmonic_mixed.bp"),
            output_path_p=str(output_dirs["pv"] / "fsi2_biharmonic_mixed_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "fsi2_biharmonic_mixed.txt"),
            **kwargs,
        )


def test_fsi2_biharmonic_mixed_quadrature_cap_changes_the_qois_slightly(output_dirs):
    dt_val = 0.0025

    def run(label, max_quadrature_degree):
        qoi_path = output_dirs["qoi"] / f"{label}.txt"
        solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{label}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{label}_p.bp"),
            qoi_path=str(qoi_path),
            direct_k_c_solve=True,
            max_quadrature_degree=max_quadrature_degree,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    exact = run("no_cap", None)
    capped = run("cap_4", 4)
    assert exact.shape == capped.shape == (3, 5)
    # The cap makes the integration of the nonlinear terms inexact, so the QoIs are not identical, but
    # they stay close. On this mesh the rows differ by up to 3e-5 relative to the uncapped values, so
    # rtol = 1e-3 leaves a margin of more than 30 while still failing for a wrong residual. The initial
    # row of zeros is the same in both.
    np.testing.assert_allclose(capped, exact, rtol=1e-3, atol=0)
    assert not np.array_equal(capped, exact)


def test_fsi2_biharmonic_mixed_reassembles_the_jacobian_in_every_iteration_without_reuse(output_dirs):
    # With gamma_reassemble=None, the Jacobian is assembled in every Newton iteration, also in the
    # first of a time step, which the default reuses from the step before.
    dt_val = 0.0025

    def run(name, gamma_reassemble):
        solve(
            mesh_path=MESH,
            T=3 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
            direct_k_c_solve=True,
            gamma_reassemble=gamma_reassemble,
            newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
        )
        return read_newton_log(output_dirs["qoi"] / f"{name}_newton.txt")

    reused, fresh = run("reused", 0.2), run("fresh", None)
    check_solver_newton_log(
        output_dirs["qoi"] / "fresh_newton.txt",
        output_dirs["qoi"] / "fresh.txt",
        ["u", "v", "p", "z"],
        always_assembled=True,
    )
    assert np.any(reused["assembled"][reused["it"] > 0] == 0)
    assert fresh["assembled"].sum() > reused["assembled"].sum()
