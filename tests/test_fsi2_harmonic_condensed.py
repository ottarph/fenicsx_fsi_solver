from pathlib import Path

import numpy as np
import pytest
from restart_helpers import (
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
    check_start_from_initial_state,
    read_checkpoint_state,
)

from xfsi_solver.solvers import fsi2_biharmonic_mixed, fsi2_harmonic_fulljacobian
from xfsi_solver.solvers.fsi2_harmonic_condensed import solve

# p lives on the fluid submesh but is checkpointed on the full mesh
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1)}

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "iterative"])
def test_fsi2_harmonic_condensed_solve(output_dirs, direct):
    dt_val = 0.0025
    qoi_path = output_dirs["qoi"] / "fsi2_harmonic_condensed.txt"
    solve(
        mesh_path=MESH,
        # Run enough steps to save at least 2 VTX snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_harmonic_condensed.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harmonic_condensed_p.bp"),
        qoi_path=str(qoi_path),
        vtx_save_every=4,
        direct_k_c_solve=direct,
    )

    # The initial state and six time steps, with t, drag, lift and the tip displacement.
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (7, 5)
    # The inflow drives the fluid in the x-direction, so the drag grows.
    assert 0.0 < rows[1, 1] < rows[-1, 1]
    assert (output_dirs["pv"] / "fsi2_harmonic_condensed.bp").exists()


def test_fsi2_harmonic_condensed_restart_reproduces_continuous_run(output_dirs, tmp_path):
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


@pytest.mark.parametrize("first_ranks,restart_ranks", [(2, 1), (1, 2)], ids=["2to1", "1to2"])
def test_fsi2_harmonic_condensed_restart_on_different_number_of_ranks(
    output_dirs, tmp_path, first_ranks, restart_ranks
):
    # As above, the states differ within the Newton tolerance, and are not bitwise identical across
    # partitions.
    check_restart_reproduces_continuous_run(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        first_ranks=first_ranks,
        restart_ranks=restart_ranks,
        solve_kwargs={"direct_k_c_solve": True},
    )


def test_fsi2_harmonic_condensed_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, MESH, output_dirs, tmp_path)


def test_fsi2_harmonic_condensed_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, MESH, output_dirs, tmp_path)


def test_fsi2_harmonic_condensed_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve, MESH, ELEMENTS, output_dirs, tmp_path, rtol=1e-6, qoi_rtol=1e-6, solve_kwargs={"direct_k_c_solve": True}
    )


def test_fsi2_harmonic_condensed_starts_from_biharmonic_checkpoint(output_dirs, tmp_path):
    # The checkpoint of the biharmonic solver also holds z, which the harmonic solver does not read
    dt_val = 0.0025
    paths = {
        name: dict(
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(output_dirs["qoi"] / f"{name}_qoi.txt"),
        )
        for name in ("biharmonic", "harmonic")
    }
    fsi2_biharmonic_mixed.solve(
        mesh_path=MESH,
        T=2 * dt_val,
        dt_val=dt_val,
        checkpoint_dir=tmp_path / "biharmonic",
        checkpoint_every=2,
        direct_k_c_solve=True,
        **paths["biharmonic"],
    )
    solve(
        mesh_path=MESH,
        T=3 * dt_val,
        dt_val=dt_val,
        initial_state_dir=tmp_path / "biharmonic",
        direct_k_c_solve=True,
        **paths["harmonic"],
    )

    rows_biharmonic = np.loadtxt(paths["biharmonic"]["qoi_path"], ndmin=2)
    rows_harmonic = np.loadtxt(paths["harmonic"]["qoi_path"], ndmin=2)
    np.testing.assert_allclose(rows_harmonic[:, 0], [2 * dt_val, 3 * dt_val], rtol=1e-12)
    np.testing.assert_allclose(rows_harmonic[0], rows_biharmonic[-1], rtol=1e-12)


def test_fsi2_harmonic_condensed_matches_fulljacobian(output_dirs, tmp_path):
    """The condensed and the full-Jacobian solvers solve the same discrete equations.

    At a converged state, the solid kinematic equation holds at every solid dof, so the condensed
    solid displacement equals u_S, and both solvers have the same residual. They differ only in the
    Newton iterates: the full Jacobian converges quadratically, to residual norms of about 1e-12,
    while the condensed solver, with J_0 and Jacobian reuse, converges linearly and stops once the
    residual norm is below the absolute tolerance 1e-7. After 8 steps from rest, the states agreed
    to about 1e-11 relative with a fresh J_0 in every Newton iteration (gamma_reassemble=0), and to
    2e-10 (u), 9e-10 (v) and 2e-11 (p) with the default reuse used here. The tolerance 1e-6, relative
    to each field's largest value, as in the restart tests of the solvers with Jacobian reuse, leaves
    room for other partitions and MUMPS versions.
    """
    dt_val = 0.0025
    n_steps = 8
    states = {}
    qois = {}
    for name, solver_solve, kwargs in [
        ("full", fsi2_harmonic_fulljacobian.solve, {}),
        ("condensed", solve, {"direct_k_c_solve": True}),
    ]:
        qoi_path = output_dirs["qoi"] / f"{name}_qoi.txt"
        solver_solve(
            mesh_path=MESH,
            T=n_steps * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(qoi_path),
            checkpoint_dir=tmp_path / name,
            checkpoint_every=n_steps,
            **kwargs,
        )
        t, step, states[name] = read_checkpoint_state(MESH, tmp_path / name, dt_val, ELEMENTS)
        assert step == n_steps
        qois[name] = np.loadtxt(qoi_path, ndmin=2)

    for name in ELEMENTS:
        full, condensed = states["full"][name].x.array, states["condensed"][name].x.array
        assert np.max(np.abs(full)) > 0.0
        np.testing.assert_allclose(condensed, full, rtol=0, atol=1e-6 * np.max(np.abs(full)), err_msg=name)
    assert qois["full"].shape == qois["condensed"].shape == (n_steps + 1, 5)
    for column, (full, condensed) in enumerate(zip(qois["full"].T, qois["condensed"].T, strict=True)):
        np.testing.assert_allclose(condensed, full, rtol=0, atol=1e-6 * np.max(np.abs(full)), err_msg=f"QoI {column}")


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "iterative"])
def test_fsi2_harmonic_condensed_plots_matrices(output_dirs, tmp_path, monkeypatch, direct):
    mesh_path = str(Path(MESH).resolve())
    # The plots go to output/figures/matrix_blocks/<script>/ in the working directory
    monkeypatch.chdir(tmp_path)
    solve(
        mesh_path=mesh_path,
        T=0.0025,
        dt_val=0.0025,
        output_path=str(output_dirs["pv"] / "fsi2_harmonic_condensed.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harmonic_condensed_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_harmonic_condensed.txt"),
        plot_matrices=True,
        direct_k_c_solve=direct,
    )

    figures = tmp_path / "output/figures/matrix_blocks/fsi2_harmonic_condensed"
    for label in ("storage", "eq1", "approximate_storage", "approximate_eq1"):
        assert (figures / f"spy_{label}.png").exists()
        assert (figures / f"block_norms_{label}.png").exists()
