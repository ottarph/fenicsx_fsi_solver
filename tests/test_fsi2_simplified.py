import numpy as np
import pytest
from newton_log_helpers import check_solver_newton_log
from restart_helpers import (
    check_initial_state_rejects_T_not_after_it,
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
    check_start_from_initial_state,
    read_checkpoint_state,
)

from xfsi_solver.mesh_motion import biharmonic_c0ip_mesh_motion, harmonic_mesh_motion
from xfsi_solver.solvers import fsi2_biharmonic_mixed
from xfsi_solver.solvers.fsi2_simplified import (
    DEFAULT_DIRECT_SOLVER_OPTIONS,
    DEFAULT_ITERATIVE_SOLVER_OPTIONS,
    solve,
)

# The checkpoints hold the displacement as one function u on the full mesh, as in fsi2_biharmonic_mixed;
# p lives on the fluid submesh but is checkpointed on the full mesh. z, the Laplacian of the displacement
# in the mixed mesh motion, is not checkpointed: every mesh motion solve computes it again.
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1)}

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


def test_fsi2_simplified_solve(output_dirs):
    dt_val = 0.0025
    qoi_path = output_dirs["qoi"] / "fsi2_simplified.txt"
    solve(
        mesh_path=MESH,
        # Run enough steps to save at least 2 VTX snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_simplified.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_simplified_p.bp"),
        qoi_path=str(qoi_path),
        vtx_save_every=4,
        newton_log_path=str(output_dirs["qoi"] / "fsi2_simplified_newton.txt"),
    )

    # The initial state and six time steps, with t, drag, lift and the tip displacement.
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (7, 5)
    # The inflow drives the fluid in the x-direction, so the drag grows, and the structure moves.
    assert 0.0 < rows[1, 1] < rows[-1, 1]
    assert np.abs(rows[-1, 3]) > 0.0
    assert (output_dirs["pv"] / "fsi2_simplified.bp").exists()
    check_solver_newton_log(output_dirs["qoi"] / "fsi2_simplified_newton.txt", qoi_path, ["v", "p"])


def test_fsi2_simplified_quadrature_cap_changes_the_qois_slightly(output_dirs):
    dt_val = 0.0025

    def run(name, max_quadrature_degree):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(qoi_path),
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


def test_fsi2_simplified_iterative_solve_matches_direct(output_dirs):
    dt_val = 0.0025

    def run(name, solver_options):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=3 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(qoi_path),
            solver_options=solver_options,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    direct = run("direct", DEFAULT_DIRECT_SOLVER_OPTIONS)
    # Fails with DIVERGED_LINEAR_SOLVE if the fieldsplit does not get the v and p splits or their options.
    iterative = run("iterative", DEFAULT_ITERATIVE_SOLVER_OPTIONS)
    # Newton stops at the same absolute residual tolerance with either linear solver, but at different
    # iterates. The QoIs differ by up to 1.3e-5 relative, in the small lift of the first step (9.5e-4),
    # so rtol = 1e-4 leaves a margin of more than 7. The initial row of zeros is the same in both.
    np.testing.assert_allclose(iterative, direct, rtol=1e-4, atol=0)


@pytest.mark.parametrize(
    "mesh_motion_type", [biharmonic_c0ip_mesh_motion, harmonic_mesh_motion], ids=["biharmonic_c0ip", "harmonic"]
)
def test_fsi2_simplified_other_mesh_motions(output_dirs, tmp_path, mesh_motion_type):
    """The other mesh motions move the fluid mesh differently from the default mixed biharmonic one,
    which changes the QoIs only slightly while the structure has barely moved: by up to 7e-7 relative
    over these steps. The checkpoints hold no field of the mesh motion, so a run can start from the
    state of a run with another mesh motion."""
    dt_val = 0.0025

    def run(name, T, **kwargs):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=T,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(qoi_path),
            **kwargs,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    mixed = run("mixed", 2 * dt_val, checkpoint_dir=tmp_path / "mixed", checkpoint_every=2)
    other = run("other", 2 * dt_val, mesh_motion_type=mesh_motion_type)
    assert other.shape == mixed.shape == (3, 5)
    np.testing.assert_allclose(other, mixed, rtol=1e-4, atol=0)

    continued = run("continued", 4 * dt_val, mesh_motion_type=mesh_motion_type, initial_state_dir=tmp_path / "mixed")
    # The initial row, at the time of the checkpoint, holds the QoIs of the state read
    assert continued.shape == (3, 5)
    np.testing.assert_allclose(continued[0], mixed[-1], rtol=1e-6, atol=0)
    assert continued[-1, 0] == pytest.approx(4 * dt_val)


@pytest.mark.parametrize("ranks", [(1, 1), (2, 3)], ids=["serial", "2_to_3_ranks"])
def test_fsi2_simplified_restart_reproduces_continuous_run(output_dirs, tmp_path, ranks):
    # The restarted run assembles the Jacobian at its first Newton iteration, where the continuous run
    # reuses the one from the time step before, so the two agree within the Newton tolerance.
    check_restart_reproduces_continuous_run(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        first_ranks=ranks[0],
        restart_ranks=ranks[1],
    )


def test_fsi2_simplified_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, MESH, output_dirs, tmp_path)


def test_fsi2_simplified_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, MESH, output_dirs, tmp_path)


def test_fsi2_simplified_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(solve, MESH, ELEMENTS, output_dirs, tmp_path, rtol=1e-6, qoi_rtol=1e-6)


def test_fsi2_simplified_initial_state_rejects_T_not_after_it(output_dirs, tmp_path):
    check_initial_state_rejects_T_not_after_it(solve, MESH, output_dirs, tmp_path)


def test_fsi2_simplified_checkpoints_interchangeable_with_mixed(output_dirs, tmp_path):
    # The two solvers compute the same iterates, up to the quadrature of the mesh motion, and their
    # checkpoints have the same layout, so their states agree, and each continues from the other's
    # checkpoint as from its own. The mesh motion of fsi2_simplified integrates each block of the mixed
    # form at its own degree, while fsi2_biharmonic_mixed integrates the terms of a residual block
    # together, at the highest degree of any of them, so u differs by 2e-6 of its largest value after
    # 4 steps.
    dt_val = 0.0025

    def run(module_solve, name, T, **kwargs):
        module_solve(
            mesh_path=MESH,
            T=T,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
            checkpoint_dir=tmp_path / name,
            checkpoint_every=4,
            **kwargs,
        )

    mixed = fsi2_biharmonic_mixed.solve
    run(solve, "simplified", 4 * dt_val)
    run(mixed, "mixed", 4 * dt_val, direct_k_c_solve=True)
    _, _, state_simplified = read_checkpoint_state(MESH, tmp_path / "simplified", dt_val, ELEMENTS)
    _, _, state_mixed = read_checkpoint_state(MESH, tmp_path / "mixed", dt_val, ELEMENTS)
    for name in ELEMENTS:
        values = state_mixed[name].x.array
        np.testing.assert_allclose(
            state_simplified[name].x.array, values, rtol=0, atol=1e-5 * np.max(np.abs(values)), err_msg=name
        )

    run(solve, "simplified_from_simplified", 8 * dt_val, initial_state_dir=tmp_path / "simplified")
    run(solve, "simplified_from_mixed", 8 * dt_val, initial_state_dir=tmp_path / "mixed")
    run(
        mixed,
        "mixed_from_simplified",
        8 * dt_val,
        initial_state_dir=tmp_path / "simplified",
        direct_k_c_solve=True,
    )
    run(mixed, "mixed_from_mixed", 8 * dt_val, initial_state_dir=tmp_path / "mixed", direct_k_c_solve=True)
    reference = np.loadtxt(output_dirs["qoi"] / "mixed_from_mixed.txt", ndmin=2)
    for name in (
        "simplified_from_simplified",
        "simplified_from_mixed",
        "mixed_from_simplified",
    ):
        rows = np.loadtxt(output_dirs["qoi"] / f"{name}.txt", ndmin=2)
        assert rows.shape == reference.shape
        for column, values in enumerate(reference.T):
            np.testing.assert_allclose(
                rows[:, column], values, rtol=0, atol=1e-6 * np.max(np.abs(values)), err_msg=f"{name}, column {column}"
            )
