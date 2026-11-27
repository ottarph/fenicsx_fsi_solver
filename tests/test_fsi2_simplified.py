import numpy as np
import pytest
from mpi4py import MPI
from newton_log_helpers import check_solver_newton_log
from restart_helpers import (
    check_initial_state_rejects_T_not_after_it,
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
    check_start_from_initial_state,
    output_path_kwargs,
    read_checkpoint_state,
)

from xfsi_solver.fsi.materials import MR3_FSI2_MATCHED, MR3_FSI2_MATCHED_C11_0
from xfsi_solver.mesh_motion import biharmonic_c0ip_mesh_motion, harmonic_mesh_motion
from xfsi_solver.petsc_options.fsi2d import DEFAULT_DIRECT_SOLVER_OPTIONS, DEFAULT_ITERATIVE_SOLVER_OPTIONS
from xfsi_solver.solvers import fsi2_biharmonic_mixed
from xfsi_solver.solvers import fsi2_simplified as fsi2_simplified_module
from xfsi_solver.solvers.fsi2_simplified import solve
from xfsi_solver.tools import vtkhdf
from xfsi_solver.tools.newton_log import read_newton_log
from xfsi_solver.tools.qoi import DET_F_QOI_HEADER

# The columns of the QoI file
NUM_COLUMNS = len(DET_F_QOI_HEADER.split("\t"))
# The columns that the first row of a run started from a checkpoint computes from the start state alone, as
# they need the previous step: the drag and lift, at t^{n+theta}
START_ROW_NEW_COLUMNS = [1, 2]

# The checkpoints hold the displacement as one function u on the full mesh, as in fsi2_biharmonic_mixed;
# p lives on the fluid submesh but is checkpointed on the full mesh. z, the Laplacian of the displacement
# in the mixed mesh motion, is not checkpointed: every mesh motion solve computes it again.
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1)}

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"

# The residual of fsi2_simplified is multiplied by dt (except the continuity term), so the default
# snes_atol = 1e-7 is 1e-7 / dt = 4e-5 on the unscaled momentum rows at dt = 0.0025. On this mesh, from
# rest, Newton then stops after one iteration per step, and two runs that take different paths to the
# same state (a restart, which assembles a fresh Jacobian, a start from a snapshot, or another initial
# guess) differ by up to 1e-5 of a QoI's largest value. The tests comparing such runs converge Newton to
# snes_atol = 1e-7 * dt instead, the tolerance of the unscaled residual, at which their QoIs agree within
# 1e-7, a tenth of what the tests allow.
CONVERGED_SOLVER_OPTIONS = DEFAULT_DIRECT_SOLVER_OPTIONS | {"snes_atol": 1e-7 * 0.0025}


def test_fsi2_simplified_solve(output_dirs):
    dt_val = 0.0025
    qoi_path = output_dirs["qoi"] / "fsi2_simplified.txt"
    solve(
        mesh_path=MESH,
        # Run enough steps to write the output at two times
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_simplified.vtkhdf"),
        qoi_path=str(qoi_path),
        output_every=4,
        newton_log_path=str(output_dirs["qoi"] / "fsi2_simplified_newton.txt"),
    )

    # The initial state and six time steps, with t, drag, lift, the tip displacement and the range of det(F) on
    # the solid and on the fluid mesh, which stays near 1 from rest.
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (7, NUM_COLUMNS)
    with open(qoi_path) as f:
        assert f.readline().lstrip("#").split() == DET_F_QOI_HEADER.split("\t")
    np.testing.assert_array_equal(rows[0, 5:9], 1.0)  # at rest
    np.testing.assert_allclose(rows[:, 5:9], 1.0, atol=1e-3)
    assert np.all(rows[1:, 5] <= rows[1:, 6]) and np.all(rows[1:, 7] <= rows[1:, 8])
    assert np.any(rows[1:, 5:9] != 1.0)
    # The inflow drives the fluid in the x-direction, so the drag grows, and the structure moves.
    assert 0.0 < rows[1, 1] < rows[-1, 1]
    assert np.abs(rows[-1, 3]) > 0.0
    # The output at steps 0 and 4, in one VTKHDF file with u, v and p
    output_path = output_dirs["pv"] / "fsi2_simplified.vtkhdf"
    np.testing.assert_allclose(vtkhdf.output_times(output_path, MPI.COMM_WORLD), [0.0, 4 * dt_val])
    assert sorted(vtkhdf.function_names(output_path, MPI.COMM_WORLD)) == ["p", "u", "v"]
    check_solver_newton_log(output_dirs["qoi"] / "fsi2_simplified_newton.txt", qoi_path, ["v", "p"])


def test_fsi2_simplified_quadrature_cap_changes_the_qois_slightly(output_dirs):
    dt_val = 0.0025

    def run(name, max_quadrature_degree):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            max_quadrature_degree=max_quadrature_degree,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    exact = run("no_cap", None)
    capped = run("cap_4", 4)
    assert exact.shape == capped.shape == (3, NUM_COLUMNS)
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
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=solver_options,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    # Since the residual is multiplied by dt, the default snes_atol = 1e-7 and ksp_atol = 1e-8 stop Newton
    # after one iteration per step and FGMRES on its absolute tolerance, so the default iterative solve
    # differs from the direct one by 2% in the small lift of the first step (9.5e-4). Both solves are
    # converged tightly here, so that the test compares the solutions, not where each solver stops.
    converged = {"snes_atol": 1e-12, "ksp_atol": 1e-14}
    direct = run("direct", DEFAULT_DIRECT_SOLVER_OPTIONS | converged)
    # Fails with DIVERGED_LINEAR_SOLVE if the fieldsplit does not get the v and p splits or their options.
    iterative = run("iterative", DEFAULT_ITERATIVE_SOLVER_OPTIONS | converged)
    # The QoIs agree to all 7 written digits; rtol = 1e-6 allows for the rounding of the last one. The
    # initial row of zeros is the same in both.
    np.testing.assert_allclose(iterative, direct, rtol=1e-6, atol=0)


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
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            **kwargs,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    mixed = run("mixed", 2 * dt_val, checkpoint_dir=tmp_path / "mixed", checkpoint_every=2)
    other = run("other", 2 * dt_val, mesh_motion_type=mesh_motion_type)
    assert other.shape == mixed.shape == (3, NUM_COLUMNS)
    np.testing.assert_allclose(other, mixed, rtol=1e-4, atol=0)

    continued = run("continued", 4 * dt_val, mesh_motion_type=mesh_motion_type, initial_state_dir=tmp_path / "mixed")
    # The initial row, at the time of the checkpoint, holds the QoIs of the state read
    assert continued.shape == (3, NUM_COLUMNS)
    same = np.setdiff1d(np.arange(NUM_COLUMNS), START_ROW_NEW_COLUMNS)
    np.testing.assert_allclose(continued[0, same], mixed[-1, same], rtol=1e-6, atol=0)
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
        solve_kwargs={"solver_options": CONVERGED_SOLVER_OPTIONS},
    )


def test_fsi2_simplified_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, MESH, output_dirs, tmp_path)


def test_fsi2_simplified_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, MESH, output_dirs, tmp_path)


def test_fsi2_simplified_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        solve_kwargs={"solver_options": CONVERGED_SOLVER_OPTIONS},
        start_row_new_columns=("drag", "lift"),
    )


def test_fsi2_simplified_initial_state_rejects_T_not_after_it(output_dirs, tmp_path):
    check_initial_state_rejects_T_not_after_it(solve, MESH, output_dirs, tmp_path)


def _snapshot_names(snapshot_dir):
    return sorted(path.name for path in snapshot_dir.iterdir())


def test_fsi2_simplified_writes_snapshots_to_start_from(output_dirs, tmp_path):
    """Snapshots are written at the steps closest to the requested times, also at the step of a restart
    checkpoint and at the initial state, and a run started from one continues the run that wrote it."""
    dt_val = 0.0025

    def run(name, **kwargs):
        solve(
            mesh_path=MESH,
            T=6 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
            solver_options=CONVERGED_SOLVER_OPTIONS,
            **kwargs,
        )
        return np.loadtxt(output_dirs["qoi"] / f"{name}.txt", ndmin=2)

    # 0.006 is closest to step 2 (t = 0.005), 0.0101 to step 4 (t = 0.01), with the restart checkpoint
    # there, and 0.0124 to step 5 (t = 0.0125)
    snapshot_dir = tmp_path / "snapshots"
    source = run(
        "source",
        snapshot_dir=snapshot_dir,
        snapshot_times=(0.0, 0.006, 0.0101, 0.0124),
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=4,
    )
    assert _snapshot_names(snapshot_dir) == ["t0.0000", "t0.0050", "t0.0100", "t0.0125"]
    assert _snapshot_names(snapshot_dir / "t0.0050") == ["checkpoint_0.vtkhdf"]
    assert _snapshot_names(tmp_path / "checkpoints") == ["checkpoint_0.vtkhdf"]

    t, step, state = read_checkpoint_state(MESH, snapshot_dir / "t0.0000", dt_val, ELEMENTS)
    assert (t, step) == (0.0, 0)
    assert not any(f.x.array.any() for f in state.values())
    t_snapshot, step, snapshot = read_checkpoint_state(MESH, snapshot_dir / "t0.0100", dt_val, ELEMENTS)
    t_checkpoint, _, checkpoint = read_checkpoint_state(MESH, tmp_path / "checkpoints", dt_val, ELEMENTS)
    assert step == 4
    assert t_snapshot == t_checkpoint == pytest.approx(4 * dt_val)
    for name in ELEMENTS:
        np.testing.assert_array_equal(snapshot[name].x.array, checkpoint[name].x.array)

    # The started run has the QoI rows of the source from t = 0.005 on, within the Newton tolerance
    started = run("started", initial_state_dir=snapshot_dir / "t0.0050")
    assert started.shape == (5, NUM_COLUMNS)
    np.testing.assert_allclose(started[:, 0], source[2:, 0], rtol=1e-12)
    for column in range(1, source.shape[1]):
        first = 1 if column in START_ROW_NEW_COLUMNS else 0
        np.testing.assert_allclose(
            started[first:, column], source[2 + first :, column], rtol=0, atol=1e-6 * np.abs(source[:, column]).max()
        )


def test_fsi2_simplified_restart_keeps_earlier_snapshots(output_dirs, tmp_path):
    """A restart from the checkpoint at step 4 keeps the snapshot of step 2, and writes those of steps 5
    and 7; a new run in the same snapshot directory removes them all."""
    dt_val = 0.0025
    snapshot_dir = tmp_path / "snapshots"
    kwargs = dict(
        mesh_path=MESH,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "snapshots.vtkhdf"),
        qoi_path=str(output_dirs["qoi"] / "snapshots.txt"),
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=4,
        snapshot_dir=snapshot_dir,
    )
    solve(T=6 * dt_val, snapshot_times=(2 * dt_val, 5 * dt_val), **kwargs)
    assert _snapshot_names(snapshot_dir) == ["t0.0050", "t0.0125"]
    solve(T=8 * dt_val, restart=True, snapshot_times=(2 * dt_val, 5 * dt_val, 7 * dt_val), **kwargs)
    assert _snapshot_names(snapshot_dir) == ["t0.0050", "t0.0125", "t0.0175"]
    for name, k in [("t0.0050", 2), ("t0.0125", 5), ("t0.0175", 7)]:
        assert read_checkpoint_state(MESH, snapshot_dir / name, dt_val, ELEMENTS)[1] == k

    solve(T=dt_val, snapshot_times=(dt_val,), **kwargs)
    assert _snapshot_names(snapshot_dir) == ["t0.0025"]


def test_fsi2_simplified_rejects_snapshot_after_T(output_dirs, tmp_path):
    with pytest.raises(ValueError, match="after the end of the run"):
        solve(
            mesh_path=MESH,
            T=0.005,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "late.vtkhdf"),
            qoi_path=str(output_dirs["qoi"] / "late.txt"),
            snapshot_dir=tmp_path / "snapshots",
            snapshot_times=(0.01,),
        )


def test_fsi2_simplified_checkpoints_interchangeable_with_mixed(output_dirs, tmp_path):
    # Without the prediction of the displacements, the two solvers discretise the same equations and
    # their checkpoints have the same layout, so their states agree, and each continues from the other's
    # checkpoint as from its own. The states differ by the quadrature of the mesh motion (fsi2_simplified
    # integrates each block of the mixed form at its own degree, fsi2_biharmonic_mixed the terms of a
    # residual block together, at the highest degree of any of them), and by the time weighting of the
    # mesh transport term and of J F^-T in the pressure term, which FSIForms theta-weights and
    # fsi2_biharmonic_mixed evaluates at the new time level, an O(dt) difference per unit time.
    dt_val = 0.0025

    def run(module_solve, name, T, **kwargs):
        module_solve(
            mesh_path=MESH,
            T=T,
            dt_val=dt_val,
            **output_path_kwargs(module_solve, output_dirs["pv"], name),
            qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
            checkpoint_dir=tmp_path / name,
            checkpoint_every=4,
            **kwargs,
        )

    mixed = fsi2_biharmonic_mixed.solve
    # fsi2_biharmonic_mixed does not scale its residual by dt, so its default snes_atol = 1e-7 is
    # the same stopping test as CONVERGED_SOLVER_OPTIONS for fsi2_simplified.
    simplified_kwargs = dict(predict_displacements=False, solver_options=CONVERGED_SOLVER_OPTIONS)
    run(solve, "simplified", 4 * dt_val, **simplified_kwargs)
    run(mixed, "mixed", 4 * dt_val, direct_k_c_solve=True)
    _, _, state_simplified = read_checkpoint_state(MESH, tmp_path / "simplified", dt_val, ELEMENTS)
    _, _, state_mixed = read_checkpoint_state(MESH, tmp_path / "mixed", dt_val, ELEMENTS)
    for name in ELEMENTS:
        values = state_mixed[name].x.array
        np.testing.assert_allclose(
            state_simplified[name].x.array, values, rtol=0, atol=1e-5 * np.max(np.abs(values)), err_msg=name
        )

    run(
        solve,
        "simplified_from_simplified",
        8 * dt_val,
        initial_state_dir=tmp_path / "simplified",
        **simplified_kwargs,
    )
    run(solve, "simplified_from_mixed", 8 * dt_val, initial_state_dir=tmp_path / "mixed", **simplified_kwargs)
    run(
        mixed,
        "mixed_from_simplified",
        8 * dt_val,
        initial_state_dir=tmp_path / "simplified",
        direct_k_c_solve=True,
    )
    run(mixed, "mixed_from_mixed", 8 * dt_val, initial_state_dir=tmp_path / "mixed", direct_k_c_solve=True)
    # Each solver continues from the other's checkpoint as from its own: the start states differ only as
    # much as the two runs to them, so the QoIs of a continuation agree with those of the same solver from
    # its own checkpoint.
    for name, reference_name in (
        ("simplified_from_mixed", "simplified_from_simplified"),
        ("mixed_from_simplified", "mixed_from_mixed"),
    ):
        reference = np.loadtxt(output_dirs["qoi"] / f"{reference_name}.txt", ndmin=2)
        rows = np.loadtxt(output_dirs["qoi"] / f"{name}.txt", ndmin=2)
        assert rows.shape == reference.shape
        for column, values in enumerate(reference.T):
            np.testing.assert_allclose(
                rows[:, column], values, rtol=0, atol=1e-5 * np.max(np.abs(values)), err_msg=f"{name}, column {column}"
            )


def test_fsi2_simplified_reassembles_the_jacobian_in_every_iteration_without_reuse(output_dirs):
    # With gamma_reassemble=None, the Jacobian is assembled in every Newton iteration, also in the
    # first of a time step, which the default reuses from the step before.
    dt_val = 0.0025

    def run(name, gamma_reassemble):
        solve(
            mesh_path=MESH,
            T=3 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
            gamma_reassemble=gamma_reassemble,
            newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
        )
        return read_newton_log(output_dirs["qoi"] / f"{name}_newton.txt")

    reused, fresh = run("reused", 0.2), run("fresh", None)
    check_solver_newton_log(
        output_dirs["qoi"] / "fresh_newton.txt", output_dirs["qoi"] / "fresh.txt", ["v", "p"], always_assembled=True
    )
    assert np.any(reused["assembled"][reused["it"] > 0] == 0)
    assert fresh["assembled"].sum() > reused["assembled"].sum()


def test_fsi2_simplified_returns_the_newton_statistics_of_the_newton_log(output_dirs):
    # The statistics solve() returns count the same iterations and assemblies as the Newton log.
    dt_val = 0.0025
    stats = solve(
        mesh_path=MESH,
        T=4 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "stats.vtkhdf"),
        qoi_path=str(output_dirs["qoi"] / "stats.txt"),
        newton_log_path=str(output_dirs["qoi"] / "stats_newton.txt"),
    )
    log = read_newton_log(output_dirs["qoi"] / "stats_newton.txt")

    assert stats["num_steps"] == 4
    assert stats["elapsed_time"] > 0.0
    assert stats["num_newton_iterations"] == sum(stats["iterations_per_step"]) == np.sum(log["it"] > 0)
    assert stats["num_assemblies"] == sum(stats["assemblies_per_step"]) == log["assembled"].sum()
    # The first step assembles the Jacobian, at least once
    assert stats["assemblies_per_step"][0] >= 1


def test_fsi2_simplified_predicted_displacements_converge_to_the_same_state(output_dirs):
    # The prediction only changes the initial guess of each Newton solve, so the runs agree within
    # the Newton tolerance. From rest, the structure hardly moves in the first steps, so the residual
    # of the initial guess hardly changes either; it is smaller with the prediction once the beam
    # oscillates (32bd021:notes/newton-convergence/implementation-log.md).
    dt_val = 0.0025

    def run(name, predict_displacements):
        solve(
            mesh_path=MESH,
            T=4 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
            newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
            predict_displacements=predict_displacements,
            solver_options=CONVERGED_SOLVER_OPTIONS,
        )
        return np.loadtxt(output_dirs["qoi"] / f"{name}.txt", ndmin=2), read_newton_log(
            output_dirs["qoi"] / f"{name}_newton.txt"
        )

    (qoi, log), (qoi_predicted, log_predicted) = run("previous", False), run("predicted", True)
    check_solver_newton_log(
        output_dirs["qoi"] / "predicted_newton.txt", output_dirs["qoi"] / "predicted.txt", ["v", "p"]
    )
    for column in range(1, qoi.shape[1]):
        np.testing.assert_allclose(
            qoi_predicted[:, column], qoi[:, column], rtol=0, atol=1e-6 * np.abs(qoi[:, column]).max()
        )
    initial = log["residual"][log["it"] == 0]
    initial_predicted = log_predicted["residual"][log_predicted["it"] == 0]
    assert np.all(initial_predicted <= initial * (1 + 1e-6))


@pytest.mark.parametrize("solver_options", ["direct", "iterative"])
def test_fsi2_simplified_skipped_dirichlet_lifting_gives_the_lifted_qois(output_dirs, solver_options):
    # During the ramp of the inflow, its values change in every step, so the initial guess departs from
    # them, and the lifted and held Newton iterates differ in the first step of each.
    dt_val = 0.0025
    options = {"direct": DEFAULT_DIRECT_SOLVER_OPTIONS, "iterative": DEFAULT_ITERATIVE_SOLVER_OPTIONS}[solver_options]

    def run(hold):
        name = f"held_{solver_options}_{hold}"
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        stats = solve(
            mesh_path=MESH,
            T=4 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=options,
            skip_dirichlet_lifting=hold,
        )
        return np.loadtxt(qoi_path, ndmin=2), stats

    lifted, lifted_stats = run(False)
    held, held_stats = run(True)
    assert held_stats["iterations_per_step"] == lifted_stats["iterations_per_step"]
    # On this mesh the QoIs differ by up to 7e-6 of the largest value of their column with the
    # iterative solver, and 3e-8 with the direct one (on 4 ranks): the Newton tolerance, well below 1e-4.
    scale = np.abs(lifted[1:, 1:]).max(axis=0)
    np.testing.assert_array_less(np.abs(held[1:, 1:] - lifted[1:, 1:]) / scale, 1e-4)
    np.testing.assert_array_equal(held[:, 0], lifted[:, 0])


def run_material(output_dirs, name, T, dt_val, **kwargs):
    """Run FSI2 on the coarse mesh with the given material and loading; return the QoI rows and statistics."""
    qoi_path = output_dirs["qoi"] / f"{name}.txt"
    statistics = solve(
        mesh_path=MESH,
        T=T,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
        qoi_path=str(qoi_path),
        **kwargs,
    )
    return np.loadtxt(qoi_path, ndmin=2), statistics


def test_fsi2_simplified_stvk_material_is_the_default(output_dirs):
    default, _ = run_material(output_dirs, "default", 2 * 0.0025, 0.0025)
    stvk, _ = run_material(output_dirs, "stvk", 2 * 0.0025, 0.0025, solid_material="stvk")
    np.testing.assert_array_equal(stvk, default)


def test_fsi2_simplified_mooney_rivlin_3_matches_svk_at_small_strain(output_dirs):
    """MR3_FSI2_MATCHED has the initial shear and bulk moduli of the FSI2 solid, so while the structure has
    barely moved, the QoIs are those of the SVK solid up to the small-strain difference of the laws: by up to
    1e-4 relative in these steps, where the solid J stays within 1e-4 of 1."""
    svk, _ = run_material(output_dirs, "svk", 4 * 0.0025, 0.0025)
    mr3, statistics = run_material(
        output_dirs, "mr3", 4 * 0.0025, 0.0025, solid_material="mooney_rivlin_3", mooney_rivlin_3=MR3_FSI2_MATCHED
    )
    assert not np.array_equal(mr3, svk)
    np.testing.assert_allclose(mr3, svk, rtol=1e-3, atol=1e-12)
    assert len(statistics["J_solid_per_step"]) == len(statistics["J_fluid_per_step"]) == 4
    for J_min, J_max in statistics["J_solid_per_step"]:
        assert 1.0 - 1e-4 < J_min <= J_max < 1.0 + 1e-4


def test_fsi2_simplified_mooney_rivlin_3_at_finite_strain(output_dirs):
    """With three times the FSI2 inflow, over t in [0, 2] at dt = 0.02, the flag bends so that the solid J
    ranges over about [0.955, 1.046] on this mesh, and the third term of the MR3 law changes the tip
    displacement by 2-6% of its largest value against the same material without it (C11 = 0). The fluid mesh
    stays valid (J > 0.85). Newton is started from a fresh Jacobian in every iteration, and converges in at
    most 9 iterations per step, as with the SVK solid at this loading. This exercises the coupled MR3 stress
    at finite strain; dt is too large for the QoIs to be accurate."""
    kwargs = dict(U_bar=3.0, max_quadrature_degree=6, gamma_reassemble=None, solid_material="mooney_rivlin_3")
    with_c11, statistics = run_material(output_dirs, "mr3_c11", 2.0, 0.02, mooney_rivlin_3=MR3_FSI2_MATCHED, **kwargs)
    without, _ = run_material(output_dirs, "mr3_c11_0", 2.0, 0.02, mooney_rivlin_3=MR3_FSI2_MATCHED_C11_0, **kwargs)

    assert with_c11.shape == without.shape == (101, NUM_COLUMNS)
    assert np.all(np.isfinite(with_c11))
    # The det(F) columns of the QoI file are the ranges the solver returns, after the initial row
    J_solid, J_fluid = with_c11[1:, 5:7], with_c11[1:, 7:9]
    np.testing.assert_allclose(J_solid, statistics["J_solid_per_step"], rtol=1e-6)  # written with 7 digits
    np.testing.assert_allclose(J_fluid, statistics["J_fluid_per_step"], rtol=1e-6)
    assert 0.9 < J_solid[:, 0].min() < 0.98 and 1.02 < J_solid[:, 1].max() < 1.1  # finite strain
    assert J_fluid[:, 0].min() > 0.85
    assert max(statistics["iterations_per_step"]) <= 12

    tip_scale = np.abs(without[:, 3:5]).max(axis=0)
    tip_change = np.abs(with_c11[:, 3:5] - without[:, 3:5]).max(axis=0) / tip_scale
    assert np.all(tip_change > 0.005) and np.all(tip_change < 0.2)


@pytest.mark.parametrize(
    "material",
    [
        {"solid_material": "neo_hookean"},
        {"solid_material": "mooney_rivlin_3"},
        {"solid_material": "mooney_rivlin_3", "mooney_rivlin_3": {"C10": 0.0, "C01": 2.5e5}},
        {"solid_material": "mooney_rivlin_3", "mooney_rivlin_3": MR3_FSI2_MATCHED | {"C2": 1.0}},
        {"solid_material": "mooney_rivlin_3", "mooney_rivlin_3": MR3_FSI2_MATCHED | {"D1": -1.0}},
        {"mooney_rivlin_3": MR3_FSI2_MATCHED},
    ],
    ids=["unknown", "no_coefficients", "missing_C11", "unknown_coefficient", "negative_D1", "without_material"],
)
def test_fsi2_simplified_rejects_an_inconsistent_material(output_dirs, material):
    with pytest.raises(ValueError):
        run_material(output_dirs, "rejected", 0.0025, 0.0025, **material)


@pytest.mark.parametrize("solid_material", ["stvk", "mooney_rivlin_3"])
def test_fsi2_simplified_stops_on_an_inverted_fluid_mesh(output_dirs, monkeypatch, solid_material):
    """A fluid mesh with det(F) <= 0 stops the run with either solid material, after the QoI row that shows it
    is written. The measurement of the fluid mesh is made to report an inverted cell from the second step on."""
    original = fsi2_simplified_module.deformation_jacobian_range
    measures = []

    def deformation_jacobian_range(u, cells, *args, **kwargs):
        measure = original(u, cells, *args, **kwargs)
        measures.append(measure)
        if len(measures) == 1:
            return measure  # the solid
        calls = []

        def fluid_measure():
            calls.append(None)
            J_min, J_max = measure()
            return (-0.25, J_max) if len(calls) >= 3 else (J_min, J_max)  # rows: initial state, step 1, step 2

        return fluid_measure

    monkeypatch.setattr(fsi2_simplified_module, "deformation_jacobian_range", deformation_jacobian_range)
    material = {"solid_material": solid_material}
    if solid_material == "mooney_rivlin_3":
        material["mooney_rivlin_3"] = MR3_FSI2_MATCHED
    qoi_path = output_dirs["qoi"] / "inverted.txt"
    with pytest.raises(
        RuntimeError, match=r"At t = 0\.0050, the smallest det\(F\) is not positive: .* -0\.25 on the fluid mesh"
    ):
        solve(
            mesh_path=MESH,
            T=4 * 0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "inverted.vtkhdf"),
            qoi_path=str(qoi_path),
            **material,
        )
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (3, NUM_COLUMNS)  # the initial state, step 1, and step 2, where the run stopped
    assert rows[-1, 7] == -0.25 and rows[1, 7] > 0.99


@pytest.mark.parametrize("solver", ["iterative", "direct"])
def test_fsi2_simplified_interface_ale_converges_to_the_same_state(output_dirs, solver):
    # The interface ALE terms only change the Jacobian, so the converged QoIs are those without them, with
    # the fieldsplit and with MUMPS LU, which then factorizes a nest Jacobian.
    dt_val = 0.0025
    base = {"iterative": DEFAULT_ITERATIVE_SOLVER_OPTIONS, "direct": DEFAULT_DIRECT_SOLVER_OPTIONS}[solver]
    converged = {"snes_atol": 1e-12, "ksp_atol": 1e-14}

    def run(name, interface_ale):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=3 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=base | converged,
            interface_ale=interface_ale,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    np.testing.assert_allclose(run("ale", True), run("J_0", False), rtol=1e-6, atol=0)


def test_fsi2_simplified_under_integrated_interface_ale_converges_to_the_same_state(output_dirs):
    # A lower quadrature degree for the interface ALE terms only changes the Jacobian, so the converged
    # QoIs are those without the terms.
    dt_val = 0.0025
    converged = {"snes_atol": 1e-12, "ksp_atol": 1e-14}

    def run(name, interface_ale):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=3 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=DEFAULT_DIRECT_SOLVER_OPTIONS | converged,
            max_quadrature_degree=5,
            interface_ale=interface_ale,
            interface_ale_quadrature_degree=2,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    np.testing.assert_allclose(run("ale_2", True), run("J_0", False), rtol=1e-6, atol=0)


def test_fsi2_simplified_momentum_row_interface_ale_converges_to_the_same_state(output_dirs):
    # The interface ALE terms in the momentum rows only still only change the Jacobian.
    dt_val = 0.0025
    converged = {"snes_atol": 1e-12, "ksp_atol": 1e-14}

    def run(name, interface_ale):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=3 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=DEFAULT_DIRECT_SOLVER_OPTIONS | converged,
            max_quadrature_degree=5,
            interface_ale=interface_ale,
            interface_ale_rows=("momentum",),
        )
        return np.loadtxt(qoi_path, ndmin=2)

    np.testing.assert_allclose(run("ale_momentum", True), run("J_0", False), rtol=1e-6, atol=0)


def test_fsi2_simplified_rejects_invalid_interface_ale_rows(output_dirs):
    # Raised before the mesh is read
    with pytest.raises(ValueError, match="interface_ale_rows"):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "invalid.vtkhdf"),
            qoi_path=str(output_dirs["qoi"] / "invalid.txt"),
            interface_ale=True,
            interface_ale_rows=(),
        )


@pytest.mark.parametrize("stress, solver", [("pressure", "direct"), ("pressure", "iterative"), ("stress", "direct")])
def test_fsi2_simplified_interface_traction_jacobian_converges_to_the_same_state(output_dirs, stress, solver):
    # The interface traction terms only change the Jacobian, so the converged QoIs are those without them,
    # with MUMPS LU on the assembled Jacobian and with the fieldsplit on a nest Jacobian.
    dt_val = 0.0025
    base = {"iterative": DEFAULT_ITERATIVE_SOLVER_OPTIONS, "direct": DEFAULT_DIRECT_SOLVER_OPTIONS}[solver]
    converged = {"snes_atol": 1e-12, "ksp_atol": 1e-14}

    def run(name, interface_traction_jacobian):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=3 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=base | converged,
            max_quadrature_degree=5,
            interface_traction_jacobian=interface_traction_jacobian,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    np.testing.assert_allclose(run(f"traction_{stress}", stress), run("J_0", None), rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"interface_traction_jacobian": "viscous"}, "interface_traction_jacobian must be"),
        ({"interface_traction_jacobian": "pressure", "interface_ale": True}, "does not go with interface_ale"),
    ],
)
def test_fsi2_simplified_rejects_invalid_interface_traction_jacobian(output_dirs, kwargs, match):
    # Raised before the mesh is read
    with pytest.raises(ValueError, match=match):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "invalid.vtkhdf"),
            qoi_path=str(output_dirs["qoi"] / "invalid.txt"),
            **kwargs,
        )
