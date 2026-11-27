import numpy as np
import pytest
from newton_log_helpers import check_solver_newton_log
from restart_helpers import check_start_from_initial_state, read_checkpoint_state

from xfsi_solver.petsc_options.fsi3d import DEFAULT_DIRECT_SOLVER_OPTIONS, DEFAULT_ITERATIVE_SOLVER_OPTIONS
from xfsi_solver.solvers.threeDfsi_simplified import solve
from xfsi_solver.tools.newton_log import read_newton_log
from xfsi_solver.tools.qoi import DET_F_QOI_HEADER_3D

# The columns of the QoI file
NUM_COLUMNS = len(DET_F_QOI_HEADER_3D.split("\t"))

VECTOR = ("Lagrange", 2, (3,))
# p lives on the fluid submesh but is checkpointed on the full mesh. No field of the mesh motion is
# checkpointed: every mesh motion solve computes it again.
ELEMENTS = {"u": VECTOR, "v": VECTOR, "p": ("Lagrange", 1)}

MESH = "data/meshes/fsi3d/mesh_sec_coarse.xdmf"

# The residual of threeDfsi_simplified is multiplied by dt (except the continuity term), so the default
# snes_atol = 1e-7 is 1e-7 / dt = 2.5e-5 on the unscaled momentum rows at dt = 0.004. On this mesh, from
# rest, Newton then stops after one iteration in the first step, so two runs that take different paths
# to the same state stop at different iterates. The tests comparing such runs converge Newton to
# snes_atol = 1e-7 * dt instead, the tolerance of the unscaled residual.
CONVERGED_SOLVER_OPTIONS = DEFAULT_DIRECT_SOLVER_OPTIONS | {"snes_atol": 1e-7 * 0.004}


@pytest.mark.slow
def test_threeDfsi_simplified_iterative_solve_matches_direct(output_dirs):
    dt_val = 0.004

    def run(name, solver_options):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=solver_options,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    # Since the residual is multiplied by dt, the default snes_atol = 1e-7 and ksp_atol = 1e-8 stop Newton
    # after one iteration in the first step, and the default iterative solve then differs from the direct
    # one by 0.2% in the drag and by more than the lift itself. Both solves are converged tightly here, so
    # that the test compares the solutions, not where each solver stops.
    converged = {"snes_atol": 1e-12, "ksp_atol": 1e-14}
    # Fails with DIVERGED_LINEAR_SOLVE if the fieldsplit does not get the v and p splits or their options.
    iterative = run("iterative", DEFAULT_ITERATIVE_SOLVER_OPTIONS | converged)
    direct = run("direct", DEFAULT_DIRECT_SOLVER_OPTIONS | converged)

    # The initial state and two time steps, with t, drag, lift, the displacement of B and the range of det(F).
    assert iterative.shape == direct.shape == (3, NUM_COLUMNS)
    # The inflow drives the fluid in the x-direction, so the drag grows.
    assert 0.0 < iterative[1, 1] < iterative[2, 1]
    # The QoIs agree to all 7 written digits; rtol = 1e-6 allows for the rounding of the last one. The
    # initial row of zeros is the same in both.
    np.testing.assert_allclose(iterative[:, :3], direct[:, :3], rtol=1e-6, atol=0)


@pytest.mark.slow
def test_threeDfsi_simplified_interface_ale_converges_to_the_same_state(output_dirs):
    # The interface ALE terms only change the Jacobian, so the converged QoIs are those without them: with
    # MUMPS LU, which then factorizes a nest Jacobian, and with the fieldsplit, here with the terms
    # integrated at quadrature degree 2.
    dt_val = 0.004
    converged = {"snes_atol": 1e-12, "ksp_atol": 1e-14}

    def run(name, solver_options, **kwargs):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=solver_options | converged,
            **kwargs,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    J_0 = run("J_0", DEFAULT_DIRECT_SOLVER_OPTIONS)
    direct = run("ale_direct", DEFAULT_DIRECT_SOLVER_OPTIONS, interface_ale=True)
    iterative = run(
        "ale_iterative", DEFAULT_ITERATIVE_SOLVER_OPTIONS, interface_ale=True, interface_ale_quadrature_degree=2
    )
    assert J_0.shape == direct.shape == iterative.shape == (3, NUM_COLUMNS)
    np.testing.assert_allclose(direct, J_0, rtol=1e-6, atol=0)
    # B_y is about 200 times smaller than B_x and B_z, so the iterative solve's residual differences show in
    # it relative to its own size: 1.1e-15 in -7.6e-10 at the first step. atol is 1e-6 of the largest
    # displacement of B, and far below the drag, lift and det(F).
    np.testing.assert_allclose(iterative, J_0, rtol=1e-6, atol=1e-6 * np.abs(J_0[:, 3:6]).max())


@pytest.mark.slow
def test_threeDfsi_simplified_prediction_options(output_dirs):
    # The default run predicts the displacements, here without Jacobian reuse, so it assembles in
    # every Newton iteration. The other starts from the previous step as it is, with the default
    # reuse. Both converge to the same states, within the Newton tolerance: their QoIs differ by at most
    # 2.7e-7 of a column's largest value.
    dt_val = 0.004

    def run(name, **kwargs):
        stats = solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
            solver_options=CONVERGED_SOLVER_OPTIONS,
            newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
            **kwargs,
        )
        check_solver_newton_log(
            output_dirs["qoi"] / f"{name}_newton.txt",
            output_dirs["qoi"] / f"{name}.txt",
            ["v", "p"],
            always_assembled=kwargs.get("gamma_reassemble", 0.1) is None,
        )
        # The statistics solve() returns count the same iterations and assemblies as the Newton log.
        log = read_newton_log(output_dirs["qoi"] / f"{name}_newton.txt")
        assert stats["num_steps"] == len(stats["step_times"]) == 2
        assert stats["num_newton_iterations"] == sum(stats["iterations_per_step"]) == np.sum(log["it"] > 0)
        assert stats["num_assemblies"] == sum(stats["assemblies_per_step"]) == log["assembled"].sum()
        assert stats["num_linear_iterations"] == sum(stats["linear_iterations_per_step"])
        return np.loadtxt(output_dirs["qoi"] / f"{name}.txt", ndmin=2), read_newton_log(
            output_dirs["qoi"] / f"{name}_newton.txt"
        )

    predicted, _ = run("predicted", gamma_reassemble=None)
    previous, _ = run("previous", predict_displacements=False)
    for column in range(1, predicted.shape[1]):
        np.testing.assert_allclose(
            predicted[:, column], previous[:, column], rtol=0, atol=1e-6 * np.abs(previous[:, column]).max()
        )


@pytest.mark.slow
def test_threeDfsi_simplified_skipped_dirichlet_lifting_gives_the_lifted_qois(output_dirs):
    # With the iterative solver, a nest Jacobian whose velocity block gets one GAMG V-cycle, which need
    # not leave the update zero on the Dirichlet dofs. The inflow changes in every step, so the lifted and
    # held iterates differ in the first Newton step of each, and the QoIs agree to the Newton tolerance.
    #
    # Since the momentum equation is multiplied by dt, the default snes_atol = 1e-7 stops Newton after one
    # or two iterations here, by a residual just above or below it (2.2e-7 lifted, 9.8e-8 held, at the
    # second step), so the two runs differ in their iteration counts and by up to the size of the lift
    # in the QoIs. Both are converged tightly instead. The held iterate starts closer, since it holds the
    # new inflow values (initial residual 9.5e-6 against 9.8e-4 at the first step), so it takes 2 and 3
    # Newton iterations where the lifted one takes 3 and 3.
    dt_val = 0.004
    converged = {"snes_atol": 1e-12, "ksp_atol": 1e-14}

    def run(name, hold):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        stats = solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=DEFAULT_ITERATIVE_SOLVER_OPTIONS | converged,
            skip_dirichlet_lifting=hold,
        )
        return np.loadtxt(qoi_path, ndmin=2), stats

    lifted, lifted_stats = run("lifted", False)
    held, held_stats = run("held", True)
    assert held.shape == lifted.shape == (3, NUM_COLUMNS)
    held_its, lifted_its = held_stats["iterations_per_step"], lifted_stats["iterations_per_step"]
    assert all(h <= lf for h, lf in zip(held_its, lifted_its, strict=True)), (held_its, lifted_its)
    for column in range(1, lifted.shape[1]):
        np.testing.assert_allclose(
            held[:, column], lifted[:, column], rtol=0, atol=1e-6 * np.abs(lifted[:, column]).max()
        )


@pytest.mark.slow
def test_threeDfsi_simplified_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        dt_val=0.004,
        rtol=1e-6,
        qoi_rtol=1e-6,
        solve_kwargs={"solver_options": CONVERGED_SOLVER_OPTIONS},
        start_row_new_columns=("drag", "lift"),
    )


@pytest.mark.slow
def test_threeDfsi_simplified_writes_snapshots(output_dirs, tmp_path):
    # 0.005 is closest to step 1 (t = 0.004); the run starts from rest, so the initial state is step 0
    dt_val = 0.004
    snapshot_dir = tmp_path / "snapshots"
    solve(
        mesh_path=MESH,
        T=2 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "snapshots.vtkhdf"),
        qoi_path=str(output_dirs["qoi"] / "snapshots.txt"),
        solver_options=DEFAULT_DIRECT_SOLVER_OPTIONS,
        snapshot_dir=snapshot_dir,
        snapshot_times=(0.0, 0.005),
    )
    assert sorted(path.name for path in snapshot_dir.iterdir()) == ["t0.0000", "t0.0040"]
    assert sorted(path.name for path in (snapshot_dir / "t0.0040").iterdir()) == ["checkpoint_0.vtkhdf"]
    t, step, state = read_checkpoint_state(MESH, snapshot_dir / "t0.0040", dt_val, ELEMENTS)
    assert step == 1
    assert t == pytest.approx(dt_val)
    assert state["v"].x.array.any()


@pytest.mark.slow
def test_threeDfsi_simplified_mooney_rivlin_3_matches_svk_at_small_strain(output_dirs):
    """With C10 + C01 = mu_s / 2 and D1 from the SVK bulk modulus, the Mooney-Rivlin solid has the initial moduli
    of the 3d benchmark solid, so from rest, where the beam has barely moved, the QoIs are those of the SVK solid
    up to the small-strain difference of the laws. The QoI file holds the range of det(F) in its last four
    columns, 1 in the initial state, and the solver returns the same values."""
    dt_val = 0.004

    def run(name, **kwargs):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        statistics = solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=CONVERGED_SOLVER_OPTIONS,
            **kwargs,
        )
        with open(qoi_path) as f:
            assert f.readline().lstrip("#").split() == DET_F_QOI_HEADER_3D.split("\t")
        return np.loadtxt(qoi_path, ndmin=2), statistics

    svk, _ = run("svk")
    mr3, statistics = run(
        "mr3", solid_material="mooney_rivlin_3", mooney_rivlin_3={"C10": 0.0, "C01": 1.0e6, "C11": 1.8e6}
    )
    assert svk.shape == mr3.shape == (3, NUM_COLUMNS)
    assert not np.array_equal(mr3, svk)
    np.testing.assert_allclose(mr3[:, :6], svk[:, :6], rtol=1e-3, atol=1e-12)
    np.testing.assert_array_equal(mr3[0, 6:10], 1.0)
    np.testing.assert_allclose(mr3[:, 6:10], 1.0, atol=1e-3)
    np.testing.assert_allclose(mr3[1:, 6:8], statistics["J_solid_per_step"], rtol=1e-6)
    np.testing.assert_allclose(mr3[1:, 8:10], statistics["J_fluid_per_step"], rtol=1e-6)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"absolute_theta_shift": 0.01, "relative_theta_shift": 1.0}, "theta_shift"),
        ({"nu_s": 0.4, "lambda_s": 8.0e6}, "nu_s and lambda_s"),
        ({"max_quadrature_degree": None}, "quadrature cap,"),
        ({"mesh_motion_max_quadrature_degree": None}, "quadrature cap on the mesh motion"),
        ({"snapshot_times": (0.004,)}, "snapshot_times requires snapshot_dir"),
        ({"solid_material": "neo_hookean"}, "solid_material must be one of"),
        ({"solid_material": "mooney_rivlin_3"}, "needs mooney_rivlin_3"),
        ({"solid_material": "mooney_rivlin_3", "mooney_rivlin_3": {"C10": 0.0, "C01": 1e6}}, "needs mooney_rivlin_3"),
        ({"solid_material": "mooney_rivlin_3", "mooney_rivlin_3": {"C10": 0.0, "C01": 1e6, "C11": -1.0}}, "C11"),
        ({"mooney_rivlin_3": {"C10": 0.0, "C01": 1e6, "C11": 0.0}}, "but solid_material is"),
        ({"interface_ale": True, "interface_ale_rows": ("pressure",)}, "interface_ale_rows"),
    ],
    ids=[
        "theta_shift",
        "nu_s_lambda_s",
        "quadrature_cap",
        "mesh_motion_quadrature_cap",
        "snapshot_without_dir",
        "unknown_material",
        "no_coefficients",
        "missing_C11",
        "negative_C11",
        "coefficients_without_material",
        "interface_ale_rows",
    ],
)
def test_threeDfsi_simplified_rejects_invalid_options(output_dirs, kwargs, match):
    # Raised before the mesh is read
    with pytest.raises(ValueError, match=match):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.004,
            dt_val=0.004,
            output_path=str(output_dirs["pv"] / "invalid.vtkhdf"),
            qoi_path=str(output_dirs["qoi"] / "invalid.txt"),
            **kwargs,
        )
