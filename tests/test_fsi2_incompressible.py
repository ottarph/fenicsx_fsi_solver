import numpy as np
import pytest
from mpi4py import MPI
from newton_log_helpers import check_solver_newton_log
from restart_helpers import check_restart_reproduces_continuous_run, check_start_from_initial_state

from xfsi_solver.petsc_options.fsi2d import (
    DEFAULT_DIRECT_SOLVER_OPTIONS,
    DEFAULT_ITERATIVE_SOLVER_OPTIONS,
    MUMPS_ON_SOLID,
)
from xfsi_solver.solvers.fsi2_incompressible import solve
from xfsi_solver.tools import vtkhdf

# The checkpoints hold the displacement as one function u on the full mesh; p and p_S live on the fluid
# and solid submeshes but are checkpointed on the full mesh.
ELEMENTS = {
    "u": ("Lagrange", 2, (2,)),
    "v": ("Lagrange", 2, (2,)),
    "p": ("Lagrange", 1),
    "p_S": ("Lagrange", 1),
}

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"

# The momentum residual is multiplied by dt, as in fsi2_simplified, so the default snes_atol = 1e-7 is
# loose on it at dt = 0.0025. The tests comparing runs that take different paths to the same state (a
# restart, which assembles a fresh Jacobian, or a start from a checkpoint) converge Newton to
# snes_atol = 1e-12 instead. At 1e-7 * dt, the tolerance of fsi2_simplified's tests, the tip
# y-displacement, still about 1e-8 in these first steps, differs by 3e-6 of its largest value after a
# restart, more than the 1e-6 the tests allow.
CONVERGED_SOLVER_OPTIONS = DEFAULT_DIRECT_SOLVER_OPTIONS | {"snes_atol": 1e-12}

# The columns of the QoI file: t, drag, lift, the tip displacement, the violation of det(F) = 1, and the
# smallest and largest det(F) of the solid and of the fluid mesh
NUM_COLUMNS = 12
AREA_CHANGE = 5
DET_F_RANGE = slice(8, 12)


def test_fsi2_incompressible_solve(output_dirs):
    dt_val = 0.0025
    qoi_path = output_dirs["qoi"] / "fsi2_incompressible.txt"
    output_path = output_dirs["pv"] / "fsi2_incompressible.vtkhdf"
    solve(
        mesh_path=MESH,
        # Run enough steps to write the output at two times
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_path),
        qoi_path=str(qoi_path),
        output_every=4,
        newton_log_path=str(output_dirs["qoi"] / "fsi2_incompressible_newton.txt"),
    )

    # The initial state and six time steps
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (7, NUM_COLUMNS)
    # The inflow drives the fluid in the x-direction, so the drag grows, and the structure moves.
    assert 0.0 < rows[1, 1] < rows[-1, 1]
    assert np.abs(rows[-1, 3]) > 0.0
    # The constraint against P1 pressures, which contain the constants, keeps the area of the solid to
    # round-off, while det(F) - 1 is small but not zero pointwise.
    assert np.all(np.abs(rows[:, AREA_CHANGE]) < 1e-12)
    assert np.all(rows[1:, AREA_CHANGE + 1 : AREA_CHANGE + 3] > 0.0)
    assert np.all(rows[:, AREA_CHANGE + 1 : AREA_CHANGE + 3] < 1e-6)
    # The range of det(F) on the solid and on the fluid mesh: 1 at rest, then min <= max near 1, the solid's
    # within its incompressibility violation
    det_F = rows[:, DET_F_RANGE]
    np.testing.assert_array_equal(det_F[0], 1.0)
    assert np.all(det_F[:, 0] <= det_F[:, 1]) and np.all(det_F[:, 2] <= det_F[:, 3])
    np.testing.assert_allclose(det_F[:, :2], 1.0, atol=1e-5)
    np.testing.assert_allclose(det_F[:, 2:], 1.0, atol=1e-3)
    assert np.any(det_F[1:] != 1.0)
    # The output at steps 0 and 4, in one VTKHDF file with u, v, p and p_S
    np.testing.assert_allclose(vtkhdf.output_times(output_path, MPI.COMM_WORLD), [0.0, 4 * dt_val])
    assert sorted(vtkhdf.function_names(output_path, MPI.COMM_WORLD)) == ["p", "p_S", "u", "v"]
    check_solver_newton_log(output_dirs["qoi"] / "fsi2_incompressible_newton.txt", qoi_path, ["v", "p", "p_S"])


def test_fsi2_incompressible_rejects_both_theta_shifts(output_dirs):
    # Raised before the mesh is read
    with pytest.raises(ValueError, match="theta_shift"):
        solve(
            mesh_path="data/meshes/does_not_exist.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "rejected.vtkhdf"),
            qoi_path=str(output_dirs["qoi"] / "rejected.txt"),
            absolute_theta_shift=0.01,
            relative_theta_shift=1.0,
        )


def test_fsi2_incompressible_default_theta_shift_is_twice_dt(output_dirs):
    # The default shift is theta = 0.5 + 2 dt, twice that of fsi2_simplified, to damp the stiff modes of
    # the solid velocity, so the default run is the run with relative_theta_shift=2.0, and not the one
    # with 1.0.
    dt_val = 0.0025

    def run(name, **kwargs):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            **kwargs,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    default = run("default_shift")
    np.testing.assert_array_equal(run("shift_2", relative_theta_shift=2.0), default)
    assert not np.array_equal(run("shift_1", relative_theta_shift=1.0), default)


def test_fsi2_incompressible_quadrature_cap_changes_the_qois_slightly(output_dirs):
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
    # they stay close: rtol = 1e-3 still fails for a wrong residual. The initial row of zeros is the
    # same in both. The solid area change is round-off on these two steps, so it is compared with an
    # absolute tolerance instead.
    other = np.arange(NUM_COLUMNS) != AREA_CHANGE
    np.testing.assert_allclose(capped[:, other], exact[:, other], rtol=1e-3, atol=0)
    np.testing.assert_allclose(capped[:, AREA_CHANGE], exact[:, AREA_CHANGE], rtol=0, atol=1e-15)
    assert not np.array_equal(capped, exact)


@pytest.mark.parametrize("solver_options", ["iterative", "mumps_on_solid"])
def test_fsi2_incompressible_iterative_solve_matches_direct(output_dirs, solver_options):
    dt_val = 0.0025

    def run(name, options, **kwargs):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=3 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            solver_options=options,
            **kwargs,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    # Both solves are converged tightly, so that the test compares the solutions, not where each solver
    # stops.
    converged = {"snes_atol": 1e-12, "ksp_atol": 1e-14}
    direct = run("direct", DEFAULT_DIRECT_SOLVER_OPTIONS | converged)
    # Fails with DIVERGED_LINEAR_SOLVE if the fieldsplit does not get the v and (p, p_S) splits or their
    # options.
    if solver_options == "iterative":
        iterative = run("iterative", DEFAULT_ITERATIVE_SOLVER_OPTIONS | converged)
    else:
        iterative = run("mumps_on_solid", MUMPS_ON_SOLID | converged, requires_v_sub_IS=True)
    # The QoIs agree to the 7 written digits, up to the rounding of the last one, except the solid area
    # change, which is round-off.
    other = np.arange(NUM_COLUMNS) != AREA_CHANGE
    np.testing.assert_allclose(iterative[:, other], direct[:, other], rtol=1e-5, atol=0)
    np.testing.assert_allclose(iterative[:, AREA_CHANGE], direct[:, AREA_CHANGE], rtol=0, atol=1e-13)


@pytest.mark.parametrize("ranks", [(1, 1), (2, 3)], ids=["serial", "2_to_3_ranks"])
def test_fsi2_incompressible_restart_reproduces_continuous_run(output_dirs, tmp_path, ranks):
    check_restart_reproduces_continuous_run(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        # The solid area change is round-off on these few steps, so only its size is compared
        qoi_column_rtol={"solid_area_change": 1.0},
        first_ranks=ranks[0],
        restart_ranks=ranks[1],
        solve_kwargs={"solver_options": CONVERGED_SOLVER_OPTIONS},
    )


def test_fsi2_incompressible_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        qoi_column_rtol={"solid_area_change": 1.0},
        solve_kwargs={"solver_options": CONVERGED_SOLVER_OPTIONS},
        start_row_new_columns=("drag", "lift"),
    )


def test_fsi2_incompressible_skipped_dirichlet_lifting_gives_the_lifted_qois(output_dirs):
    # During the ramp of the inflow, its values change in every step, so the initial guess departs from
    # them, and the lifted and held Newton iterates differ in the first step of each.
    dt_val = 0.0025

    def run(hold):
        name = f"held_{hold}"
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        stats = solve(
            mesh_path=MESH,
            T=4 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
            qoi_path=str(qoi_path),
            skip_dirichlet_lifting=hold,
        )
        return np.loadtxt(qoi_path, ndmin=2), stats

    lifted, lifted_stats = run(False)
    held, held_stats = run(True)
    assert held_stats["iterations_per_step"] == lifted_stats["iterations_per_step"]
    # The QoIs agree to the Newton tolerance, well below 1e-4 of the largest value of their column,
    # except the solid area change, which is round-off.
    other = np.arange(1, NUM_COLUMNS) != AREA_CHANGE
    scale = np.abs(lifted[1:, 1:]).max(axis=0)[other]
    np.testing.assert_array_less(np.abs(held[1:, 1:] - lifted[1:, 1:])[:, other] / scale, 1e-4)
    np.testing.assert_array_equal(held[:, 0], lifted[:, 0])
