import numpy as np
import pytest
from newton_log_helpers import check_solver_newton_log
from restart_helpers import check_start_from_initial_state

from xfsi_solver.solvers.threeDfsi_simplified import (
    DEFAULT_DIRECT_SOLVER_OPTIONS,
    DEFAULT_ITERATIVE_SOLVER_OPTIONS,
    solve,
)
from xfsi_solver.tools.newton_log import read_newton_log

VECTOR = ("Lagrange", 2, (3,))
# p lives on the fluid submesh but is checkpointed on the full mesh. No field of the mesh motion is
# checkpointed: every mesh motion solve computes it again.
ELEMENTS = {"u": VECTOR, "v": VECTOR, "p": ("Lagrange", 1)}

MESH = "data/meshes/fsi3d/mesh_sec_coarse.xdmf"


@pytest.mark.slow
def test_threeDfsi_simplified_iterative_solve_matches_direct(output_dirs):
    dt_val = 0.004

    def run(name, solver_options):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(qoi_path),
            solver_options=solver_options,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    # Fails with DIVERGED_LINEAR_SOLVE if the fieldsplit does not get the v and p splits or their options.
    iterative = run("iterative", DEFAULT_ITERATIVE_SOLVER_OPTIONS)
    direct = run("direct", DEFAULT_DIRECT_SOLVER_OPTIONS)

    # The initial state and two time steps, with t, drag, lift and the displacement of B.
    assert iterative.shape == direct.shape == (3, 6)
    # The inflow drives the fluid in the x-direction, so the drag grows.
    assert 0.0 < iterative[1, 1] < iterative[2, 1]
    # Newton stops at the same absolute residual tolerance with either linear solver, but at different
    # iterates. The initial row of zeros is the same in both.
    np.testing.assert_allclose(iterative[:, :3], direct[:, :3], rtol=1e-4, atol=0)


@pytest.mark.slow
def test_threeDfsi_simplified_prediction_and_reuse_options(output_dirs):
    # The default run predicts the displacements, here without Jacobian reuse, so it assembles in
    # every Newton iteration. The other starts from the previous step as it is and skips the reuse
    # test at the first iteration, so it never assembles at the second. Both converge to the same
    # states, within the Newton tolerance.
    dt_val = 0.004

    def run(name, **kwargs):
        solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
            solver_options=DEFAULT_DIRECT_SOLVER_OPTIONS,
            newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
            **kwargs,
        )
        check_solver_newton_log(
            output_dirs["qoi"] / f"{name}_newton.txt",
            output_dirs["qoi"] / f"{name}.txt",
            ["v", "p"],
            always_assembled=kwargs.get("gamma_reassemble", 0.2) is None,
        )
        return np.loadtxt(output_dirs["qoi"] / f"{name}.txt", ndmin=2), read_newton_log(
            output_dirs["qoi"] / f"{name}_newton.txt"
        )

    predicted, _ = run("predicted", gamma_reassemble=None)
    previous, log = run("previous", predict_displacements=False, skip_first_reuse_test=True)
    assert np.all(log["assembled"][log["it"] == 2] == 0)
    for column in range(1, predicted.shape[1]):
        np.testing.assert_allclose(
            predicted[:, column], previous[:, column], rtol=0, atol=1e-6 * np.abs(previous[:, column]).max()
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
        solve_kwargs={"solver_options": DEFAULT_DIRECT_SOLVER_OPTIONS},
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"absolute_theta_shift": 0.01, "relative_theta_shift": 1.0}, "theta_shift"),
        ({"nu_s": 0.4, "lambda_s": 8.0e6}, "nu_s and lambda_s"),
        ({"max_quadrature_degree": None}, "quadrature cap,"),
        ({"mesh_motion_max_quadrature_degree": None}, "quadrature cap on the mesh motion"),
    ],
    ids=["theta_shift", "nu_s_lambda_s", "quadrature_cap", "mesh_motion_quadrature_cap"],
)
def test_threeDfsi_simplified_rejects_invalid_options(output_dirs, kwargs, match):
    # Raised before the mesh is read
    with pytest.raises(ValueError, match=match):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.004,
            dt_val=0.004,
            output_path=str(output_dirs["pv"] / "invalid.bp"),
            output_path_p=str(output_dirs["pv"] / "invalid_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "invalid.txt"),
            **kwargs,
        )
