import numpy as np
import pytest
from newton_log_helpers import check_solver_newton_log
from restart_helpers import check_restart_reproduces_continuous_run, check_start_from_initial_state

from xfsi_solver.solvers import fsi2_biharmonic_fulljacobian
from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import solve
from xfsi_solver.tools.newton_log import read_newton_log

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"

# p and z live on the fluid submesh but are checkpointed on the full mesh
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1), "z": ("Lagrange", 2, (2,))}


@pytest.mark.parametrize("mesh_motion_scaling", [None, 1.0e-9], ids=["unscaled", "scaled"])
def test_fsi2_biharmonic_fulljacobian_initialguess_solve(output_dirs, mesh_motion_scaling):
    dt_val = 0.0025
    name = "fsi2_biharmonic_fulljacobian_initialguess"
    solve(
        mesh_path=MESH,
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        vtx_save_every=4,
        mesh_motion_scaling=mesh_motion_scaling,
        newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
    )

    assert (output_dirs["pv"] / f"{name}.bp").exists()
    check_solver_newton_log(
        output_dirs["qoi"] / f"{name}_newton.txt",
        output_dirs["qoi"] / f"{name}.txt",
        ["u", "v", "p", "z"],
        always_assembled=True,
    )


def run(solve_function, output_dirs, name, num_steps, **kwargs):
    """The QoIs and Newton log of ``num_steps`` steps from rest on the coarse mesh."""
    dt_val = 0.0025
    solve_function(
        mesh_path=MESH,
        T=num_steps * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
        **kwargs,
    )
    return np.loadtxt(output_dirs["qoi"] / f"{name}.txt", ndmin=2), read_newton_log(
        output_dirs["qoi"] / f"{name}_newton.txt"
    )


def test_fsi2_biharmonic_fulljacobian_initialguess_without_prediction_is_fulljacobian(output_dirs):
    qoi, log = run(fsi2_biharmonic_fulljacobian.solve, output_dirs, "full", 3)
    qoi_off, log_off = run(solve, output_dirs, "off", 3, predict_displacements=False)
    np.testing.assert_array_equal(qoi_off, qoi)
    np.testing.assert_array_equal(log_off["residual"], log["residual"])


def test_fsi2_biharmonic_fulljacobian_initialguess_converges_to_the_same_state(output_dirs):
    # The prediction only changes the initial guess of each Newton solve, so the runs agree within
    # the Newton tolerance. Unlike in fsi2_simplified, the residual of the initial guess is not smaller
    # with the prediction: the solid stress is taken at the predicted displacement, which the old
    # velocity does not balance. The first Newton step from it gets closer, though: on this mesh, from
    # the second step on, the residual after it is 3 to 20 times smaller, more as the structure speeds
    # up. The first step starts from rest, where the prediction changes nothing.
    qoi, log = run(solve, output_dirs, "previous", 4, predict_displacements=False)
    qoi_predicted, log_predicted = run(solve, output_dirs, "predicted", 4)
    for column in range(1, qoi.shape[1]):
        np.testing.assert_allclose(
            qoi_predicted[:, column], qoi[:, column], rtol=0, atol=1e-6 * np.abs(qoi[:, column]).max()
        )
    after_first = log["residual"][log["it"] == 1]
    after_first_predicted = log_predicted["residual"][log_predicted["it"] == 1]
    np.testing.assert_allclose(after_first_predicted[0], after_first[0], rtol=1e-6)
    assert np.all(after_first_predicted[1:] < 0.5 * after_first[1:])
    assert len(log_predicted) <= len(log)


def test_fsi2_biharmonic_fulljacobian_initialguess_restart_reproduces_continuous_run(output_dirs, tmp_path):
    check_restart_reproduces_continuous_run(solve, MESH, ELEMENTS, output_dirs, tmp_path)


@pytest.mark.parametrize("first_ranks,restart_ranks", [(2, 1), (1, 2)], ids=["2to1", "1to2"])
def test_fsi2_biharmonic_fulljacobian_initialguess_restart_on_different_number_of_ranks(
    output_dirs, tmp_path, first_ranks, restart_ranks
):
    # Not bitwise identical across partitions; QoIs are printed to 6 digits, so the last one may flip
    check_restart_reproduces_continuous_run(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        qoi_rtol=1e-5,
        first_ranks=first_ranks,
        restart_ranks=restart_ranks,
    )


def test_fsi2_biharmonic_fulljacobian_initialguess_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(solve, MESH, ELEMENTS, output_dirs, tmp_path)
