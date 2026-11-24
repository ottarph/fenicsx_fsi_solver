import numpy as np
import pytest
from newton_log_helpers import check_solver_newton_log
from restart_helpers import check_restart_reproduces_continuous_run, check_start_from_initial_state

from xfsi_solver.solvers import fsi2_biharmonic_fulljacobian
from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import NEST_GROUPS, solve
from xfsi_solver.tools.newton_log import read_newton_log

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"

# p and z live on the fluid submesh but are checkpointed on the full mesh
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1), "z": ("Lagrange", 2, (2,))}


def test_fsi2_biharmonic_fulljacobian_initialguess_solve(output_dirs):
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
        newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
    )

    assert (output_dirs["pv"] / f"{name}.bp").exists()
    check_solver_newton_log(
        output_dirs["qoi"] / f"{name}_newton.txt",
        output_dirs["qoi"] / f"{name}.txt",
        ["u", "v", "p", "z"],
        always_assembled=True,
    )


def test_fsi2_biharmonic_fulljacobian_initialguess_nest_jacobian(output_dirs):
    calls = []

    def callback(snes, nest_view):
        J = snes.getJacobian()[0]
        assert nest_view.names == list(NEST_GROUPS)
        # The nest keeps the numbering of the Jacobian, so it acts on the same vectors.
        x = J.createVecRight()
        x.setRandom()
        expected, result = J.createVecLeft(), J.createVecLeft()
        J.mult(x, expected)
        nest_view.nest.mult(x, result)
        assert np.allclose(result.array, expected.array, rtol=1e-12, atol=1e-12 * expected.norm())

        def norm(row, col):
            block = nest_view.block(row, col)
            return 0.0 if block is None else block.norm()

        # The interface rows of u hold only the solid kinematic relation, in u_s, u_int and v.
        for col in ["p", "z_int", "z_f", "u_f"]:
            assert norm("u_int", col) == 0.0
        assert norm("u_int", "u_int") > 0.0 and norm("u_int", "v") > 0.0
        # The mesh motion rows do not reach the solid off the interface, nor v and p.
        for row in ["z_int", "z_f", "u_f"]:
            assert nest_view.block(row, "u_s") is None
            assert nest_view.block(row, "v") is None and nest_view.block(row, "p") is None
        sizes = {name: index_set.getSize() for name, index_set in zip(NEST_GROUPS, nest_view.index_sets, strict=True)}
        assert sum(sizes.values()) == J.getSize()[0]
        # u and z have the same element, on both sides of the same interface.
        assert sizes["u_int"] == sizes["z_int"] > 0
        calls.append(sizes)

    dt_val = 0.0025
    name = "nest"
    solve(
        mesh_path=MESH,
        T=2 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
        jacobian_callback=callback,
    )
    # Once per Newton iteration, with the same groups.
    log = read_newton_log(output_dirs["qoi"] / f"{name}_newton.txt")
    assert len(calls) == np.sum(log["it"] > 0)
    assert all(sizes == calls[0] for sizes in calls)


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
