import numpy as np
import pytest
from newton_log_helpers import check_solver_newton_log
from restart_helpers import check_restart_reproduces_continuous_run, check_start_from_initial_state

from xfsi_solver.solvers import fsi2_harmonic_fulljacobian
from xfsi_solver.solvers.fsi2_harmonic_fulljacobian_initialguess import NEST_GROUPS, solve
from xfsi_solver.tools.condensation import (
    check_condensation,
    condense_solid_exact,
    condense_u_s,
    drop_ale_derivatives,
)
from xfsi_solver.tools.newton_log import read_newton_log

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"

# p lives on the fluid submesh but is checkpointed on the full mesh
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1)}


def run(output_dirs, name, num_steps, solve_function=solve, **kwargs):
    """The QoIs of ``num_steps`` steps from rest on the coarse mesh, and the Newton log if kept."""
    dt_val = 0.0025
    if solve_function is solve:
        kwargs.setdefault("newton_log_path", str(output_dirs["qoi"] / f"{name}_newton.txt"))
    solve_function(
        mesh_path=MESH,
        T=num_steps * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        **kwargs,
    )
    qoi = np.loadtxt(output_dirs["qoi"] / f"{name}.txt", ndmin=2)
    log_path = output_dirs["qoi"] / f"{name}_newton.txt"
    return qoi, read_newton_log(log_path) if log_path.exists() else None


@pytest.mark.parametrize("condense_solid", [False, True], ids=["full", "condensed"])
def test_fsi2_harmonic_fulljacobian_initialguess_solve(output_dirs, condense_solid):
    name = "fsi2_harmonic_fulljacobian_initialguess"
    run(output_dirs, name, 6, vtx_save_every=4, condense_solid=condense_solid)
    assert (output_dirs["pv"] / f"{name}.bp").exists()
    check_solver_newton_log(
        output_dirs["qoi"] / f"{name}_newton.txt",
        output_dirs["qoi"] / f"{name}.txt",
        ["u", "v", "p"],
        always_assembled=True,
    )


def test_fsi2_harmonic_fulljacobian_initialguess_without_options_is_fulljacobian(output_dirs):
    qoi, _ = run(output_dirs, "full", 3, solve_function=fsi2_harmonic_fulljacobian.solve)
    qoi_off, _ = run(output_dirs, "off", 3, predict_displacements=False)
    np.testing.assert_allclose(qoi_off, qoi, rtol=1e-12, atol=0)


def test_fsi2_harmonic_fulljacobian_initialguess_options_converge_to_the_same_state(output_dirs):
    # The prediction only changes the initial guess, and the condensation only the solid
    # displacement in the solid stress, which the linear kinematic relation equals from the first
    # Newton step on. So all runs agree within the Newton tolerance.
    qoi, log = run(output_dirs, "previous", 4, predict_displacements=False)
    variants = [
        ("predicted", {}),
        ("condensed", {"condense_solid": True}),
    ]
    for name, kwargs in variants:
        qoi_other, log_other = run(output_dirs, name, 4, **kwargs)
        for column in range(1, qoi.shape[1]):
            np.testing.assert_allclose(
                qoi_other[:, column], qoi[:, column], rtol=0, atol=1e-6 * np.abs(qoi[:, column]).max()
            )
        assert len(log_other) <= len(log)


@pytest.mark.parametrize("condense_solid", [False, True], ids=["full", "condensed"])
def test_fsi2_harmonic_fulljacobian_initialguess_nest_condensations(output_dirs, condense_solid):
    errors = []

    def callback(snes, nest_view):
        assert nest_view.names == list(NEST_GROUPS)
        J = snes.getJacobian()[0]
        x = J.createVecRight()
        x.setRandom()
        expected, result = J.createVecLeft(), J.createVecLeft()
        J.mult(x, expected)
        nest_view.nest.mult(x, result)
        assert np.allclose(result.array, expected.array, rtol=1e-12, atol=1e-12 * expected.norm())

        # The solid stress is the only coupling of v to u_s, and the condensation moves it to (v, v).
        assert (nest_view.block("v", "u_s") is None) == condense_solid
        sizes = {name: index_set.getSize() for name, index_set in zip(NEST_GROUPS, nest_view.index_sets, strict=True)}
        assert sum(sizes.values()) == J.getSize()[0] and min(sizes.values()) > 0

        S, _, made = condense_u_s(nest_view)
        schur_error = check_condensation(nest_view, ["u_s"], S)
        for obj in (S, *made):
            obj.destroy()
        S, _, made, relation_error = condense_solid_exact(nest_view)
        exact_error = check_condensation(nest_view, ["u_s", "u_int"], S)
        for obj in (S, *made):
            obj.destroy()
        errors.append((schur_error, exact_error, relation_error))

    run(output_dirs, "nest", 2, condense_solid=condense_solid, jacobian_callback=callback)
    assert len(errors) > 0
    for schur_error, exact_error, relation_error in errors:
        assert schur_error < 1e-12
        assert exact_error < 1e-9
        assert relation_error < 1e-14


def test_fsi2_harmonic_fulljacobian_initialguess_J_0_converges_to_the_same_state(output_dirs):
    # Dropping the ALE derivatives of (v, p) only changes the Jacobian, so with the solid condensed in
    # UFL, Newton converges, more slowly, to the same state as with the full Jacobian. The (v, p)
    # block is then the Jacobian of fsi2_simplified, and the displacement rows only follow it.
    dropped = []

    def J_0(snes, nest_view):
        dropped.append(drop_ale_derivatives(nest_view))
        for row in ["v", "p"]:
            for col in ["u_s", "u_int", "u_f"]:
                block = nest_view.block(row, col)
                assert block is None or block.norm() == 0.0
        assert nest_view.block("v", "v").norm() > 0.0

    qoi, log = run(output_dirs, "full", 4, condense_solid=True)
    qoi_J_0, log_J_0 = run(output_dirs, "J_0", 4, condense_solid=True, jacobian_callback=J_0)
    for column in range(1, qoi.shape[1]):
        np.testing.assert_allclose(qoi_J_0[:, column], qoi[:, column], rtol=0, atol=1e-6 * np.abs(qoi[:, column]).max())
    # From rest the fluid ALE derivatives are small but not zero, so J_0 needs at least as many iterations.
    assert len(dropped) == np.sum(log_J_0["it"] > 0) and max(dropped) > 0.0
    assert len(log_J_0) >= len(log)


def test_fsi2_harmonic_fulljacobian_initialguess_restart_reproduces_continuous_run(output_dirs, tmp_path):
    check_restart_reproduces_continuous_run(
        solve, MESH, ELEMENTS, output_dirs, tmp_path, solve_kwargs={"condense_solid": True}
    )


def test_fsi2_harmonic_fulljacobian_initialguess_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve, MESH, ELEMENTS, output_dirs, tmp_path, solve_kwargs={"move_reference_domain": True}
    )
