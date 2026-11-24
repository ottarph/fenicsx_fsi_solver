import numpy as np
import pytest
from newton_log_helpers import check_solver_newton_log
from restart_helpers import check_restart_reproduces_continuous_run, check_start_from_initial_state

from xfsi_solver.solvers import fsi2_harmonic_fulljacobian
from xfsi_solver.solvers.fsi2_harmonic_fulljacobian_initialguess import NEST_GROUPS, solve
from xfsi_solver.tools.condensation import (
    check_condensation,
    condense_schur,
    condense_solid_exact,
    condense_u_s,
    count_nonzeros,
    drop_ale_derivatives,
    use_schur_solve,
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
    # The same Newton steps, but the momentum rows are multiplied by dt here, so Newton stops earlier
    # and the QoIs agree to the Newton tolerance only: on this mesh to 5e-6 of the largest value of a
    # column.
    qoi, _ = run(output_dirs, "full", 3, solve_function=fsi2_harmonic_fulljacobian.solve)
    qoi_off, _ = run(output_dirs, "off", 3, predict_displacements=False)
    for column in range(1, qoi.shape[1]):
        np.testing.assert_allclose(qoi_off[:, column], qoi[:, column], rtol=0, atol=1e-5 * np.abs(qoi[:, column]).max())


def test_fsi2_harmonic_fulljacobian_initialguess_options_converge_to_the_same_state(output_dirs):
    # The prediction only changes the initial guess, and the condensation only the solid
    # displacement in the solid stress, which the linear kinematic relation equals from the first
    # Newton step on. So all runs agree within the Newton tolerance, on this mesh to 4e-6 of the
    # largest value of a column.
    qoi, log = run(output_dirs, "previous", 4, predict_displacements=False)
    variants = [
        ("predicted", {}),
        ("condensed", {"condense_solid": True}),
    ]
    for name, kwargs in variants:
        qoi_other, log_other = run(output_dirs, name, 4, **kwargs)
        for column in range(1, qoi.shape[1]):
            np.testing.assert_allclose(
                qoi_other[:, column], qoi[:, column], rtol=0, atol=1e-5 * np.abs(qoi[:, column]).max()
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
        # The mesh displacement, u_int and u_f, exactly and with the diagonal of J[E, E] as its
        # inverse, which is only an approximation, but keeps S sparse.
        mesh_errors, mesh_nonzeros = {}, {}
        for inverse in ("exact", "diagonal"):
            S, S_blocks, made = condense_schur(nest_view, ["u_int", "u_f"], inverse)
            mesh_errors[inverse] = check_condensation(nest_view, ["u_int", "u_f"], S)
            mesh_nonzeros[inverse] = count_nonzeros(S_blocks)
            for obj in (S, *made):
                obj.destroy()
        errors.append((schur_error, exact_error, relation_error, mesh_errors, mesh_nonzeros))

    run(output_dirs, "nest", 2, condense_solid=condense_solid, jacobian_callback=callback)
    assert len(errors) > 0
    for schur_error, exact_error, relation_error, mesh_errors, mesh_nonzeros in errors:
        assert schur_error < 1e-12
        assert exact_error < 1e-9
        assert relation_error < 1e-14
        assert mesh_errors["exact"] < 1e-12
        # On this mesh 7e-3 (with condense_solid) and 4e-2 (without) from the exact Schur complement.
        assert 1e-4 < mesh_errors["diagonal"] < 0.1
        assert mesh_nonzeros["diagonal"] <= mesh_nonzeros["exact"]


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
        np.testing.assert_allclose(qoi_J_0[:, column], qoi[:, column], rtol=0, atol=1e-5 * np.abs(qoi[:, column]).max())
    # From rest the fluid ALE derivatives are small but not zero, so J_0 needs at least as many iterations.
    assert len(dropped) == np.sum(log_J_0["it"] > 0) and max(dropped) > 0.0
    assert len(log_J_0) >= len(log)


@pytest.mark.parametrize("inverse", ["exact", "diagonal"])
def test_fsi2_harmonic_fulljacobian_initialguess_schur_mesh_solve(output_dirs, inverse):
    # Condensing u_int and u_f out in the linear solver changes the Newton steps only with the diagonal
    # inverse, so both converge to the state that LU of the whole Jacobian gives. The diagonal one
    # converges linearly and stops just below snes_atol, where LU has overshot, so with the default
    # snes_atol = 1e-7 the QoIs from rest differ by up to 4e-3 of the largest value of a column. With
    # 1e-11 both are converged far enough to agree to 1e-5.
    pc_types = []

    def schur(snes, nest_view):
        use_schur_solve(snes, nest_view, ["u_int", "u_f"], inverse)
        pc_types.append(snes.getKSP().getPC().getType())

    qoi, _ = run(output_dirs, "lu", 3, condense_solid=True, snes_atol=1e-11)
    qoi_schur, _ = run(output_dirs, "schur", 3, condense_solid=True, snes_atol=1e-11, jacobian_callback=schur)
    assert set(pc_types) == {"fieldsplit"}
    for column in range(1, qoi.shape[1]):
        np.testing.assert_allclose(
            qoi_schur[:, column], qoi[:, column], rtol=0, atol=1e-5 * np.abs(qoi[:, column]).max()
        )


def test_fsi2_harmonic_fulljacobian_initialguess_restart_reproduces_continuous_run(output_dirs, tmp_path):
    check_restart_reproduces_continuous_run(
        solve, MESH, ELEMENTS, output_dirs, tmp_path, solve_kwargs={"condense_solid": True}
    )


def test_fsi2_harmonic_fulljacobian_initialguess_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve, MESH, ELEMENTS, output_dirs, tmp_path, solve_kwargs={"move_reference_domain": True}
    )
