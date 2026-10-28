import numpy as np
import pytest
from restart_helpers import (
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
    check_start_from_initial_state,
)

from xfsi_solver.solvers.fsi2_harmonic_lagrange import solve

# All fields live on the fluid or solid submesh but are checkpointed on the full mesh. The interface
# Lagrange multipliers are not checkpointed (see the solver), and restarting them from zero changes u_f
# by ~1e-6 (as much as tightening the Newton tolerance does) and the other fields by < 1e-9 relative.
# The interface gaps, measured on u_f, may then change in their last printed digit (1.4e-6 relative to
# their largest value seen with DOLFINx main), so they get the tolerance of u_f
VECTOR = ("Lagrange", 2, (2,))
ELEMENTS = {"u_f": VECTOR, "v_f": VECTOR, "u_s": VECTOR, "v_s": VECTOR, "p": ("Lagrange", 1)}
RESTART_TOLERANCES = dict(
    rtol=1e-8,
    field_rtol={"u_f": 1e-5},
    qoi_column_rtol={"interface_u_gap": 1e-5, "interface_v_gap": 1e-5},
)


def test_fsi2_harmonic_lagrange_solve(output_dirs):
    dt_val = 0.0025
    solve(
        mesh_path="data/meshes/fsi2/mesh_sec_coarse.xdmf",
        # Run enough steps to save at least 2 VTX snapshots
        T=10 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_harmonic_lagrange.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harmonic_lagrange_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_harmonic_lagrange.txt"),
        vtx_save_every=8,
    )

    assert (output_dirs["qoi"] / "fsi2_harmonic_lagrange.txt").exists()
    assert (output_dirs["pv"] / "fsi2_harmonic_lagrange.bp").exists()


def test_fsi2_harmonic_lagrange_quadrature_cap_changes_the_qois_slightly(output_dirs):
    dt_val = 0.0025

    def run(name, max_quadrature_degree):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path="data/meshes/fsi2/mesh_sec_coarse.xdmf",
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
    assert exact.shape == capped.shape == (3, 7)
    # The cap makes the integration of the nonlinear terms inexact, so the QoIs are not identical, but
    # they stay close. On this mesh the time, drag, lift and tip displacement columns differ by up to
    # 5e-5 relative to the uncapped values, so rtol = 1e-3 leaves a margin of 20 while still failing for a
    # wrong residual. The interface gaps are zero up to the accuracy of the solve: u_gap is at the level
    # 1e-10 and changes by 3e-13 (relative 3e-3, which is noise), v_gap is 6e-6 and changes by up to 2e-11,
    # so they are compared with atol = 1e-9. The initial row of zeros is the same in both.
    np.testing.assert_allclose(capped[:, :5], exact[:, :5], rtol=1e-3, atol=0)
    np.testing.assert_allclose(capped[:, 5:], exact[:, 5:], rtol=0, atol=1e-9)
    assert not np.array_equal(capped, exact)


def test_fsi2_harmonic_lagrange_restart_reproduces_continuous_run(output_dirs, tmp_path):
    check_restart_reproduces_continuous_run(
        solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", ELEMENTS, output_dirs, tmp_path, **RESTART_TOLERANCES
    )


@pytest.mark.parametrize("first_ranks,restart_ranks", [(2, 1), (1, 2)], ids=["2to1", "1to2"])
def test_fsi2_harmonic_lagrange_restart_on_different_number_of_ranks(output_dirs, tmp_path, first_ranks, restart_ranks):
    # Not bitwise identical across partitions; QoIs are printed to 6 digits, so the last one may flip.
    # A different partition perturbs the loosely resolved u_f more (up to ~3e-6 relative seen), and the
    # interface gap diagnostics are at the solver tolerance (~1e-8), so only their leading digits agree
    # (up to ~5e-3 relative seen). With DOLFINx main, the u gap after the first step, 1.1e-10 on one rank,
    # is 2.4e-10 on two, 3e-2 of its largest value: both zero up to the accuracy of the solve, so the gaps
    # are compared to 1e-1 of their largest values (4e-10 for u). The physical fields and QoIs keep their
    # tolerances
    check_restart_reproduces_continuous_run(
        solve,
        "data/meshes/fsi2/mesh_sec_coarse.xdmf",
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=RESTART_TOLERANCES["rtol"],
        field_rtol={"u_f": 1e-4},
        qoi_rtol=1e-5,
        qoi_column_rtol={"interface_u_gap": 1e-1, "interface_v_gap": 1e-1},
        first_ranks=first_ranks,
        restart_ranks=restart_ranks,
    )


def test_fsi2_harmonic_lagrange_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)


def test_fsi2_harmonic_lagrange_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)


def test_fsi2_harmonic_lagrange_starts_from_initial_state(output_dirs, tmp_path):
    # Two runs from the same initial state, one restarted after a step, both with the multipliers from
    # zero. u_f is still small here (largest value 8e-8), and resolved only to about the final Newton
    # residual (5e-12): its runs differed by 1.9e-7 relative with DOLFINx 0.11, but by 2.8e-5 (2.3e-12) with
    # DOLFINx main, at the same residuals, so u_f is compared to 1e-4, as across partitions
    check_start_from_initial_state(
        solve,
        "data/meshes/fsi2/mesh_sec_coarse.xdmf",
        ELEMENTS,
        output_dirs,
        tmp_path,
        **RESTART_TOLERANCES | {"field_rtol": {"u_f": 1e-4}},
    )


def test_fsi2_harmonic_lagrange_rejects_both_theta_shifts(output_dirs):
    with pytest.raises(ValueError, match="theta_shift"):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "fsi2_harmonic_lagrange.bp"),
            output_path_p=str(output_dirs["pv"] / "fsi2_harmonic_lagrange_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "fsi2_harmonic_lagrange.txt"),
            absolute_theta_shift=0.01,
            relative_theta_shift=1.0,
        )


def test_fsi2_harmonic_lagrange_rejects_both_nu_s_and_lambda_s(output_dirs):
    with pytest.raises(ValueError, match="nu_s and lambda_s"):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "fsi2_harmonic_lagrange.bp"),
            output_path_p=str(output_dirs["pv"] / "fsi2_harmonic_lagrange_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "fsi2_harmonic_lagrange.txt"),
            nu_s=0.4,
            lambda_s=2.0e6,
        )
