import numpy as np
import pytest
from restart_helpers import (
    check_initial_state_rejects_T_not_after_it,
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
    check_start_from_initial_state,
)

from xfsi_solver.solvers.fsi2_harmonic import solve

# p lives on the fluid submesh but is checkpointed on the full mesh
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1)}


def test_fsi2_harmonic_solve(output_dirs):
    dt_val = 0.0025
    solve(
        # smaller stand-in for the default "mesh_quad_fine_sec.xdmf" mesh
        mesh_path="data/meshes/fsi2/mesh_sec_coarse.xdmf",
        # Run enough steps to save at least 2 VTX snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_harmonic.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harmonic_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_harmonic.txt"),
        vtx_save_every=4,
    )

    assert (output_dirs["qoi"] / "fsi2_harmonic.txt").exists()
    assert (output_dirs["pv"] / "fsi2_harmonic.bp").exists()


def test_fsi2_harmonic_quadrature_cap_changes_the_qois_slightly(output_dirs):
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
    assert exact.shape == capped.shape == (3, 5)
    # The cap makes the integration of the nonlinear terms inexact, so the QoIs are not identical, but
    # they stay close. On this mesh the rows differ by up to 3e-5 relative to the uncapped values, so
    # rtol = 1e-3 leaves a margin of more than 30 while still failing for a wrong residual. The initial
    # row of zeros is the same in both.
    np.testing.assert_allclose(capped, exact, rtol=1e-3, atol=0)
    assert not np.array_equal(capped, exact)


def test_fsi2_harmonic_writes_no_vtx_by_default(output_dirs):
    dt_val = 0.0025
    solve(
        mesh_path="data/meshes/fsi2/mesh_sec_coarse.xdmf",
        T=2 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_harmonic.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harmonic_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_harmonic.txt"),
    )

    assert (output_dirs["qoi"] / "fsi2_harmonic.txt").exists()
    assert not any(output_dirs["pv"].iterdir())


def test_fsi2_harmonic_restart_reproduces_continuous_run(output_dirs, tmp_path):
    check_restart_reproduces_continuous_run(
        solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", ELEMENTS, output_dirs, tmp_path
    )


@pytest.mark.parametrize("first_ranks,restart_ranks", [(2, 1), (1, 2)], ids=["2to1", "1to2"])
def test_fsi2_harmonic_restart_on_different_number_of_ranks(output_dirs, tmp_path, first_ranks, restart_ranks):
    # Not bitwise identical across partitions; QoIs are printed to 6 digits, so the last one may flip
    check_restart_reproduces_continuous_run(
        solve,
        "data/meshes/fsi2/mesh_sec_coarse.xdmf",
        ELEMENTS,
        output_dirs,
        tmp_path,
        qoi_rtol=1e-5,
        first_ranks=first_ranks,
        restart_ranks=restart_ranks,
    )


def test_fsi2_harmonic_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)


def test_fsi2_harmonic_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)


def test_fsi2_harmonic_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", ELEMENTS, output_dirs, tmp_path)


def test_fsi2_harmonic_initial_state_rejects_T_not_after_it(output_dirs, tmp_path):
    check_initial_state_rejects_T_not_after_it(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)


def test_fsi2_harmonic_rejects_both_theta_shifts(output_dirs):
    with pytest.raises(ValueError, match="theta_shift"):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "fsi2_harmonic.bp"),
            output_path_p=str(output_dirs["pv"] / "fsi2_harmonic_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "fsi2_harmonic.txt"),
            absolute_theta_shift=0.01,
            relative_theta_shift=1.0,
        )


def test_fsi2_harmonic_rejects_both_nu_s_and_lambda_s(output_dirs):
    with pytest.raises(ValueError, match="nu_s and lambda_s"):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "fsi2_harmonic.bp"),
            output_path_p=str(output_dirs["pv"] / "fsi2_harmonic_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "fsi2_harmonic.txt"),
            nu_s=0.4,
            lambda_s=2.0e6,
        )
