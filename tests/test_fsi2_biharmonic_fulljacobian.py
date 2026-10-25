from pathlib import Path

import pytest
from restart_helpers import (
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
    check_start_from_initial_state,
)

from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian import solve

# p and z live on the fluid submesh but are checkpointed on the full mesh
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1), "z": ("Lagrange", 2, (2,))}


# Without and with the row scaling of the mesh-motion equations in the factorization
SCALINGS = pytest.mark.parametrize("mesh_motion_scaling", [None, 1.0e-9], ids=["unscaled", "scaled"])


@SCALINGS
def test_fsi2_biharmonic_fulljacobian_solve(output_dirs, mesh_motion_scaling):
    dt_val = 0.0025
    solve(
        mesh_path="data/meshes/fsi2/mesh_sec_coarse.xdmf",
        # Run enough steps to save at least 2 VTX snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_biharmonic_fulljacobian.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_biharmonic_fulljacobian_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_biharmonic_fulljacobian.txt"),
        vtx_save_every=4,
        mesh_motion_scaling=mesh_motion_scaling,
    )

    assert (output_dirs["qoi"] / "fsi2_biharmonic_fulljacobian.txt").exists()
    assert (output_dirs["pv"] / "fsi2_biharmonic_fulljacobian.bp").exists()


@SCALINGS
def test_fsi2_biharmonic_fulljacobian_restart_reproduces_continuous_run(output_dirs, tmp_path, mesh_motion_scaling):
    check_restart_reproduces_continuous_run(
        solve,
        "data/meshes/fsi2/mesh_sec_coarse.xdmf",
        ELEMENTS,
        output_dirs,
        tmp_path,
        solve_kwargs={"mesh_motion_scaling": mesh_motion_scaling},
    )


@pytest.mark.parametrize("first_ranks,restart_ranks", [(2, 1), (1, 2)], ids=["2to1", "1to2"])
@SCALINGS
def test_fsi2_biharmonic_fulljacobian_restart_on_different_number_of_ranks(
    output_dirs, tmp_path, first_ranks, restart_ranks, mesh_motion_scaling
):
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
        solve_kwargs={"mesh_motion_scaling": mesh_motion_scaling},
    )


def test_fsi2_biharmonic_fulljacobian_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)


def test_fsi2_biharmonic_fulljacobian_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)


def test_fsi2_biharmonic_fulljacobian_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", ELEMENTS, output_dirs, tmp_path)


def test_fsi2_biharmonic_fulljacobian_plots_matrices(output_dirs, tmp_path, monkeypatch):
    mesh_path = str(Path("data/meshes/fsi2/mesh_sec_coarse.xdmf").resolve())
    # The plots go to output/figures/matrix_blocks/<script>/ in the working directory
    monkeypatch.chdir(tmp_path)
    solve(
        mesh_path=mesh_path,
        T=0.0025,
        dt_val=0.0025,
        output_path=str(output_dirs["pv"] / "fsi2_biharmonic_fulljacobian.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_biharmonic_fulljacobian_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_biharmonic_fulljacobian.txt"),
        plot_matrices=True,
    )

    figures = tmp_path / "output/figures/matrix_blocks/fsi2_biharmonic_fulljacobian"
    for label in ("storage", "eq1", "split"):
        assert (figures / f"spy_{label}.png").exists()
        assert (figures / f"block_norms_{label}.png").exists()


def test_fsi2_biharmonic_fulljacobian_rejects_both_theta_shifts(output_dirs):
    with pytest.raises(ValueError, match="theta_shift"):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "fsi2_biharmonic_fulljacobian.bp"),
            output_path_p=str(output_dirs["pv"] / "fsi2_biharmonic_fulljacobian_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "fsi2_biharmonic_fulljacobian.txt"),
            absolute_theta_shift=0.01,
            relative_theta_shift=1.0,
        )


def test_fsi2_biharmonic_fulljacobian_rejects_both_nu_s_and_lambda_s(output_dirs):
    with pytest.raises(ValueError, match="nu_s and lambda_s"):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "fsi2_biharmonic_fulljacobian.bp"),
            output_path_p=str(output_dirs["pv"] / "fsi2_biharmonic_fulljacobian_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "fsi2_biharmonic_fulljacobian.txt"),
            nu_s=0.4,
            lambda_s=2.0e6,
        )
