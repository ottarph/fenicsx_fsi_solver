import pytest
from restart_helpers import (
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
)

from xfsi_solver.solvers.fsi2_harmonic import solve

ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1)}


@pytest.mark.parametrize(
    "cell_type,mesh_path",
    [("tri", "data/meshes/fsi2/mesh_coarse.xdmf"), ("quad", "data/meshes/fsi2/mesh_quad_coarse.xdmf")],
    ids=["tri", "quad"],
)
def test_fsi2_harmonic_solve(output_dirs, cell_type, mesh_path):
    dt_val = 0.0025
    solve(
        mesh_path=mesh_path,
        # save_every=4 in the solver, so run enough steps to save at least 2 snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"fsi2_harm_{cell_type}.bp"),
        output_path_p=str(output_dirs["pv"] / f"fsi2_harm_p_{cell_type}.bp"),
        qoi_path=str(output_dirs["qoi"] / f"fsi2_harm_qoi_{cell_type}.txt"),
    )

    assert (output_dirs["qoi"] / f"fsi2_harm_qoi_{cell_type}.txt").exists()


def test_fsi2_harmonic_restart_reproduces_continuous_run(output_dirs, tmp_path):
    check_restart_reproduces_continuous_run(solve, "data/meshes/fsi2/mesh_coarse.xdmf", ELEMENTS, output_dirs, tmp_path)


@pytest.mark.parametrize("first_ranks,restart_ranks", [(2, 1), (1, 2)], ids=["2to1", "1to2"])
def test_fsi2_harmonic_restart_on_different_number_of_ranks(output_dirs, tmp_path, first_ranks, restart_ranks):
    # Not bitwise identical across partitions; QoIs are printed to 6 digits, so the last one may flip
    check_restart_reproduces_continuous_run(
        solve,
        "data/meshes/fsi2/mesh_coarse.xdmf",
        ELEMENTS,
        output_dirs,
        tmp_path,
        qoi_rtol=1e-5,
        first_ranks=first_ranks,
        restart_ranks=restart_ranks,
    )


def test_fsi2_harmonic_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, "data/meshes/fsi2/mesh_coarse.xdmf", output_dirs, tmp_path)


def test_fsi2_harmonic_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, "data/meshes/fsi2/mesh_coarse.xdmf", output_dirs, tmp_path)
