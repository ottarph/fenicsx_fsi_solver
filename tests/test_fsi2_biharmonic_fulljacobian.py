import pytest
from restart_helpers import (
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
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
        # save_every=4 in the solver, so run enough steps to save at least 2 snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_biharmonic_fulljacobian.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_biharmonic_fulljacobian_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_biharmonic_fulljacobian.txt"),
        mesh_motion_scaling=mesh_motion_scaling,
    )

    assert (output_dirs["qoi"] / "fsi2_biharmonic_fulljacobian.txt").exists()


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
