import pytest
from restart_helpers import (
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
)

from xfsi_solver.solvers.fsi2_harmonic_lagrange import solve

# All fields live on the fluid or solid submesh but are checkpointed on the full mesh. The interface
# Lagrange multipliers are not checkpointed (see the solver), and restarting them from zero changes u_f
# by ~1e-6 (as much as tightening the Newton tolerance does) and the other fields by < 1e-9 relative
VECTOR = ("Lagrange", 2, (2,))
ELEMENTS = {"u_f": VECTOR, "v_f": VECTOR, "u_s": VECTOR, "v_s": VECTOR, "p": ("Lagrange", 1)}
RESTART_TOLERANCES = dict(rtol=1e-8, field_rtol={"u_f": 1e-5})


def test_fsi2_harmonic_lagrange_solve(output_dirs):
    dt_val = 0.0025
    solve(
        mesh_path="data/meshes/fsi2/mesh_sec_coarse.xdmf",
        # save_every=8 in the solver, so run enough steps to save at least 2 snapshots
        T=10 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_harm_lg.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harm_lg_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "fsi2_harm_lg_qoi.txt"),
    )

    assert (output_dirs["qoi"] / "fsi2_harm_lg_qoi.txt").exists()


def test_fsi2_harmonic_lagrange_restart_reproduces_continuous_run(output_dirs, tmp_path):
    check_restart_reproduces_continuous_run(
        solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", ELEMENTS, output_dirs, tmp_path, **RESTART_TOLERANCES
    )


@pytest.mark.parametrize("first_ranks,restart_ranks", [(2, 1), (1, 2)], ids=["2to1", "1to2"])
def test_fsi2_harmonic_lagrange_restart_on_different_number_of_ranks(output_dirs, tmp_path, first_ranks, restart_ranks):
    # Not bitwise identical across partitions; QoIs are printed to 6 digits, so the last one may flip.
    # A different partition perturbs the loosely resolved u_f more (up to ~3e-6 relative seen), and the
    # interface gap diagnostics are at the solver tolerance (~1e-8), so only their leading digits agree
    # (up to ~5e-3 relative seen); the physical fields and QoIs keep their tolerances
    check_restart_reproduces_continuous_run(
        solve,
        "data/meshes/fsi2/mesh_sec_coarse.xdmf",
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=RESTART_TOLERANCES["rtol"],
        field_rtol={"u_f": 1e-4},
        qoi_rtol=1e-5,
        qoi_column_rtol={"interface_u_gap": 1e-2, "interface_v_gap": 1e-2},
        first_ranks=first_ranks,
        restart_ranks=restart_ranks,
    )


def test_fsi2_harmonic_lagrange_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)


def test_fsi2_harmonic_lagrange_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, "data/meshes/fsi2/mesh_sec_coarse.xdmf", output_dirs, tmp_path)
