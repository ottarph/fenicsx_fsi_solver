import numpy as np
import pytest
from restart_helpers import (
    check_initial_state_rejects_T_not_after_it,
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
    check_start_from_initial_state,
    read_checkpoint_state,
)

from xfsi_solver.solvers import fsi2_biharmonic_mixed
from xfsi_solver.solvers.fsi2_biharmonic_mixed_twice_simplified import solve

# The checkpoints hold the displacement as one function u on the full mesh, as in fsi2_biharmonic_mixed;
# p and z live on the fluid submesh but are checkpointed on the full mesh
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1), "z": ("Lagrange", 2, (2,))}

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


def test_fsi2_biharmonic_mixed_twice_simplified_solve(output_dirs):
    dt_val = 0.0025
    qoi_path = output_dirs["qoi"] / "fsi2_biharmonic_mixed_twice_simplified.txt"
    solve(
        mesh_path=MESH,
        # Run enough steps to save at least 2 VTX snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_biharmonic_mixed_twice_simplified.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_biharmonic_mixed_twice_simplified_p.bp"),
        qoi_path=str(qoi_path),
        vtx_save_every=4,
    )

    # The initial state and six time steps, with t, drag, lift and the tip displacement.
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (7, 5)
    # The inflow drives the fluid in the x-direction, so the drag grows, and the structure moves.
    assert 0.0 < rows[1, 1] < rows[-1, 1]
    assert np.abs(rows[-1, 3]) > 0.0
    assert (output_dirs["pv"] / "fsi2_biharmonic_mixed_twice_simplified.bp").exists()


@pytest.mark.parametrize("ranks", [(1, 1), (2, 3)], ids=["serial", "2_to_3_ranks"])
def test_fsi2_biharmonic_mixed_twice_simplified_restart_reproduces_continuous_run(output_dirs, tmp_path, ranks):
    # The restarted run assembles the Jacobian at its first Newton iteration, where the continuous run
    # reuses the one from the time step before, so the two agree within the Newton tolerance.
    check_restart_reproduces_continuous_run(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        first_ranks=ranks[0],
        restart_ranks=ranks[1],
    )


def test_fsi2_biharmonic_mixed_twice_simplified_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, MESH, output_dirs, tmp_path)


def test_fsi2_biharmonic_mixed_twice_simplified_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, MESH, output_dirs, tmp_path)


def test_fsi2_biharmonic_mixed_twice_simplified_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(solve, MESH, ELEMENTS, output_dirs, tmp_path, rtol=1e-6, qoi_rtol=1e-6)


def test_fsi2_biharmonic_mixed_twice_simplified_initial_state_rejects_T_not_after_it(output_dirs, tmp_path):
    check_initial_state_rejects_T_not_after_it(solve, MESH, output_dirs, tmp_path)


def test_fsi2_biharmonic_mixed_twice_simplified_checkpoints_interchangeable_with_mixed(output_dirs, tmp_path):
    # The two solvers compute the same iterates, and their checkpoints have the same layout, so their
    # states agree, and each continues from the other's checkpoint as from its own.
    dt_val = 0.0025

    def run(module_solve, name, T, **kwargs):
        module_solve(
            mesh_path=MESH,
            T=T,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
            checkpoint_dir=tmp_path / name,
            checkpoint_every=4,
            **kwargs,
        )

    mixed = fsi2_biharmonic_mixed.solve
    run(solve, "twice_simplified", 4 * dt_val)
    run(mixed, "mixed", 4 * dt_val, direct_k_c_solve=True)
    _, _, state_twice_simplified = read_checkpoint_state(MESH, tmp_path / "twice_simplified", dt_val, ELEMENTS)
    _, _, state_mixed = read_checkpoint_state(MESH, tmp_path / "mixed", dt_val, ELEMENTS)
    for name in ELEMENTS:
        values = state_mixed[name].x.array
        np.testing.assert_allclose(
            state_twice_simplified[name].x.array, values, rtol=0, atol=1e-6 * np.max(np.abs(values)), err_msg=name
        )

    run(solve, "twice_simplified_from_twice_simplified", 8 * dt_val, initial_state_dir=tmp_path / "twice_simplified")
    run(solve, "twice_simplified_from_mixed", 8 * dt_val, initial_state_dir=tmp_path / "mixed")
    run(
        mixed,
        "mixed_from_twice_simplified",
        8 * dt_val,
        initial_state_dir=tmp_path / "twice_simplified",
        direct_k_c_solve=True,
    )
    run(mixed, "mixed_from_mixed", 8 * dt_val, initial_state_dir=tmp_path / "mixed", direct_k_c_solve=True)
    reference = np.loadtxt(output_dirs["qoi"] / "mixed_from_mixed.txt", ndmin=2)
    for name in (
        "twice_simplified_from_twice_simplified",
        "twice_simplified_from_mixed",
        "mixed_from_twice_simplified",
    ):
        rows = np.loadtxt(output_dirs["qoi"] / f"{name}.txt", ndmin=2)
        assert rows.shape == reference.shape
        for column, values in enumerate(reference.T):
            np.testing.assert_allclose(
                rows[:, column], values, rtol=0, atol=1e-6 * np.max(np.abs(values)), err_msg=f"{name}, column {column}"
            )
