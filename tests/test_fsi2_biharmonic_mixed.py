import numpy as np
import pytest
from restart_helpers import (
    check_restart_rejects_different_dt,
    check_restart_rejects_T_not_after_checkpoint,
    check_restart_reproduces_continuous_run,
)

from xfsi_solver.solvers.fsi2_biharmonic_mixed import solve

# p and z live on the fluid submesh but are checkpointed on the full mesh
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1), "z": ("Lagrange", 2, (2,))}

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "iterative"])
def test_fsi2_biharmonic_mixed_solve(output_dirs, direct):
    dt_val = 0.0025
    qoi_path = output_dirs["qoi"] / "fsi2_biharmonic_mixed.txt"
    solve(
        mesh_path=MESH,
        # Run enough steps to save at least 2 VTX snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_biharmonic_mixed.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_biharmonic_mixed_p.bp"),
        qoi_path=str(qoi_path),
        vtx_save_every=4,
        direct_k_c_solve=direct,
    )

    # The initial state and six time steps, with t, drag, lift and the tip displacement.
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (7, 5)
    # The inflow drives the fluid in the x-direction, so the drag grows.
    assert 0.0 < rows[1, 1] < rows[-1, 1]
    assert (output_dirs["pv"] / "fsi2_biharmonic_mixed.bp").exists()


def test_fsi2_biharmonic_mixed_restart_reproduces_continuous_run(output_dirs, tmp_path):
    # The restarted run assembles the approximate Jacobian at its first Newton iteration, where the
    # continuous run reuses the one from the time step before, so the two converge to states that
    # differ within the Newton tolerance (about 1e-8 relative).
    check_restart_reproduces_continuous_run(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        solve_kwargs={"direct_k_c_solve": True},
    )


def test_fsi2_biharmonic_mixed_restart_rejects_different_dt(output_dirs, tmp_path):
    check_restart_rejects_different_dt(solve, MESH, output_dirs, tmp_path)


def test_fsi2_biharmonic_mixed_restart_rejects_T_not_after_checkpoint(output_dirs, tmp_path):
    check_restart_rejects_T_not_after_checkpoint(solve, MESH, output_dirs, tmp_path)
