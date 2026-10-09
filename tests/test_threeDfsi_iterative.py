import numpy as np
import pytest

from xfsi_solver.solvers.threeDfsi_iterative import solve


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "iterative"])
def test_threeDfsi_iterative_solve(output_dirs, direct):
    dt_val = 0.004
    qoi_path = output_dirs["qoi"] / "fsi3d_iterative_qoi.txt"
    solve(
        mesh_path="data/meshes/fsi3d/mesh_sec_coarse.xdmf",
        T=2 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi3d_iterative.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi3d_iterative_p.bp"),
        qoi_path=str(qoi_path),
        direct_k_c_solve=direct,
        block_preconditioned_k_c_solve=not direct,
        cahouet_chabard_schur_preconditioner=not direct,
    )

    # The initial state and two time steps, with t, drag, lift and the displacement of B.
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (3, 6)
    # The inflow drives the fluid in the x-direction, so the drag grows.
    assert 0.0 < rows[1, 1] < rows[2, 1]
