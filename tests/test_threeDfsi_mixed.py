import numpy as np
import pytest

from xfsi_solver.solvers.threeDfsi_mixed import solve


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "iterative"])
def test_threeDfsi_mixed_solve(output_dirs, direct):
    dt_val = 0.004
    qoi_path = output_dirs["qoi"] / "threeDfsi_mixed.txt"
    solve(
        mesh_path="data/meshes/fsi3d/mesh_sec_coarse.xdmf",
        T=2 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "threeDfsi_mixed.bp"),
        output_path_p=str(output_dirs["pv"] / "threeDfsi_mixed_p.bp"),
        qoi_path=str(qoi_path),
        direct_k_c_solve=direct,
    )

    # The initial state and two time steps, with t, drag, lift and the displacement of B.
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (3, 6)
    # The inflow drives the fluid in the x-direction, so the drag grows.
    assert 0.0 < rows[1, 1] < rows[2, 1]
