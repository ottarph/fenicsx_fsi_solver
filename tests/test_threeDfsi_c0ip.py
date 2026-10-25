import numpy as np
import pytest
from restart_helpers import check_start_from_initial_state

from xfsi_solver.solvers.threeDfsi_c0ip import solve

VECTOR = ("Lagrange", 2, (3,))
# p lives on the fluid submesh but is checkpointed on the full mesh
ELEMENTS = {"u": VECTOR, "v": VECTOR, "p": ("Lagrange", 1)}


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "iterative"])
def test_threeDfsi_c0ip_solve(output_dirs, direct):
    dt_val = 0.004
    qoi_path = output_dirs["qoi"] / "threeDfsi_c0ip.txt"
    solve(
        mesh_path="data/meshes/fsi3d/mesh_sec_coarse.xdmf",
        T=2 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "threeDfsi_c0ip.bp"),
        output_path_p=str(output_dirs["pv"] / "threeDfsi_c0ip_p.bp"),
        qoi_path=str(qoi_path),
        direct_k_c_solve=direct,
    )

    # The initial state and two time steps, with t, drag, lift and the displacement of B.
    rows = np.loadtxt(qoi_path, ndmin=2)
    assert rows.shape == (3, 6)
    # The inflow drives the fluid in the x-direction, so the drag grows.
    assert 0.0 < rows[1, 1] < rows[2, 1]


def test_threeDfsi_c0ip_k_c_options_replace_the_defaults(output_dirs):
    # One FGMRES iteration is not enough for K_c, so the run fails on the K_c solve, which shows
    # that the option reached it.
    dt_val = 0.004
    with pytest.raises(RuntimeError, match="solver_fieldsplit_q_c_ DIVERGED_(ITS|MAX_IT)"):
        solve(
            mesh_path="data/meshes/fsi3d/mesh_sec_coarse.xdmf",
            T=dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / "threeDfsi_c0ip_k_c_options.bp"),
            output_path_p=str(output_dirs["pv"] / "threeDfsi_c0ip_k_c_options_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "threeDfsi_c0ip_k_c_options.txt"),
            k_c_options={"ksp_max_it": 1},
        )


def test_threeDfsi_c0ip_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve,
        "data/meshes/fsi3d/mesh_sec_coarse.xdmf",
        ELEMENTS,
        output_dirs,
        tmp_path,
        dt_val=0.004,
        rtol=1e-6,
        qoi_rtol=1e-6,
        solve_kwargs={"direct_k_c_solve": True},
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"absolute_theta_shift": 0.01, "relative_theta_shift": 1.0}, "theta_shift"),
        ({"nu_s": 0.4, "lambda_s": 8.0e6}, "nu_s and lambda_s"),
    ],
    ids=["theta_shift", "nu_s_lambda_s"],
)
def test_threeDfsi_c0ip_rejects_ambiguous_options(output_dirs, kwargs, match):
    # Raised before the mesh is read
    with pytest.raises(ValueError, match=match):
        solve(
            mesh_path="nonexistent.xdmf",
            T=0.004,
            dt_val=0.004,
            output_path=str(output_dirs["pv"] / "ambiguous.bp"),
            output_path_p=str(output_dirs["pv"] / "ambiguous_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "ambiguous.txt"),
            **kwargs,
        )
