import numpy as np
import pytest
from restart_helpers import check_start_from_initial_state

from xfsi_solver.solvers.fsi2_biharmonic_mixed_incompressible import solve

VECTOR = ("Lagrange", 2, (2,))
# p and p_S live on the fluid and solid submeshes but are checkpointed on the full mesh
ELEMENTS = {"u": VECTOR, "v": VECTOR, "p": ("Lagrange", 1), "p_S": ("Lagrange", 1), "z": VECTOR}

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


def test_fsi2_biharmonic_mixed_incompressible_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        rtol=1e-6,
        qoi_rtol=1e-6,
        # The solid area change is round-off (about 1e-17) on these few steps, so only its size is compared
        qoi_column_rtol={"solid_area_change": 1.0},
        output_kwargs={"output_path_p_S": "_p_S.bp"},
        solve_kwargs={"direct_k_c_solve": True},
    )


def test_fsi2_biharmonic_mixed_incompressible_rejects_both_theta_shifts(output_dirs):
    # Raised before the mesh is read
    with pytest.raises(ValueError, match="theta_shift"):
        solve(
            mesh_path="data/meshes/does_not_exist.xdmf",
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "rejected.bp"),
            output_path_p=str(output_dirs["pv"] / "rejected_p.bp"),
            output_path_p_S=str(output_dirs["pv"] / "rejected_p_S.bp"),
            qoi_path=str(output_dirs["qoi"] / "rejected.txt"),
            absolute_theta_shift=0.01,
            relative_theta_shift=1.0,
        )


def test_fsi2_biharmonic_mixed_incompressible_quadrature_cap_changes_the_qois_slightly(output_dirs):
    dt_val = 0.0025

    def run(label, max_quadrature_degree):
        qoi_path = output_dirs["qoi"] / f"{label}.txt"
        solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{label}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{label}_p.bp"),
            output_path_p_S=str(output_dirs["pv"] / f"{label}_p_S.bp"),
            qoi_path=str(qoi_path),
            direct_k_c_solve=True,
            max_quadrature_degree=max_quadrature_degree,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    exact = run("no_cap", None)
    capped = run("cap_4", 4)
    assert exact.shape == capped.shape == (3, 8)
    # The cap makes the integration of the nonlinear terms inexact, so the QoIs are not identical, but
    # they stay close. On this mesh the rows differ by up to 7e-5 relative to the uncapped values (the
    # largest for det_F_max), so rtol = 1e-3 leaves a margin of more than 10 while still failing for a
    # wrong residual. The initial row of zeros is the same in both. The solid area change is round-off
    # (below 1e-17) on these two steps and differs by that much, so it is compared with an absolute
    # tolerance of 1e-15 instead.
    area_change = 5
    other = np.arange(8) != area_change
    np.testing.assert_allclose(capped[:, other], exact[:, other], rtol=1e-3, atol=0)
    np.testing.assert_allclose(capped[:, area_change], exact[:, area_change], rtol=0, atol=1e-15)
    assert not np.array_equal(capped, exact)
