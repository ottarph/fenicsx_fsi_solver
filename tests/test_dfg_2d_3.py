import numpy as np
import pytest

from xfsi_solver.solvers.dfg_2d_3 import solve


@pytest.mark.parametrize(
    "cell_type,mesh_path",
    [
        ("tri", "data/meshes/dfg2d_alt/mesh_tri_coarse.xdmf"),
        ("quad", "data/meshes/dfg2d_alt/mesh_quad_coarse.xdmf"),
    ],
    ids=["tri", "quad"],
)
def test_dfg_2d_3_solve(output_dirs, cell_type, mesh_path):
    dt_val = 1 / 400
    solve(
        mesh_path=mesh_path,
        # Run enough steps to save at least 2 VTX snapshots
        T=21 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"dfg_2d_3_{cell_type}.bp"),
        output_path_p=str(output_dirs["pv"] / f"dfg_2d_3_p_{cell_type}.bp"),
        drag_path=str(output_dirs["qoi"] / f"dfg_2d_3_drag_{cell_type}.txt"),
        lift_path=str(output_dirs["qoi"] / f"dfg_2d_3_lift_{cell_type}.txt"),
        drag_plot_path=str(output_dirs["figures"] / f"dfg_2d_3_drag_{cell_type}.png"),
        lift_plot_path=str(output_dirs["figures"] / f"dfg_2d_3_lift_{cell_type}.png"),
        drag_coeff_plot_path=str(output_dirs["figures"] / f"dfg_2d_3_drag_coeff_{cell_type}.png"),
        lift_coeff_plot_path=str(output_dirs["figures"] / f"dfg_2d_3_lift_coeff_{cell_type}.png"),
        vtx_save_every=10,
    )

    assert (output_dirs["qoi"] / f"dfg_2d_3_drag_{cell_type}.txt").exists()
    assert (output_dirs["figures"] / f"dfg_2d_3_drag_coeff_{cell_type}.png").exists()
    assert (output_dirs["pv"] / f"dfg_2d_3_{cell_type}.bp").exists()


@pytest.mark.parametrize(
    "cell_type,mesh_path",
    [
        ("tri", "data/meshes/dfg2d_alt/mesh_tri_coarse.xdmf"),
        ("quad", "data/meshes/dfg2d_alt/mesh_quad_coarse.xdmf"),
    ],
    ids=["tri", "quad"],
)
def test_dfg_2d_3_quadrature_cap_changes_the_qois_slightly(output_dirs, cell_type, mesh_path):
    dt_val = 1 / 400

    def run(name, max_quadrature_degree):
        paths = {key: str(output_dirs["qoi"] / f"{name}_{key}.txt") for key in ("drag", "lift")}
        solve(
            mesh_path=mesh_path,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            drag_path=paths["drag"],
            lift_path=paths["lift"],
            drag_plot_path=str(output_dirs["figures"] / f"{name}_drag.png"),
            lift_plot_path=str(output_dirs["figures"] / f"{name}_lift.png"),
            drag_coeff_plot_path=str(output_dirs["figures"] / f"{name}_drag_coeff.png"),
            lift_coeff_plot_path=str(output_dirs["figures"] / f"{name}_lift_coeff.png"),
            max_quadrature_degree=max_quadrature_degree,
        )
        return np.hstack([np.loadtxt(paths["drag"], ndmin=2), np.loadtxt(paths["lift"], ndmin=2)[:, 1:]])

    exact = run(f"no_cap_{cell_type}", None)
    capped = run(f"cap_4_{cell_type}", 4)
    assert exact.shape == capped.shape == (2, 3)
    # The cap of 4 is below the degree 5 that UFL estimates for the convection term with P2 velocity. It
    # makes the integration of the nonlinear terms inexact, so the drag and lift are not identical, but
    # they stay close. The files hold only 5 significant digits (%.4e), and on these meshes the entries
    # differ by up to 6e-5 relative to the uncapped values, so rtol = 1e-3 leaves a margin of more than 15
    # while still failing for a wrong residual. The time column and the initial state are the same in both.
    np.testing.assert_allclose(capped, exact, rtol=1e-3, atol=0)
    assert not np.array_equal(capped, exact)


def test_dfg_2d_3_solve_rejects_both_theta_shifts(output_dirs):
    with pytest.raises(ValueError):
        solve(
            mesh_path="nonexistent_mesh.xdmf",
            T=1 / 400,
            dt_val=1 / 400,
            output_path=str(output_dirs["pv"] / "dfg_2d_3_theta.bp"),
            output_path_p=str(output_dirs["pv"] / "dfg_2d_3_p_theta.bp"),
            drag_path=str(output_dirs["qoi"] / "dfg_2d_3_drag_theta.txt"),
            lift_path=str(output_dirs["qoi"] / "dfg_2d_3_lift_theta.txt"),
            drag_plot_path=str(output_dirs["figures"] / "dfg_2d_3_drag_theta.png"),
            lift_plot_path=str(output_dirs["figures"] / "dfg_2d_3_lift_theta.png"),
            drag_coeff_plot_path=str(output_dirs["figures"] / "dfg_2d_3_drag_coeff_theta.png"),
            lift_coeff_plot_path=str(output_dirs["figures"] / "dfg_2d_3_lift_coeff_theta.png"),
            absolute_theta_shift=0.01,
            relative_theta_shift=1.0,
        )
