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
        # save_every=10 in the solver, so run enough steps to save at least 2 snapshots
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
    )

    assert (output_dirs["qoi"] / f"dfg_2d_3_drag_{cell_type}.txt").exists()
    assert (output_dirs["figures"] / f"dfg_2d_3_drag_coeff_{cell_type}.png").exists()
