from xfsi_solver.solvers.dfg_2d_3 import solve


def test_dfg_2d_3_solve(output_dirs):
    dt_val = 1 / 400
    solve(
        mesh_path="data/meshes/dfg2d_alt/mesh_quad.xdmf",
        T=3 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "dfg_2d_3.bp"),
        output_path_p=str(output_dirs["pv"] / "dfg_2d_3_p.bp"),
        drag_path=str(output_dirs["qoi"] / "dfg_2d_3_drag.txt"),
        lift_path=str(output_dirs["qoi"] / "dfg_2d_3_lift.txt"),
        drag_plot_path=str(output_dirs["figures"] / "dfg_2d_3_drag.png"),
        lift_plot_path=str(output_dirs["figures"] / "dfg_2d_3_lift.png"),
        drag_coeff_plot_path=str(output_dirs["figures"] / "dfg_2d_3_drag_coeff.png"),
        lift_coeff_plot_path=str(output_dirs["figures"] / "dfg_2d_3_lift_coeff.png"),
    )

    assert (output_dirs["qoi"] / "dfg_2d_3_drag.txt").exists()
    assert (output_dirs["figures"] / "dfg_2d_3_drag_coeff.png").exists()
