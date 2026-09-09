import pytest

from xfsi_solver.solvers.fsi2_harmonic import solve


@pytest.mark.parametrize(
    "cell_type,mesh_path",
    [("tri", "data/meshes/fsi2/mesh_coarse.xdmf"), ("quad", "data/meshes/fsi2/mesh_quad_coarse.xdmf")],
    ids=["tri", "quad"],
)
def test_fsi2_harmonic_solve(output_dirs, cell_type, mesh_path):
    dt_val = 0.0025
    solve(
        mesh_path=mesh_path,
        # save_every=4 in the solver, so run enough steps to save at least 2 snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"fsi2_harm_{cell_type}.bp"),
        output_path_p=str(output_dirs["pv"] / f"fsi2_harm_p_{cell_type}.bp"),
        disp_path=str(output_dirs["qoi"] / f"fsi2_harm_Adisp_{cell_type}.txt"),
    )

    assert (output_dirs["qoi"] / f"fsi2_harm_Adisp_{cell_type}.txt").exists()
