import pytest

from xfsi_solver.solvers.fsi2_harmonic import solve


@pytest.mark.parametrize(
    "mesh_path",
    ["data/meshes/fsi2/mesh_coarse.xdmf", "data/meshes/fsi2/mesh_quad_coarse.xdmf"],
    ids=["tri", "quad"],
)
def test_fsi2_harmonic_solve(output_dirs, mesh_path):
    dt_val = 0.0025
    solve(
        mesh_path=mesh_path,
        # save_every=4 in the solver, so run enough steps to save at least 2 snapshots
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_harm.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harm_p.bp"),
        disp_path=str(output_dirs["qoi"] / "fsi2_harm_Adisp.txt"),
    )

    assert (output_dirs["qoi"] / "fsi2_harm_Adisp.txt").exists()
