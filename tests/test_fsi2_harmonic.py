from fsi2_harmonic import solve


def test_fsi2_harmonic_solve(output_dirs):
    dt_val = 0.0025
    solve(
        mesh_path="data/meshes/fsi2/mesh.xdmf",
        T=3 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "fsi2_harm.bp"),
        output_path_p=str(output_dirs["pv"] / "fsi2_harm_p.bp"),
        disp_path=str(output_dirs["qoi"] / "fsi2_harm_Adisp.txt"),
    )

    assert (output_dirs["qoi"] / "fsi2_harm_Adisp.txt").exists()
