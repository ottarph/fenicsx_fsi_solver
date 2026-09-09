from xfsi_solver.component_solvers.navier_stokes_ALE_fsi2_mm_bih import solve


def test_navier_stokes_ale_fsi2_mm_bih_solve(output_dirs):
    output_path = output_dirs["pv"] / "navier_stokes_ale_fsi2_mm_bih.bp"
    solve(
        mesh_path="data/meshes/fsi2/mesh_coarse.xdmf",
        dt_val=0.0025,
        bd_dset_path="data/fsi2_boundary/",
        output_path=str(output_path),
        max_steps=3,
    )

    assert output_path.exists()
