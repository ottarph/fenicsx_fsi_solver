import numpy as np

from xfsi_solver.solvers import fsi2_biharmonic_mixed, fsi2_biharmonic_mixed_simplified

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


def test_fsi2_biharmonic_mixed_simplified_matches_mixed(output_dirs):
    # With exact solves in the u_S and m splits, the Python PCs of the simplified solver give the
    # same Newton corrections as the rows of u in fsi2_biharmonic_mixed, so the two solvers agree
    # to round-off, which is below the precision of the QoI file.
    dt_val = 0.0025
    rows = {}
    for module in (fsi2_biharmonic_mixed, fsi2_biharmonic_mixed_simplified):
        name = module.__name__.rsplit(".", 1)[-1]
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        module.solve(
            mesh_path=MESH,
            T=6 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(qoi_path),
            vtx_save_every=4,
            direct_k_c_solve=True,
        )
        rows[name] = np.loadtxt(qoi_path, ndmin=2)

    simplified = rows["fsi2_biharmonic_mixed_simplified"]
    # The initial state and six time steps, with t, drag, lift and the tip displacement.
    assert simplified.shape == (7, 5)
    # The inflow drives the fluid in the x-direction, so the drag grows, and the structure moves.
    assert 0.0 < simplified[1, 1] < simplified[-1, 1]
    assert np.abs(simplified[-1, 3]) > 0.0
    np.testing.assert_allclose(simplified, rows["fsi2_biharmonic_mixed"], rtol=1e-6, atol=1e-12)
    assert (output_dirs["pv"] / "fsi2_biharmonic_mixed_simplified.bp").exists()
