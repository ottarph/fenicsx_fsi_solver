import numpy as np

from xfsi_solver.solvers import fsi2_biharmonic_mixed, fsi2_biharmonic_mixed_intermediate_simplified

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


def test_fsi2_biharmonic_mixed_intermediate_simplified_matches_mixed(output_dirs):
    # With exact solves in the u_S and m splits, the Python PCs of the simplified solver give the
    # same Newton corrections as the rows of u in fsi2_biharmonic_mixed, so the two solvers agree
    # to round-off, which is below the precision of the QoI file.
    dt_val = 0.0025
    rows = {}
    for module in (fsi2_biharmonic_mixed, fsi2_biharmonic_mixed_intermediate_simplified):
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

    simplified = rows["fsi2_biharmonic_mixed_intermediate_simplified"]
    # The initial state and six time steps, with t, drag, lift and the tip displacement.
    assert simplified.shape == (7, 5)
    # The inflow drives the fluid in the x-direction, so the drag grows, and the structure moves.
    assert 0.0 < simplified[1, 1] < simplified[-1, 1]
    assert np.abs(simplified[-1, 3]) > 0.0
    np.testing.assert_allclose(simplified, rows["fsi2_biharmonic_mixed"], rtol=1e-6, atol=1e-12)
    assert (output_dirs["pv"] / "fsi2_biharmonic_mixed_intermediate_simplified.bp").exists()


def test_fsi2_biharmonic_mixed_intermediate_simplified_quadrature_cap_changes_the_qois_slightly(output_dirs):
    dt_val = 0.0025

    def run(label, max_quadrature_degree):
        qoi_path = output_dirs["qoi"] / f"{label}.txt"
        fsi2_biharmonic_mixed_intermediate_simplified.solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{label}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{label}_p.bp"),
            qoi_path=str(qoi_path),
            direct_k_c_solve=True,
            max_quadrature_degree=max_quadrature_degree,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    exact = run("no_cap", None)
    capped = run("cap_4", 4)
    assert exact.shape == capped.shape == (3, 5)
    # The cap makes the integration of the nonlinear terms inexact, so the QoIs are not identical, but
    # they stay close. On this mesh the rows differ by up to 3e-5 relative to the uncapped values, so
    # rtol = 1e-3 leaves a margin of more than 30 while still failing for a wrong residual. The initial
    # row of zeros is the same in both.
    np.testing.assert_allclose(capped, exact, rtol=1e-3, atol=0)
    assert not np.array_equal(capped, exact)
