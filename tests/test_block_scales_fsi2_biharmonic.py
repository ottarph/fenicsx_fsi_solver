import numpy as np

from xfsi_solver.scripts.block_scales_fsi2_biharmonic import block_scales_hook
from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import NEST_GROUPS, solve

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


def test_block_scales_hook(output_dirs):
    records = []
    dt_val = 0.0025
    name = "block_scales"
    solve(
        mesh_path=MESH,
        T=2 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        jacobian_callback=block_scales_hook(records),
    )
    assert len(records) >= 2
    record = records[0]
    assert record["iteration"] == 0
    for group in NEST_GROUPS:
        for kind in ("state", "residual"):
            norms = record[kind][group]
            assert norms["n"] > 0
            assert np.isfinite(norms["l2"]) and 0.0 <= norms["max"] <= norms["l2"]
    # The solid stress and the solid kinematic relation are in every Jacobian, with entries.
    for block in ("v,u_s", "u_s,u_s", "z_f,u_f", "u_f,z_f"):
        stats = record["jacobian"][block]
        assert stats["nnz"] > 0 and 0.0 < stats["geomean"] <= stats["max"]
