# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Generate developed-motion states of the shared-space FSI2 solver for benchmarks.

Runs the reference ``full/direct`` solver and saves restart states every
``--every`` time units, e.g.::

    conda run --no-capture-output -n xfsi_solver python -m \\
        xfsi_solver.scripts.fsi2_harmonic_diffmesh_checkpoints \\
        --mesh data/meshes/fsi2/mesh_sec_coarse.xdmf --dt 0.0025 --T 6 --every 0.5 \\
        --out output/checkpoints/mesh_sec_coarse_dt0.0025
"""

import argparse
from pathlib import Path

from xfsi_solver.solvers.fsi2_harmonic_diffmesh import SolverConfig, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", default="data/meshes/fsi2/mesh_sec_coarse.xdmf")
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--T", type=float, default=6.0)
    parser.add_argument("--every", type=float, default=0.5)
    parser.add_argument("--out", default="output/checkpoints/mesh_sec_coarse_dt0.0025")
    parser.add_argument("--restart", default=None, help="state file to continue from")
    args = parser.parse_args()

    out = Path(args.out)
    solve(
        mesh_path=args.mesh, T=args.T, dt_val=args.dt,
        output_path=str(out / "pv" / "u_v.bp") if args.restart is None else str(out / "pv" / "u_v_restart.bp"),
        output_path_p=str(out / "pv" / "p.bp") if args.restart is None else str(out / "pv" / "p_restart.bp"),
        qoi_path=str(out / ("qoi.txt" if args.restart is None else f"qoi_from_{Path(args.restart).stem}.txt")),
        config=SolverConfig(snes_monitor=False),
        checkpoint_dir=out, checkpoint_every=args.every, initial_state=args.restart,
    )


if __name__ == "__main__":
    main()
