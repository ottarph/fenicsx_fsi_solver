# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""FSI2 with the compressible three-term Mooney-Rivlin solid in place of St. Venant-Kirchhoff.

This is a modified FSI2 benchmark: the published SVK reference values are a context for its results, not
targets. The default coefficients are ``MR3_FSI2_MATCHED``, an artificial benchmark material with the initial
shear and bulk moduli of the FSI2 solid (C10 + C01 = mu_s / 2, D1 = 2 / (lambda_s + 2 mu_s / 3)) and the C11
of a tissue fit. ``--D1`` overrides the D1 taken from the SVK bulk modulus. Every output goes to its own
folder ``output/fsi2_mr3/<run-name>/``, which must not exist yet: the QoIs (``qoi.txt``), with the range of
det(F) of the solid and the fluid mesh in its last four columns, the Newton log (``newton.txt``), a summary of
the run (``summary.json``), and with ``--output-every`` the fields (``fields.vtkhdf``).

Run from the repository root, for example from rest on the coarse test mesh::

    python examples/fsi2_mooney_rivlin_3.py --run-name coarse_smoke --mesh data/meshes/fsi2/mesh_sec_coarse.xdmf \\
        --T 0.1

or with MPI on mesh_sec from the state of an earlier run, with D1 given explicitly::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -n 5 python -u examples/fsi2_mooney_rivlin_3.py \\
        --run-name from_t15 --initial-state-dir output/snapshots/fsi2_simplified/t15.0000 --T 16 \\
        --D1 8.571428571428571e-7
"""

import argparse
import json
from pathlib import Path

from mpi4py import MPI

from xfsi_solver.fsi.materials import MR3_FSI2_MATCHED
from xfsi_solver.solvers.fsi2_simplified import solve


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True, help="the output folder output/fsi2_mr3/<run-name>")
    parser.add_argument("--mesh", default="data/meshes/fsi2/mesh_sec.xdmf")
    parser.add_argument("--T", type=float, required=True, help="end time")
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--initial-state-dir", help="start from the latest checkpoint in this directory")
    parser.add_argument("--C10", type=float, default=MR3_FSI2_MATCHED["C10"], help="Pa")
    parser.add_argument("--C01", type=float, default=MR3_FSI2_MATCHED["C01"], help="Pa")
    parser.add_argument("--C11", type=float, default=MR3_FSI2_MATCHED["C11"], help="Pa")
    parser.add_argument("--D1", type=float, help="1/Pa; by default 2 / K with K = lambda_s + 2 mu_s / 3 of FSI2")
    parser.add_argument("--svk", action="store_true", help="use the default St. Venant-Kirchhoff solid instead")
    parser.add_argument(
        "--max-quadrature-degree", type=int, default=5, help="cap of the quadrature degree; 0 for no cap"
    )
    parser.add_argument(
        "--gamma-reassemble",
        type=float,
        help="reuse the Jacobian while Newton contracts by this factor; by default a fresh one every iteration",
    )
    parser.add_argument("--output-every", type=int, help="write the fields every this many steps")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    folder = Path("output/fsi2_mr3") / args.run_name
    exists = folder.exists() if comm.rank == 0 else None
    if comm.bcast(exists, root=0):
        raise FileExistsError(f"{folder} exists; give a new --run-name")
    comm.barrier()
    if comm.rank == 0:
        folder.mkdir(parents=True)

    if args.svk:
        material = {}
    else:
        coefficients = {"C10": args.C10, "C01": args.C01, "C11": args.C11}
        if args.D1 is not None:
            coefficients["D1"] = args.D1
        material = {"solid_material": "mooney_rivlin_3", "mooney_rivlin_3": coefficients}
    statistics = solve(
        mesh_path=args.mesh,
        T=args.T,
        dt_val=args.dt,
        output_path=str(folder / "fields.vtkhdf"),
        qoi_path=str(folder / "qoi.txt"),
        newton_log_path=str(folder / "newton.txt"),
        initial_state_dir=args.initial_state_dir,
        output_every=args.output_every,
        max_quadrature_degree=args.max_quadrature_degree or None,
        gamma_reassemble=args.gamma_reassemble,
        **material,
    )

    if comm.rank == 0:
        summary = {key: value for key, value in vars(args).items()} | {
            "num_ranks": comm.size,
            "elapsed_time": statistics["elapsed_time"],
            "num_steps": statistics["num_steps"],
            "num_newton_iterations": statistics["num_newton_iterations"],
            "max_iterations_per_step": max(statistics["iterations_per_step"], default=0),
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
