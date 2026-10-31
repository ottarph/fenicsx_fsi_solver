# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""One run of the sweep over the Jacobian reuse parameter ``gamma_reassemble`` of ``fsi2_simplified``.

Runs ``fsi2_simplified.solve`` on the FSI2 benchmark with MUMPS LU, the displacement prediction and
the given ``gamma_reassemble``, and writes into ``--run-dir`` the QoIs, the Newton log (one row per
Newton iterate, with whether its Jacobian was assembled and its wall time), checkpoints, and
``stats.json``: the statistics ``solve`` returns, the number of ranks, and the PETSc event times
(maximum over the ranks) of the assembly and factorization of the Jacobian and of the other parts
of an iteration. Run it under MPI, one run per directory:

    mpirun -n 2 python -u -m xfsi_solver.scripts.jacobian_reuse_sweep --gamma 0.2 --run-dir output/sweep/g0.2

``--gamma none`` reassembles the Jacobian in every Newton iteration.
"""

import argparse
import json
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.solvers.fsi2_simplified import DEFAULT_DIRECT_SOLVER_OPTIONS, solve

# The PETSc events of a Newton iteration: SNESJacobianEval assembles the Jacobian (and is also logged,
# with negligible time, when the Jacobian callback reuses it), PCSetUp and MatLUFactor* factorize it
# (and, once per run, the mesh motion), SNESFunctionEval assembles the residual, KSPSolve solves with
# the factorization, MatMatSolve solves the mesh motion.
EVENTS = [
    "SNESSolve",
    "SNESJacobianEval",
    "SNESFunctionEval",
    "KSPSolve",
    "PCSetUp",
    "MatLUFactorSym",
    "MatLUFactorNum",
    "MatSolve",
    "MatMatSolve",
]


def parse_gamma(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gamma", type=parse_gamma, required=True, help="gamma_reassemble, or 'none'")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mesh", default="data/meshes/fsi2/mesh_sec.xdmf")
    parser.add_argument("--T", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--max-quadrature-degree", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=400)
    parser.add_argument("--initial-state-dir", default=None, help="start from the latest checkpoint in this directory")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    run_dir = args.run_dir
    if comm.rank == 0:
        if (run_dir / "stats.json").exists():
            raise SystemExit(f"{run_dir} holds a finished run; choose another --run-dir")
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    comm.barrier()

    PETSc.Log.begin()
    stats = solve(
        mesh_path=args.mesh,
        T=args.T,
        dt_val=args.dt,
        output_path=str(run_dir / "u_v.bp"),
        output_path_p=str(run_dir / "p.bp"),
        qoi_path=str(run_dir / "qoi.txt"),
        newton_log_path=str(run_dir / "newton.txt"),
        checkpoint_dir=str(run_dir / "checkpoints"),
        checkpoint_every=args.checkpoint_every,
        initial_state_dir=args.initial_state_dir,
        gamma_reassemble=args.gamma,
        max_quadrature_degree=args.max_quadrature_degree,
        solver_options=DEFAULT_DIRECT_SOLVER_OPTIONS,
        predict_displacements=True,
    )

    events = {}
    for name in EVENTS:
        info = PETSc.Log.Event(name).getPerfInfo()
        events[name] = {"count": int(info["count"]), "time": comm.allreduce(info["time"], op=MPI.MAX)}

    if comm.rank == 0:
        record = {
            "gamma_reassemble": args.gamma,
            "num_ranks": comm.size,
            "mesh": args.mesh,
            "T": args.T,
            "dt": args.dt,
            "max_quadrature_degree": args.max_quadrature_degree,
            "initial_state_dir": args.initial_state_dir,
            **stats,
            "events": events,
        }
        (run_dir / "stats.json").write_text(json.dumps(record, indent=1))
        print(f"\nWrote {run_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
