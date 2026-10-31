# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""One run of the sweep over the Jacobian reuse parameter ``gamma_reassemble`` of ``fsi2_simplified``
or ``threeDfsi_simplified``.

With ``--solver fsi2_simplified``, the default, runs ``fsi2_simplified.solve`` on the FSI2 benchmark
with MUMPS LU; with ``--solver threeDfsi_simplified``, ``threeDfsi_simplified.solve`` on the 3d
benchmark with ``MUMPS_ON_SOLID`` (FGMRES, with MUMPS LU on the solid velocity dofs), the quadrature
cap also on the mesh motion, and by default the medium mesh with dt = 0.004. Both with the
displacement prediction and the given ``gamma_reassemble``. It writes into ``--run-dir`` the QoIs,
the Newton log (one row per Newton iterate, with whether its Jacobian was assembled and its wall
time), checkpoints, and ``stats.json``: the statistics ``solve`` returns, the number of ranks, and
the PETSc event times (maximum over the ranks) of the assembly and factorization of the Jacobian and
of the other parts of an iteration. Run it under MPI, one run per directory:

    mpirun -n 2 python -u -m xfsi_solver.scripts.jacobian_reuse_sweep --gamma 0.2 --run-dir output/sweep/g0.2

``--gamma none`` reassembles the Jacobian in every Newton iteration. The defaults of ``--mesh``,
``--T`` and ``--dt`` depend on the solver.
"""

import argparse
import json
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.solvers import fsi2_simplified, threeDfsi_simplified

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
    "PCApply",
    "MatSOR",
]

# Per solver: its solve, the defaults of --mesh, --T and --dt, and the arguments fixed for the sweep
SOLVERS = {
    "fsi2_simplified": (
        fsi2_simplified.solve,
        {"mesh": "data/meshes/fsi2/mesh_sec.xdmf", "T": 15.0, "dt": 0.0025},
        {"solver_options": fsi2_simplified.DEFAULT_DIRECT_SOLVER_OPTIONS},
    ),
    "threeDfsi_simplified": (
        threeDfsi_simplified.solve,
        {"mesh": "data/meshes/fsi3d/mesh_sec_medium.xdmf", "T": 10.352, "dt": 0.004},
        {"solver_options": threeDfsi_simplified.MUMPS_ON_SOLID, "requires_v_sub_IS": True},
    ),
}


def parse_gamma(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gamma", type=parse_gamma, required=True, help="gamma_reassemble, or 'none'")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--solver", choices=list(SOLVERS), default="fsi2_simplified")
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--T", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--max-quadrature-degree", type=int, default=6, help="also caps the 3d mesh motion")
    parser.add_argument("--checkpoint-every", type=int, default=400)
    parser.add_argument("--initial-state-dir", default=None, help="start from the latest checkpoint in this directory")
    args = parser.parse_args()
    solve, defaults, solver_kwargs = SOLVERS[args.solver]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.solver == "threeDfsi_simplified":
        solver_kwargs = {**solver_kwargs, "mesh_motion_max_quadrature_degree": args.max_quadrature_degree}

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
        predict_displacements=True,
        **solver_kwargs,
    )

    events = {}
    for name in EVENTS:
        info = PETSc.Log.Event(name).getPerfInfo()
        events[name] = {"count": int(info["count"]), "time": comm.allreduce(info["time"], op=MPI.MAX)}

    if comm.rank == 0:
        record = {
            "solver": args.solver,
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
