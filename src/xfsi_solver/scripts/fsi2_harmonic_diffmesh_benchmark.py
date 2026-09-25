# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Benchmark the linear/nonlinear solver modes of the shared-space FSI2 solver.

Every mode runs the same time steps from the same state, in its own process
(so that the reported peak memory belongs to that mode), and is compared
with ``full/direct``::

    conda run --no-capture-output -n xfsi_solver python -m \\
        xfsi_solver.scripts.fsi2_harmonic_diffmesh_benchmark \\
        --mesh data/meshes/fsi2/mesh_sec_coarse.xdmf --dt 0.0025 --steps 20 \\
        --state output/checkpoints/mesh_sec_coarse_dt0.0025/state_t6.0000.npz \\
        --out output/benchmarks/coarse_t6_dt0.0025

``--state`` is a file written by
:mod:`xfsi_solver.scripts.fsi2_harmonic_diffmesh_checkpoints` (states are
per number of ranks), or omitted to start at t = 0. ``--np`` runs every mode
with ``mpiexec -n NP``. Results are written to ``OUT/results.json`` and
``OUT/results.md``.

``--solver stiffened_elastic`` benchmarks
:mod:`xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh` instead (states
from its ``--checkpoint-every``, mesh-law parameters ``--mesh-*``), with
accepted-time steps ``t0 + n dt``, n = 1..steps. The modes
``no_ale/fieldsplit(u lu)``, ``(vp lu)`` and ``(u+vp lu)`` replace the
displacement and/or momentum-pressure subsolvers of the production
fieldsplit by LU, to isolate the effect of the approximations.
"""

import argparse
import json
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np

MODES = {
    "full/direct": dict(jacobian_mode="full", linear_solver="direct"),
    "no_ale/direct": dict(jacobian_mode="no_ale", linear_solver="direct"),
    "full/fieldsplit(no_ale P)": dict(jacobian_mode="full", preconditioner_mode="no_ale", linear_solver="fieldsplit"),
    "no_ale/fieldsplit": dict(jacobian_mode="no_ale", linear_solver="fieldsplit"),
}
# diagnostic variants of the production fieldsplit (not run by default)
FIELDSPLIT_VARIANTS = {
    "no_ale/fieldsplit(u lu)": dict(displacement="lu"),
    "no_ale/fieldsplit(vp lu)": dict(momentum="lu"),
    "no_ale/fieldsplit(u+vp lu)": dict(displacement="lu", momentum="lu"),
    "no_ale/fieldsplit(u gamg)": dict(displacement_fluid="gamg"),
    "no_ale/fieldsplit(u amg)": dict(displacement_fluid="amg"),
}
ALL_MODES = MODES | {name: dict(jacobian_mode="no_ale", linear_solver="fieldsplit") for name in FIELDSPLIT_VARIANTS}


def run_mode(args):
    """Run one mode (in this process, possibly under MPI) and write its statistics."""
    from mpi4py import MPI

    from xfsi_solver.solvers.fsi2_harmonic_diffmesh import SolverConfig, save_state, solve, state_path
    from xfsi_solver.solvers.fsi2_harmonic_diffmesh_fieldsplit import FieldSplitConfig

    out = Path(args.out) / _dirname(args.mode)
    t0 = float(np.load(state_path(args.state))["t_next"]) if args.state else 0.0
    config = SolverConfig(**ALL_MODES[args.mode], snes_atol=args.snes_atol, snes_rtol=args.snes_rtol,
                          ksp_rtol=args.ksp_rtol, snes_monitor=False,
                          fieldsplit=FieldSplitConfig(**FIELDSPLIT_VARIANTS.get(args.mode, {})))
    if args.solver == "harmonic":
        result = solve(args.mesh, T=t0 + (args.steps - 0.5) * args.dt, dt_val=args.dt,
                       output_path=str(out / "uv.bp"), output_path_p=str(out / "p.bp"),
                       qoi_path=str(out / "qoi.txt"), config=config, initial_state=args.state)
    else:
        from xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh import MeshMotionConfig
        from xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh import solve as elastic_solve

        mesh_config = MeshMotionConfig(mesh_stiffening_exponent=args.mesh_stiffening_exponent,
                                       mesh_poisson_ratio=args.mesh_poisson_ratio)
        result = elastic_solve(args.mesh, T=t0 + args.steps * args.dt, dt_val=args.dt, output_dir=out,
                               config=config, mesh_config=mesh_config, initial_state=args.state,
                               output_every=args.steps + 1)
    save_state(result.problem, 0.0, out / "final_state.npz")

    peak_rss = MPI.COMM_WORLD.allreduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, op=MPI.MAX)
    steps = result.steps
    solves = [s for step in steps for s in step.linear_solves]
    timings = {}
    for step in steps:
        for key, value in step.timings.items():
            timings[key] = timings.get(key, 0.0) + value
    pc_stats = {}
    for step in steps:
        for name, stats in step.preconditioner_statistics.items():
            for key, value in stats.items():
                pc_stats.setdefault(name, {}).setdefault(key, 0)
                pc_stats[name][key] += value
    stats = {
        "mode": args.mode,
        "ranks": MPI.COMM_WORLD.size,
        "t0": t0,
        "steps": len(steps),
        "newton_iterations": [s.snes_iterations for s in steps],
        "final_residuals": [float(s.residual_history[-1]) for s in steps],
        "final_field_residuals": [s.field_residuals[-1].tolist() for s in steps],
        "linear_iterations": [d["iterations"] for d in solves],
        "true_relative_residuals": [d["true_relative_residual"] for d in solves],
        "step_times": [s.time for s in steps],
        "timings": timings,
        "preconditioner_statistics": pc_stats,
        "min_J_fluid": min((s.diagnostics["geometry"]["fluid"]["J_min"] for s in steps if s.diagnostics),
                           default=None),
        "peak_rss_mb_max_rank": peak_rss / 1024.0,
        "qoi": [[s.t, s.drag, s.lift, *s.tip_displacement.tolist()] for s in steps],
    }
    if MPI.COMM_WORLD.rank == 0:
        (out / "stats.json").write_text(json.dumps(stats, indent=1))


def summarize(args, modes):
    out = Path(args.out)
    results = {mode: json.loads((out / _dirname(mode) / "stats.json").read_text()) for mode in modes}
    reference = results.get("full/direct")
    ranks = next(iter(results.values()))["ranks"]
    rows = []
    for mode, r in results.items():
        row = {
            "mode": mode,
            "newton/step": float(np.mean(r["newton_iterations"])),
            "newton max": int(np.max(r["newton_iterations"])),
            "krylov/solve": float(np.mean(r["linear_iterations"])) if r["linear_iterations"] else 0.0,
            "krylov max": int(np.max(r["linear_iterations"])) if r["linear_iterations"] else 0,
            "s/step": float(np.mean(r["step_times"])),
            "s/step (excl. first)": float(np.mean(r["step_times"][1:])) if len(r["step_times"]) > 1 else float("nan"),
            "peak RSS MB": r["peak_rss_mb_max_rank"],
        }
        row |= {f"t_{k}": v / r["steps"] for k, v in r["timings"].items()}
        if reference is not None:
            q, q_ref = np.array(r["qoi"]), np.array(reference["qoi"])
            n = min(len(q), len(q_ref))
            for j, name in ((1, "drag"), (2, "lift"), (4, "A_y")):
                scale = max(np.max(np.abs(q_ref[:n, j])), 1e-12)
                row[f"max rel err {name}"] = float(np.max(np.abs(q[:n, j] - q_ref[:n, j])) / scale)
            errors = _field_errors(out / _dirname(mode), out / _dirname("full/direct"), ranks)
            row |= {f"rel err {k}": v for k, v in errors.items()}
        rows.append(row)

    (out / "results.json").write_text(json.dumps({"args": vars(args), "rows": rows, "raw": results}, indent=1))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(k)) for k in keys) + " |")
    header = (f"solver {args.solver}"
              + (f" (chi {args.mesh_stiffening_exponent}, nu {args.mesh_poisson_ratio})"
                 if args.solver != "harmonic" else "")
              + f", mesh `{args.mesh}`, dt {args.dt}, {args.steps} steps from "
              f"{'t = 0' if not args.state else '`' + args.state + '`'}, {ranks} rank(s), "
              f"snes_atol {args.snes_atol}, ksp_rtol {args.ksp_rtol}\n\n")
    (out / "results.md").write_text(header + "\n".join(lines) + "\n")
    print(header + "\n".join(lines))


def _field_errors(path, ref_path, ranks):
    diff = {"u": 0.0, "v": 0.0, "p": 0.0}
    norm = {"u": 0.0, "v": 0.0, "p": 0.0}
    for rank in range(ranks):
        suffix = "" if ranks == 1 else f"_rank{rank}of{ranks}"
        a = np.load(path / f"final_state{suffix}.npz")
        b = np.load(ref_path / f"final_state{suffix}.npz")
        for k in diff:
            diff[k] += float(np.sum((a[k] - b[k]) ** 2))
            norm[k] += float(np.sum(b[k] ** 2))
    # ghost entries are counted on every rank that holds them; the relative error is unaffected in serial
    return {k: np.sqrt(diff[k] / max(norm[k], 1e-300)) for k in diff}


def _dirname(mode):
    return mode.replace("/", "_").replace("(", "").replace(")", "").replace(" ", "_")


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", default="data/meshes/fsi2/mesh_sec_coarse.xdmf")
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--state", default=None)
    parser.add_argument("--snes-atol", type=float, default=1e-9)
    parser.add_argument("--snes-rtol", type=float, default=1e-12)
    parser.add_argument("--ksp-rtol", type=float, default=1e-6)
    parser.add_argument("--np", type=int, default=1)
    parser.add_argument("--modes", default=",".join(MODES), help=f"comma-separated, of {list(ALL_MODES)}")
    parser.add_argument("--solver", choices=("harmonic", "stiffened_elastic"), default="harmonic")
    parser.add_argument("--mesh-stiffening-exponent", type=float, default=1.25)
    parser.add_argument("--mesh-poisson-ratio", type=float, default=0.45)
    parser.add_argument("--out", default="output/benchmarks/default")
    parser.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.mode is not None:
        run_mode(args)
        return

    modes = args.modes.split(",")
    for mode in modes:
        if mode not in ALL_MODES:
            raise ValueError(f"Unknown mode {mode!r}, expected one of {list(ALL_MODES)}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for mode in modes:
        cmd = [sys.executable, "-m", "xfsi_solver.scripts.fsi2_harmonic_diffmesh_benchmark", *sys.argv[1:],
               "--mode", mode]
        if args.np > 1:
            cmd = ["mpiexec", "-n", str(args.np), *cmd]
        print(f"=== {mode}", flush=True)
        log = Path(args.out) / f"{_dirname(mode)}.log"
        with open(log, "w") as f:
            completed = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            print(f"mode {mode} failed, see {log}", flush=True)
            modes = [m for m in modes if m != mode]
    if modes:
        summarize(args, modes)


if __name__ == "__main__":
    main()
