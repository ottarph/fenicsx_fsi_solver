# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Analysis of the gamma_reassemble runs of threeDfsi_simplified from a developed state
(notes/jacobian-reuse, jobs/jacobian_reuse_3d.sh).

Reads output/jacobian_reuse_3d/g<gamma>/: 88 steps from the threeDfsi_mixed_medium checkpoint at
t = 10 on the medium mesh, one run at a time on 28 ranks, so their wall times compare directly. For
each gamma: the Newton iterations, Jacobian assemblies and FGMRES iterations per step, the measured
wall time per step (without the first, which also sets up the mesh motion factorization), the mean
wall time of a Newton iteration that assembles the Jacobian (and sets up the preconditioner) and of
one that reuses it, and the QoIs at the end against those of gamma = 0.2.

Prints the tables and writes tables_3d.md. Run from the repository root:

    python notes/jacobian-reuse/analyze_3d.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import gamma_label, iteration_costs, load_sweep  # noqa: E402

RUNS = Path("output/jacobian_reuse_3d")
NOTES = Path(__file__).parent
QOI_NAMES = ["drag", "lift", "B_x", "B_y", "B_z"]


def main():
    runs, unfinished = load_sweep(RUNS)
    lines = []
    by_gamma = {r["gamma_reassemble"]: r for r in runs}
    reference = by_gamma.get(0.2)
    ref_step = np.mean(reference["step_times"][1:]) if reference else np.nan

    lines.append("## threeDfsi_simplified, medium mesh, 28 ranks, 88 steps from t = 10\n")
    lines.append(
        "| gamma | Newton iterations / step | assemblies / step | FGMRES / solve | s / step, after the first | "
        "vs 0.2 | t_A (s) | t_R (s) | first step (s) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        run_dir = RUNS / f"g{gamma_label(r['gamma_reassemble']).lower()}"
        assembled, reused, _, _ = iteration_costs(run_dir)
        n, n_it = r["num_steps"], r["num_newton_iterations"]
        per_step = np.mean(r["step_times"][1:])
        t_a = f"{assembled.mean():.3f}" if assembled.size else "-"
        t_r = f"{reused.mean():.3f}" if reused.size else "-"
        lines.append(
            f"| {gamma_label(r['gamma_reassemble'])} | {n_it / n:.2f} | {r['num_assemblies'] / n:.3f} | "
            f"{r['num_linear_iterations'] / n_it:.1f} | {per_step:.2f} | {per_step / ref_step - 1:+.1%} | {t_a} | "
            f"{t_r} | {r['step_times'][0]:.1f} |"
        )
    for name, t_last in unfinished:
        lines.append(f"| {name} | did not finish; last logged t = {t_last:.4f} | | | | | | | |")

    events = ["SNESJacobianEval", "PCSetUp", "SNESFunctionEval", "KSPSolve", "MatMatSolve"]
    lines.append("\nPETSc event times per step (s, maximum over the ranks, all steps):\n")
    lines.append("| gamma | " + " | ".join(events) + " |")
    lines.append("|---|" + "---|" * len(events))
    for r in runs:
        cells = [f"{r['events'][e]['time'] / r['num_steps']:.3f}" for e in events]
        lines.append(f"| {gamma_label(r['gamma_reassemble'])} | " + " | ".join(cells) + " |")

    if reference is not None:
        ref_qoi = np.loadtxt(RUNS / "g0.2" / "qoi.txt", ndmin=2)
        lines.append("\nLargest difference of the QoIs from gamma = 0.2 over the run, relative to their range:\n")
        lines.append("| gamma | " + " | ".join(QOI_NAMES) + " |")
        lines.append("|---|" + "---|" * len(QOI_NAMES))
        for r in runs:
            qoi = np.loadtxt(RUNS / f"g{gamma_label(r['gamma_reassemble']).lower()}" / "qoi.txt", ndmin=2)
            m = min(len(qoi), len(ref_qoi))
            cells = []
            for j in range(1, 1 + len(QOI_NAMES)):
                span = np.ptp(ref_qoi[:m, j]) or 1.0
                cells.append(f"{np.abs(qoi[:m, j] - ref_qoi[:m, j]).max() / span:.1e}")
            lines.append(f"| {gamma_label(r['gamma_reassemble'])} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    (NOTES / "tables_3d.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
