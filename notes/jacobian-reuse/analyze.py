# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Analysis of the gamma_reassemble sweep of fsi2_simplified (notes/jacobian-reuse).

Reads the sweep runs, output/jacobian_reuse_sweep/g<gamma>/, for the Newton iterations N_it(gamma) and
Jacobian assemblies N_A(gamma) over t = 0 to 15, and the calibration runs,
output/jacobian_reuse_calibration/p<ranks>_g<gamma>/, for the wall time t_A(p) of a Newton iteration
that assembles and refactorizes the Jacobian, t_R(p) of one that reuses it, and the time per step
outside the Newton iterations o(p). The predicted wall time of the sweep runs on p ranks is

    E(gamma, p) = N_A(gamma) t_A(p) + (N_it(gamma) - N_A(gamma)) t_R(p) + S o(p).

The same model over a window of time steps, with the counts of those steps, gives the best gamma
per stage: the start-up, from rest until the oscillation of the beam has developed, at t_dev, and
the developed flow after it, which is what a run restarted from a developed state solves (up to the
one assembly in its first step). t_dev is the first peak of the tip displacement A_y whose amplitude
is within DEVELOPED_TOLERANCE of the mean amplitude of the peaks over the last LAST_SECONDS seconds.
The best gamma is also given per window of WINDOW seconds.

Prints the tables, writes tables.md and the figures to plots/. Run from the repository root:

    python notes/jacobian-reuse/analyze.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xfsi_solver.tools.newton_log import read_newton_log

SWEEP = Path("output/jacobian_reuse_sweep")
CALIBRATION = Path("output/jacobian_reuse_calibration")
RESTART = Path("output/jacobian_reuse_restart")
NOTES = Path(__file__).parent
PLOTS = NOTES / "plots"

DEVELOPED_TOLERANCE = 0.05
LAST_SECONDS = 2.0
WINDOW = 1.0

INK, INK_SECONDARY, GRID = "#1f1f1e", "#5f5e5a", "#e4e2dc"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# Ordinal blue ramp (steps 250 to 700) for the rank counts, light to dark with more ranks
RANK_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]


def gamma_label(gamma):
    return "None" if gamma is None else f"{gamma:g}"


def load_sweep(directory=None):
    """The finished runs in ``directory`` (by default the sweep), sorted by gamma with None (every
    iteration assembles) first, and the unfinished ones with how far they got."""
    runs, unfinished = [], []
    for run_dir in sorted((SWEEP if directory is None else directory).glob("g*")):
        stats_path = run_dir / "stats.json"
        if stats_path.exists():
            runs.append(json.loads(stats_path.read_text()))
            continue
        log = read_newton_log(run_dir / "newton.txt") if (run_dir / "newton.txt").exists() else None
        t_last = float(log["t"][-1]) if log is not None and log.size else 0.0
        unfinished.append((run_dir.name[1:], t_last))
    runs.sort(key=lambda r: -1.0 if r["gamma_reassemble"] is None else r["gamma_reassemble"])
    return runs, unfinished


def iteration_costs(run_dir):
    """The wall times of the Newton iterations of a calibration run, split by whether the iteration
    assembled the Jacobian, without the first step (which also does the symbolic factorizations),
    and the time per step outside the iterations."""
    stats = json.loads((run_dir / "stats.json").read_text())
    log = read_newton_log(run_dir / "newton.txt")
    # The time column is the wall time since the start of the step, so the time of iteration k is
    # the difference to the row of iteration k - 1 of the same step.
    dt_iter = np.diff(log["time"])
    is_iteration = log["it"][1:] > 0
    first_step = log["t"][1:] == log["t"][0]
    durations = dt_iter[is_iteration]
    assembled = log["assembled"][1:][is_iteration].astype(bool)
    later = ~first_step[is_iteration]
    # The time per step outside the Newton iterations (time-step setup, prediction, QoIs), from
    # all steps, since the elapsed time covers all of them
    newton_time = log["time"][np.r_[log["it"][1:] == 0, True]].sum()
    outside = (stats["elapsed_time"] - newton_time) / stats["num_steps"]
    return durations[assembled & later], durations[~assembled & later], outside, stats


def load_calibration():
    """Per rank count: t_A, t_R and o, with the measured and predicted elapsed time of each run."""
    by_ranks = {}
    for run_dir in sorted(CALIBRATION.glob("p*_g*")):
        if not (run_dir / "stats.json").exists():
            continue
        assembled, reused, outside, stats = iteration_costs(run_dir)
        entry = by_ranks.setdefault(stats["num_ranks"], {"assembled": [], "reused": [], "outside": [], "runs": []})
        entry["assembled"].append(assembled)
        entry["reused"].append(reused)
        entry["outside"].append(outside)
        entry["runs"].append(stats)
    costs = {}
    for ranks, entry in sorted(by_ranks.items()):
        assembled, reused = np.concatenate(entry["assembled"]), np.concatenate(entry["reused"])
        costs[ranks] = {
            "t_A": float(np.mean(assembled)) if assembled.size else np.nan,
            "t_R": float(np.mean(reused)) if reused.size else np.nan,
            "o": float(np.mean(entry["outside"])),
            "n_A": assembled.size,
            "n_R": reused.size,
            "runs": entry["runs"],
        }
    return costs


def predicted_time(run, cost, steps=None):
    """The predicted wall time of the run, or of the time steps selected by the boolean mask
    ``steps``, on the ranks of ``cost``."""
    if steps is None:
        n_a, n_it, n_steps = run["num_assemblies"], run["num_newton_iterations"], run["num_steps"]
    else:
        n_a = int(np.sum(np.asarray(run["assemblies_per_step"])[steps]))
        n_it = int(np.sum(np.asarray(run["iterations_per_step"])[steps]))
        n_steps = int(np.sum(steps))
    return n_a * cost["t_A"] + (n_it - n_a) * cost["t_R"] + n_steps * cost["o"]


def step_times(run):
    """The time at the end of each time step of a sweep run (which starts from rest at t = 0)."""
    return run["dt"] * np.arange(1, run["num_steps"] + 1)


def developed_time(qoi_path):
    """t_dev, the first peak of the tip displacement A_y whose amplitude is within
    DEVELOPED_TOLERANCE of the mean amplitude of the peaks over the last LAST_SECONDS, and the
    peaks (t, amplitude)."""
    q = np.loadtxt(qoi_path, ndmin=2)
    t, a_y = q[:, 0], q[:, 4]
    mean = np.mean(a_y[t >= t[-1] - LAST_SECONDS])
    dev = np.abs(a_y - mean)
    is_peak = (dev[1:-1] >= dev[:-2]) & (dev[1:-1] > dev[2:])
    peak_t, peak_amp = t[1:-1][is_peak], dev[1:-1][is_peak]
    final_amp = np.mean(peak_amp[peak_t >= t[-1] - LAST_SECONDS])
    developed = np.abs(peak_amp - final_amp) <= DEVELOPED_TOLERANCE * final_amp
    # The first peak from which all later peaks stay within the tolerance
    first = len(developed) - np.argmin(developed[::-1]) if not developed.all() else 0
    if first == len(peak_t):
        raise ValueError(f"The oscillation in {qoi_path} has not developed by t = {t[-1]:g}")
    return float(peak_t[first]), peak_t, peak_amp, final_amp, t, a_y


def plot_development(t, a_y, peak_t, peak_amp, final_amp, t_dev):
    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    ax.plot(t, a_y * 1e3, color=BLUE, linewidth=1)
    ax.axvline(t_dev, color=INK_SECONDARY, linewidth=1.2, linestyle="--")
    ax.annotate(
        f"t_dev = {t_dev:.2f}",
        (t_dev, ax.get_ylim()[1]),
        xytext=(-4, -12),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    ax.set_xlabel("t (s)", color=INK)
    ax.set_ylabel("A_y (mm)", color=INK)
    ax.set_title(
        f"Tip displacement; developed once the peak amplitude stays within {DEVELOPED_TOLERANCE:.0%} of "
        f"its final {final_amp * 1e3:.1f} mm",
        color=INK,
        fontsize=10,
        loc="left",
    )
    style(ax)
    fig.savefig(PLOTS / "development.svg")
    fig.savefig(PLOTS / "development.png", dpi=150)
    plt.close(fig)


def best_over(runs, cost, steps):
    """The predicted times of the runs over the steps ``steps`` on the ranks of ``cost``, and the index
    of the best."""
    e = np.array([predicted_time(r, cost, steps) for r in runs])
    return e, int(np.argmin(e))


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def gamma_axis(runs):
    """x positions for the gammas: None (always reassemble) at a separate spot left of the rest."""
    gammas = [r["gamma_reassemble"] for r in runs]
    finite = [g for g in gammas if g is not None]
    x_none = min(finite) / 2.5 if finite else 1.0
    return np.array([x_none if g is None else g for g in gammas]), x_none


def plot_counts(runs):
    x, x_none = gamma_axis(runs)
    steps = runs[0]["num_steps"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "num_assemblies", "Jacobian assemblies per time step"),
        (axes[1], "num_newton_iterations", "Newton iterations per time step"),
    ]:
        y = np.array([r[key] for r in runs]) / steps
        ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=5)
        ax.set_xscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels([gamma_label(r["gamma_reassemble"]) for r in runs], rotation=60, fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("gamma_reassemble", color=INK)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_ylim(bottom=0)
        style(ax)
    fig.suptitle(f"FSI2, t = 0 to 15 ({steps} steps), MUMPS LU, displacement prediction", color=INK, fontsize=11)
    fig.savefig(PLOTS / "counts.svg")
    fig.savefig(PLOTS / "counts.png", dpi=150)
    plt.close(fig)


def plot_costs(costs):
    ranks = np.array(sorted(costs))
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    for key, color, label in [
        ("t_A", BLUE, "iteration that assembles + refactorizes"),
        ("t_R", ORANGE, "iteration that reuses"),
    ]:
        y = [costs[p][key] for p in ranks]
        ax.plot(ranks, y, color=color, linewidth=2, marker="o", markersize=5)
        ax.annotate(
            label, (ranks[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(p) for p in ranks])
    ax.minorticks_off()
    ax.set_xlabel("MPI ranks", color=INK)
    ax.set_ylabel("wall time per iteration (s)", color=INK)
    style(ax)
    fig.savefig(PLOTS / "iteration_cost.svg")
    fig.savefig(PLOTS / "iteration_cost.png", dpi=150)
    plt.close(fig)


def plot_predicted(runs, costs):
    x, _ = gamma_axis(runs)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    ranks = sorted(costs)
    ramp = RANK_RAMP[-len(ranks) :] if len(ranks) <= len(RANK_RAMP) else RANK_RAMP
    for color, p in zip(ramp, ranks, strict=False):
        e = np.array([predicted_time(r, costs[p]) for r in runs])
        rel = e / e.min()
        ax.plot(x, rel, color=color, linewidth=2, marker="o", markersize=4, label=f"{p} ranks")
        i = int(np.argmin(e))
        ax.plot(
            x[i], rel[i], marker="o", markersize=9, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.2
        )
    ax.set_xscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([gamma_label(r["gamma_reassemble"]) for r in runs], rotation=60, fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("gamma_reassemble", color=INK)
    ax.set_ylabel("predicted wall time / minimum", color=INK)
    ax.set_title(
        "Predicted wall time over t = 0 to 15, relative to the best gamma (circled)", color=INK, fontsize=11, loc="left"
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style(ax)
    fig.savefig(PLOTS / "predicted_time.svg")
    fig.savefig(PLOTS / "predicted_time.png", dpi=150)
    plt.close(fig)


def main():
    PLOTS.mkdir(exist_ok=True)
    runs, unfinished = load_sweep()
    costs = load_calibration()
    lines = []

    lines.append("## Counts over t = 0 to 15\n")
    lines.append("| gamma | steps | Newton iterations | per step | assemblies | per step | assembled share |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in runs:
        s, n_it, n_a = r["num_steps"], r["num_newton_iterations"], r["num_assemblies"]
        lines.append(
            f"| {gamma_label(r['gamma_reassemble'])} | {s} | {n_it} | {n_it / s:.2f} | {n_a} | {n_a / s:.3f} | "
            f"{n_a / n_it:.3f} |"
        )
    for name, t_last in unfinished:
        lines.append(f"| {name} | did not finish; last logged t = {t_last:.4f} | | | | | |")

    if costs:
        lines.append("\n## Costs per iteration (calibration, 200 steps from t = 15, idle machine)\n")
        lines.append(
            "| ranks | t_A (s) | t_R (s) | t_A / t_R | o (s/step) | iterations A / R | elapsed, measured / model |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for p, c in costs.items():
            checks = ", ".join(
                f"{gamma_label(s['gamma_reassemble'])}: {s['elapsed_time']:.1f} / {predicted_time(s, c):.1f}"
                for s in c["runs"]
            )
            lines.append(
                f"| {p} | {c['t_A']:.4f} | {c['t_R']:.4f} | {c['t_A'] / c['t_R']:.2f} | {c['o']:.4f} | "
                f"{c['n_A']} / {c['n_R']} | {checks} |"
            )

    if costs and runs:
        lines.append("\n## Predicted wall time over t = 0 to 15 (s)\n")
        ranks = list(costs)
        lines.append("| gamma | " + " | ".join(f"{p} ranks" for p in ranks) + " |")
        lines.append("|---|" + "---|" * len(ranks))
        table = np.array([[predicted_time(r, costs[p]) for p in ranks] for r in runs])
        best = table.argmin(axis=0)
        for i, r in enumerate(runs):
            cells = [f"**{table[i, j]:.0f}**" if best[j] == i else f"{table[i, j]:.0f}" for j in range(len(ranks))]
            lines.append(f"| {gamma_label(r['gamma_reassemble'])} | " + " | ".join(cells) + " |")
        lines.append("\n## Best gamma per rank count\n")
        lines.append("| ranks | best gamma | predicted time (s) | vs gamma = 0.2 | vs None | gammas within 2% |")
        lines.append("|---|---|---|---|---|---|")
        gammas = [r["gamma_reassemble"] for r in runs]
        for j, p in enumerate(ranks):
            col = table[:, j]
            e_02 = col[gammas.index(0.2)] if 0.2 in gammas else np.nan
            e_none = col[gammas.index(None)] if None in gammas else np.nan
            near = ", ".join(gamma_label(g) for g, e in zip(gammas, col, strict=True) if e <= 1.02 * col.min())
            lines.append(
                f"| {p} | {gamma_label(gammas[best[j]])} | {col.min():.0f} | {col.min() / e_02 - 1:+.1%} | "
                f"{col.min() / e_none - 1:+.1%} | {near} |"
            )
        plot_predicted(runs, costs)
        plot_costs(costs)
    if runs:
        plot_counts(runs)

    if costs and runs:
        t_dev, peak_t, peak_amp, final_amp, t_q, a_y = developed_time(SWEEP / "g0.2" / "qoi.txt")
        plot_development(t_q, a_y, peak_t, peak_amp, final_amp, t_dev)
        gammas = [r["gamma_reassemble"] for r in runs]
        t_steps = step_times(runs[0])
        start_up, developed = t_steps <= t_dev, t_steps > t_dev
        lines.append(f"\n## Stages: start-up t = 0 to {t_dev:.2f}, developed t = {t_dev:.2f} to 15\n")
        lines.append(
            f"t_dev: first peak of A_y from which the amplitude stays within {DEVELOPED_TOLERANCE:.0%} of the mean "
            f"amplitude over the last {LAST_SECONDS:g} s ({final_amp * 1e3:.2f} mm), from the gamma = 0.2 run.\n"
        )
        lines.append("Counts per step in each stage:\n")
        lines.append("| gamma | start-up: iterations | assemblies | developed: iterations | assemblies |")
        lines.append("|---|---|---|---|---|")
        for r in runs:
            it, asm = np.asarray(r["iterations_per_step"]), np.asarray(r["assemblies_per_step"])
            lines.append(
                f"| {gamma_label(r['gamma_reassemble'])} | {it[start_up].mean():.2f} | {asm[start_up].mean():.3f} | "
                f"{it[developed].mean():.2f} | {asm[developed].mean():.3f} |"
            )
        lines.append("\nBest gamma per stage, and the gain of switching gamma at t_dev over the best single gamma:\n")
        lines.append(
            "| ranks | start-up: best | time (s) | within 2% | developed: best | time (s) | within 2% | "
            "developed, per 1000 steps (s) | switching gain |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for p, c in costs.items():
            e_s, i_s = best_over(runs, c, start_up)
            e_d, i_d = best_over(runs, c, developed)
            e_all, _ = best_over(runs, c, np.ones_like(start_up))
            near_s = ", ".join(gamma_label(g) for g, e in zip(gammas, e_s, strict=True) if e <= 1.02 * e_s.min())
            near_d = ", ".join(gamma_label(g) for g, e in zip(gammas, e_d, strict=True) if e <= 1.02 * e_d.min())
            gain = 1 - (e_s.min() + e_d.min()) / e_all.min()
            lines.append(
                f"| {p} | {gamma_label(gammas[i_s])} | {e_s.min():.0f} | {near_s} | {gamma_label(gammas[i_d])} | "
                f"{e_d.min():.0f} | {near_d} | {1000 * e_d.min() / developed.sum():.1f} | {gain:.1%} |"
            )

        restarts, restarts_unfinished = load_sweep(RESTART)
        if restarts or restarts_unfinished:
            t0 = restarts[0]["T"] - restarts[0]["num_steps"] * restarts[0]["dt"] if restarts else 10.0
            window = t_steps > t0 + 1e-9
            lines.append(f"\n## Restarts from the developed state at t = {t0:g} to 15\n")
            lines.append(
                "Started from the gamma = 0.2 run's checkpoint at t = 10, so the first step assembles. The counts of "
                "the continuous sweep run over the same steps are given for comparison.\n"
            )
            lines.append(
                "| gamma | restart: iterations/step | assemblies/step | continuous: iterations/step | assemblies/step |"
            )
            lines.append("|---|---|---|---|---|")
            by_gamma = {r["gamma_reassemble"]: r for r in runs}
            for r in restarts:
                it, asm, n = r["num_newton_iterations"], r["num_assemblies"], r["num_steps"]
                cont = by_gamma.get(r["gamma_reassemble"])
                if cont is not None:
                    c_it = np.asarray(cont["iterations_per_step"])[window].mean()
                    c_asm = np.asarray(cont["assemblies_per_step"])[window].mean()
                    cont_cells = f"{c_it:.2f} | {c_asm:.3f}"
                else:
                    cont_cells = "- | -"
                lines.append(f"| {gamma_label(r['gamma_reassemble'])} | {it / n:.2f} | {asm / n:.3f} | {cont_cells} |")
            for name, t_last in restarts_unfinished:
                lines.append(f"| {name} | failed at t = {t_last:.4f} | | | |")
            if restarts:
                lines.append("\nPredicted time of the restarts over t = 10 to 15 (s), best in bold:\n")
                lines.append("| gamma | " + " | ".join(f"{p} ranks" for p in costs) + " |")
                lines.append("|---|" + "---|" * len(costs))
                table = np.array([[predicted_time(r, c) for c in costs.values()] for r in restarts])
                best = table.argmin(axis=0)
                for i, r in enumerate(restarts):
                    cells = [
                        f"**{table[i, j]:.0f}**" if best[j] == i else f"{table[i, j]:.0f}" for j in range(len(costs))
                    ]
                    lines.append(f"| {gamma_label(r['gamma_reassemble'])} | " + " | ".join(cells) + " |")

        lines.append(f"\n## Best gamma per window of {WINDOW:g} s\n")
        ranks = list(costs)
        lines.append("| window | " + " | ".join(f"{p} ranks" for p in ranks) + " |")
        lines.append("|---|" + "---|" * len(ranks))
        edges = np.arange(0.0, t_steps[-1] + 1e-9, WINDOW)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            window = (t_steps > a + 1e-9) & (t_steps <= b + 1e-9)
            cells = [gamma_label(gammas[best_over(runs, costs[p], window)[1]]) for p in ranks]
            lines.append(f"| {a:g}-{b:g} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    (NOTES / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
