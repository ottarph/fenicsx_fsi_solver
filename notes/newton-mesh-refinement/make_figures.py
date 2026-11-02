"""Make the figures of report.md from the Newton logs of the runs in output/newton_refinement/.

    python notes/newton-mesh-refinement/make_figures.py [figure_dir]

The runs are one period (200 steps of dt = 0.0025 from t = 16.5525) on each mesh of the family
data/meshes/fsi2/refinement/mesh_sec_ref{0,1,2}, from the settled states, in the configurations
j0_pred, j0_nopred (J_0 reassembled in every iteration, with and without the prediction, tight
relative tolerance), full_pred (the full Jacobian with the prediction, tight relative tolerance) and
j0_reuse (J_0 with the prediction and reuse, gamma 0.2, default tolerances). Writes SVG and PNG
figures into figure_dir, by default plots/ next to this file.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from xfsi_solver.scripts.plot_newton_logs import _RC, SERIES_COLORS, TEXT_SECONDARY, Run

ROOT = Path("output/newton_refinement")
LEVELS = [0, 1, 2]
H = {level: 2.0**-level for level in LEVELS}  # mesh size relative to ref0
LEVEL_LABELS = {0: "ref0 (h)", 1: "ref1 (h/2)", 2: "ref2 (h/4)"}
# Ratios only between residuals above FLOOR * ||R_0|| of their step, clear of round-off.
FLOOR = 1.0e-9

figure_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "plots"
figure_dir.mkdir(parents=True, exist_ok=True)


def run(kind: str, level: int) -> Run | None:
    """The run, or None if it has no log yet, or too few steps to read, while it starts."""
    run_dir = ROOT / f"{kind}_ref{level}"
    if not (run_dir / f"{kind}_ref{level}_newton.txt").exists():
        return None
    try:
        return Run(f"{kind}, ref{level}", run_dir)
    except ValueError:
        return None


def ratios(r: Run, first: bool) -> np.ndarray:
    """Contraction ||R_(k+1)|| / ||R_k|| of every step, in the first iteration or the later ones."""
    out = []
    for step in r.steps:
        res = step["residual"]
        ks = [0] if first else range(1, len(res) - 1)
        out += [res[k + 1] / res[k] for k in ks if k + 1 < len(res) and res[k + 1] > FLOOR * res[0]]
    return np.array(out)


def step_rates(r: Run) -> np.ndarray:
    """Median contraction of the later iterations of each step, NaN for a step without any."""
    rates = []
    for step in r.steps:
        res = step["residual"]
        rs = [res[k + 1] / res[k] for k in range(1, len(res) - 1) if res[k + 1] > FLOOR * res[0]]
        rates.append(np.median(rs) if rs else np.nan)
    return np.array(rates)


def plot_rates_vs_h(path: Path) -> None:
    fig, (ax_rate, ax_its) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    series = [("j0_pred", False, "$J_0$, predicted: later iterations", SERIES_COLORS[0]),
              ("j0_nopred", False, "$J_0$, not predicted: later iterations", SERIES_COLORS[1]),
              ("j0_pred", True, "$J_0$, predicted: first iteration", SERIES_COLORS[2])]  # fmt: skip
    for kind, first, label, color in series:
        hs, med, lo, hi = [], [], [], []
        for level in LEVELS:
            r = run(kind, level)
            if r is None:
                continue
            values = ratios(r, first)
            hs.append(H[level])
            q = np.percentile(values, [25, 50, 75])
            lo.append(q[1] - q[0])
            med.append(q[1])
            hi.append(q[2] - q[1])
        ax_rate.errorbar(hs, med, yerr=[lo, hi], color=color, marker="o", markersize=5, capsize=3, label=label)
    ax_rate.set_xscale("log", base=2)
    ax_rate.set_yscale("log")
    ax_rate.set_xticks([H[level] for level in LEVELS], ["h", "h/2", "h/4"])
    ax_rate.invert_xaxis()
    ax_rate.set_xlabel("mesh size")
    ax_rate.set_ylabel(r"contraction $\|R_{k+1}\| / \|R_k\|$")
    ax_rate.set_title("Contraction per Newton iteration: median and IQR", loc="left", fontsize=9)
    ax_rate.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=8)

    its_series = [("j0_pred", "$J_0$, predicted", SERIES_COLORS[0]),
                  ("j0_nopred", "$J_0$, not predicted", SERIES_COLORS[1]),
                  ("full_pred", "full Jacobian, predicted", SERIES_COLORS[3]),
                  ("j0_reuse", r"$J_0$, predicted, reused ($\gamma$=0.2)", SERIES_COLORS[4])]  # fmt: skip
    for kind, label, color in its_series:
        hs, its = [], []
        for level in LEVELS:
            r = run(kind, level)
            if r is not None:
                hs.append(H[level])
                its.append(r.iterations.mean())
        ax_its.plot(hs, its, color=color, marker="o", markersize=5, label=label)
    ax_its.set_xscale("log", base=2)
    ax_its.set_xticks([H[level] for level in LEVELS], ["h", "h/2", "h/4"])
    ax_its.invert_xaxis()
    ax_its.set_xlabel("mesh size")
    ax_its.set_ylabel("Newton iterations per step")
    ax_its.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_its.set_title("Mean Newton iterations per step (tight tolerance except reuse)", loc="left", fontsize=9)
    ax_its.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_residual_histories(path: Path) -> None:
    kinds = [("j0_pred", "$J_0$, predicted"), ("full_pred", "full Jacobian, predicted")]
    fig, axes = plt.subplots(1, len(kinds), figsize=(9.0, 3.6), sharey=True)
    for ax, (kind, title) in zip(axes, kinds, strict=True):
        max_it = 0
        for level, color in zip(LEVELS, SERIES_COLORS, strict=False):
            r = run(kind, level)
            if r is None:
                continue
            n = int(r.iterations.max())
            values = r.residuals(n) / r.residuals(n)[:, :1]
            reached = np.sum(~np.isnan(values), axis=0) >= len(values) / 2
            k = np.arange(n + 1)[reached]
            max_it = max(max_it, k[-1])
            lower, median, upper = (np.nanpercentile(values[:, reached], q, axis=0) for q in (25, 50, 75))
            ax.fill_between(k, lower, upper, color=color, alpha=0.18, linewidth=0)
            ax.plot(k, median, color=color, marker="o", markersize=4, label=LEVEL_LABELS[level])
        ax.set_yscale("log")
        ax.set_xticks(range(max_it + 1))
        ax.set_xlabel("Newton iteration $k$")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{title}: residual relative to the first", loc="left", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel(r"$\|R_k\| / \|R_0\|$")
    fig.savefig(path)
    plt.close(fig)


def plot_rate_over_period(path: Path) -> None:
    fig, (ax_disp, ax_rate) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True, height_ratios=[1, 1.6])
    for level, color in zip(LEVELS, SERIES_COLORS, strict=False):
        r = run("j0_pred", level)
        if r is None:
            continue
        ax_disp.plot(r.qoi[1:, 0], r.qoi[1:, 4], color=color, linewidth=1.2, label=LEVEL_LABELS[level])
        ax_rate.plot(r.times, step_rates(r), color=color, linewidth=1.0, label=LEVEL_LABELS[level])
    ax_disp.set_ylabel("tip $y$-disp. [m]")
    ax_disp.set_title(
        "Tip displacement, and median contraction of the later iterations of each step ($J_0$, predicted)",
        loc="left",
        fontsize=9,
    )
    ax_rate.set_yscale("log")
    ax_rate.set_ylabel("contraction")
    ax_rate.set_xlabel("$t$ [s]")
    handles, labels = ax_rate.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=3, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_reuse_iterations(path: Path) -> None:
    fig, (ax_its, ax_asm) = plt.subplots(2, 1, figsize=(6.4, 3.8), sharex=True, height_ratios=[1.6, 0.6])
    plotted = []
    for level, color in zip(LEVELS, SERIES_COLORS, strict=False):
        r = run("j0_reuse", level)
        if r is None:
            continue
        plotted.append(level)
        ax_its.step(r.times, r.iterations, where="mid", color=color, linewidth=1.0,
                    label=f"{LEVEL_LABELS[level]} (mean {r.iterations.mean():.2f})")  # fmt: skip
        assembly_times = [t for t, step in zip(r.times, r.steps, strict=True) if np.any(step["assembled"])]
        ax_asm.eventplot(assembly_times, lineoffsets=len(plotted) - 1, linelengths=0.7, linewidths=1.0, colors=color)
    ax_its.set_ylabel("Newton iterations")
    ax_its.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_its.set_title(r"Newton iterations per step with reuse ($J_0$, predicted, $\gamma$=0.2)", loc="left", fontsize=9)
    ax_asm.set_yticks(range(len(plotted)), [""] * len(plotted))
    ax_asm.set_ylabel("assembly", rotation=0, ha="right", va="center", color=TEXT_SECONDARY)
    ax_asm.grid(axis="y", visible=False)
    ax_asm.set_xlabel("$t$ [s]")
    handles, labels = ax_its.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=3, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


with plt.rc_context(_RC):
    for suffix in ("svg", "png"):
        plot_rates_vs_h(figure_dir / f"rates_vs_h.{suffix}")
        plot_residual_histories(figure_dir / f"residual_histories.{suffix}")
        plot_rate_over_period(figure_dir / f"rate_over_period.{suffix}")
        plot_reuse_iterations(figure_dir / f"reuse_iterations.{suffix}")
