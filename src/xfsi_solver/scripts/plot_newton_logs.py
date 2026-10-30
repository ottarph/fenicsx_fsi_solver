# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Plots of the Newton convergence of FSI2 runs, from their Newton logs (``xfsi_solver.tools.newton_log``).

    python -m xfsi_solver.scripts.plot_newton_logs <figure_dir> <label>=<run_dir> ... [-- <label>=<run_dir> ...]

Each ``run_dir`` holds ``<name>_newton.txt`` and ``<name>_qoi.txt``, named after the directory.
The runs after ``--`` are shown only in the cost figure, with the others. The figures compare the
runs:

- ``residual_history``: the residual norm against the Newton iteration, median and interquartile
  range over the time steps, with the SNES ``atol``,
- ``contraction_by_age``: the contraction factor ||R_(k+1)|| / ||R_k|| of the Newton iterations
  against the age of their Jacobian, the number of time steps since it was assembled, for the
  first iteration of a time step and the later ones, for the runs with Jacobian reuse,
- ``iterations_over_time``: the tip displacement, the Newton iterations of every time step and the
  time steps in which the runs with reuse assemble the Jacobian,
- ``cost``: the mean wall time per time step, and per Newton iteration with a new and with a reused
  Jacobian.

The first time step of every run is left out, since it may start from the state of another
formulation.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from xfsi_solver.tools.newton_log import read_newton_log  # noqa: E402

# Categorical series colors, in a fixed order, and the ink of text, axes and reference lines.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SNES_ATOL = 1.0e-7

_RC = {
    "font.size": 9,
    "axes.edgecolor": TEXT_SECONDARY,
    "axes.labelcolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "path.simplify": True,
}


class Run:
    """The Newton log and the QoIs of one run, without its first time step."""

    def __init__(self, label: str, run_dir: str | Path):
        run_dir = Path(run_dir)
        self.label = label
        rows = read_newton_log(run_dir / f"{run_dir.name}_newton.txt")
        self.rows = rows[rows["t"] > rows["t"][0] + 1e-9]
        self.qoi = np.loadtxt(run_dir / f"{run_dir.name}_qoi.txt", ndmin=2)
        self.times, starts = np.unique(self.rows["t"], return_index=True)
        self.steps = [self.rows[s:e] for s, e in zip(starts, [*starts[1:], len(self.rows)], strict=True)]

    @property
    def iterations(self) -> np.ndarray:
        """Newton iterations of each time step."""
        return np.array([len(step) - 1 for step in self.steps])

    @property
    def step_times(self) -> np.ndarray:
        """Wall time of each time step, from the residual of its initial guess on."""
        return np.array([step["time"][-1] for step in self.steps])

    def residuals(self, max_it: int) -> np.ndarray:
        """Residual norms of every time step at iterations 0 to ``max_it``, ``nan`` after convergence."""
        values = np.full((len(self.steps), max_it + 1), np.nan)
        for i, step in enumerate(self.steps):
            n = min(len(step), max_it + 1)
            values[i, :n] = step["residual"][:n]
        return values

    def iteration_records(self) -> dict[str, np.ndarray]:
        """One entry per Newton iteration k -> k + 1, under the keys

        - ``k``: the iteration it starts from, 0 for the first of its time step,
        - ``contraction``: ||R_(k+1)|| / ||R_k||,
        - ``assembled``: whether the Jacobian was assembled for this iteration,
        - ``age``: the number of time steps since that Jacobian was assembled, 0 within the step it
          was assembled in, and -1 for a Jacobian inherited from the left-out first time step,
        - ``duration``: the wall time of the iteration.
        """
        records = {key: [] for key in ("k", "contraction", "assembled", "age", "duration")}
        age = -1
        for step in self.steps:
            if age >= 0:
                age += 1
            for k in range(1, len(step)):
                if step["assembled"][k]:
                    age = 0
                records["k"].append(k - 1)
                records["contraction"].append(step["residual"][k] / step["residual"][k - 1])
                records["assembled"].append(bool(step["assembled"][k]))
                records["age"].append(age)
                records["duration"].append(step["time"][k] - step["time"][k - 1])
        return {key: np.array(values) for key, values in records.items()}


def _label(text: str) -> str:
    """``text`` with J_0 typeset as math."""
    return text.replace("J_0", "$J_0$")


def plot_residual_history(runs: list[Run], path: Path) -> None:
    max_it = max(int(run.iterations.max()) for run in runs)
    last_plotted = 0
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for run, color in zip(runs, SERIES_COLORS, strict=False):
        values = run.residuals(max_it)
        # Iterations that at least half the steps reach. Steps that have converged drop out, so the
        # last points are over the slower steps.
        reached = np.sum(~np.isnan(values), axis=0) >= len(values) / 2
        k = np.arange(max_it + 1)[reached]
        last_plotted = max(last_plotted, k[-1])
        lower, median, upper = (np.nanpercentile(values[:, reached], q, axis=0) for q in (25, 50, 75))
        ax.fill_between(k, lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.plot(k, median, color=color, marker="o", markersize=4, label=_label(run.label))
    ax.axhline(SNES_ATOL, color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
    ax.annotate("snes_atol", (-0.35, SNES_ATOL), xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                color=TEXT_SECONDARY, fontsize=8)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("Newton iteration $k$")
    ax.set_ylabel(r"$\|R_k\|_2$")
    ax.set_xticks(range(last_plotted + 1))
    ax.set_xlim(-0.4, last_plotted + 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_title("Residual per Newton iteration: median and interquartile range over time steps",
                 loc="left", fontsize=9, color=TEXT)  # fmt: skip
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def plot_contraction_by_age(runs: list[Run], reference: Run | None, gamma: float, path: Path) -> None:
    """Contraction factor against the age of the Jacobian in time steps for the runs with reuse,
    separately for the first Newton iteration of a time step and the later ones, with the medians of
    the run ``reference``, which assembles in every iteration."""
    width = 6.0 if len(runs) == 1 else 4.6
    fig, axes = plt.subplots(1, len(runs), figsize=(width * len(runs), 3.8), sharey=True, squeeze=False)
    kinds = [("first iteration of a step", lambda k: k == 0, SERIES_COLORS[3], -0.15),
             ("later iterations", lambda k: k > 0, SERIES_COLORS[2], 0.15)]  # fmt: skip
    for ax, run in zip(axes[0], runs, strict=True):
        records = run.iteration_records()
        valid = records["age"] >= 0
        ages = np.arange(records["age"].max() + 1)
        for name, selects, color, offset in kinds:
            mask = valid & selects(records["k"])
            present = [a for a in ages if np.any(mask & (records["age"] == a))]
            quartiles = np.array([np.percentile(records["contraction"][mask & (records["age"] == a)], [25, 50, 75])
                                  for a in present])  # fmt: skip
            x = np.array(present) + offset
            ax.vlines(x, quartiles[:, 0], quartiles[:, 2], color=color, linewidth=2.0, alpha=0.5)
            ax.plot(x, quartiles[:, 1], color=color, marker="o", markersize=4, linewidth=1.2, label=name)
        if reference is not None:
            ref = reference.iteration_records()
            for _name, selects, color, _ in kinds:
                ax.axhline(np.median(ref["contraction"][selects(ref["k"])]), color=color, linestyle=":",
                           linewidth=1.2)  # fmt: skip
            ax.annotate(f"dotted: the same, {_label(reference.label)}", (ages[-1] + 0.4, 1.0e-3), ha="right",
                        va="bottom", fontsize=8, color=TEXT_SECONDARY)  # fmt: skip
        ax.axhline(gamma, color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
        ax.annotate(rf"$\gamma$ = {gamma:g}: reassemble above", (ages[-1] + 0.4, gamma), xytext=(0, 3),
                    textcoords="offset points", ha="right", va="bottom", color=TEXT_SECONDARY, fontsize=8)  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylim(1.0e-3, 0.6)
        ax.set_xlim(-0.6, ages[-1] + 0.6)
        ax.set_xticks(ages)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("age of the Jacobian (time steps since it was assembled)")
        if len(runs) == 1:
            ax.set_title(f"Contraction per Newton iteration, {_label(run.label)}: median and interquartile range",
                         loc="left", fontsize=9)  # fmt: skip
        else:
            ax.set_title(_label(run.label), loc="left", fontsize=9)
    if len(runs) > 1:
        fig.suptitle("Contraction per Newton iteration: median and interquartile range", x=0.01, ha="left", fontsize=9)
    axes[0, 0].set_ylabel(r"contraction $\|R_{k+1}\| / \|R_k\|$")
    # Below the axes, clear of the reference lines, which can lie anywhere.
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncols=2, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_iterations_over_time(runs: list[Run], path: Path) -> None:
    reusing = [run for run in runs if reuses_jacobian(run)]
    num_panels = 3 if reusing else 2
    fig, axes = plt.subplots(num_panels, 1, figsize=(6.4, 5.2 if reusing else 4.6), sharex=True,
                             height_ratios=[1, 1.4, 0.35][:num_panels])  # fmt: skip
    ax_disp, ax_its = axes[0], axes[1]
    qoi = runs[0].qoi
    ax_disp.plot(qoi[1:, 0], qoi[1:, 4], color=TEXT_SECONDARY, linewidth=1.2)
    ax_disp.set_ylabel("tip $y$-disp. [m]")
    ax_disp.set_title(f"Tip displacement ({_label(runs[0].label)}) and Newton iterations per time step",
                      loc="left", fontsize=9)  # fmt: skip
    for run, color in zip(runs, SERIES_COLORS, strict=False):
        ax_its.step(run.times, run.iterations, where="mid", color=color, linewidth=1.2,
                    label=f"{_label(run.label)} (mean {run.iterations.mean():.2f})")  # fmt: skip
    ax_its.set_ylabel("Newton iterations")
    ax_its.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if reusing:
        ax_asm = axes[2]
        for i, run in enumerate(reusing):
            assembly_times = [t for t, step in zip(run.times, run.steps, strict=True) if np.any(step["assembled"])]
            color = SERIES_COLORS[runs.index(run) % len(SERIES_COLORS)]
            ax_asm.eventplot(assembly_times, lineoffsets=i, linelengths=0.7, linewidths=1.0, colors=color)
        ax_asm.set_yticks(range(len(reusing)), [""] * len(reusing))
        ax_asm.set_ylabel("assembly", rotation=0, ha="right", va="center")
        ax_asm.grid(axis="y", visible=False)
    axes[-1].set_xlabel("$t$ [s]")
    # Below the figure, since the step lines fill the panel at any height.
    handles, labels = ax_its.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=2, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_cost(runs: list[Run], path: Path) -> None:
    fig, (ax_step, ax_it) = plt.subplots(1, 2, figsize=(8.0, 0.55 * len(runs) + 1.4), sharey=True)
    y = np.arange(len(runs))[::-1]
    labels = [_label(run.label) for run in runs]
    step_means = [run.step_times.mean() for run in runs]
    ax_step.barh(y, step_means, height=0.6, color=SERIES_COLORS[0])
    for yi, value in zip(y, step_means, strict=True):
        ax_step.annotate(f"{value:.3f} s", (value, yi), xytext=(3, 0), textcoords="offset points", va="center",
                         fontsize=8, color=TEXT)  # fmt: skip
    ax_step.set_yticks(y, labels)
    ax_step.set_xlabel("mean wall time per time step [s]")
    ax_step.grid(axis="y", visible=False)
    ax_step.set_xlim(0, 1.18 * max(step_means))

    # One bar per kind of iteration the run has: with a Jacobian assembled for it, and reusing one.
    height = 0.36
    longest = 0.0
    for yi, run in zip(y, runs, strict=True):
        records = run.iteration_records()
        assembled, duration = records["assembled"], records["duration"]
        for offset, mask, color in [
            (height / 2, assembled, SERIES_COLORS[1]),
            (-height / 2, ~assembled, SERIES_COLORS[2]),
        ]:
            if not np.any(mask):
                continue
            value = duration[mask].mean()
            longest = max(longest, value)
            ax_it.barh(yi + offset, value, height=0.92 * height, color=color)
            ax_it.annotate(f"{value:.3f} s (n={np.count_nonzero(mask)})", (value, yi + offset), xytext=(3, 0),
                           textcoords="offset points", va="center", fontsize=7, color=TEXT)  # fmt: skip
    handles = [
        matplotlib.patches.Patch(color=SERIES_COLORS[1], label="new Jacobian"),
        matplotlib.patches.Patch(color=SERIES_COLORS[2], label="reused Jacobian"),
    ]
    ax_it.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2, fontsize=8)
    ax_it.set_xlabel("mean wall time per Newton iteration [s]")
    ax_it.grid(axis="y", visible=False)
    ax_it.set_xlim(0, 1.5 * longest)
    fig.suptitle("Cost of the runs", x=0.01, ha="left", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def summary_table(runs: list[Run]) -> str:
    """A Markdown table of the iterations, assemblies, times and median contraction factors."""
    lines = [
        "| run | its/step | max its | assemblies | time/step (s) | rate 0→1 | 1→2 | 2→3 | 3→4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        values = run.residuals(4)
        rates = [np.nanmedian(values[:, k + 1] / values[:, k]) for k in range(4)]
        rate_cells = " | ".join("" if np.isnan(r) else f"{r:.1e}" for r in rates)
        lines.append(
            f"| {run.label} | {run.iterations.mean():.2f} | {run.iterations.max()} | "
            f"{int(run.rows['assembled'].sum())} | {run.step_times.mean():.3f} | {rate_cells} |"
        )
    return "\n".join(lines)


def reuses_jacobian(run: Run) -> bool:
    """Whether some Newton iteration of ``run`` reuses the Jacobian of an earlier one."""
    return bool(np.any(run.rows["assembled"][run.rows["it"] > 0] == 0))


def plot_newton_comparison(
    runs: list[Run], figure_dir: str | Path, cost_runs: list[Run] | None = None, gamma: float = 0.2
) -> None:
    """Write the figures of the module docstring for ``runs`` into ``figure_dir``, as PNG and SVG.

    The cost figure shows ``cost_runs``, by default ``runs``. The contraction figure shows the runs
    that reuse the Jacobian, with the reassembly threshold ``gamma``, against the last run that does
    not, if any.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    reusing = [run for run in runs if reuses_jacobian(run)]
    fresh = [run for run in runs if not reuses_jacobian(run)]
    with plt.rc_context(_RC):
        for suffix in ("png", "svg"):
            plot_residual_history(runs, figure_dir / f"residual_history.{suffix}")
            plot_iterations_over_time(runs, figure_dir / f"iterations_over_time.{suffix}")
            plot_cost(cost_runs or runs, figure_dir / f"cost.{suffix}")
            if reusing:
                reference = fresh[-1] if fresh else None
                plot_contraction_by_age(reusing, reference, gamma, figure_dir / f"contraction_by_age.{suffix}")


def main() -> None:
    figure_dir, *specs = sys.argv[1:]
    extra = specs[specs.index("--") + 1 :] if "--" in specs else []
    specs = specs[: specs.index("--")] if "--" in specs else specs
    runs = [Run(*spec.rsplit("=", 1)) for spec in specs]
    cost_runs = runs + [Run(*spec.rsplit("=", 1)) for spec in extra]
    plot_newton_comparison(runs, figure_dir, cost_runs=cost_runs)
    print(summary_table(cost_runs))


if __name__ == "__main__":
    main()
