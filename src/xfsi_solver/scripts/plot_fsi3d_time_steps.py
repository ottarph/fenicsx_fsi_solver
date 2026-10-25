# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Compare runs of the 3d FSI benchmark with different time steps with each other and with Failer & Richter.

Failer & Richter (J. Sci. Comput. 82:28, 2020, Sect. 5.4) run to t = 8 with k = 0.004, continue from there with
k = 0.004, 0.002 and 0.001, and give the mean and amplitude of each QoI over [9, 10] (eq. (16): mean = (max + min) / 2,
amplitude = (max - min) / 2) in the tables of their Fig. 3. This script reads QoI files of threeDfsi_c0ip or
threeDfsi_mixed (columns t, drag, lift, B_x, B_y, B_z) and writes, as PDF and PNG, to
output/figures/plot_fsi3d_time_steps:

- fsi3d_time_steps: every QoI over the window for each run (--run), with the range from min to max and the mean of
  the paper's finest mesh (level 3) at --paper-k, and optionally further runs for comparison (--compare), such as
  a run on another mesh, drawn dashed in gray.
- fsi3d_time_step_differences: the difference of each run from the last --run, in percent of that run's amplitude,
  at the times of the coarser run. The last run should be the one with the smallest time step.

It also prints a table of the mean and amplitude of each QoI and the frequency of B_y, for the runs and for the
paper's level 3 at all three time steps. Run from the repository root, for example:

    python -m xfsi_solver.scripts.plot_fsi3d_time_steps \\
        --run "fine, dt = 0.004" runs/fine_dt0.004.txt \\
        --run "fine, dt = 0.002" runs/fine_dt0.002.txt \\
        --run "fine, dt = 0.001" runs/fine_dt0.001.txt \\
        --compare "medium, dt = 0.004" output/qoi/threeDfsi_mixed_medium.txt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xfsi_solver.component_benchmarks.qoi_analysis import mean_and_amplitude

FIGURE_DIR = Path("output/figures/plot_fsi3d_time_steps")

FAILER_RICHTER_2020 = (
    "L. Failer and T. Richter. A parallel Newton multigrid framework for monolithic fluid-structure interactions. "
    "J. Sci. Comput. 82:28, 2020. doi:10.1007/s10915-019-01113-y"
)

# Fig. 3 of Failer & Richter, level 3: (mean, amplitude) over [9, 10] for each time step k. Displacements in m (the
# paper lists them times 1e3), drag and lift as forces (the paper lists the drag times 1e-2).
PAPER_LEVEL_3 = {
    0.004: {
        "drag": (185.7, 3.6),
        "lift": (-0.704, 41.347),
        "B_x": (-2.176e-3, 2.419e-3),
        "B_y": (2.766e-3, 25.687e-3),
        "B_z": (-0.196e-3, 0.196e-3),
    },
    0.002: {
        "drag": (185.7, 3.5),
        "lift": (-0.728, 41.407),
        "B_x": (-2.161e-3, 2.401e-3),
        "B_y": (2.750e-3, 25.643e-3),
        "B_z": (0.490e-3, 0.881e-3),
    },
    0.001: {
        "drag": (185.5, 3.5),
        "lift": (-0.717, 41.299),
        "B_x": (-2.143e-3, 2.383e-3),
        "B_y": (2.699e-3, 25.594e-3),
        "B_z": (0.486e-3, 0.877e-3),
    },
}

# (column name, axis label, scale of the plotted and printed values)
QOIS = [
    ("drag", "Drag [N]", 1.0),
    ("lift", "Lift [N]", 1.0),
    ("B_x", "$B_x$ [mm]", 1e3),
    ("B_y", "$B_y$ [mm]", 1e3),
    ("B_z", "$B_z$ [mm]", 1e3),
]

# Categorical colors of the runs, in a fixed order; comparison runs and the paper are gray.
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY = "#8a8984"
PAPER_RANGE_COLOR = "#d9d8d3"
GRID_COLOR = "#e8e7e2"


def load_qois(path):
    """The columns of a QoI file, by the names in its header."""
    with open(path) as f:
        names = f.readline().lstrip("#").split()
    data = np.loadtxt(path, ndmin=2)
    return {name: data[:, i] for i, name in enumerate(names)}


def restrict(qois, window):
    """The rows of qois with t in the closed interval window."""
    t0, t1 = window
    mask = (qois["t"] >= t0 - 1e-9) & (qois["t"] <= t1 + 1e-9)
    return {name: column[mask] for name, column in qois.items()}


def frequency(t, y):
    """Mean frequency of y from its upward crossings of (max + min) / 2, located by linear interpolation."""
    z = y - 0.5 * (y.max() + y.min())
    i = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
    crossings = t[i] - z[i] * (t[i + 1] - t[i]) / (z[i + 1] - z[i])
    if len(crossings) < 2:
        return None
    return (len(crossings) - 1) / (crossings[-1] - crossings[0])


def plot_runs(runs, compare, paper_k):
    fig, axes = plt.subplots(len(QOIS), 1, figsize=(7.0, 9.5), sharex=True)
    for ax, (name, label, scale) in zip(axes, QOIS, strict=True):
        mean, amplitude = PAPER_LEVEL_3[paper_k][name]
        ax.axhspan(
            (mean - amplitude) * scale,
            (mean + amplitude) * scale,
            color=PAPER_RANGE_COLOR,
            alpha=0.6,
            lw=0,
            label=f"Failer & Richter, level 3, k = {paper_k}: min to max",
        )
        ax.axhline(mean * scale, color=GRAY, lw=0.8, ls=":", label=f"Failer & Richter, level 3, k = {paper_k}: mean")
        for run_label, qois in compare.items():
            ax.plot(qois["t"], qois[name] * scale, color=GRAY, lw=1.0, ls="--", label=run_label)
        for (run_label, qois), color in zip(runs.items(), RUN_COLORS, strict=False):
            ax.plot(qois["t"], qois[name] * scale, color=color, lw=1.3, label=run_label)
        ax.set_ylabel(label)
        ax.grid(axis="y", color=GRID_COLOR, lw=0.6)
    return fig, axes


def plot_differences(runs):
    """The difference of each run from the last one, in percent of the last run's amplitude."""
    *others, (reference_label, reference) = runs.items()
    fig, axes = plt.subplots(len(QOIS), 1, figsize=(7.0, 8.5), sharex=True)
    for ax, (name, label, _) in zip(axes, QOIS, strict=True):
        _, reference_amplitude = mean_and_amplitude(reference[name])
        for (run_label, qois), color in zip(others, RUN_COLORS, strict=False):
            difference = qois[name] - np.interp(qois["t"], reference["t"], reference[name])
            ax.plot(
                qois["t"],
                100.0 * difference / reference_amplitude,
                color=color,
                lw=1.3,
                label=f"{run_label} minus {reference_label}",
            )
        ax.axhline(0.0, color=GRAY, lw=0.8)
        ax.set_ylabel(label.split(" [")[0] + "\n[% of amplitude]")
        ax.grid(axis="y", color=GRID_COLOR, lw=0.6)
    return fig, axes


def finish(fig, axes, window, title, name, output_dir):
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(*window)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.03, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def summary_rows(runs):
    """(label, [(mean, amplitude) of each QoI, scaled], frequency of B_y or None) of each run and of the paper."""
    rows = []
    for label, qois in runs.items():
        cells = [tuple(scale * v for v in mean_and_amplitude(qois[name])) for name, _, scale in QOIS]
        rows.append((label, cells, frequency(qois["t"], qois["B_y"])))
    for k, values in PAPER_LEVEL_3.items():
        cells = [tuple(scale * v for v in values[name]) for name, _, scale in QOIS]
        rows.append((f"Failer & Richter, level 3, k = {k}", cells, None))
    return rows


def print_summary(rows, window):
    print(f"Mean ± amplitude over [{window[0]:g}, {window[1]:g}], displacements in mm, and the frequency of B_y:\n")
    print("| run | " + " | ".join(label for _, label, _ in QOIS) + " | f(B_y) [Hz] |")
    print("|---" * (len(QOIS) + 2) + "|")
    for label, cells, f in rows:
        values = " | ".join(f"{mean:.4g} ± {amplitude:.4g}" for mean, amplitude in cells)
        print(f"| {label} | {values} | {'' if f is None else f'{f:.3f}'} |")
    print(f"\nFailer & Richter: {FAILER_RICHTER_2020}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        required=True,
        metavar=("LABEL", "QOI_FILE"),
        help="a run to compare, in order of decreasing time step; repeat for each run",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        action="append",
        default=[],
        metavar=("LABEL", "QOI_FILE"),
        help="a further run drawn for comparison, such as one on another mesh",
    )
    parser.add_argument("--window", nargs=2, type=float, default=(9.0, 10.0), metavar=("T0", "T1"))
    parser.add_argument("--paper-k", type=float, default=0.001, choices=sorted(PAPER_LEVEL_3))
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args(argv)
    if len(args.run) > len(RUN_COLORS):
        parser.error(f"at most {len(RUN_COLORS)} runs, one per color")

    window = tuple(args.window)
    runs = {label: restrict(load_qois(path), window) for label, path in args.run}
    compare = {label: restrict(load_qois(path), window) for label, path in args.compare}

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plot_runs(runs, compare, args.paper_k)
    title = f"3d FSI benchmark, t in [{window[0]:g}, {window[1]:g}]"
    finish(fig, axes, window, title, "fsi3d_time_steps", args.output_dir)
    if len(runs) > 1:
        fig, axes = plot_differences(runs)
        title = f"Differences from {list(runs)[-1]}, relative to its amplitude"
        finish(fig, axes, window, title, "fsi3d_time_step_differences", args.output_dir)

    print_summary(summary_rows({**runs, **compare}), window)


if __name__ == "__main__":
    main()
