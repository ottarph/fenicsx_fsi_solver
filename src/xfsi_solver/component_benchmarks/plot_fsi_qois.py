# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Plot the displacement of the point A and the drag and lift for the FSI tests, together with the reference values.

The lift is shown with the sign of the benchmark proposal (see compare_fsi_qois.py). Writes, as PDF and SVG, to
output/figures/component_benchmarks, with the suffix of the QoI files appended to each name:

- fsi1_qois: the steady test, over the pseudo-time steps, with the reference value.
- fsi2_qois, fsi3_qois: the whole run, with the reference minimum and maximum (mean -/+ amplitude).
- fsi2_qois_last_1s, fsi3_qois_last_0.5s: the end of the run, over the time windows of the plots in the benchmark
  proposal.

Run from the repository root, after the FSI solver has written its QoI files:

    python -m xfsi_solver.component_benchmarks.plot_fsi_qois
    python -m xfsi_solver.component_benchmarks.plot_fsi_qois --suffix _coarse   # reads fsi_1_coarse.txt, ...
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from xfsi_solver.component_benchmarks.compare_fsi_qois import REFERENCE, TESTS, load_qois, qoi_path

FIGURE_DIR = Path("output/figures/component_benchmarks")

Y_LABELS = {
    "ux": "$u_x$ of A [$10^{-3}$ m]",
    "uy": "$u_y$ of A [$10^{-3}$ m]",
    "drag": "Drag",
    "lift": "Lift",
}

# Length of the time window at the end of the run in the plots of the benchmark proposal.
LAST_WINDOW = {"FSI2": 1.0, "FSI3": 0.5}


def save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg"):
        fig.savefig(FIGURE_DIR / f"{name}.{extension}")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / name}.pdf and .svg")


def plot_steady(test, path, suffix):
    """The quantities over the pseudo-time steps, which approach the steady solution."""
    t, quantities = load_qois(path)
    steps = range(len(t))
    reference = REFERENCE[test]

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True, layout="constrained")
    for ax, (name, values) in zip(axes, quantities.items(), strict=True):
        ax.plot(steps, values, "ko-", label="computed")
        ax.axhline(reference[name], color="r", linestyle="--", label="reference")
        ax.set_ylabel(Y_LABELS[name])
        ax.grid()
    axes[-1].set_xlabel("Pseudo-time step")
    fig.suptitle(f"{test}: displacement of the point A and forces on the cylinder and the flag")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)
    save(fig, f"{test.lower()}_qois{suffix}")


def plot_periodic(test, path, suffix, t_start=None, name_suffix=""):
    """The quantities over time from t_start (default: the start of the run), with the reference minimum and maximum."""
    t, quantities = load_qois(path)
    reference = REFERENCE[test]
    if t_start is not None:
        shown = t >= t_start
        t = t[shown]
        quantities = {name: values[shown] for name, values in quantities.items()}

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    for ax, (name, values) in zip(axes, quantities.items(), strict=True):
        mean, amplitude, _ = reference[name]
        ax.plot(t, values, "k-", label="computed")
        ax.axhline(mean + amplitude, color="r", linestyle="--", label="reference min and max")
        ax.axhline(mean - amplitude, color="r", linestyle="--")
        ax.set_ylabel(Y_LABELS[name])
        ax.grid()
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{test}: displacement of the point A and forces on the cylinder and the flag")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)
    save(fig, f"{test.lower()}_qois{suffix}{name_suffix}")


def plot_all(suffix=""):
    """Plot every FSI test whose QoI file fsi_<n><suffix>.txt exists."""
    for test in TESTS:
        path = qoi_path(test, suffix)
        if not path.is_file():
            print(f"{test}: no QoI file at {path}")
            continue
        if test == "FSI1":
            plot_steady(test, path, suffix)
        else:
            window = LAST_WINDOW[test]
            t_end = load_qois(path)[0][-1]
            plot_periodic(test, path, suffix)
            plot_periodic(test, path, suffix, t_start=t_end - window, name_suffix=f"_last_{window:g}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suffix", default="", help="suffix of the QoI file names, e.g. _coarse for fsi_1_coarse.txt")
    plot_all(parser.parse_args().suffix)


if __name__ == "__main__":
    main()
