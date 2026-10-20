# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Plot the displacement of the point A and the drag and lift for the FSI tests, together with published values.

The published values are those that compare_fsi_qois.py compares with: for FSI1 and FSI3, the mean over the methods
of Turek et al. (2010) and their monolithic finite element methods 2a, 2b and 3; for FSI2, the reference values of
Turek & Hron (2006). See references.py for the full citations. The lift is shown with the sign of the benchmark
proposal. Writes, as PDF and SVG, to output/figures/component_benchmarks, with the suffix of the QoI files appended to
each name:

- fsi1_qois: the steady test, over the pseudo-time steps, with the published values.
- fsi2_qois, fsi3_qois: the whole run, with the published minima and maxima (mean -/+ amplitude).
- fsi2_qois_last_1s, fsi3_qois_last_0.5s: the end of the run, over the time windows of the plots in the benchmark
  proposal.

Run from the repository root, after the FSI solver has written its QoI files:

    python -m xfsi_solver.component_benchmarks.plot_fsi_qois
    python -m xfsi_solver.component_benchmarks.plot_fsi_qois --suffix _coarse   # reads fsi_1_coarse.txt, ...
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from xfsi_solver.component_benchmarks.compare_fsi_qois import TESTS, load_qois, published_results, qoi_path

FIGURE_DIR = Path("output/figures/component_benchmarks")

Y_LABELS = {
    "ux": "$u_x$ of A [$10^{-3}$ m]",
    "uy": "$u_y$ of A [$10^{-3}$ m]",
    "drag": "Drag",
    "lift": "Lift",
}

# Line color and style of each published result, in the order of published_results.
REFERENCE_STYLES = [("tab:red", "--"), ("tab:orange", ":"), ("tab:green", "-."), ("tab:blue", (0, (6, 2)))]

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
    published = published_results(test)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True, layout="constrained")
    for ax, (name, values) in zip(axes, quantities.items(), strict=True):
        ax.plot(steps, values, "ko-", label="computed")
        for (label, result), (color, linestyle) in zip(published.items(), REFERENCE_STYLES, strict=False):
            if result[name] is not None:
                ax.axhline(result[name], color=color, linestyle=linestyle, linewidth=1, label=label)
        ax.set_ylabel(Y_LABELS[name])
        ax.grid()
    axes[-1].set_xlabel("Pseudo-time step")
    fig.suptitle(f"{test}: displacement of the point A and forces on the cylinder and the flag")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False, fontsize="small")
    save(fig, f"{test.lower()}_qois{suffix}")


def plot_periodic(test, path, suffix, t_start=None, name_suffix=""):
    """The quantities over time from t_start (default: the start of the run), with the published minima and maxima."""
    t, quantities = load_qois(path)
    published = published_results(test)
    if t_start is not None:
        shown = t >= t_start
        t = t[shown]
        quantities = {name: values[shown] for name, values in quantities.items()}

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    for ax, (name, values) in zip(axes, quantities.items(), strict=True):
        ax.plot(t, values, "k-", label="computed")
        for (label, result), (color, linestyle) in zip(published.items(), REFERENCE_STYLES, strict=False):
            if result[name] is None:
                continue
            mean, amplitude, _ = result[name]
            style = {"color": color, "linestyle": linestyle, "linewidth": 1}
            ax.axhline(mean + amplitude, label=f"{label}, min and max", **style)
            ax.axhline(mean - amplitude, **style)
        ax.set_ylabel(Y_LABELS[name])
        ax.grid()
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{test}: displacement of the point A and forces on the cylinder and the flag")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False, fontsize="small")
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
