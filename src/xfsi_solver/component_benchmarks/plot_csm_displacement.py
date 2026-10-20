# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Plot the displacement of the point A over time for the CSM tests, together with the reference values.

Writes, as PDF and SVG, to output/figures/component_benchmarks:

- csm1_displacement, csm2_displacement: the static tests, over the pseudo-time steps, with the reference value.
- csm3_displacement: the whole CSM3 run, with the reference minimum and maximum (mean -/+ amplitude).
- csm3_displacement_last_2s: the last two seconds of CSM3, as in the plots of the benchmark proposal.

Run from the repository root, after the CSM solver has written its QoI files:

    python -m xfsi_solver.component_benchmarks.plot_csm_displacement
"""

from pathlib import Path

import matplotlib.pyplot as plt

from xfsi_solver.component_benchmarks.compare_csm_qois import QOI_FILES, REFERENCE, load_qois
from xfsi_solver.component_benchmarks.references import TUREK_HRON_2006_SHORT

FIGURE_DIR = Path("output/figures/component_benchmarks")


def save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg"):
        fig.savefig(FIGURE_DIR / f"{name}.{extension}")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / name}.pdf and .svg")


def plot_static(test, path):
    """Displacement over the pseudo-time steps, which approach the static solution."""
    t, ux, uy = load_qois(path)
    steps = range(len(t))
    reference = REFERENCE[test]

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for ax, u, component in ((ax_x, ux, "x"), (ax_y, uy, "y")):
        ax.plot(steps, u, "ko-", label="computed")
        ax.axhline(reference[f"u{component}"], color="r", linestyle="--", label=TUREK_HRON_2006_SHORT)
        ax.set_ylabel(f"$u_{component}$ of A [$10^{{-3}}$ m]")
        ax.grid()
    ax_y.set_xlabel("Pseudo-time step")
    fig.suptitle(f"{test}: displacement of the point A")
    ax_x.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)
    save(fig, f"{test.lower()}_displacement")


def plot_periodic(test, path, t_start=None, name_suffix=""):
    """Displacement over time from t_start (default: the start of the run), with the reference minimum and maximum."""
    t, ux, uy = load_qois(path)
    reference = REFERENCE[test]
    if t_start is not None:
        shown = t >= t_start
        t, ux, uy = t[shown], ux[shown], uy[shown]

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for ax, u, component in ((ax_x, ux, "x"), (ax_y, uy, "y")):
        mean = reference[f"u{component}_mean"]
        amplitude = reference[f"u{component}_amplitude"]
        ax.plot(t, u, "k-", label="computed")
        ax.axhline(mean + amplitude, color="r", linestyle="--", label=f"{TUREK_HRON_2006_SHORT}, min and max")
        ax.axhline(mean - amplitude, color="r", linestyle="--")
        ax.set_ylabel(f"$u_{component}$ of A [$10^{{-3}}$ m]")
        ax.grid()
    ax_y.set_xlabel("Time [s]")
    fig.suptitle(f"{test}: displacement of the point A")
    ax_x.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)
    save(fig, f"{test.lower()}_displacement{name_suffix}")


def main():
    for test, path in QOI_FILES.items():
        if not path.is_file():
            print(f"{test}: no QoI file at {path}")
            continue
        if test == "CSM3":
            t_end = load_qois(path)[0][-1]
            plot_periodic(test, path)
            plot_periodic(test, path, t_start=t_end - 2.0, name_suffix="_last_2s")
        else:
            plot_static(test, path)


if __name__ == "__main__":
    main()
