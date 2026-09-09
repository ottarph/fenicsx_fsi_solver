# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Simplify plotted paths (dropping vertices that don't visibly change the
# line) so the SVG output stays small despite the qoi series often having
# several thousand points.
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0

# (column index into a qoi/reference array, y-axis label, output filename stem)
_QOI_SPECS = [
    (1, "Drag", "drag"),
    (2, "lift", "lift"),
    (3, "Tip $x$-displacement [m]", "x-disp"),
    (4, "Tip $y$-displacement [m]", "y-disp"),
]


def plot_qois(
    biharmonic_path: str | os.PathLike | None = None,
    harmonic_path: str | os.PathLike | None = None,
    reference_path: str | os.PathLike | None = None,
    output_dir: str | os.PathLike = "output/figures",
):
    """Plot drag, lift, and tip displacement over time for whichever of the
    biharmonic/harmonic solver QOI files and the FSI2 reference data are
    given. Any of the three inputs may be omitted (``None``) to plot only
    the series that are actually available.
    """

    series = []

    if biharmonic_path is not None:
        qois_bih = np.loadtxt(biharmonic_path)
        series.append(("biharmonic", "k-", qois_bih))

    if harmonic_path is not None:
        qois_harm = np.loadtxt(harmonic_path)
        series.append(("harmonic", "b--", qois_harm))

    if reference_path is not None:
        ref = np.loadtxt(reference_path)
        qois_ref = np.column_stack([
            ref[:, 0],           # time
            ref[:, 4] + ref[:, 6],    # drag
            -ref[:, 5] - ref[:, 7],   # lift
            ref[:, 10],          # tip x-displacement
            ref[:, 11],          # tip y-displacement
        ])
        series.append(("reference", "r:", qois_ref))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for col, ylabel, stem in _QOI_SPECS:
        plt.figure(figsize=(10, 5))
        for label, style, qois in series:
            plt.plot(qois[:, 0], qois[:, col], style, label=label)
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        if series:
            plt.legend()
        plt.savefig(output_dir / f"{stem}.pdf")
        plt.savefig(output_dir / f"{stem}.svg")
        plt.close()


def main():
    plot_qois(
        biharmonic_path="output/qoi/fsi2_biharm_qoi.txt",
        harmonic_path="output/qoi/fsi2_harm_dm_qoi.txt",
        reference_path="data/fsi2_reference.txt",
    )


if __name__ == "__main__":
    main()
