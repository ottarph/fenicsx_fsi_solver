# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Path simplification (dropping vertices that don't visibly change the
# line) applied only to the SVG save below, so it stays small despite the
# qoi series often having several thousand points, without also reducing
# the PDF output's precision.
_SVG_RC = {"path.simplify": True, "path.simplify_threshold": 1.0}

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
        series.append(("reference", "r:", _load_reference(reference_path)))

    _plot_series(series, output_dir)


def plot_restricted_comparison(
    biharmonic_path: str | os.PathLike | None = None,
    restricted_path: str | os.PathLike | None = None,
    reference_path: str | os.PathLike | None = None,
    output_dir: str | os.PathLike = "output/figures",
    filename_prefix: str = "restr_compare_",
):
    """Plot drag, lift, and tip displacement over time for the base
    biharmonic solver and the biharmonic solver with restricted test
    functions, optionally together with the FSI2 reference data. Any of the
    three inputs may be omitted (``None``). Figures are written with
    ``filename_prefix`` prepended so they don't overwrite those from
    :func:`plot_qois`.
    """

    series = []

    if biharmonic_path is not None:
        series.append(("biharmonic", "k-", np.loadtxt(biharmonic_path)))

    if restricted_path is not None:
        series.append(("biharmonic, restricted test functions", "b--", np.loadtxt(restricted_path)))

    if reference_path is not None:
        series.append(("reference", "r:", _load_reference(reference_path)))

    _plot_series(series, output_dir, filename_prefix)


def plot_c0ip_comparison(
    biharmonic_path: str | os.PathLike | None = None,
    c0ip_path: str | os.PathLike | None = None,
    reference_path: str | os.PathLike | None = None,
    output_dir: str | os.PathLike = "output/figures",
    filename_prefix: str = "c0ip_compare_",
):
    """Plot drag, lift, and tip displacement over time for the standard
    biharmonic solver and the C0 interior penalty (C0IP) biharmonic solver,
    optionally together with the FSI2 reference data. Any of the three inputs
    may be omitted (``None``). Figures are written with ``filename_prefix``
    prepended so they don't overwrite those from :func:`plot_qois`.
    """

    series = []

    if biharmonic_path is not None:
        series.append(("biharmonic", "k-", np.loadtxt(biharmonic_path)))

    if c0ip_path is not None:
        series.append(("biharmonic, C0IP", "b--", np.loadtxt(c0ip_path)))

    if reference_path is not None:
        series.append(("reference", "r:", _load_reference(reference_path)))

    _plot_series(series, output_dir, filename_prefix)


def _load_reference(reference_path: str | os.PathLike) -> np.ndarray:
    """Load the FSI2 reference data into the same column layout as the
    solver qoi files: time, drag, lift, tip x- and y-displacement.
    """
    ref = np.loadtxt(reference_path)
    return np.column_stack(
        [
            ref[:, 0],  # time
            ref[:, 4] + ref[:, 6],  # drag
            -ref[:, 5] - ref[:, 7],  # lift
            ref[:, 10],  # tip x-displacement
            ref[:, 11],  # tip y-displacement
        ]
    )


def _plot_series(
    series: list[tuple[str, str, np.ndarray]],
    output_dir: str | os.PathLike,
    filename_prefix: str = "",
):
    """Write one PDF and one SVG figure per qoi in ``_QOI_SPECS``, each
    showing every ``(label, style, qois)`` entry in ``series``.
    """
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
        plt.savefig(output_dir / f"{filename_prefix}{stem}.pdf")
        with matplotlib.rc_context(_SVG_RC):
            plt.savefig(output_dir / f"{filename_prefix}{stem}.svg")
        plt.close()


def main():
    plot_qois(
        biharmonic_path="output/qoi/fsi2_biharm_qoi.txt",
        harmonic_path="output/qoi/fsi2_harm_dm_qoi.txt",
        reference_path="data/fsi2_reference.txt",
    )
    plot_restricted_comparison(
        biharmonic_path="output/qoi/fsi2_biharm_qoi.txt",
        restricted_path="output/qoi/fsi2_biharm_qoi_restr.txt",
        reference_path="data/fsi2_reference.txt",
    )
    plot_c0ip_comparison(
        biharmonic_path="output/qoi/fsi2_biharm_qoi.txt",
        c0ip_path="output/qoi/fsi2_biharm_c0ip.txt",
        reference_path="data/fsi2_reference.txt",
    )


if __name__ == "__main__":
    main()
