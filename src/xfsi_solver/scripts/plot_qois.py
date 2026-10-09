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

# The same for the 3d benchmark, whose qoi files have the displacement of the point B in
# columns 3 to 5.
_QOI_SPECS_3D = [
    (1, "Drag", "drag"),
    (2, "Lift", "lift"),
    (3, "$B$ $x$-displacement [m]", "x-disp"),
    (4, "$B$ $y$-displacement [m]", "y-disp"),
    (5, "$B$ $z$-displacement [m]", "z-disp"),
]

# Reference values of the 3d benchmark, as (mean, amplitude) over t in [9, 10], on mesh
# level 3 with k = 0.001 (Failer & Richter, J. Sci. Comput. 82:28, 2020, Fig. 3), by
# column of the qoi files.
_REFERENCE_3D = {
    1: (185.5, 3.5),  # drag
    2: (-0.717, 41.299),  # lift
    3: (-2.143e-3, 2.383e-3),  # B x-displacement
    4: (2.699e-3, 25.594e-3),  # B y-displacement
    5: (0.486e-3, 0.877e-3),  # B z-displacement
}
_REFERENCE_3D_INTERVAL = (9.0, 10.0)


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

    if (qois_bih := _load_qois(biharmonic_path)) is not None:
        series.append(("biharmonic", "k-", qois_bih))

    if (qois_harm := _load_qois(harmonic_path)) is not None:
        series.append(("harmonic", "b--", qois_harm))

    if (reference := _load_reference(reference_path)) is not None:
        series.append(("reference", "r:", reference))

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

    if (qois_bih := _load_qois(biharmonic_path)) is not None:
        series.append(("biharmonic", "k-", qois_bih))

    if (qois_restr := _load_qois(restricted_path)) is not None:
        series.append(("biharmonic, restricted test functions", "b--", qois_restr))

    if (reference := _load_reference(reference_path)) is not None:
        series.append(("reference", "r:", reference))

    _plot_series(series, output_dir, filename_prefix)


def plot_3d_qois(
    qoi_path: str | os.PathLike | None = None,
    output_dir: str | os.PathLike = "output/figures",
    filename_prefix: str = "fsi3d_",
):
    """Plot drag, lift, and the displacement of the point B over time for the 3d
    benchmark, with the reference ranges of Failer & Richter over t in [9, 10]. The
    qoi file may still be written by a running simulation. Figures are written with
    ``filename_prefix`` prepended so they don't overwrite the 2d figures.
    """
    qois = _load_qois(qoi_path)
    if qois is None:
        return
    _plot_series([("3d benchmark", "k-", qois)], output_dir, filename_prefix, _QOI_SPECS_3D, _REFERENCE_3D)


def _load_qois(qoi_path: str | os.PathLike | None) -> np.ndarray | None:
    """Load a solver qoi file, or return ``None`` with a message if there is no path, no
    file, or no rows to plot yet. A last line that a running simulation is still writing
    is dropped.
    """
    if qoi_path is None:
        return None
    if not Path(qoi_path).is_file():
        print(f"Skipping {qoi_path}: file not found")
        return None
    rows = [line.split() for line in Path(qoi_path).read_text().splitlines() if line and not line.startswith("#")]
    if rows and len(rows[-1]) != len(rows[0]):
        rows = rows[:-1]
    if len(rows) < 2:
        print(f"Skipping {qoi_path}: fewer than two rows to plot")
        return None
    return np.array(rows, dtype=float)


def _load_reference(reference_path: str | os.PathLike | None) -> np.ndarray | None:
    """Load the FSI2 reference data into the same column layout as the
    solver qoi files: time, drag, lift, tip x- and y-displacement, or
    return ``None`` with a message if there is no path or no file.
    """
    if reference_path is None:
        return None
    if not Path(reference_path).is_file():
        print(f"Skipping {reference_path}: file not found")
        return None
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
    qoi_specs: list[tuple[int, str, str]] = _QOI_SPECS,
    reference_ranges: dict[int, tuple[float, float]] | None = None,
):
    """Write one PDF and one SVG figure per qoi in ``qoi_specs``, each
    showing every ``(label, style, qois)`` entry in ``series``, and the
    ``(mean, amplitude)`` of ``reference_ranges`` over
    ``_REFERENCE_3D_INTERVAL`` as horizontal lines at the minimum and the
    maximum. Without any series, no figures are written.
    """
    if not series:
        print(f"Nothing to plot for {Path(output_dir) / filename_prefix}*")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for col, ylabel, stem in qoi_specs:
        plt.figure(figsize=(10, 5))
        for label, style, qois in series:
            plt.plot(qois[:, 0], qois[:, col], style, label=label)
        if reference_ranges is not None:
            # Plotted as lines, not with hlines, so that the legend avoids them.
            mean, amplitude = reference_ranges[col]
            plt.plot(
                _REFERENCE_3D_INTERVAL, [mean + amplitude] * 2, "r--", label="Failer & Richter (2020), min and max"
            )
            plt.plot(_REFERENCE_3D_INTERVAL, [mean - amplitude] * 2, "r--")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
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
    plot_3d_qois(qoi_path="output/qoi/fsi3d_iterative_qoi_coarse.txt")
    plot_3d_qois(qoi_path="output/qoi/fsi3d_iterative_qoi_medium.txt", filename_prefix="fsi3d_medium_")


if __name__ == "__main__":
    main()
