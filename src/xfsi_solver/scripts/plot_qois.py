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

# The FSI2i benchmark, with an incompressible solid, has the violation of det(F) = 1 in the solid
# in columns 5 to 7.
_QOI_SPECS_INCOMPRESSIBLE = _QOI_SPECS + [
    (5, "Relative solid area change", "solid-area-change"),
    (6, r"RMS of $\det F - 1$ in the solid", "det-F-rms"),
    (7, r"Maximum of $|\det F - 1|$ in the solid", "det-F-max"),
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
_REFERENCE_3D_LABEL = "Failer & Richter (2020), min and max"

# Reference values of the FSI2i benchmark, as (mean, amplitude) of the last computed oscillation,
# on mesh level J = 5 with the GCsI2(E, I) scheme and tau = 0.0025 (Wichrowski et al., Int. J.
# Numer. Methods Eng. 124 (2023), Table 5), by column of the qoi files. There are none for the drag
# and the lift. The time of the last oscillation is not given, so they are drawn over the last
# _REFERENCE_INCOMPRESSIBLE_DURATION seconds of the run.
_REFERENCE_INCOMPRESSIBLE = {
    3: (-14.72e-3, 12.55e-3),  # A x-displacement
    4: (1.23e-3, 80.77e-3),  # A y-displacement
}
_REFERENCE_INCOMPRESSIBLE_DURATION = 2.0
_REFERENCE_INCOMPRESSIBLE_LABEL = r"Wichrowski et al. (2023), $\tau = 0.0025$, min and max"


QOI_DIR = Path("output/qoi")
FIGURE_DIR = Path("output/figures/qoi")
FSI2_REFERENCE = Path("data/fsi2_reference.txt")

_STYLES = ["k-", "b--", "g-.", "m:"]


def solver_script(run: str) -> str:
    """The solver script that wrote the QoI file ``<run>.txt``: the longest name of a module in
    xfsi_solver.solvers that ``run`` starts with, as in ``threeDfsi_mixed`` for
    ``threeDfsi_mixed_medium``."""
    import pkgutil

    import xfsi_solver.solvers

    scripts = [m.name for m in pkgutil.iter_modules(xfsi_solver.solvers.__path__)]
    matches = [name for name in scripts if run == name or run.startswith(name + "_")]
    if not matches:
        raise ValueError(f"{run!r} does not start with the name of a solver script in xfsi_solver.solvers")
    return max(matches, key=len)


def figure_prefix(runs: list[str], figure_dir: str | os.PathLike = FIGURE_DIR) -> tuple[Path, str]:
    """The folder and file-name prefix of the QoI figures of ``runs``: the folder of the solver
    script of the first run, and as prefix the rest of its name (such as the mesh), followed by
    ``compare_<other runs>_`` when there are other runs."""
    script = solver_script(runs[0])
    parts = [runs[0][len(script) + 1 :]] if runs[0] != script else []
    if runs[1:]:
        parts.append("compare_" + "_".join(runs[1:]))
    return Path(figure_dir) / script, "".join(part + "_" for part in parts)


def plot_comparison(
    runs: list[str],
    reference_path: str | os.PathLike | None = FSI2_REFERENCE,
    qoi_dir: str | os.PathLike = QOI_DIR,
    figure_dir: str | os.PathLike = FIGURE_DIR,
):
    """Plot drag, lift, and tip displacement over time for the FSI2 runs whose QoI files are
    ``<qoi_dir>/<run>.txt``, together with the FSI2 reference data, in the folder of the solver
    script of the first run (see :func:`figure_prefix`). Runs without a QoI file, and the
    reference with ``reference_path=None``, are left out.
    """
    series = []
    for run, style in zip(runs, _STYLES, strict=False):
        if (qois := _load_qois(Path(qoi_dir) / f"{run}.txt")) is not None:
            series.append((run, style, qois))
    if (reference := _load_reference(reference_path)) is not None:
        series.append(("reference", "r:", reference))
    _plot_series(series, *figure_prefix(runs, figure_dir))


def plot_3d_qois(
    run: str,
    qoi_dir: str | os.PathLike = QOI_DIR,
    figure_dir: str | os.PathLike = FIGURE_DIR,
):
    """Plot drag, lift, and the displacement of the point B over time for a run of the 3d
    benchmark, with the reference ranges of Failer & Richter over t in [9, 10], in the folder of
    its solver script. The qoi file may still be written by a running simulation.
    """
    qois = _load_qois(Path(qoi_dir) / f"{run}.txt")
    if qois is None:
        return
    _plot_series(
        [(run, "k-", qois)],
        *figure_prefix([run], figure_dir),
        _QOI_SPECS_3D,
        _REFERENCE_3D,
        _REFERENCE_3D_INTERVAL,
        _REFERENCE_3D_LABEL,
    )


def plot_incompressible_qois(
    run: str,
    qoi_dir: str | os.PathLike = QOI_DIR,
    figure_dir: str | os.PathLike = FIGURE_DIR,
):
    """Plot drag, lift, tip displacement and the violation of det(F) = 1 in the solid over time for
    a run of the FSI2i benchmark, with an incompressible solid, in the folder of its solver script,
    with the reference ranges of the tip displacement of Wichrowski et al. over the last
    ``_REFERENCE_INCOMPRESSIBLE_DURATION`` seconds of the run. The violation is left out for QoI
    files written before it was measured. The qoi file may still be written by a running simulation.
    """
    qois = _load_qois(Path(qoi_dir) / f"{run}.txt")
    if qois is None:
        return
    specs = [spec for spec in _QOI_SPECS_INCOMPRESSIBLE if spec[0] < qois.shape[1]]
    t_end = qois[-1, 0]
    _plot_series(
        [(run, "k-", qois)],
        *figure_prefix([run], figure_dir),
        specs,
        _REFERENCE_INCOMPRESSIBLE,
        (max(t_end - _REFERENCE_INCOMPRESSIBLE_DURATION, qois[0, 0]), t_end),
        _REFERENCE_INCOMPRESSIBLE_LABEL,
    )


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
    folder: Path,
    prefix: str,
    qoi_specs: list[tuple[int, str, str]] = _QOI_SPECS,
    reference_ranges: dict[int, tuple[float, float]] | None = None,
    reference_interval: tuple[float, float] | None = None,
    reference_label: str | None = None,
):
    """Write one PDF and one SVG figure per qoi in ``qoi_specs``, to ``<folder>/<prefix><stem>.pdf``
    and ``.svg``, each showing every ``(label, style, qois)`` entry in ``series``, and the
    ``(mean, amplitude)`` of ``reference_ranges`` over ``reference_interval`` as horizontal
    lines at the minimum and the maximum, labelled ``reference_label``, for the qois that have
    one. Without any series, no figures are written.
    """
    if not series:
        print(f"Nothing to plot for {folder}/{prefix}*")
        return

    folder.mkdir(parents=True, exist_ok=True)

    for col, ylabel, stem in qoi_specs:
        plt.figure(figsize=(10, 5))
        for label, style, qois in series:
            plt.plot(qois[:, 0], qois[:, col], style, label=label)
        if reference_ranges is not None and col in reference_ranges:
            # Plotted as lines, not with hlines, so that the legend avoids them.
            mean, amplitude = reference_ranges[col]
            plt.plot(reference_interval, [mean + amplitude] * 2, "r--", label=reference_label)
            plt.plot(reference_interval, [mean - amplitude] * 2, "r--")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(folder / f"{prefix}{stem}.pdf")
        with matplotlib.rc_context(_SVG_RC):
            plt.savefig(folder / f"{prefix}{stem}.svg")
        plt.close()
    print(f"Saved {folder}/{prefix}*.pdf and .svg")


def main():
    # Every FSI2 run on its own, with the reference, the FSI2i runs on their own, and the 3d runs
    # with their reference ranges.
    for qoi_path in sorted(QOI_DIR.glob("*.txt")):
        try:
            script = solver_script(qoi_path.stem)
        except ValueError:
            continue  # a QoI file of a solver that no longer exists
        if script == "fsi2_biharmonic_mixed_incompressible":
            plot_incompressible_qois(qoi_path.stem)
        elif script.startswith("fsi2_"):
            plot_comparison([qoi_path.stem])
        elif script.startswith("threeDfsi_"):
            plot_3d_qois(qoi_path.stem)

    # Comparisons between the solvers
    plot_comparison(["fsi2_biharmonic_simple", "fsi2_harmonic"])
    plot_comparison(["fsi2_biharmonic_fulljacobian", "fsi2_biharmonic_simple"])
    plot_comparison(["fsi2_biharmonic_c0ip", "fsi2_biharmonic_simple"])
    plot_comparison(["fsi2_biharmonic_c0ip", "fsi2_biharmonic_mixed"])


if __name__ == "__main__":
    main()
