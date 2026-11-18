# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Plot the convergence studies of the moving-domain Taylor-Green vortex (``taylor_green_ale``).

Each study file ``<name>_<pair>.txt`` holds a row of n, dt, the velocity error and the pressure errors at
t^{n+1} and at t^{n+theta} for each run. The errors are plotted against h = 1/n for a name starting with
"spatial", and against dt otherwise, with dashed reference slopes through the last point.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_study(files, out_path, slopes=None, pressure_at_theta=False):
    """Plot the velocity and pressure errors of the study ``files`` side by side to ``out_path``.

    ``slopes`` maps an element pair to the reference slopes (velocity, pressure) drawn for it. The pressure
    error is that against the exact pressure at t^{n+1}, or with ``pressure_at_theta`` at t^{n+theta}.
    """
    pressure_column = 4 if pressure_at_theta else 3
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for path in files:
        path = Path(path)
        name, pair = path.stem.rsplit("_", 1)
        table = np.atleast_2d(np.loadtxt(path))
        spatial = name.startswith("spatial")
        step = 1.0 / table[:, 0] if spatial else table[:, 1]
        for k, (ax, column, field) in enumerate(zip(axes, (2, pressure_column), ("velocity", "pressure"), strict=True)):
            (line,) = ax.loglog(step, table[:, column], "o-", label=f"{pair} ({name})")
            if slopes is not None and pair in slopes:
                rate = slopes[pair][k]
                reference = table[-1, column] * (step / step[-1]) ** rate
                ax.loglog(step, reference, "--", color=line.get_color(), label=f"order {rate}")
            ax.set_xlabel("h" if spatial else "dt")
            ax.set_ylabel(f"L2 error of the {field}")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize="small")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--slope",
        nargs=3,
        action="append",
        metavar=("PAIR", "VELOCITY", "PRESSURE"),
        default=[],
        help="reference slopes for an element pair, e.g. --slope Q2Q1 3 2",
    )
    parser.add_argument("--pressure-at-theta", action="store_true", help="the pressure error at t^{n+theta}")
    args = parser.parse_args()
    slopes = {pair: (float(rate_v), float(rate_p)) for pair, rate_v, rate_p in args.slope}
    plot_study(args.files, args.out, slopes, args.pressure_at_theta)


if __name__ == "__main__":
    main()
