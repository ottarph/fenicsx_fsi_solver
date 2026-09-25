# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Mean, amplitude and frequency of FSI2 quantities of interest over a time window.

In the FSI2 benchmark convention a periodic signal is reported as
``mean +- amplitude [frequency]``, with ``mean = (max + min) / 2`` and
``amplitude = (max - min) / 2`` of the oscillation. Here the extrema are
taken per period (between successive upward crossings of the window mean,
with hysteresis against noise) and averaged over the complete periods in the
window; the frequency is the inverse mean period. Example::

    conda run --no-capture-output -n xfsi_solver python -m \\
        xfsi_solver.scripts.fsi2_qoi_statistics --window 12 15 \\
        output/fsi2_stiffened_elastic/<run>/qoi.txt output/qoi/fsi2_biharm_qoi.txt

prints a table with the published reference (``data/fsi2_reference.txt``,
available for t in [10, 14.62]) and the relative deviation of every input
from the first one.
"""

import argparse
import json
from pathlib import Path

import numpy as np

QUANTITIES = ("drag", "lift", "A_x", "A_y")
REFERENCE = "data/fsi2_reference.txt"


def load_qoi(path) -> dict:
    """``t`` and the quantities of a QoI file (``t, drag, lift, A_x, A_y``)."""
    data = np.loadtxt(path, ndmin=2)
    return {"t": data[:, 0], **{name: data[:, 1 + k] for k, name in enumerate(QUANTITIES)}}


def load_reference(path=REFERENCE) -> dict:
    """The published FSI2 point data: drag and lift are the sums of their two contributions."""
    data = np.loadtxt(path)
    return {"t": data[:, 0], "drag": data[:, 4] + data[:, 6], "lift": data[:, 5] + data[:, 7],
            "A_x": data[:, 10], "A_y": data[:, 11]}


def periodic_statistics(t: np.ndarray, y: np.ndarray, window: tuple[float, float]) -> dict:
    """Per-period mean, amplitude and frequency of ``y`` for ``t`` in ``window``."""
    mask = (t >= window[0]) & (t <= window[1])
    t, y = t[mask], y[mask]
    if t.size < 3:
        raise ValueError(f"No data in window {window}")
    level = 0.5 * (y.max() + y.min())
    # upward crossings of the mean level, each after the signal has fallen below
    # level - hysteresis (so that noise and secondary peaks do not add crossings)
    hysteresis = 0.25 * (y.max() - y.min())
    up, armed = [], False
    for k in range(y.size - 1):
        if y[k] < level - hysteresis:
            armed = True
        if armed and y[k] < level <= y[k + 1]:
            up.append(k)
            armed = False
    up = np.array(up, dtype=int)
    # linear interpolation of the crossing times
    crossings = t[up] + (level - y[up]) * (t[up + 1] - t[up]) / (y[up + 1] - y[up])
    if crossings.size < 2:
        return {"mean": float(level), "amplitude": float(0.5 * (y.max() - y.min())), "frequency": float("nan"),
                "periods": 0, "min": float(y.min()), "max": float(y.max())}
    highs, lows = [], []
    for a, b in zip(up[:-1], up[1:], strict=True):
        highs.append(y[a:b + 2].max())
        lows.append(y[a:b + 2].min())
    highs, lows = np.array(highs), np.array(lows)
    return {"mean": float(np.mean(0.5 * (highs + lows))), "amplitude": float(np.mean(0.5 * (highs - lows))),
            "frequency": float(1.0 / np.mean(np.diff(crossings))), "periods": int(crossings.size - 1),
            "min": float(lows.min()), "max": float(highs.max())}


def statistics(qoi: dict, window) -> dict:
    return {name: periodic_statistics(qoi["t"], qoi[name], window) for name in QUANTITIES}


def deviations(stats: dict, reference: dict) -> dict:
    """Deviations from ``reference``: of the mean relative to the amplitude, of amplitude and frequency relative."""
    out = {}
    for name in QUANTITIES:
        s, r = stats[name], reference[name]
        out[name] = {"mean": (s["mean"] - r["mean"]) / r["amplitude"],
                     "amplitude": (s["amplitude"] - r["amplitude"]) / r["amplitude"],
                     "frequency": (s["frequency"] - r["frequency"]) / r["frequency"]}
    return out


def format_table(rows: dict) -> str:
    lines = ["| run | " + " | ".join(QUANTITIES) + " |", "|---|" + "---|" * len(QUANTITIES)]
    for label, stats in rows.items():
        cells = []
        for name in QUANTITIES:
            s = stats[name]
            cells.append(f"{s['mean']:.4g} +- {s['amplitude']:.4g} [{s['frequency']:.4g}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("qoi", nargs="+", help="QoI files; deviations are relative to the first")
    parser.add_argument("--window", nargs=2, type=float, default=(12.0, 14.6))
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--json", default=None, help="write the statistics to this file")
    args = parser.parse_args()

    rows = {}
    if args.reference and Path(args.reference).exists():
        reference = load_reference(args.reference)
        if reference["t"][0] <= args.window[0] and reference["t"][-1] >= args.window[1]:
            rows["published reference"] = statistics(reference, args.window)
    for path in args.qoi:
        rows[path] = statistics(load_qoi(path), args.window)
    print(f"window {args.window}\n")
    print(format_table(rows))
    base = rows[args.qoi[0]]
    print("\ndeviation from", args.qoi[0], "(mean / amplitude, amplitude and frequency relative):")
    for label, stats in rows.items():
        if label != args.qoi[0]:
            print(label, json.dumps({k: {q: round(v, 4) for q, v in d.items()}
                                     for k, d in deviations(stats, base).items()}))
    if args.json:
        Path(args.json).write_text(json.dumps({"window": args.window, "statistics": rows}, indent=1))


if __name__ == "__main__":
    main()
