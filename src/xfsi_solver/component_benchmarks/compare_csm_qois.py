# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Print the comparison quantities of the CSM tests next to the reference values of Turek & Hron.

The quantities are those of Section 3 of the benchmark proposal, for the displacement of the point A at the tip of
the beam:

- CSM1 and CSM2 are static, so they are compared by the displacement of the last (converged) row.
- CSM3 oscillates, so it is compared by the mean, amplitude and frequency over the last full period:
  mean = (max + min) / 2, amplitude = (max - min) / 2, frequency = 1 / period.

Run from the repository root, after the CSM solver has written its QoI files:

    python -m xfsi_solver.component_benchmarks.compare_csm_qois
"""

from pathlib import Path

import numpy as np

QOI_DIR = Path("output/qoi/component_benchmarks")

# The QoI file of each test, as written by csm_benchmark.py.
QOI_FILES = {
    "CSM1": QOI_DIR / "csm_1.txt",
    "CSM2": QOI_DIR / "csm_2.txt",
    "CSM3": QOI_DIR / "csm_3.txt",
}

# Reference values ("ref." rows of Tables 9, 10 and 11), in 10^-3 m and Hz.
REFERENCE = {
    "CSM1": {"ux": -7.187, "uy": -66.10},
    "CSM2": {"ux": -0.4690, "uy": -16.97},
    "CSM3": {
        "ux_mean": -14.305,
        "ux_amplitude": 14.305,
        "uy_mean": -63.607,
        "uy_amplitude": 65.160,
        "frequency": 1.0995,
    },
}


def load_qois(path):
    """Time and displacement of the point A, in s and 10^-3 m, from a QoI file of csm_benchmark.py.

    Its columns are t, two unused columns, A_x and A_y.
    """
    data = np.loadtxt(path)
    t = data[:, 0]
    ux = data[:, 3] * 1e3
    uy = data[:, 4] * 1e3
    return t, ux, uy


def times_of_minima(t, y):
    """Times of the local minima of y, each refined by fitting a parabola through it and its two neighbours."""
    dt = t[1] - t[0]
    times = []
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] <= y[i + 1]:
            before, at, after = y[i - 1], y[i], y[i + 1]
            shift = 0.5 * (before - after) / (before - 2 * at + after)
            times.append(t[i] + shift * dt)
    return np.array(times)


def last_period(t, uy):
    """Start and end time of the last full period of the oscillation, from one minimum of uy to the next."""
    minima = times_of_minima(t, uy)
    if len(minima) < 2:
        raise ValueError("Fewer than two minima of uy: the run does not cover a full period yet")
    return minima[-2], minima[-1]


def mean_and_amplitude(y):
    return 0.5 * (y.max() + y.min()), 0.5 * (y.max() - y.min())


def relative_difference(computed, reference):
    """Difference in percent of the reference, negative when computed is smaller in magnitude than reference."""
    return 100.0 * (computed - reference) / reference


def print_row(name, computed, reference):
    print(f"  {name:22s} {computed:12.4f} {reference:12.4f} {relative_difference(computed, reference):+11.2f} %")


def print_header():
    print(f"  {'':22s} {'computed':>12s} {'reference':>12s} {'difference':>13s}")


def compare_static(test, path):
    t, ux, uy = load_qois(path)
    reference = REFERENCE[test]
    last_change = np.hypot(ux[-1] - ux[-2], uy[-1] - uy[-2])

    print(f"{test}: static, last of {len(t)} rows (t = {t[-1]:g} s)")
    print_header()
    print_row("ux of A [1e-3 m]", ux[-1], reference["ux"])
    print_row("uy of A [1e-3 m]", uy[-1], reference["uy"])
    print(f"  change over the last step: {last_change:.2e} x 1e-3 m")


def compare_periodic(test, path):
    t, ux, uy = load_qois(path)
    reference = REFERENCE[test]
    start, end = last_period(t, uy)
    in_period = (t >= start) & (t <= end)
    ux_mean, ux_amplitude = mean_and_amplitude(ux[in_period])
    uy_mean, uy_amplitude = mean_and_amplitude(uy[in_period])
    frequency = 1.0 / (end - start)

    print(f"{test}: last full period, t = {start:.3f} s to {end:.3f} s (run ends at t = {t[-1]:g} s)")
    print_header()
    print_row("ux of A, mean [1e-3 m]", ux_mean, reference["ux_mean"])
    print_row("ux of A, amplitude", ux_amplitude, reference["ux_amplitude"])
    print_row("uy of A, mean [1e-3 m]", uy_mean, reference["uy_mean"])
    print_row("uy of A, amplitude", uy_amplitude, reference["uy_amplitude"])
    print_row("frequency [Hz]", frequency, reference["frequency"])


def main():
    for test, path in QOI_FILES.items():
        if not path.is_file():
            print(f"{test}: no QoI file at {path}\n")
            continue
        if test == "CSM3":
            compare_periodic(test, path)
        else:
            compare_static(test, path)
        print()


if __name__ == "__main__":
    main()
