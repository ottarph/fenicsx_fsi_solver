# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Print the comparison quantities of the FSI tests next to the reference values of Turek & Hron.

The quantities are the displacement of the point A at the tip of the flag, and the drag and lift on the cylinder and
the flag together:

- FSI1 is steady, so it is compared by the last (converged) row.
- FSI2 and FSI3 oscillate, so they are compared by the mean, amplitude and frequency over the last full period of the
  y-displacement uy of A. The x-displacement ux and the drag oscillate twice in that period, since the flag passes its
  middle position twice, so their frequency is twice that of uy and the lift.

The lift is shown with the sign of the benchmark proposal: drag_lift_forms in xfsi_solver.tools.qoi returns minus
that, so the lift column of the QoI files is negated when loaded.

Run from the repository root, after the FSI solver has written its QoI files:

    python -m xfsi_solver.component_benchmarks.compare_fsi_qois
    python -m xfsi_solver.component_benchmarks.compare_fsi_qois --suffix _coarse   # reads fsi_1_coarse.txt, ...
"""

import argparse
from pathlib import Path

import numpy as np

from xfsi_solver.component_benchmarks.qoi_analysis import last_period, mean_and_amplitude, print_header, print_row

QOI_DIR = Path("output/qoi/component_benchmarks")

TESTS = ("FSI1", "FSI2", "FSI3")

# Reference values ("ref." rows of Tables 13, 14 and 15): displacements in 10^-3 m, forces per unit depth, Hz.
# Each periodic quantity is (mean, amplitude, frequency).
REFERENCE = {
    "FSI1": {"ux": 0.0227, "uy": 0.8209, "drag": 14.295, "lift": 0.7638},
    "FSI2": {
        "ux": (-14.58, 12.44, 3.8),
        "uy": (1.23, 80.6, 2.0),
        "drag": (208.83, 73.75, 3.8),
        "lift": (0.88, 234.2, 2.0),
    },
    "FSI3": {
        "ux": (-2.69, 2.53, 10.9),
        "uy": (1.48, 34.38, 5.3),
        "drag": (457.3, 22.66, 10.9),
        "lift": (2.22, 149.78, 5.3),
    },
}

# Number of oscillations of each quantity in one period of uy.
OSCILLATIONS_PER_PERIOD = {"ux": 2, "uy": 1, "drag": 2, "lift": 1}

LABELS = {"ux": "ux of A [1e-3 m]", "uy": "uy of A [1e-3 m]", "drag": "drag", "lift": "lift"}


def qoi_path(test, suffix=""):
    """The QoI file of a test, as written by fsi2D_benchmark.py: FSI1 -> fsi_1<suffix>.txt."""
    return QOI_DIR / f"fsi_{test[-1]}{suffix}.txt"


def load_qois(path):
    """Time [s] and the quantities ux, uy [1e-3 m], drag and lift (with the sign of the benchmark proposal) as a dict.

    The columns of a QoI file of fsi2D_benchmark.py are t, drag, minus the lift, A_x and A_y.
    """
    data = np.loadtxt(path)
    t = data[:, 0]
    quantities = {
        "ux": data[:, 3] * 1e3,
        "uy": data[:, 4] * 1e3,
        "drag": data[:, 1],
        "lift": -data[:, 2],
    }
    return t, quantities


def compare_steady(test, path):
    t, quantities = load_qois(path)
    reference = REFERENCE[test]

    print(f"{test}: steady, last of {len(t)} rows ({path})")
    print_header()
    for name, values in quantities.items():
        print_row(LABELS[name], values[-1], reference[name])


def compare_periodic(test, path):
    t, quantities = load_qois(path)
    reference = REFERENCE[test]
    start, end = last_period(t, quantities["uy"])
    in_period = (t >= start) & (t <= end)

    print(f"{test}: last full period of uy, t = {start:.3f} s to {end:.3f} s (run ends at t = {t[-1]:g} s, {path})")
    print_header()
    for name, values in quantities.items():
        mean, amplitude = mean_and_amplitude(values[in_period])
        frequency = OSCILLATIONS_PER_PERIOD[name] / (end - start)
        reference_mean, reference_amplitude, reference_frequency = reference[name]
        print_row(f"{LABELS[name]}, mean", mean, reference_mean)
        print_row(f"{name}, amplitude", amplitude, reference_amplitude)
        print_row(f"{name}, frequency [Hz]", frequency, reference_frequency)


def compare_all(suffix=""):
    """Print the comparison of every FSI test whose QoI file fsi_<n><suffix>.txt exists."""
    for test in TESTS:
        path = qoi_path(test, suffix)
        if not path.is_file():
            print(f"{test}: no QoI file at {path}\n")
            continue
        if test == "FSI1":
            compare_steady(test, path)
        else:
            compare_periodic(test, path)
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suffix", default="", help="suffix of the QoI file names, e.g. _coarse for fsi_1_coarse.txt")
    compare_all(parser.parse_args().suffix)


if __name__ == "__main__":
    main()
