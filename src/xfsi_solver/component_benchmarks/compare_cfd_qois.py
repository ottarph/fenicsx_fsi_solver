# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Print the comparison quantities of the CFD tests next to the reference values of Turek & Hron.

The quantities are the drag and lift on the cylinder and the flag together:

- CFD1 and CFD2 are steady, so they are compared by the forces of the last (converged) row.
- CFD3 oscillates, so it is compared by the mean, amplitude and frequency over the last full period of the lift,
  with the drag taken over the same period.

The lift is shown with the sign of the benchmark proposal: drag_lift_forms in xfsi_solver.tools.qoi returns minus
that, so the lift column of the QoI files is negated when loaded.

Run from the repository root, after the CFD solver has written its QoI files:

    python -m xfsi_solver.component_benchmarks.compare_cfd_qois
    python -m xfsi_solver.component_benchmarks.compare_cfd_qois --suffix _fine   # reads cfd_1_fine.txt, ...
"""

import argparse
from pathlib import Path

import numpy as np

from xfsi_solver.component_benchmarks.qoi_analysis import last_period, mean_and_amplitude, print_header, print_row
from xfsi_solver.component_benchmarks.references import TUREK_HRON_2006_SHORT, print_references

QOI_DIR = Path("output/qoi/component_benchmarks")

TESTS = ("CFD1", "CFD2", "CFD3")

# Reference values of Turek & Hron (2006), see references.py: the "ref." rows of Tables 5, 6 and 7, forces per unit
# depth and Hz.
REFERENCE = {
    "CFD1": {"drag": 14.29, "lift": 1.119},
    "CFD2": {"drag": 136.7, "lift": 10.53},
    "CFD3": {
        "drag_mean": 439.45,
        "drag_amplitude": 5.6183,
        "lift_mean": -11.893,
        "lift_amplitude": 437.81,
        "frequency": 4.3956,
    },
}

# The table of Turek & Hron (2006) with the results of each test.
REFERENCE_TABLE = {"CFD1": 5, "CFD2": 6, "CFD3": 7}


def qoi_path(test, suffix=""):
    """The QoI file of a test, as written by cfd_benchmark.py: CFD1 -> cfd_1<suffix>.txt."""
    return QOI_DIR / f"cfd_{test[-1]}{suffix}.txt"


def load_qois(path):
    """Time, drag and lift (with the sign of the benchmark proposal) from a QoI file of cfd_benchmark.py.

    Its columns are t, drag, minus the lift, and two unused columns.
    """
    data = np.loadtxt(path)
    t = data[:, 0]
    drag = data[:, 1]
    lift = -data[:, 2]
    return t, drag, lift


def compare_steady(test, path):
    t, drag, lift = load_qois(path)
    reference = REFERENCE[test]
    last_change = np.hypot(drag[-1] - drag[-2], lift[-1] - lift[-2])

    print(f"{test}: steady, last of {len(t)} rows ({path})")
    print(f"  Compared with the reference values of {TUREK_HRON_2006_SHORT}, Table {REFERENCE_TABLE[test]}.")
    print_header()
    print_row("drag", drag[-1], reference["drag"])
    print_row("lift", lift[-1], reference["lift"])
    print(f"  change over the last step: {last_change:.2e}")


def compare_periodic(test, path):
    t, drag, lift = load_qois(path)
    reference = REFERENCE[test]
    start, end = last_period(t, lift)
    in_period = (t >= start) & (t <= end)
    drag_mean, drag_amplitude = mean_and_amplitude(drag[in_period])
    lift_mean, lift_amplitude = mean_and_amplitude(lift[in_period])
    frequency = 1.0 / (end - start)

    print(f"{test}: last full period, t = {start:.3f} s to {end:.3f} s (run ends at t = {t[-1]:g} s, {path})")
    print(f"  Compared with the reference values of {TUREK_HRON_2006_SHORT}, Table {REFERENCE_TABLE[test]}.")
    print_header()
    print_row("drag, mean", drag_mean, reference["drag_mean"])
    print_row("drag, amplitude", drag_amplitude, reference["drag_amplitude"])
    print_row("lift, mean", lift_mean, reference["lift_mean"])
    print_row("lift, amplitude", lift_amplitude, reference["lift_amplitude"])
    print_row("frequency [Hz]", frequency, reference["frequency"])


def compare_all(suffix=""):
    """Print the comparison of every CFD test whose QoI file cfd_<n><suffix>.txt exists."""
    for test in TESTS:
        path = qoi_path(test, suffix)
        if not path.is_file():
            print(f"{test}: no QoI file at {path}\n")
            continue
        if test == "CFD3":
            compare_periodic(test, path)
        else:
            compare_steady(test, path)
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suffix", default="", help="suffix of the QoI file names, e.g. _fine for cfd_1_fine.txt")
    compare_all(parser.parse_args().suffix)
    print_references()


if __name__ == "__main__":
    main()
