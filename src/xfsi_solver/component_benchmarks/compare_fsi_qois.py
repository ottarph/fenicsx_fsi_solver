# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Print the comparison quantities of the FSI tests next to published values.

The quantities are the displacement of the point A at the tip of the flag, and the drag and lift on the cylinder and
the flag together:

- FSI1 is steady, so it is compared by the last (converged) row.
- FSI2 and FSI3 oscillate, so they are compared by the mean, amplitude and frequency over the last full period of the
  y-displacement uy of A. The x-displacement ux and the drag oscillate twice in that period, since the flag passes its
  middle position twice, so their frequency is twice that of uy and the lift.

The published values (see references.py):

- FSI1 and FSI3 are compared with the results of seven groups in Turek et al. (2010), Tables 2 and 3: with the mean
  over the finest result of each method, with its range over the methods, and with the finest results of the
  monolithic finite element methods 2a, 2b (Rannacher) and 3 (Turek/Hron), see turek_2010_fsi_results.py.
- FSI2, which Turek et al. (2010) do not report, is compared with the reference values of Turek & Hron (2006),
  Table 14.

The lift is shown with the sign of the benchmark proposal: drag_lift_forms in xfsi_solver.tools.qoi returns minus
that, so the lift column of the QoI files is negated when loaded.

Run from the repository root, after the FSI solver has written its QoI files:

    python -m xfsi_solver.component_benchmarks.compare_fsi_qois
    python -m xfsi_solver.component_benchmarks.compare_fsi_qois --suffix _coarse   # reads fsi_1_coarse.txt, ...
"""

import argparse
from pathlib import Path

import numpy as np

from xfsi_solver.component_benchmarks import turek_2010_fsi_results as turek_2010
from xfsi_solver.component_benchmarks.qoi_analysis import difference_cell, last_period, mean_and_amplitude, print_table
from xfsi_solver.component_benchmarks.references import (
    TUREK_ET_AL_2010_SHORT,
    TUREK_HRON_2006_SHORT,
    print_references,
)

QOI_DIR = Path("output/qoi/component_benchmarks")

TESTS = ("FSI1", "FSI2", "FSI3")
QUANTITIES = ("ux", "uy", "drag", "lift")
PERIODIC_PARTS = ("mean", "amplitude", "frequency [Hz]")

# Reference values of FSI2 ("ref." row of Table 14 of Turek & Hron (2006)): displacements in 10^-3 m, forces per unit
# depth, frequencies in Hz, each quantity as (mean, amplitude, frequency).
TUREK_HRON_2006_FSI2 = {
    "ux": (-14.58, 12.44, 3.8),
    "uy": (1.23, 80.6, 2.0),
    "drag": (208.83, 73.75, 3.8),
    "lift": (0.88, 234.2, 2.0),
}

# Number of oscillations of each quantity in one period of uy.
OSCILLATIONS_PER_PERIOD = {"ux": 2, "uy": 1, "drag": 2, "lift": 1}


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


def is_periodic(test):
    return test != "FSI1"


def computed_result(test, t, quantities):
    """The comparison quantities of a run: the last value of each quantity for a steady test, and (mean, amplitude,
    frequency) over the last full period of uy for a periodic test, with that period as (start, end)."""
    if not is_periodic(test):
        return {name: values[-1] for name, values in quantities.items()}, None
    start, end = last_period(t, quantities["uy"])
    in_period = (t >= start) & (t <= end)
    result = {}
    for name, values in quantities.items():
        mean, amplitude = mean_and_amplitude(values[in_period])
        result[name] = (mean, amplitude, OSCILLATIONS_PER_PERIOD[name] / (end - start))
    return result, (start, end)


def finest_results(test):
    """{method: its finest result} of the methods of Turek et al. (2010) with results for test (FSI1 or FSI3)."""
    if test == "FSI1":
        return {method: turek_2010.fsi1_finest(method) for method in turek_2010.FSI1_TABLE}
    return {method: turek_2010.fsi3_finest(method) for method in turek_2010.FSI3_TABLE}


def mean_over_methods(test):
    """The mean over the finest results of the methods of Turek et al. (2010), in the structure of a result."""
    statistics = turek_2010.statistics_over_methods(finest_results(test))
    if is_periodic(test):
        return {name: tuple(None if part is None else part[0] for part in value) for name, value in statistics.items()}
    return {name: value[0] for name, value in statistics.items()}


def published_results(test):
    """The published results a test is compared with, as {label: result}, each result like computed_result's, with
    None for the values not given. Also used for the reference lines in plot_fsi_qois.py."""
    if test == "FSI2":
        return {TUREK_HRON_2006_SHORT: TUREK_HRON_2006_FSI2}
    results = {f"{TUREK_ET_AL_2010_SHORT}, mean of methods": mean_over_methods(test)}
    for method in turek_2010.MONOLITHIC_FEM_METHODS:
        group = turek_2010.METHODS[method][0]
        results[f"{TUREK_ET_AL_2010_SHORT}, method {method} ({group})"] = finest_results(test)[method]
    return results


def rows(result, periodic):
    """{row name: value} of a result, one row per quantity, or per mean, amplitude and frequency of each quantity."""
    flat = {}
    for name in QUANTITIES:
        value = result.get(name)
        if periodic:
            for part, part_value in zip(PERIODIC_PARTS, value or (None,) * 3, strict=True):
                flat[f"{name}, {part}"] = part_value
        else:
            flat[name] = value
    return flat


def compare_with_turek_hron_2006(computed, periodic):
    reference = rows(TUREK_HRON_2006_FSI2, periodic)
    print(f"  Compared with the reference values of {TUREK_HRON_2006_SHORT}, Table 14.")
    print_table(
        ["computed", "reference", "difference"],
        [(name, [value, reference[name], difference_cell(value, reference[name])]) for name, value in computed.items()],
    )


def compare_with_turek_et_al_2010(test, computed, periodic):
    finest = finest_results(test)
    methods = list(finest)
    statistics = rows(turek_2010.statistics_over_methods(finest), periodic)
    table_number = 2 if test == "FSI1" else 3

    print(f"  Compared with {TUREK_ET_AL_2010_SHORT}, Table {table_number}: the mean, minimum and maximum over the")
    print(f"  finest result of each of the methods {', '.join(methods)}.")
    table_rows = []
    for name, value in computed.items():
        mean, minimum, maximum = statistics[name] or (None, None, None)
        table_rows.append((name, [value, mean, minimum, maximum, difference_cell(value, mean)]))
    print_table(["computed", "mean", "minimum", "maximum", "vs mean"], table_rows)

    monolithic = {method: rows(finest[method], periodic) for method in turek_2010.MONOLITHIC_FEM_METHODS}
    print(
        f"\n  Compared with {TUREK_ET_AL_2010_SHORT}, Table {table_number}: the finest results of the monolithic "
        "finite element methods"
    )
    for method in monolithic:
        group, approach = turek_2010.METHODS[method]
        print(f"    {method} ({group}): {approach}")
    table_rows = [
        (
            name,
            [
                value,
                *(monolithic[method][name] for method in monolithic),
                difference_cell(value, monolithic["3"][name]),
            ],
        )
        for name, value in computed.items()
    ]
    print_table(["computed", *(f"method {method}" for method in monolithic), "vs method 3"], table_rows)


def compare(test, path):
    t, quantities = load_qois(path)
    result, period = computed_result(test, t, quantities)
    periodic = is_periodic(test)
    computed = rows(result, periodic)

    if periodic:
        start, end = period
        print(f"{test}: last full period of uy, t = {start:.3f} s to {end:.3f} s (run ends at t = {t[-1]:g} s, {path})")
    else:
        print(f"{test}: steady, last of {len(t)} rows ({path})")
    print("  Displacements of A in 1e-3 m, lift with the sign of the benchmark proposal.")

    if test == "FSI2":
        compare_with_turek_hron_2006(computed, periodic)
    else:
        compare_with_turek_et_al_2010(test, computed, periodic)


def compare_all(suffix=""):
    """Print the comparison of every FSI test whose QoI file fsi_<n><suffix>.txt exists."""
    for test in TESTS:
        path = qoi_path(test, suffix)
        if not path.is_file():
            print(f"{test}: no QoI file at {path}\n")
            continue
        compare(test, path)
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suffix", default="", help="suffix of the QoI file names, e.g. _coarse for fsi_1_coarse.txt")
    compare_all(parser.parse_args().suffix)
    print_references()


if __name__ == "__main__":
    main()
