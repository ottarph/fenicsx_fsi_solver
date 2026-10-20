# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Helpers shared by the scripts that compare the component benchmarks with the reference values of Turek & Hron.

The comparison quantities of a periodic test are taken over its last full period, as in Section 3 of the benchmark
proposal: mean = (max + min) / 2, amplitude = (max - min) / 2, frequency = 1 / period.
"""

import numpy as np


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


def last_period(t, y):
    """Start and end time of the last full period of the oscillation of y, from one minimum of y to the next."""
    minima = times_of_minima(t, y)
    if len(minima) < 2:
        raise ValueError("Fewer than two minima: the run does not cover a full period yet")
    return minima[-2], minima[-1]


def mean_and_amplitude(y):
    return 0.5 * (y.max() + y.min()), 0.5 * (y.max() - y.min())


def relative_difference(computed, reference):
    """Difference in percent of the reference, negative when computed is smaller in magnitude than reference."""
    return 100.0 * (computed - reference) / reference


def print_table(column_names, rows, name_width=24, column_width=12):
    """Print rows of (name, values) under column_names: numbers with 4 decimals, strings as they are, None as "-"."""

    def cell(value):
        if value is None:
            return f"{'-':>{column_width}s}"
        if isinstance(value, str):
            return f"{value:>{column_width}s}"
        return f"{value:{column_width}.4f}"

    print(f"  {'':{name_width}s}" + "".join(f" {name:>{column_width}s}" for name in column_names))
    for name, values in rows:
        print(f"  {name:{name_width}s}" + "".join(f" {cell(value)}" for value in values))


def difference_cell(computed, reference):
    """The relative difference of computed to reference as a table cell, or None without a reference."""
    if computed is None or reference is None or reference == 0:
        return None
    return f"{relative_difference(computed, reference):+.2f} %"


def print_header():
    print(f"  {'':24s} {'computed':>12s} {'reference':>12s} {'difference':>13s}")


def print_row(name, computed, reference):
    print(f"  {name:24s} {computed:12.4f} {reference:12.4f} {relative_difference(computed, reference):+11.2f} %")
