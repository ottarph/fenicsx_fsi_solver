# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Results of seven groups for the FSI1 and FSI3 benchmarks, from Tables 2 and 3 of

    S. Turek, J. Hron, M. Razzaq, H. Wobker and M. Schäfer. Numerical benchmarking of fluid-structure interaction: a
    comparison of different discretization and solution approaches. In H.-J. Bungartz, M. Mehl and M. Schäfer (eds.),
    Fluid Structure Interaction II, LNCSE 73, pp. 413-424. Springer, 2010. doi:10.1007/978-3-642-14206-2_15

The tables are transcribed row by row, in the units of the paper: in Table 2 (FSI1), u1 in 10^-5 m and u2 in 10^-4 m;
in Table 3 (FSI3), u1 and u2 in 10^-3 m. u1 and u2 are the x- and y-displacement of the point A, and the periodic
quantities are (mean, amplitude). f1 and f2 are the frequencies of u1 and u2. None marks a value the paper does not
give. The rows of a method go from the coarsest to the finest mesh, and for each mesh from the largest to the smallest
time step, so its last row is its finest result.

The functions below return the values in the units of compare_fsi_qois.py: displacements in 10^-3 m.
"""

from typing import NamedTuple

import numpy as np

# Method label: (group, approach), as described in Section 4 of the paper.
METHODS = {
    "1": ("Schäfer", "partitioned, implicit: finite volumes (FASTEST) and FEAP"),
    "2a": ("Rannacher", "monolithic FEM, Eulerian (interface capturing), stabilized Q1/Q1"),
    "2b": ("Rannacher", "monolithic FEM, ALE, Q2/P1 discontinuous"),
    "3": ("Turek/Hron", "monolithic FEM, ALE, Q2/P1"),
    "4": ("Breuer", "partitioned: finite volumes (FASTEST-3D) and Carat"),
    "5": ("Krafczyk/Rank", "partitioned, explicit: lattice Boltzmann (VirtualFluids) and p-FEM (AdhoC)"),
    "6": ("Wall", "partitioned, iterative staggered: stabilized Q2Q2 fluid and EAS solid"),
    "7": ("Bletzinger", "partitioned: OpenFOAM and CARAT++ with a shell model"),
}

# The monolithic finite element methods, the closest to the solver of fsi2D_benchmark.py.
MONOLITHIC_FEM_METHODS = ("2a", "2b", "3")


class FSI1Row(NamedTuple):
    unknowns: int
    u1: float | None  # [10^-5 m]
    u2: float | None  # [10^-4 m]
    drag: float | None
    lift: float | None


class FSI3Row(NamedTuple):
    unknowns: int
    dt: float
    u1: tuple[float, float]  # [10^-3 m]
    u2: tuple[float, float]  # [10^-3 m]
    drag: tuple[float, float] | None
    lift: tuple[float, float] | None
    f1: float
    f2: float


# Table 2: results for the steady benchmark FSI1. Method 4 has no FSI1 results.
FSI1_TABLE = {
    "1": [
        FSI1Row(82722, None, None, 14.2770, 0.77200),
        FSI1Row(322338, None, None, 14.2890, 0.76900),
    ],
    "2a": [
        FSI1Row(11250, 2.4800, 7.7800, None, None),
    ],
    "2b": [
        FSI1Row(19488, 2.2821, 8.1957, 14.2382, 0.76481),
        FSI1Row(29512, 2.2793, 8.2201, 14.2263, 0.76420),
        FSI1Row(51016, 2.2733, 8.1867, 14.2408, 0.76400),
        FSI1Row(93992, 2.2710, 8.1702, 14.2500, 0.76392),
        FSI1Row(179912, 2.2700, 8.1609, 14.2561, 0.76389),
        FSI1Row(351720, 2.2695, 8.1556, 14.2603, 0.76388),
    ],
    "3": [
        FSI1Row(19488, 2.2871, 8.1930, 14.2736, 0.76175),
        FSI1Row(76672, 2.2774, 8.2042, 14.2918, 0.76305),
        FSI1Row(304128, 2.2732, 8.2071, 14.2948, 0.76356),
        FSI1Row(1211392, 2.2716, 8.2081, 14.2949, 0.76370),
        FSI1Row(4835328, 2.2708, 8.2085, 14.2945, 0.76374),
        FSI1Row(19320832, 2.2705, 8.2088, 14.2943, 0.76375),
    ],
    "5": [
        FSI1Row(884736, 2.1270, 11.0800, 14.3179, 0.85491),
        FSI1Row(3538944, 2.1990, 8.3370, 14.3127, 0.75138),
        FSI1Row(14155776, 2.2160, 8.2010, 14.3815, 0.75170),
    ],
    "6": [
        FSI1Row(7059, 2.5396, 8.8691, 14.2800, 0.73690),
        FSI1Row(19991, 2.2630, 8.2935, 14.2970, 0.76687),
        FSI1Row(77643, 2.2676, 8.2347, 14.2940, 0.76545),
        FSI1Row(164262, 2.2680, 8.2310, 14.2940, 0.76487),
    ],
    "7": [
        FSI1Row(217500, 2.2640, 8.2800, 14.3510, 0.76351),
    ],
}

# Table 3: results for the unsteady benchmark FSI3.
FSI3_TABLE = {
    "1": [
        FSI3Row(61318, 1.0e-3, (-2.54, 2.41), (1.45, 32.80), (450.3, 23.51), (-0.10, 143.0), 10.90, 5.13),
        FSI3Row(237286, 2.0e-3, (-2.88, 2.73), (1.53, 34.94), (458.6, 27.18), (2.08, 153.1), 10.60, 5.30),
        FSI3Row(237286, 1.0e-3, (-2.87, 2.73), (1.54, 34.94), (458.6, 27.31), (2.00, 153.3), 10.34, 5.91),
        FSI3Row(237286, 5.0e-4, (-2.86, 2.72), (1.53, 34.90), (458.6, 27.27), (2.01, 153.4), 12.16, 6.08),
        FSI3Row(941158, 1.0e-3, (-2.91, 2.77), (1.47, 35.26), (459.9, 27.92), (1.84, 157.7), 11.63, 4.98),
    ],
    "2a": [
        FSI3Row(11250, 5.0e-3, (-2.48, 2.24), (1.27, 36.50), None, None, 10.10, 5.10),
    ],
    "2b": [
        FSI3Row(7176, 5.0e-3, (-2.44, 2.32), (1.02, 31.82), (473.5, 56.97), (8.08, 283.8), 11.07, 5.29),
        FSI3Row(7176, 2.0e-3, (-2.48, 2.39), (0.92, 32.81), (471.3, 62.28), (6.11, 298.6), 10.73, 5.35),
        FSI3Row(7176, 1.0e-3, (-2.58, 2.49), (0.94, 33.19), (470.4, 64.02), (4.65, 300.3), 10.69, 5.36),
        FSI3Row(27744, 5.0e-3, (-2.43, 2.27), (1.41, 31.73), (483.7, 22.31), (2.21, 149.0), 10.53, 5.37),
        FSI3Row(27744, 2.0e-3, (-2.63, 2.61), (1.46, 33.46), (483.3, 24.48), (2.08, 161.2), 10.66, 5.43),
        FSI3Row(27744, 1.0e-3, (-2.80, 2.64), (1.45, 34.12), (483.0, 25.67), (2.21, 165.3), 10.75, 5.41),
        FSI3Row(42024, 2.5e-3, (-2.40, 2.26), (1.39, 31.71), (448.7, 21.16), (1.84, 141.3), 10.72, 5.42),
        FSI3Row(42024, 1.0e-3, (-2.53, 2.38), (1.40, 32.49), (449.7, 22.24), (1.61, 142.8), 10.77, 5.44),
        FSI3Row(42024, 5.0e-4, (-2.57, 2.42), (1.42, 32.81), (450.1, 22.49), (1.49, 143.7), 10.79, 5.42),
        FSI3Row(72696, 2.5e-3, (-2.64, 2.48), (1.38, 33.25), (451.1, 24.57), (2.04, 150.6), 10.73, 5.38),
        FSI3Row(72696, 1.0e-3, (-2.79, 2.62), (1.28, 34.61), (452.0, 25.78), (1.91, 152.7), 10.78, 5.42),
        FSI3Row(72696, 5.0e-4, (-2.84, 2.67), (1.28, 34.61), (452.4, 26.19), (2.36, 152.7), 10.84, 5.42),
    ],
    "3": [
        FSI3Row(19488, 1.0e-3, (-3.02, 2.83), (1.41, 35.47), (458.2, 28.32), (2.41, 145.6), 10.75, 5.37),
        FSI3Row(19488, 5.0e-4, (-3.02, 2.85), (1.42, 35.63), (458.7, 28.78), (2.23, 146.0), 10.75, 5.37),
        FSI3Row(19488, 2.5e-4, (-3.02, 2.85), (1.32, 35.73), (458.7, 28.80), (2.23, 146.0), 10.74, 5.33),
        FSI3Row(76672, 1.0e-3, (-2.78, 2.62), (1.44, 34.36), (459.1, 26.63), (2.41, 151.3), 10.93, 5.46),
        FSI3Row(76672, 5.0e-4, (-2.78, 2.62), (1.44, 34.35), (459.1, 26.62), (2.39, 150.7), 10.92, 5.46),
        FSI3Row(76672, 2.5e-4, (-2.77, 2.61), (1.43, 34.43), (459.1, 26.50), (2.36, 149.9), 10.93, 5.46),
        FSI3Row(304128, 1.0e-3, (-2.86, 2.70), (1.45, 34.93), (460.2, 27.65), (2.47, 154.9), 10.95, 5.47),
        FSI3Row(304128, 5.0e-4, (-2.86, 2.70), (1.45, 34.90), (460.2, 27.47), (2.37, 153.8), 10.92, 5.46),
        FSI3Row(304128, 2.5e-4, (-2.88, 2.72), (1.47, 34.99), (460.5, 27.74), (2.50, 153.9), 10.93, 5.46),
    ],
    "4": [
        FSI3Row(81120, 9.0e-5, (-5.18, 5.04), (1.12, 45.10), (477.0, 48.00), (7.00, 223.0), 10.14, 4.99),
        FSI3Row(324480, 2.0e-5, (-4.54, 4.34), (1.50, 42.50), (467.5, 39.50), (16.20, 188.7), 10.12, 5.05),
    ],
    "5": [
        FSI3Row(2480814, 5.1e-5, (-2.88, 2.71), (1.48, 35.10), (463.0, 31.30), (1.81, 154.0), 11.00, 5.50),
    ],
    "6": [
        FSI3Row(7059, 5.0e-4, (-1.60, 1.60), (1.50, 25.90), (525.0, 22.50), (-0.55, 106.0), 10.90, 5.45),
        FSI3Row(27147, 5.0e-4, (-2.00, 1.89), (1.45, 29.00), (434.0, 17.50), (2.53, 88.6), 10.60, 5.30),
    ],
    "7": [
        FSI3Row(271740, 5.0e-4, (-3.04, 2.87), (1.55, 36.63), (474.9, 28.12), (3.86, 165.9), 10.99, 5.51),
    ],
}


def fsi1_finest(method):
    """The finest FSI1 result of a method: {"ux", "uy" [10^-3 m], "drag", "lift"}, None where not given."""
    row = FSI1_TABLE[method][-1]
    return {
        "ux": None if row.u1 is None else row.u1 * 1e-2,
        "uy": None if row.u2 is None else row.u2 * 1e-1,
        "drag": row.drag,
        "lift": row.lift,
    }


def fsi3_finest(method):
    """The finest FSI3 result of a method: {"ux", "uy" [10^-3 m], "drag", "lift"}, each (mean, amplitude, frequency).

    The paper gives the frequencies of the displacements only, so those of the drag and lift are None, and both are
    None for a method without force values.
    """
    row = FSI3_TABLE[method][-1]
    return {
        "ux": (*row.u1, row.f1),
        "uy": (*row.u2, row.f2),
        "drag": None if row.drag is None else (*row.drag, None),
        "lift": None if row.lift is None else (*row.lift, None),
    }


def statistics_over_methods(finest_results):
    """Mean, minimum and maximum of each value over the finest results of all methods that give it.

    finest_results maps a method to its result, as returned by fsi1_finest or fsi3_finest. Returns the same
    structure, with (mean, min, max) in place of each value.
    """

    def stats(values):
        values = [value for value in values if value is not None]
        return (np.mean(values), np.min(values), np.max(values)) if values else None

    names = next(iter(finest_results.values())).keys()
    statistics = {}
    for name in names:
        results = [result[name] for result in finest_results.values() if result[name] is not None]
        if results and isinstance(results[0], tuple):
            statistics[name] = tuple(stats([result[i] for result in results]) for i in range(len(results[0])))
        else:
            statistics[name] = stats(results)
    return statistics
