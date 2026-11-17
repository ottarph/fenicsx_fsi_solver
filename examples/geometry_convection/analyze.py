"""Newton convergence rates, cost and QoI differences of the geometry-convection runs.

Usage: python analyze.py <run prefix> ...  (the output prefixes given to the drivers, e.g.
output/geometry_convection/gc_full_o1; the first is the reference of the QoI comparison)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def newton_steps(prefix):
    """The residual norms of each time step, in order, from the Newton log."""
    data = np.loadtxt(f"{prefix}_newton.txt", comments="#", usecols=(0, 1, 2, 4))
    steps = defaultdict(list)
    assembled = defaultdict(int)
    for t, _it, asm, res in data:
        steps[t].append(res)
        assembled[t] += int(asm)
    return [np.array(steps[t]) for t in sorted(steps)], [assembled[t] for t in sorted(steps)]


def summarize(prefix):
    steps, assembled = newton_steps(prefix)
    res = json.loads(Path(f"{prefix}_res.json").read_text()) if Path(f"{prefix}_res.json").exists() else None
    its = np.array([len(s) - 1 for s in steps])
    print(f"\n== {prefix}: {len(steps)} steps")
    print(
        f"  Newton its/step: mean {its.mean():.2f}, min {its.min()}, max {its.max()}, "
        f"histogram {dict(zip(*np.unique(its, return_counts=True), strict=True))}"
    )
    print(f"  Jacobian assemblies: {sum(assembled)} ({sum(assembled) / len(steps):.2f}/step)")
    if res is not None:
        print(f"  Time: {res['elapsed_time']:.1f} s ({res['elapsed_time'] / res['num_steps'] * 1e3:.1f} ms/step)")
    # Contraction ratio ||R_l|| / ||R_{l-1}|| at iteration l, over the steps that reach it
    print("  contraction ||R_l||/||R_(l-1)||   median   [10%, 90%]   steps")
    max_l = its.max()
    for it in range(1, min(max_l, 6) + 1):
        r = np.array([s[it] / s[it - 1] for s in steps if len(s) > it])
        print(
            f"    l = {it}:                       {np.median(r):8.2e} [{np.quantile(r, 0.1):.1e}, "
            f"{np.quantile(r, 0.9):.1e}]  {r.size}"
        )
    # Order estimate p_l = log(r_l) / log(r_(l-1)), r_l = ||R_l|| / ||R_(l-1)||, i.e. ||R_l|| ~ C ||R_(l-1)||^p
    print("  order p_l = log(||R_l||/||R_(l-1)||) / log(||R_(l-1)||/||R_(l-2)||)   median  steps")
    for it in range(2, min(max_l, 5) + 1):
        p = np.array([np.log(s[it] / s[it - 1]) / np.log(s[it - 1] / s[it - 2]) for s in steps if len(s) > it])
        print(f"    l = {it}:   {np.median(p):5.2f}  {p.size}")
    return steps


def qoi(prefix):
    return np.loadtxt(f"{prefix}_qoi.txt", comments="#")


if __name__ == "__main__":
    prefixes = sys.argv[1:]
    for prefix in prefixes:
        summarize(prefix)
    ref = prefixes[0]
    q_ref = qoi(ref)
    ranges = q_ref[:, 1:].max(axis=0) - q_ref[:, 1:].min(axis=0)
    print(f"\nQoI: max deviation from {ref} over the rows / its peak-to-peak range (drag, lift, u_x, u_y)")
    for prefix in prefixes[1:]:
        q = qoi(prefix)
        n = min(len(q), len(q_ref))
        dev = np.abs(q[:n, 1:] - q_ref[:n, 1:]).max(axis=0) / ranges
        print(f"  {prefix}: " + ", ".join(f"{d:.1e}" for d in dev))
