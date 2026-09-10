# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""A prescribed (closed-form, no PDE solve) fluid-domain deformation.

Per notes/remeshing/implementation-plan.md §2/§6/§7 Phase 1: for the
standalone fluid-domain remeshing prototype, the fluid-solid interface (the
flag boundary) is deformed by a function *we choose*, standing in for
"whatever the structural displacement would have been" -- not the output of
solving anything. This module builds that function as a single closed-form
formula valid over the whole fluid domain, so that:

- On the flag surface, it produces a cantilever-like bending profile (zero
  at the clamped root, growing towards the tip) scaled by ``amplitude``.
- It decays smoothly to exactly zero away from the flag, in particular on
  all the *fixed* fluid boundary pieces (inflow, outflow, channel walls,
  the obstacle) -- checked numerically in
  tests/test_remeshing_deformation.py -- so a remesh never needs to move
  those.

No dolfinx.fem.petsc solver is used anywhere in this module.
"""

import numpy as np

from xfsi_solver.remeshing import fsi2_geometry as geo

#: Radius (in metres) of the compactly-supported deformation envelope
#: around the flag centerline segment. Chosen comfortably smaller than the
#: distance from the flag to the channel walls/inflow/outflow (>= ~0.2 m),
#: so the envelope -- and hence the whole prescribed deformation -- is
#: *exactly* zero there, not just small (checked in
#: tests/test_remeshing_deformation.py).
ENVELOPE_RADIUS = 0.15


def prescribed_interface_deformation(amplitude: float):
    """Return a callable ``f(x) -> array`` for ``dfx.fem.Function.interpolate``.

    ``f`` gives a smooth, closed-form vertical bending deformation
    concentrated on and near the FSI2 flag, decaying to exactly zero away
    from it, with ``amplitude`` controlling the tip deflection (metres).
    """

    def f(x: np.ndarray) -> np.ndarray:
        px, py = x[0], x[1]

        # Fraction along the flag, from clamped root (0) to tip (1); clipped
        # so it is exactly 0 for x <= FLAG_LEFT (in particular, everywhere
        # on the obstacle boundary, since the obstacle's x-extent barely
        # reaches FLAG_LEFT at all).
        t = np.clip((px - geo.FLAG_LEFT) / (geo.FLAG_RIGHT - geo.FLAG_LEFT), 0.0, 1.0)

        # Nearest point on the (undeformed) flag centerline segment, and a
        # C^infinity, compactly-supported envelope of distance to it: ~1 on
        # the flag surface itself (half-thickness << ENVELOPE_RADIUS),
        # smoothly decaying to *exactly* 0 at distance ENVELOPE_RADIUS.
        nearest_x = np.clip(px, geo.FLAG_LEFT, geo.FLAG_RIGHT)
        dist = np.sqrt((px - nearest_x) ** 2 + (py - geo.C_Y) ** 2)
        r = dist / ENVELOPE_RADIUS
        inside = r < 1.0
        r_safe = np.where(inside, r, 0.0)  # avoid 1/(1 - r**2) for r >= 1
        envelope = np.where(inside, np.exp(1.0 - 1.0 / (1.0 - r_safe**2)), 0.0)

        dy = amplitude * t**2 * envelope

        values = np.zeros((2, x.shape[1]), dtype=x.dtype)
        values[1] = dy
        return values

    return f
