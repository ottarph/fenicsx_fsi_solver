# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""A prescribed (closed-form, no PDE solve) fluid-domain deformation.

Per notes/remeshing/implementation-plan.md §2/§6/§7 Phase 1: for the
standalone fluid-domain remeshing prototype, the fluid-solid interface (the
flag boundary) is deformed by a function *we choose*, standing in for
"whatever the structural displacement would have been" -- not the output of
solving anything. This module builds that function as a single closed-form
formula valid over the whole domain, so that:

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


def _bump(r: np.ndarray) -> np.ndarray:
    """C-infinity, compactly-supported bump: 1 at r=0, smoothly decaying to
    *exactly* 0 at r>=1."""
    inside = r < 1.0
    r_safe = np.where(inside, r, 0.0)  # avoid 1/(1 - r**2) for r >= 1
    return np.where(inside, np.exp(1.0 - 1.0 / (1.0 - r_safe**2)), 0.0)


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
        envelope = _bump(dist / ENVELOPE_RADIUS)

        dy = amplitude * t**2 * envelope

        values = np.zeros((2, x.shape[1]), dtype=x.dtype)
        values[1] = dy
        return values

    return f


def incremental_interface_deformation(step_amplitude: float, interface_points: np.ndarray):
    """Like :func:`prescribed_interface_deformation`, but for applying one
    *incremental* bending step directly to the mesh's own *current*
    geometry, across any number of remesh events, without needing to track
    a separate material/reference-coordinate field.

    ``prescribed_interface_deformation`` decays with distance from the
    *original, undeformed* flag centerline (a fixed line at ``y=geo.C_Y``)
    -- correct for a single application starting from the true t=0 mesh,
    but wrong once that mesh has already been bent and remeshed: querying
    it at the new mesh's own (already-bent) coordinates double-counts the
    prior bending. There is no cheap way to instead recover each new mesh
    node's *original* t=0 position after a remesh (an early version of the
    Phase 4 loop tried transferring a "reference coordinates" field via
    ``transfer.transfer_field`` for this, which is mathematically wrong -- a
    field that is the identity everywhere transfers to exactly "the query
    point itself" again by construction, regardless of which mesh cell is
    used, so it can never recover a *different* reference position; doing
    this properly would need to actually solve an extension problem, which
    is out of scope for this solver-free phase).

    Instead, this decays with distance from the *current* interface (the
    literal set of points the caller passes in, e.g. from that mesh's own
    ``facet_tags`` for ``solid_fluid_interface``) rather than a fixed line.
    Since x never moves under this pure-vertical deformation, ``t(x)`` is
    still exactly the same fixed-geometry fraction-along-the-flag used
    above; only the spatial envelope needs to "follow" the interface.

    Args:
        step_amplitude: incremental tip deflection (metres) to add this step.
        interface_points: (n, >=2) array of physical coordinates of the
            *current* interface boundary (e.g. from
            ``V.tabulate_dof_coordinates()`` restricted to interface dofs).
    """
    interface_xy = np.asarray(interface_points)[:, :2]

    def f(x: np.ndarray) -> np.ndarray:
        px, py = x[0], x[1]
        t = np.clip((px - geo.FLAG_LEFT) / (geo.FLAG_RIGHT - geo.FLAG_LEFT), 0.0, 1.0)

        query = np.stack([px, py], axis=-1)  # (n, 2)
        dist = np.linalg.norm(query[:, None, :] - interface_xy[None, :, :], axis=-1).min(axis=1)
        envelope = _bump(dist / ENVELOPE_RADIUS)

        dy = step_amplitude * t**2 * envelope

        values = np.zeros((2, x.shape[1]), dtype=x.dtype)
        values[1] = dy
        return values

    return f
