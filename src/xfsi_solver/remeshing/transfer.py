# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Transfer a field from the old mesh onto a freshly regenerated one.

Implements notes/remeshing/implementation-plan.md §5: nonmatching-mesh
interpolation via ``dolfinx.fem.create_interpolation_data`` /
``Function.interpolate_nonmatching``, the same pattern already used (for a
different purpose -- bringing in a precomputed boundary displacement) in
``component_solvers/navier_stokes_ALE_fsi2_mm.py``.
"""

import dolfinx as dfx
import numpy as np

#: Default padding for create_interpolation_data. The new mesh's boundary
#: is deliberately built to coincide with the old mesh's current physical
#: boundary (discrete_mesh.py), but whenever a curved boundary is
#: *resampled* to a different discretization by the two meshes, they
#: approximate it with different polygons, so points near it can miss
#: their nominal source cell by an amount comparable to the *local mesh
#: resolution there*, not just floating-point error -- and silently stay
#: at 0 (uninterpolated) rather than raising, which is easy to miss.
#:
#: As of discrete_mesh.py's ``PINNED_BOUNDARIES``, most boundaries
#: (``solid_fluid_interface``, ``inflow``, ``outflow``, ``obstacle``,
#: ``solid_obstacle_interface``) are carried forward node-for-node rather
#: than resampled, specifically so this mismatch doesn't arise for them --
#: ``channel_side`` is the one boundary this can still happen to by
#: default. The empirical calibration below predates that pinning and was
#: measured with the obstacle boundary resampled (as
#: ``test_default_padding_matters`` in ``tests/test_remeshing_transfer.py``
#: still reproduces directly, via ``monkeypatch``, since a resampled curved
#: boundary is the scenario this padding is actually for): with
#: SizingField(size_near=0.02, size_far=0.06), 1e-6 left points near the
#: (fine, ~0.02) obstacle boundary un-interpolated; even 5e-3 left points
#: near the (coarse, ~0.06) far-field boundary un-interpolated; 1e-2
#: resolved both down to ordinary P2 interpolation error (~1e-5). Treat
#: this default as needing to be checked against the coarsest cell size
#: actually in play, not assumed correct for every sizing configuration.
DEFAULT_PADDING = 1e-2


def transfer_field(f_from: dfx.fem.Function, V_to: dfx.fem.FunctionSpace, padding: float = DEFAULT_PADDING):
    """Interpolate ``f_from`` (defined on one mesh) onto ``V_to`` (defined
    on a different, non-matching mesh -- e.g. the old vs. a regenerated
    regenerated mesh), returning a new :class:`dolfinx.fem.Function` on ``V_to``.
    """
    f_to = dfx.fem.Function(V_to)
    mesh_to = V_to.mesh
    cells = np.arange(mesh_to.topology.index_map(mesh_to.topology.dim).size_local, dtype=np.int32)
    interpolation_data = dfx.fem.create_interpolation_data(V_to, f_from.function_space, cells, padding=padding)
    f_to.interpolate_nonmatching(f_from, cells, interpolation_data)
    return f_to
