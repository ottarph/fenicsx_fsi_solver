# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Transfer a field from the old fluid mesh onto a freshly regenerated one.

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
#: boundary (discrete_mesh.py), but the two meshes approximate curved
#: boundaries (the obstacle, and the interface once deformed) with
#: *different* polygons, so points near those boundaries can miss their
#: nominal source cell by an amount comparable to the *local mesh
#: resolution there*, not just floating-point error -- and silently stay
#: at 0 (uninterpolated) rather than raising, which is easy to miss.
#: Empirically (see notes/remeshing/implementation-plan.md §8): with
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
    fluid mesh), returning a new :class:`dolfinx.fem.Function` on ``V_to``.
    """
    f_to = dfx.fem.Function(V_to)
    mesh_to = V_to.mesh
    cells = np.arange(mesh_to.topology.index_map(mesh_to.topology.dim).size_local, dtype=np.int32)
    interpolation_data = dfx.fem.create_interpolation_data(V_to, f_from.function_space, cells, padding=padding)
    f_to.interpolate_nonmatching(f_from, cells, interpolation_data)
    return f_to
