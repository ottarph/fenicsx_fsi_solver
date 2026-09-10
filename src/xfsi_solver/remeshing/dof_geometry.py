# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Map between mesh-geometry node order and a CG1 function space's dof order.

These are *not* the same ordering -- confirmed empirically on the FSI2
fluid mesh (``mesh.geometry.x`` vs.
``dfx.fem.functionspace(mesh, ("CG", 1)).tabulate_dof_coordinates()``: same
set of points, different order) -- which is exactly the bug that broke an
early version of the Phase 1 prescribed-deformation code here (a CG1
``Function``'s ``.x.array`` was added directly to ``mesh.geometry.x``,
silently scrambling the mesh). ``discrete_mesh.regenerate_fluid_mesh`` needs
geometry-node order (it feeds gmsh raw coordinates); ``quality.MeshQuality``
needs CG1 dof order (it warps a CG1 ``Function``). ``loop.py`` (Phase 4)
needs to convert between the two every step, so the mapping is built once
here rather than re-derived ad hoc.

The map is recovered from *local* (per-cell) vertex ordering, which DOLFINx
keeps consistent between a mesh's geometry dofmap and a same-degree nodal
Lagrange space's dofmap, even though the *global* numberings differ.
"""

import dolfinx as dfx
import numpy as np


def geometry_to_dof_permutation(mesh: dfx.mesh.Mesh, V: dfx.fem.FunctionSpace) -> np.ndarray:
    """``perm`` such that ``V``-dof/block index ``perm[g]`` is the same
    physical point as geometry node ``g``, i.e. ``dof_coords[perm] ==
    mesh.geometry.x`` and (the inverse use) ``arr_in_dof_order[perm] ==
    arr_in_geometry_order``.

    ``V`` must be a CG1 (possibly vector/blocked) space on ``mesh``, using
    the same nodal points as ``mesh``'s (assumed degree-1) geometry.
    """
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    geometry_dofmap = mesh.geometry.dofmaps[0][:num_cells]
    v_dofmap = V.dofmap.list[:num_cells]
    if geometry_dofmap.shape != v_dofmap.shape:
        raise ValueError(
            f"geometry dofmap shape {geometry_dofmap.shape} does not match V's dofmap "
            f"shape {v_dofmap.shape}; V must be CG1 on the same (degree-1) mesh"
        )

    perm = np.full(mesh.geometry.x.shape[0], -1, dtype=np.int64)
    perm[geometry_dofmap.reshape(-1)] = v_dofmap.reshape(-1)
    if np.any(perm < 0):
        raise RuntimeError("geometry_to_dof_permutation: some geometry nodes were never mapped")
    return perm


def to_dof_order(array_in_geometry_order: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder a (num_geometry_nodes, ...) array into CG1 dof order."""
    out = np.empty_like(array_in_geometry_order)
    out[perm] = array_in_geometry_order
    return out


def to_geometry_order(array_in_dof_order: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder a (num_dofs, ...) array into geometry-node order."""
    return array_in_dof_order[perm]
