# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Functions on a mesh pieced together from functions on cell submeshes of it."""

import dolfinx
import numpy as np


def all_cells(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    """The local cells of ``mesh``, owned and ghost."""
    index_map = mesh.topology.index_map(mesh.topology.dim)
    return np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)


class SubmeshComposite:
    """A function ``parent`` on a mesh that holds ``parts``, functions with the same element on cell
    submeshes of the mesh, each on the cells of its submesh.

    ``parts`` is a list of ``(function, entity_map)`` pairs, with the entity map returned by
    ``dolfinx.mesh.create_submesh``. Values are copied between ``parent`` and the parts cell by cell,
    which is exact since the elements match. Where submeshes share dofs, on their common facets, the
    part listed last gives the value of ``parent``.
    """

    def __init__(self, parent: dolfinx.fem.Function, parts: list[tuple[dolfinx.fem.Function, dolfinx.mesh.EntityMap]]):
        self.parent = parent
        self._parts = []
        for f, entity_map in parts:
            sub_cells = all_cells(f.function_space.mesh)
            parent_cells = entity_map.sub_topology_to_topology(sub_cells, inverse=False)
            self._parts.append((f, sub_cells, parent_cells))

    def gather(self) -> None:
        """``parent`` <- the parts."""
        for f, sub_cells, parent_cells in self._parts:
            self.parent.interpolate(f, cells0=sub_cells, cells1=parent_cells)
        self.parent.x.scatter_forward()

    def scatter(self) -> None:
        """The parts <- ``parent``."""
        for f, sub_cells, parent_cells in self._parts:
            f.interpolate(self.parent, cells0=parent_cells, cells1=sub_cells)
