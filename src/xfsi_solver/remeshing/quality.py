# Copyright (c) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT
#
# Vendored from https://github.com/ottarph/pvmeshquality (MIT License,
# commit as of 2026-09-10), with the standalone plotting demo dropped. Smoke
# -tested against this repo's pinned dolfinx==0.11.0 / pyvista==0.48.4 and
# found to work unmodified. See notes/remeshing/implementation-plan.md §3/§4
# for how this is used as the remeshing trigger's quality metric.

import dolfinx as dfx
import numpy as np
import pyvista as pv

DFX_TO_PV_CELLTYPE = {
    dfx.mesh.basix.CellType.point:         (pv.CellType.VERTEX,     np.int8(-1)),
    dfx.mesh.basix.CellType.interval:      (pv.CellType.LINE,       np.int8( 1)),
    dfx.mesh.basix.CellType.triangle:      (pv.CellType.TRIANGLE,   np.int8( 1)),
    dfx.mesh.basix.CellType.quadrilateral: (pv.CellType.QUAD,       np.int8( 1)),
    dfx.mesh.basix.CellType.tetrahedron:   (pv.CellType.TETRA,      np.int8( 1)),
    dfx.mesh.basix.CellType.pyramid:       (pv.CellType.PYRAMID,    np.int8( 1)),
    dfx.mesh.basix.CellType.prism:         (pv.CellType.WEDGE,      np.int8( 1)),
    dfx.mesh.basix.CellType.hexahedron:    (pv.CellType.HEXAHEDRON, np.int8( 1)),
}

class MeshQuality:

    def __init__(
        self,
        quality_measure: str = "scaled_jacobian",
        mesh: dfx.mesh.Mesh | None = None,
        fspace: dfx.fem.FunctionSpace | None = None,
    ):
        # Note, scaled Jacobian for triangles can never be negative, due to the implementation,
        # https://github.com/sandialabs/verdict/blob/master/V_TriMetric.cpp#L601

        self.quality_measure = quality_measure
        self.fspace = fspace
        self.mesh = mesh

        assert mesh is not None or fspace is not None, "One of mesh or fspace must be provided."

        if isinstance(fspace, dfx.fem.FunctionSpace):
            assert mesh is None or mesh == fspace.mesh, "If both mesh and fspace are provided, they must be consistent."
            assert len(fspace.value_shape) == 1, "Only vector-valued function spaces are supported."
            assert fspace.ufl_element().degree == 1, "Only CG1 function spaces are supported."
            self._val_dim: int = fspace.value_shape[0]
            self._geo_dim: int = fspace.mesh.geometry.dim

            self.grid = pv.UnstructuredGrid(*dfx.plot.vtk_mesh(fspace))

            self._base_orientation = 1.0
            self._base_orientation = np.sign(self.__call__())

        else:
            top, celltype, geometry = dfx.plot.vtk_mesh(mesh)

            mesh_celltype = mesh.basix_cell()
            assert mesh_celltype in DFX_TO_PV_CELLTYPE, f"Unsupported cell type {mesh_celltype}."
            assert mesh_celltype not in (
                dfx.mesh.basix.CellType.point,
                dfx.mesh.basix.CellType.interval,
            ), "Meshes of point or interval cells are not supported."

            celltype_pv = np.full(celltype.shape, DFX_TO_PV_CELLTYPE[mesh_celltype][0])


            self.grid = pv.UnstructuredGrid(top, celltype_pv, geometry)

            self._base_orientation = 1.0
            self._base_orientation = np.sign(self.__call__())

        return

    def __call__(self, u: dfx.fem.Function | np.ndarray | None = None) -> np.ndarray:
        """
        Compute the mesh quality of `self.mesh` deformed by u. If `np.ndarray`- inputs are used,
        these should be in the ordering given by uh.x.array.reshape(-1,d) for uh a d-dimensional CG1 function.
        If `u` is `None`, the undeformed mesh quality is returned.

        Args:
            u (dfx.fem.Function | np.ndarray): Function to deform mesh by.
        Returns:
            np.ndarray: The mesh quality of all cells in deformed mesh.
        """

        if u is not None:
            assert isinstance(self.fspace, dfx.fem.FunctionSpace), "fspace must be a FunctionSpace to use u != None."

            if isinstance(u, dfx.fem.Function):
                uh: np.ndarray = u.x.array.reshape(-1, self._val_dim)
            elif isinstance(u, np.ndarray):
                uh: np.ndarray = u
            else:
                raise ValueError(f"Invalid type {type(u)} for u.")

            if self._val_dim == 2:
                uh = np.concatenate( (uh, np.zeros_like(uh[:,[0]], dtype=np.float64)) , axis=1)

            self.grid["uh"] = uh

            warped = self.grid.warp_by_vector("uh")
            quality = warped.cell_quality(quality_measure=self.quality_measure)
            return np.copy(quality.cell_data[self.quality_measure]) * self._base_orientation

        else:
            warped = self.grid
            quality = warped.cell_quality(quality_measure=self.quality_measure)
            return np.copy(quality.cell_data[self.quality_measure]) * self._base_orientation
