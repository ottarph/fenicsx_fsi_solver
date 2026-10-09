# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Penalty parameters of the C^0 interior penalty (C0IP) method for the biharmonic equation."""

import basix
import dolfinx
import numpy as np
import ufl


def get_dg_penalty_parameters_triangle(mesh: dolfinx.mesh.Mesh, a_val: float, p: int, vol=None, h=None):
    """Interior and boundary facet penalty parameters on a triangle mesh, for elements of degree p.

    The cell volumes vol and facet areas h default to ufl.CellVolume and ufl.FacetArea, which
    are only supported on affine meshes. On curved meshes, pass those from compute_cell_volumes
    and compute_facet_areas.

    Local parameter selection proposed in *Local parameter selection in the C^0 interior
    penalty method for the biharmonic equation* by Bringmann, Carstensen, and Streitberger, 2023.
    """
    assert mesh.basix_cell() == basix.CellType.triangle

    h = ufl.FacetArea(mesh) if h is None else h
    vol = ufl.CellVolume(mesh) if vol is None else vol

    alpha_base = dolfinx.fem.Constant(mesh, a_val)

    # Note that h("+") == h("-"), since h is the facet area, not cell volume.
    sigma = 3.0 * alpha_base * p * (p - 1) / 8.0 * h("+") ** 2 * 2 * ufl.avg(1 / vol)
    sigma_boundary = 3.0 * alpha_base * p * (p - 1) * h**2 / 2 * (1 / vol)

    return sigma, sigma_boundary


def get_dg_penalty_parameters_quadrilateral(mesh: dolfinx.mesh.Mesh, a_val: float, p: int, vol, h):
    """Interior and boundary facet penalty parameters on a quadrilateral mesh, for elements of
    degree p, with the cell volumes vol and facet areas h from compute_cell_volumes and
    compute_facet_areas, since ufl.CellVolume and ufl.FacetArea are not supported on quadrilaterals.

    Local parameter selection proposed in *Local parameter selection in the C^0 interior
    penalty method for the biharmonic equation* by Bringmann, Carstensen, and Streitberger, 2023.
    """
    assert mesh.basix_cell() == basix.CellType.quadrilateral

    alpha_base = dolfinx.fem.Constant(mesh, a_val)

    # Note that h("+") == h("-"), since h is the facet area, not cell volume.
    sigma = alpha_base * (p - 1) ** 2 * h("+") ** 2 * 2 * ufl.avg(1 / vol)
    sigma_boundary = 4.0 * alpha_base * (p - 1) ** 2 * h**2 * (1 / vol)

    return sigma, sigma_boundary


def compute_cell_volumes(mesh: dolfinx.mesh.Mesh):
    """DG0 function of the cell volumes of mesh."""
    DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    v = ufl.TestFunction(DG0)
    b = dolfinx.fem.assemble_vector(dolfinx.fem.form(v * ufl.dx))

    vol = dolfinx.fem.Function(DG0, name="cell_volume")
    vol.x.array[:] = b.array[:]
    vol.x.scatter_forward()

    return vol


def compute_facet_areas(mesh: dolfinx.mesh.Mesh):
    """DG0 function of the facet areas of mesh on the submesh of all its facets, and the
    entity map of that submesh, to pass in the entity_maps of the forms using it.

    Following the approach in
    https://fenicsproject.discourse.group/t/ufl-facetarea-for-quadrilaterals/17451/4
    """
    tdim = mesh.topology.dim
    facets = dolfinx.mesh.locate_entities(mesh, tdim - 1, lambda x: np.full_like(x[0], True, dtype=bool))
    submesh_facets, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, tdim - 1, facets)

    Ve = dolfinx.fem.functionspace(submesh_facets, ("DG", 0))
    facet_area_h = dolfinx.fem.Function(Ve)
    child_facets = np.arange(facet_area_h.x.array.shape[0], dtype=np.int32)

    facet_area_h.x.array[:] = submesh_facets.h(submesh_facets.geometry.dim - 1, child_facets)
    facet_area_h.x.scatter_forward()

    return facet_area_h, entity_map
