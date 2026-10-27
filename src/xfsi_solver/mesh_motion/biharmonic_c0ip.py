# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Biharmonic mesh motion in C0 interior penalty (C0IP) form, with the displacement as the only unknown."""

import basix
import dolfinx
import ufl

from xfsi_solver.mesh_motion.problem import MeshMotion
from xfsi_solver.tools.dg_optpar import (
    compute_cell_volumes,
    compute_facet_diameters_and_areas,
    get_dg_penalty_parameters_quadrilateral,
    get_dg_penalty_parameters_tetrahedron,
    get_dg_penalty_parameters_triangle,
)


def c0ip_penalty_parameters(mesh: dolfinx.mesh.Mesh, a_val: float, degree: int):
    """The interior and boundary facet penalty parameters of tools/dg_optpar.py for the cells of
    ``mesh`` and elements of degree ``degree``, the facet diameters they are divided by, and the
    entity map of the facet submesh those live on, for the entity_maps of the forms."""
    vol = compute_cell_volumes(mesh)
    h_facet, area_facet, facet_entity_map = compute_facet_diameters_and_areas(mesh)
    cell = mesh.basix_cell()
    # In 2d, a facet's diameter is also its area, as get_dg_penalty_parameters_triangle and _quadrilateral take.
    if cell == basix.CellType.triangle:
        beta, beta_boundary = get_dg_penalty_parameters_triangle(mesh, a_val=a_val, p=degree, vol=vol, h=h_facet)
    elif cell == basix.CellType.quadrilateral:
        beta, beta_boundary = get_dg_penalty_parameters_quadrilateral(mesh, a_val=a_val, p=degree, vol=vol, h=h_facet)
    elif cell == basix.CellType.tetrahedron:
        beta, beta_boundary = get_dg_penalty_parameters_tetrahedron(
            mesh, a_val=a_val, p=degree, vol=vol, h=h_facet, area=area_facet
        )
    else:
        raise NotImplementedError(f"C0IP penalty parameters are not implemented for {cell} cells")
    return beta, beta_boundary, h_facet, facet_entity_map


def biharmonic_c0ip_mesh_motion(
    u: dolfinx.fem.Function,
    bcs: list[dolfinx.fem.DirichletBC],
    *,
    a_val: float = 2.0,
    solver_options: dict | None = None,
    petsc_options_prefix: str = "mesh_motion_",
) -> MeshMotion:
    """The displacement ``u`` of the mesh it lives on, solving the biharmonic equation in C0IP form,
    as in fsi2_biharmonic_c0ip.py, with the Dirichlet conditions ``bcs`` on u, which should hold on
    the whole boundary, and a zero normal derivative of u on the boundary imposed by Nitsche's
    method, as the mixed form imposes it naturally.

    u needs continuous Lagrange elements of degree 2 or more. The penalty parameters are those of
    tools/dg_optpar.py with ``a_val``, for triangles, quadrilaterals and tetrahedra. The matrix is
    symmetric positive definite. ``solver_options`` and ``petsc_options_prefix`` are those of
    :class:`MeshMotion`.
    """
    U = u.function_space
    mesh = U.mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    degree = U.element.basix_element.degree
    if degree < 2:
        raise ValueError(f"The C0IP mesh motion needs elements of degree 2 or more, not {degree}")
    beta, beta_boundary, h_facet, facet_entity_map = c0ip_penalty_parameters(mesh, a_val, degree)

    du = ufl.TestFunction(U)
    n = ufl.FacetNormal(mesh)
    # The facet diameters live on a submesh of the facets, so the measures need the domain.
    dx = ufl.Measure("dx", domain=mesh)
    dS = ufl.Measure("dS", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    def lap(f):
        return ufl.div(ufl.grad(f))

    F = ufl.inner(lap(u), lap(du)) * dx
    F -= ufl.inner(ufl.avg(lap(u)), ufl.jump(ufl.grad(du), n)) * dS
    F -= ufl.inner(ufl.avg(lap(du)), ufl.jump(ufl.grad(u), n)) * dS
    F += beta / h_facet("+") * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(du), n)) * dS
    F -= ufl.inner(lap(u), ufl.grad(du) * n) * ds
    F -= ufl.inner(lap(du), ufl.grad(u) * n) * ds
    F += beta_boundary / h_facet * ufl.inner(ufl.grad(u) * n, ufl.grad(du) * n) * ds

    return MeshMotion(
        F,
        [u],
        bcs,
        solver_options=solver_options,
        petsc_options_prefix=petsc_options_prefix,
        entity_maps=[facet_entity_map],
    )
