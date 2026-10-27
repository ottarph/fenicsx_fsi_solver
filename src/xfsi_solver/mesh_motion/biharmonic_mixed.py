# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Biharmonic mesh motion in mixed form, with the Laplacian of the displacement as a second unknown."""

import dolfinx
import ufl

from xfsi_solver.mesh_motion.problem import MeshMotion
from xfsi_solver.tools.quadrature import cap_quadrature_degree


def biharmonic_mixed_mesh_motion(
    u: dolfinx.fem.Function,
    bcs: list[dolfinx.fem.DirichletBC],
    *,
    z: dolfinx.fem.Function | None = None,
    solver_options: dict | None = None,
    petsc_options_prefix: str = "mesh_motion_",
    max_quadrature_degree: int | None = None,
) -> MeshMotion:
    """The displacement ``u`` of the mesh it lives on, solving the biharmonic equation in mixed form,

        (z, dz) - (grad(u), grad(dz)) = 0,
        -(grad(z), grad(du)) = 0,

    with the Dirichlet conditions ``bcs`` on u, which should hold on the whole boundary. z = div(grad(u))
    has no boundary conditions, so the first equation imposes a zero normal derivative of u on the
    boundary weakly. ``z`` is a function to hold z, on the mesh of u (a new one, with the element of
    u, if None), for example to checkpoint it. The unknowns of the solve are (z, u), in this order.
    ``solver_options`` and ``petsc_options_prefix`` are those of :class:`MeshMotion`.
    ``max_quadrature_degree`` caps the quadrature degree of the forms, as
    ``tools.quadrature.cap_quadrature_degree`` does; with None, every integral keeps the degree UFL
    estimates.
    """
    U = u.function_space
    mesh = U.mesh
    if z is None:
        z = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, U.ufl_element()), name="z")
    if z.function_space.dofmap.index_map_bs != U.dofmap.index_map_bs:
        raise ValueError("The mixed biharmonic mesh motion needs z and u with the same block size")
    dz, du = ufl.TestFunctions(ufl.MixedFunctionSpace(z.function_space, U))
    dx = ufl.Measure("dx", domain=mesh)

    # The zero term gives the (u, u) block, where dolfinx puts the unit diagonal of the Dirichlet
    # rows; without it, those rows are zero and the factorization fails. ufl.extract_blocks orders
    # the residual blocks by the test functions, (z, u), as the unknowns are ordered.
    F = ufl.extract_blocks(
        (
            ufl.inner(z, dz)
            - ufl.inner(ufl.grad(u), ufl.grad(dz))
            - ufl.inner(ufl.grad(z), ufl.grad(du))
            + dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(u, du)
        )
        * dx
    )
    F = [cap_quadrature_degree(form, max_quadrature_degree) for form in F]
    return MeshMotion(F, [z, u], bcs, solver_options=solver_options, petsc_options_prefix=petsc_options_prefix)
