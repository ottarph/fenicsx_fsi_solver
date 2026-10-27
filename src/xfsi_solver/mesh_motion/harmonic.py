# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Harmonic mesh motion, without scaling."""

import dolfinx
import ufl

from xfsi_solver.mesh_motion.problem import MeshMotion


def harmonic_mesh_motion(
    u: dolfinx.fem.Function,
    bcs: list[dolfinx.fem.DirichletBC],
    *,
    solver_options: dict | None = None,
    petsc_options_prefix: str = "mesh_motion_",
) -> MeshMotion:
    """The displacement ``u`` of the mesh it lives on, solving -div(grad(u)) = 0 with the Dirichlet
    conditions ``bcs``, which should hold on the whole boundary.

    The Laplacian has no scaling, such as a stiffness growing towards small or distorted cells.
    ``solver_options`` and ``petsc_options_prefix`` are those of :class:`MeshMotion`.
    """
    U = u.function_space
    mesh = U.mesh
    du = ufl.TestFunction(U)
    F = ufl.inner(ufl.grad(u), ufl.grad(du)) * ufl.Measure("dx", domain=mesh)
    return MeshMotion(F, [u], bcs, solver_options=solver_options, petsc_options_prefix=petsc_options_prefix)
