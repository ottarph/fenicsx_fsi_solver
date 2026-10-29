# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Mesh motion problems: the displacement of a mesh, such as the fluid mesh of an FSI problem, from
its values on the boundary. Each function sets up a :class:`MeshMotion`, a NonlinearProblem solved by
one Newton step with an explicitly lagged Jacobian, so that its matrix is assembled once and later
solves only compute the residual for the new boundary values, by a product with the matrix of the
linear form. By default, the matrix is factorized by MUMPS LU on one displacement component, applied
to all of them."""

from xfsi_solver.mesh_motion.biharmonic_c0ip import biharmonic_c0ip_mesh_motion
from xfsi_solver.mesh_motion.biharmonic_mixed import biharmonic_mixed_mesh_motion
from xfsi_solver.mesh_motion.harmonic import harmonic_mesh_motion
from xfsi_solver.mesh_motion.problem import MeshMotion, default_mesh_motion_solver_options

__all__ = [
    "MeshMotion",
    "biharmonic_c0ip_mesh_motion",
    "biharmonic_mixed_mesh_motion",
    "default_mesh_motion_solver_options",
    "harmonic_mesh_motion",
]
