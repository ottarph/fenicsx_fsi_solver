# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import dolfinx
import dolfinx.fem.petsc
import ufl


class Solid:
    def STVK(u: dolfinx.fem.Function, lambda_: dolfinx.fem.Constant, mu: dolfinx.fem.Constant):
        Id = ufl.Identity(u.ufl_shape[0])
        F = Id + ufl.grad(u)
        E = 0.5 * (F.T * F - Id)
        J = ufl.det(F)
        sigma = ufl.inv(J) * F * (lambda_ * ufl.tr(E) * Id + 2.0 * mu * E) * F.T
        return sigma

    def IMR_elastic(u: dolfinx.fem.Function, c1: dolfinx.fem.Constant, c2: dolfinx.fem.Constant):
        """Elastic Cauchy stress of an incompressible Mooney-Rivlin material.

        For B = F F^T, return 2 c1 B - 2 c2 B^-1. The coefficients use the
        convention W = c1 (I1 - 3) + c2 (I2 - 3), with small-strain shear
        modulus 2 (c1 + c2). Setting c2 = 0 gives the neo-Hookean model.

        This law assumes det(F) = 1; the solver must impose that constraint
        separately. For a 2D displacement, return the in-plane stress of a
        plane-strain material with unit out-of-plane stretch.
        """
        Id = ufl.Identity(u.ufl_shape[0])
        F = Id + ufl.grad(u)
        B = F * F.T
        return 2.0 * c1 * B - 2.0 * c2 * ufl.inv(B)

    def IMR_pressure(p: dolfinx.fem.Function):
        """Cauchy pressure stress -p I, with pressure positive in compression."""
        Id = ufl.Identity(p.ufl_domain().geometric_dimension)
        return -p * Id

    def IMR(
        u: dolfinx.fem.Function,
        p: dolfinx.fem.Function,
        c1: dolfinx.fem.Constant,
        c2: dolfinx.fem.Constant,
    ):
        """Full incompressible Mooney-Rivlin Cauchy stress, including pressure.

        Use IMR_elastic and IMR_pressure separately to time-average the elastic
        part while keeping pressure fully implicit. With this pressure
        convention, the undeformed stress-free state has p = 2 (c1 - c2).
        Convert to first Piola stress with J sigma F^-T, as for STVK.
        """
        return Solid.IMR_elastic(u, c1, c2) + Solid.IMR_pressure(p)


class Fluid:
    def NS_pressure(p: dolfinx.fem.Function):
        Id = ufl.Identity(p.ufl_domain().geometric_dimension)
        sigma = -p * Id
        return sigma

    def NS_velocity(
        u: dolfinx.fem.Function, v: dolfinx.fem.Function, nu: dolfinx.fem.Constant, rho: dolfinx.fem.Constant
    ):
        Id = ufl.Identity(u.ufl_shape[0])
        F = Id + ufl.grad(u)
        sigma = rho * nu * (ufl.grad(v) * ufl.inv(F) + ufl.inv(F).T * ufl.grad(v).T)
        return sigma

    def NS_velocity_eulerian(v: dolfinx.fem.Function, nu: dolfinx.fem.Constant, rho: dolfinx.fem.Constant):
        sigma = rho * nu * (ufl.grad(v) + ufl.grad(v).T)
        return sigma

    def NS(
        u: dolfinx.fem.Function,
        v: dolfinx.fem.Function,
        p: dolfinx.fem.Function,
        nu: dolfinx.fem.Constant,
        rho: dolfinx.fem.Constant,
    ):
        return Fluid.NS_velocity(u, v, nu, rho) + Fluid.NS_pressure(p)

    def NS_eulerian(
        v: dolfinx.fem.Function, p: dolfinx.fem.Function, nu: dolfinx.fem.Constant, rho: dolfinx.fem.Constant
    ):
        return Fluid.NS_velocity_eulerian(v, nu, rho) + Fluid.NS_pressure(p)
