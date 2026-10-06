# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import dolfinx
import ufl


class Solid:
    def STVK(u: dolfinx.fem.Function, lambda_: dolfinx.fem.Constant, mu: dolfinx.fem.Constant):
        Id = ufl.Identity(u.ufl_shape[0])
        F = Id + ufl.grad(u)
        E = 0.5 * (F.T * F - Id)
        J = ufl.det(F)
        sigma = ufl.inv(J) * F * (lambda_ * ufl.tr(E) * Id + 2.0 * mu * E) * F.T
        return sigma


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
