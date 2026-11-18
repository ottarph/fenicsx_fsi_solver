# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Variational forms of the monolithic ALE FSI problem in velocity-pressure form.

The unknowns are the velocity v on the whole mesh and the pressure p on the fluid. The solid
displacement is condensed out through the theta-scheme, u_s = u_s_old + dt * ((1 - theta) * v_old +
theta * v), and the fluid displacement u_f, which moves the mesh, is a coefficient of the forms that
the caller updates by a mesh motion solve.

Every term that depends on the time level is theta-weighted, including the mesh transport term and the
geometric factor J F^-T of the pressure term, so that the scheme is second order in time for theta =
1/2 + O(dt) on a moving mesh as well. The pressure is the implicit multiplier of the step (it
approximates the pressure at t^{n+theta}), and the mass equation A_I is imposed at t^{n+1}, following
Wick (2011).
"""

from dataclasses import dataclass

import dolfinx
import dolfinx.fem.petsc
import ufl

from xfsi_solver.fsi.materials import Fluid, Solid
from xfsi_solver.tools.parameters import SOLID_MATERIALS


def _deformation(u):
    """Return the deformation gradient F = I + grad(u) and its determinant J."""
    F = ufl.Identity(u.ufl_shape[0]) + ufl.grad(u)
    return F, ufl.det(F)


@dataclass(frozen=True)
class FSIForms:
    """The groups of terms of the FSI residual, tested with ``dv`` (momentum) and ``dp`` (mass).

    ``dx_fluid`` and ``dx_solid`` are the cell measures of the fluid and solid on the parent mesh. The
    material parameters, ``dt`` and ``theta`` are constants, so changing their values needs no new forms.
    """

    dv: ufl.Argument
    dp: ufl.Argument
    dx_fluid: ufl.Measure
    dx_solid: ufl.Measure
    rho_f: dolfinx.fem.Constant
    nu_f: dolfinx.fem.Constant
    rho_s: dolfinx.fem.Constant
    mu_s: dolfinx.fem.Constant
    lambda_s: dolfinx.fem.Constant
    dt: dolfinx.fem.Constant
    theta: dolfinx.fem.Constant
    solid_material: str = "stvk"
    mooney_rivlin_3: dict | None = None

    def __post_init__(self):
        if self.solid_material not in SOLID_MATERIALS:
            raise ValueError(f"solid_material must be one of {SOLID_MATERIALS}, not {self.solid_material!r}")
        if self.solid_material == "mooney_rivlin_3":
            if self.mooney_rivlin_3 is None or set(self.mooney_rivlin_3) != {"C10", "C01", "C11", "D1"}:
                raise ValueError("solid_material='mooney_rivlin_3' needs mooney_rivlin_3 with C10, C01, C11 and D1")
        elif self.mooney_rivlin_3 is not None:
            raise ValueError(f"mooney_rivlin_3 is given, but solid_material is {self.solid_material!r}")

    def _mesh_transport(self, v, u_f, w):
        """The ALE transport by the mesh velocity w, -rho_f J grad(v) F^-1 w, at one time level."""
        F, J = _deformation(u_f)
        return -self.rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * w, self.dv) * self.dx_fluid

    def A_T_mesh_transport(self, v, v_old, u_f, u_f_old):
        """The mesh transport term, theta-weighted between the new and old time levels, with the mesh
        velocity w = (u_f - u_f_old) / dt of the step in both. Evaluated at the new time level only, it
        makes the scheme first order in time on a moving mesh."""
        w = (u_f - u_f_old) / self.dt
        return self.theta * self._mesh_transport(v, u_f, w) + (1.0 - self.theta) * self._mesh_transport(
            v_old, u_f_old, w
        )

    def A_T_mass(self, v, v_old, u_f, u_f_old):
        _, J = _deformation(u_f)
        _, J_old = _deformation(u_f_old)
        J_mid = 0.5 * (J + J_old)

        residual = self.rho_f * J_mid * ufl.inner((v - v_old) / self.dt, self.dv) * self.dx_fluid
        residual += self.rho_s * ufl.inner((v - v_old) / self.dt, self.dv) * self.dx_solid
        return residual

    def A_T(self, v, v_old, u_f, u_f_old):
        return self.A_T_mass(v, v_old, u_f, u_f_old) + self.A_T_mesh_transport(v, v_old, u_f, u_f_old)

    def A_I(self, v, u_f):
        F, J = _deformation(u_f)
        return ufl.div(J * ufl.inv(F) * v) * self.dp * self.dx_fluid

    def A_E_fluid_viscous(self, v, u_f):
        F, J = _deformation(u_f)
        sigma = Fluid.NS_velocity(u_f, v, self.nu_f, self.rho_f)
        return ufl.inner(J * sigma * ufl.inv(F).T, ufl.grad(self.dv)) * self.dx_fluid

    def A_E_fluid_transport(self, v, u_f):
        F, J = _deformation(u_f)
        return self.rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, self.dv) * self.dx_fluid

    def A_E_fluid(self, v, u_f):
        return self.A_E_fluid_viscous(v, u_f) + self.A_E_fluid_transport(v, u_f)

    def solid_piola(self, u_s):
        """The first Piola-Kirchhoff stress of the solid at the displacement u_s.

        The St. Venant-Kirchhoff law gives the Cauchy stress sigma, pulled back as J sigma F^-T; the
        Mooney-Rivlin law gives the first Piola stress itself.
        """
        if self.solid_material == "mooney_rivlin_3":
            return Solid.MR3_piola(u_s, **self.mooney_rivlin_3)
        F, J = _deformation(u_s)
        sigma = Solid.STVK(u_s, self.lambda_s, self.mu_s)
        return J * sigma * ufl.inv(F).T

    def A_E_solid(self, u_s):
        return ufl.inner(self.solid_piola(u_s), ufl.grad(self.dv)) * self.dx_solid

    def A_P(self, p, u_f, u_f_old):
        """The pressure term with the implicit pressure p and the geometric factor J F^-T theta-weighted
        between the new and old time levels. With J F^-T at the new level only, the scheme is first
        order in time on a moving mesh."""
        F, J = _deformation(u_f)
        F_old, J_old = _deformation(u_f_old)
        geometry = self.theta * J * ufl.inv(F).T + (1.0 - self.theta) * J_old * ufl.inv(F_old).T
        return ufl.inner(Fluid.NS_pressure(p) * geometry, ufl.grad(self.dv)) * self.dx_fluid

    def build_residual(self, v, v_old, p, u_f, u_f_old, u_s_old):
        """Return the residual of one theta-scheme time step, with u_s condensed out.

        The momentum equation is multiplied by dt; the mass equation A_I is not.
        """
        dt, theta = self.dt, self.theta

        residual = dt * self.A_T(v, v_old, u_f, u_f_old)
        residual += self.A_I(v, u_f)
        residual += dt * self.A_P(p, u_f, u_f_old)
        residual += dt * theta * self.A_E_fluid(v, u_f)
        residual += dt * (1.0 - theta) * self.A_E_fluid(v_old, u_f_old)

        u_s_condensed = u_s_old + dt * ((1.0 - theta) * v_old + theta * v)
        residual += dt * theta * self.A_E_solid(u_s_condensed)
        residual += dt * (1.0 - theta) * self.A_E_solid(u_s_old)

        return residual
