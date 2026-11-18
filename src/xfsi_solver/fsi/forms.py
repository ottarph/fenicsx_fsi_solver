# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Variational forms of the monolithic ALE FSI problem in velocity-pressure form.

The unknowns are the velocity v on the whole mesh and the pressure p on the fluid. The solid
displacement is condensed out through the theta-scheme, u_s = u_s_old + dt * ((1 - theta) * v_old +
theta * v), and the fluid displacement u_f, which moves the mesh, is a coefficient of the forms that
the caller updates by a mesh motion solve.
"""

from dataclasses import dataclass

import dolfinx
import dolfinx.fem.petsc
import ufl

from xfsi_solver.fsi.materials import Fluid, Solid

SOLID_MATERIALS = ("stvk", "mooney_rivlin_3")


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

    def A_T_mesh_transport(self, v, v_old, u_f, u_f_old):
        F, J = _deformation(u_f)
        w = (u_f - u_f_old) / self.dt  # mesh velocity
        return -self.rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * w, self.dv) * self.dx_fluid

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

    def A_P(self, p, u_f):
        F, J = _deformation(u_f)
        return J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(self.dv)) * self.dx_fluid

    def build_residual(self, v, v_old, p, u_f, u_f_old, u_s_old):
        """Return the residual of one theta-scheme time step, with u_s condensed out.

        The momentum equation is multiplied by dt; the mass equation A_I is not.
        """
        dt, theta = self.dt, self.theta

        residual = dt * self.A_T(v, v_old, u_f, u_f_old)
        residual += self.A_I(v, u_f)
        residual += dt * self.A_P(p, u_f)
        residual += dt * theta * self.A_E_fluid(v, u_f)
        residual += dt * (1.0 - theta) * self.A_E_fluid(v_old, u_f_old)

        u_s_condensed = u_s_old + dt * ((1.0 - theta) * v_old + theta * v)
        residual += dt * theta * self.A_E_solid(u_s_condensed)
        residual += dt * (1.0 - theta) * self.A_E_solid(u_s_old)

        return residual
