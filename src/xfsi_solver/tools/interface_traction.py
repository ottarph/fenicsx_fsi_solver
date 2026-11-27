# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""The interface traction terms of the Jacobian of the solvers on (v, p) only (fsi2_simplified,
threeDfsi_simplified).

These solvers set the interface values of the fluid displacement u_f by the kinematic relation,
u_f = u_f_old + dt * (theta * v + (1 - theta) * v_old), and their Jacobian is that of the residual with
respect to (v, p) alone, so it lacks the derivative of the fluid's force on the interface with respect to
the interface's motion.

By the Piola identity, div(cof(F)) = 0 with cof(F) = det(F) F^-T and F = I + grad(u_f), a fluid stress
sigma acts on the interface through the traction sigma cof(F) N . dv on the reference interface, with N
its outward normal from the fluid. cof(F) N, the deformed normal times the stretch of the interface
(Nanson's formula), depends only on the tangential derivatives of u_f there, so only on its interface
values. The terms here are

    dt * theta * sigma (cof(F(u_interface)) - cof(F(u_f))) N . dv

on the fluid side of the interface, with u_interface = u_f_old + dt * (theta * v + (1 - theta) * v_old).
The two cofactors agree wherever u_f holds the kinematic relation on the interface, so the terms are zero,
but their derivative with respect to v is the derivative of the traction's geometric factor through the
interface's motion, which the Jacobian then holds. They are added to the form the Jacobian is derived
from, not to the residual, so the solution does not depend on them.

With sigma = -p I, the derivative of the pressure term that grows with the mean fluid pressure is all in
this factor: what is left out, the derivative of grad(p) . cof(F) dv inside the fluid, depends on grad(p)
only. So, like the interface ALE terms of ``xfsi_solver.tools.interface_ale``, the terms keep Newton's
convergence independent of the mean pressure, at the cost of one integral over the interface facets. They
leave out the derivatives of the other terms with respect to the interface's motion, which the interface
ALE terms hold, and the derivative of the viscous stress itself, which needs the normal derivative of u_f.
The interface ALE terms already hold the derivative of the traction's geometric factor, so the two do not
go together.
"""

import dolfinx
import dolfinx.fem.forms
import ufl

from xfsi_solver.fsi.forms import FSIForms
from xfsi_solver.fsi.materials import Fluid
from xfsi_solver.tools.quadrature import fix_quadrature_degree

#: The stresses whose interface traction can be added to the Jacobian: the pressure, -p I, and the full
#: fluid stress, the pressure and the viscous stress.
STRESSES = ("pressure", "stress")


def check_interface_traction_jacobian(stress, interface_ale: bool) -> None:
    """Raise ``ValueError`` unless ``stress`` is None or one of ``STRESSES``, and not given with
    ``interface_ale``."""
    if stress is not None and stress not in STRESSES:
        raise ValueError(f"interface_traction_jacobian must be None or one of {STRESSES}, not {stress!r}")
    if stress is not None and interface_ale:
        raise ValueError(
            "interface_traction_jacobian does not go with interface_ale, whose terms already hold the derivative "
            "of the interface traction"
        )


def cofactor(u):
    """cof(F) = det(F) F^-T of F = I + grad(u); UFL writes it as a polynomial in grad(u), without an inverse."""
    return ufl.cofac(ufl.Identity(u.ufl_shape[0]) + ufl.grad(u))


def interface_traction_form(stress, forms: FSIForms, v, v_old, p, u_f, u_f_old, ds_interface):
    """The interface traction terms of ``stress`` (one of ``STRESSES``, or None for none), as a form in the
    momentum rows: dt * theta * sigma (cof(F(u_interface)) - cof(F(u_f))) N . dv over ``ds_interface``, the
    fluid side of the interface, with the test function, dt, theta and the fluid's parameters of ``forms``.
    They are zero wherever u_f holds the kinematic relation on the interface."""
    if stress is None:
        return None
    dt, theta = forms.dt, forms.theta
    u_interface = u_f_old + dt * (theta * v + (1.0 - theta) * v_old)
    sigma = -p * ufl.Identity(v.ufl_shape[0])
    if stress == "stress":
        sigma += Fluid.NS_velocity(u_f, v, forms.nu_f, forms.rho_f)
    n = ufl.FacetNormal(ds_interface.ufl_domain())
    return dt * theta * ufl.inner(sigma * (cofactor(u_interface) - cofactor(u_f)) * n, forms.dv) * ds_interface


def jacobian_blocks(residual_blocks, unknowns, traction=None, max_quadrature_degree=None):
    """The Jacobian blocks of ``residual_blocks`` with respect to ``unknowns``, with the interface traction
    terms ``traction`` (from ``interface_traction_form``, or None) added to the first, momentum, block before
    differentiating. ``traction`` gets its quadrature degree as the residual blocks do from
    ``fix_quadrature_degree``, with ``max_quadrature_degree``."""
    if traction is not None:
        traction = fix_quadrature_degree(traction, max_quadrature_degree)
        residual_blocks = [residual_blocks[0] + traction, *residual_blocks[1:]]
    return dolfinx.fem.forms.derivative_block(residual_blocks, unknowns)


def interface_traction_description(stress) -> str:
    """The interface traction terms, for the summary at the end of a run."""
    if stress is None:
        return "off"
    traction = "the pressure's" if stress == "pressure" else "the fluid stress's (pressure and viscous)"
    return f"on, the derivative of {traction} interface traction through the interface's motion"
