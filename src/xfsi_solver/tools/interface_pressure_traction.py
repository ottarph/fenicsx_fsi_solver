# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""The interface pressure traction terms of the Jacobian of the solvers on the velocity and pressures only
(fsi2_simplified, threeDfsi_simplified, fsi2_incompressible).

These solvers set the interface values of the fluid displacement u_f by the kinematic relation,
u_f = u_f_old + dt * (theta * v + (1 - theta) * v_old), and their Jacobian is that of the residual with
respect to (v, p) alone (and the solid pressure p_S in fsi2_incompressible), so it lacks the derivative
of the fluid pressure's force on the interface with respect to the interface's motion.

By the Piola identity, div(cof(F)) = 0 with cof(F) = det(F) F^-T and F = I + grad(u_f), the pressure term
of the momentum equation, -p cof(F) : grad(dv) on the fluid, equals grad(p) . cof(F) dv on the fluid minus
the traction p cof(F) N . dv on its boundary, with N the outward normal from the fluid. On the interface,
cof(F) N, the deformed normal times the stretch of the interface (Nanson's formula), depends only on the
tangential derivatives of u_f, so only on its interface values. The terms here are

    dt * theta * (-p) (cof(F(u_interface)) - cof(F(u_f))) N . dv

on the fluid side of the interface, with u_interface = u_f_old + dt * (theta * v + (1 - theta) * v_old).
The two cofactors agree wherever u_f holds the kinematic relation on the interface, so the terms are zero,
but their derivative with respect to v is the derivative of the pressure traction's geometric factor
through the interface's motion, which the Jacobian then holds. They are added to the form the Jacobian is
derived from, not to the residual, so the solution does not depend on them.

The part of the pressure term's missing derivative that grows with the mean fluid pressure is all in this
factor: what is left out, the derivative of grad(p) . cof(F) dv inside the fluid, depends on grad(p) only.
So, like the interface ALE terms of ``xfsi_solver.tools.interface_ale``, the terms keep Newton's
convergence independent of the mean pressure, at the cost of one integral over the interface facets. They
leave out the derivatives of the other terms with respect to the interface's motion, which the interface
ALE terms hold. The interface ALE terms already hold the derivative of the pressure traction's geometric
factor, so the two do not go together.
"""

import dolfinx
import dolfinx.fem.forms
import ufl
import ufl.algorithms

from xfsi_solver.tools.quadrature import fix_quadrature_degree


def check_interface_pressure_traction_jacobian(enabled, interface_ale: bool, quadrature_degree=None) -> None:
    """Raise ``ValueError`` unless ``enabled`` is a bool, and not True together with ``interface_ale``, and
    ``quadrature_degree`` is None or a non-negative integer."""
    if quadrature_degree is not None and (
        isinstance(quadrature_degree, bool) or not isinstance(quadrature_degree, int) or quadrature_degree < 0
    ):
        raise ValueError(
            "interface_pressure_traction_quadrature_degree must be None or a non-negative integer, "
            f"not {quadrature_degree!r}"
        )
    if not isinstance(enabled, bool):
        raise ValueError(f"interface_pressure_traction_jacobian must be True or False, not {enabled!r}")
    if enabled and interface_ale:
        raise ValueError(
            "interface_pressure_traction_jacobian does not go with interface_ale, whose terms already hold the "
            "derivative of the interface pressure traction"
        )


def cofactor(u):
    """cof(F) = det(F) F^-T of F = I + grad(u); UFL writes it as a polynomial in grad(u), without an inverse."""
    return ufl.cofac(ufl.Identity(u.ufl_shape[0]) + ufl.grad(u))


def interface_pressure_traction_form(
    dv, dt, theta, v, v_old, p, u_f, u_f_old, ds_interface, quadrature_degree: int | None = None
):
    """The interface pressure traction terms, as a form in the momentum rows, with the test function ``dv``:
    dt * theta * (-p) (cof(F(u_interface)) - cof(F(u_f))) N . dv over ``ds_interface``, the fluid side of the
    interface. They are zero wherever u_f holds the kinematic relation on the interface.

    With ``quadrature_degree``, they are integrated with that quadrature degree, which makes their
    derivative an approximation of that of the pressure traction, cheaper to assemble at a low degree.
    Otherwise ``jacobian_blocks`` sets the degree UFL estimates for them, capped as the residual's."""
    u_interface = u_f_old + dt * (theta * v + (1.0 - theta) * v_old)
    n = ufl.FacetNormal(ds_interface.ufl_domain())
    form = dt * theta * ufl.inner(-p * (cofactor(u_interface) - cofactor(u_f)) * n, dv) * ds_interface
    if quadrature_degree is None:
        return form
    return ufl.Form(
        [
            integral.reconstruct(metadata=integral.metadata() | {"quadrature_degree": quadrature_degree})
            for integral in form.integrals()
        ]
    )


def jacobian_blocks(residual_blocks, unknowns, traction=None, max_quadrature_degree=None):
    """The Jacobian blocks of ``residual_blocks`` with respect to ``unknowns``, with the interface pressure
    traction terms ``traction`` (from ``interface_pressure_traction_form``, or None) added to the first,
    momentum, block before differentiating. Unless its quadrature degree is set already, ``traction`` gets
    it as the residual blocks do from ``fix_quadrature_degree``, with ``max_quadrature_degree``.

    dolfinx needs the facet integrals of a form to share their subdomain data, and the interface measure of
    ``traction`` has its own. The derivatives of the momentum row are therefore expanded, which drops the
    facet integrals of the residual that do not depend on the unknowns, such as an outlet traction. Facet
    integrals that do depend on them must be on the same measure as ``traction``."""
    if traction is None:
        return dolfinx.fem.forms.derivative_block(residual_blocks, unknowns)
    traction = fix_quadrature_degree(traction, max_quadrature_degree)
    momentum, *others = dolfinx.fem.forms.derivative_block(
        [residual_blocks[0] + traction, *residual_blocks[1:]], unknowns
    )
    return [[ufl.algorithms.expand_derivatives(block) for block in momentum], *others]


def interface_pressure_traction_description(enabled: bool, quadrature_degree: int | None = None) -> str:
    """The interface pressure traction terms, for the summary at the end of a run."""
    if not enabled:
        return "off"
    description = "on, the derivative of the pressure's interface traction through the interface's motion"
    if quadrature_degree is None:
        return f"{description},\n    integrated with the degree UFL estimates, capped as the residual's"
    return f"{description},\n    integrated with quadrature degree {quadrature_degree}"
