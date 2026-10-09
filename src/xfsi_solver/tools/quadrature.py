# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""A cap on the quadrature degree of UFL forms.

UFL estimates the polynomial degree of each integrand and FFCx integrates it exactly up to
that degree. For the nonlinear ALE and St. Venant-Kirchhoff terms, with inv(F) and det(F) of
P2 displacements on a second order mesh, the estimate is high, and in 3d the number of
quadrature points per tetrahedron grows quickly with the degree. Capping the degree makes the
integration of these terms inexact (a variational crime), so the cap should stay well above
the degree of the linear terms, which are then still integrated exactly.
"""

import ufl


def estimated_degree(form: ufl.Form) -> int:
    """The highest quadrature degree that UFL estimates for the integrals of ``form``.

    This is the estimate FFCx uses when no quadrature degree is set, computed as in
    https://jsdokken.com/FEniCS-workshop/src/unified_form_language/ufl_forms.html.
    """
    form_data = ufl.algorithms.compute_form_data(
        form,
        do_apply_function_pullbacks=True,
        do_apply_integral_scaling=True,
        do_apply_geometry_lowering=True,
        preserve_geometry_types=(ufl.classes.Jacobian,),
    )
    # An integrand that UFL simplifies to zero leaves no integrals, and degree 0.
    return max(
        (
            integral.metadata()["estimated_polynomial_degree"]
            for integral_data in form_data.integral_data
            for integral in integral_data.integrals
        ),
        default=0,
    )


def cap_quadrature_degree(form: ufl.Form, max_degree: int) -> ufl.Form:
    """``form``, with the quadrature degree set to ``max_degree`` in the integrals for which
    UFL estimates a higher degree. The other integrals, and those with a quadrature degree
    already set, are unchanged. Forms derived from the result, such as its derivative, keep the
    quadrature degree of each integral.
    """
    integrals = []
    for integral in form.integrals():
        metadata = integral.metadata()
        if "quadrature_degree" not in metadata and estimated_degree(ufl.Form([integral])) > max_degree:
            integral = integral.reconstruct(metadata=metadata | {"quadrature_degree": max_degree})
        integrals.append(integral)
    return ufl.Form(integrals)
