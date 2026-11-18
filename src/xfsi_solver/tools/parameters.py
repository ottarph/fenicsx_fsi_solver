# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Resolve the solver arguments that can be given in more than one, mutually exclusive, way."""

import math

# The solid materials of xfsi_solver.fsi.forms.FSIForms.
SOLID_MATERIALS = ("stvk", "mooney_rivlin_3")


def theta_value(dt_val, absolute_theta_shift, relative_theta_shift, default_relative_theta_shift) -> float:
    """Theta of the shifted Crank-Nicolson time stepping.

    theta = 0.5 + ``absolute_theta_shift``, or theta = 0.5 + ``relative_theta_shift`` * ``dt_val``. With
    neither shift given, theta = 0.5 + ``default_relative_theta_shift`` * ``dt_val``. Raises ``ValueError``
    if both are given.
    """
    if absolute_theta_shift is not None and relative_theta_shift is not None:
        raise ValueError("Give at most one of absolute_theta_shift and relative_theta_shift")
    if absolute_theta_shift is not None:
        return 0.5 + absolute_theta_shift
    if relative_theta_shift is None:
        relative_theta_shift = default_relative_theta_shift
    return 0.5 + relative_theta_shift * dt_val


def lame_lambda(mu_s, nu_s, lambda_s) -> float:
    """First Lamé parameter of the solid, ``lambda_s`` itself or computed from the shear modulus ``mu_s`` and
    the Poisson ratio ``nu_s`` as 2 mu_s nu_s / (1 - 2 nu_s). Raises ``ValueError`` unless exactly one of
    ``nu_s`` and ``lambda_s`` is given.

    The formula is evaluated as -mu_s / (1 - 0.5 / nu_s), which gives exactly 2e6 for the FSI2 values
    mu_s = 5e5, nu_s = 0.4, where 2 mu_s nu_s / (1 - 2 nu_s) is off in the last bit.
    """
    if nu_s is not None and lambda_s is not None:
        raise ValueError("Give at most one of nu_s and lambda_s")
    if lambda_s is not None:
        return lambda_s
    if nu_s is None:
        raise ValueError("Give one of nu_s and lambda_s")
    return -mu_s / (1.0 - 0.5 / nu_s)


def mooney_rivlin_3_coefficients(C10, C01, C11, D1, mu_s, lambda_s) -> dict:
    """The coefficients ``C10``, ``C01``, ``C11`` and ``D1`` of the compressible three-term Mooney-Rivlin
    solid (``xfsi_solver.fsi.materials.Solid.MR3_energy``), in Pa and 1/Pa, with D1 resolved.

    The Cij set the distortional (shear) response, with initial shear modulus 2 (C10 + C01). ``D1`` sets the
    volumetric one, with initial bulk modulus K0 = 2 / D1. With ``D1=None``, D1 is taken from the bulk
    modulus of the St. Venant-Kirchhoff parameters of the solver, K = lambda_s + 2 mu_s / 3, as D1 = 2 / K,
    so that only the shear response differs from that of the SVK solid; this is a provisional choice for
    lack of a measured D1, which an explicit ``D1`` overrides.

    Raises ``ValueError`` unless every coefficient is finite, C10, C01 and C11 are nonnegative, C10 + C01 is
    positive and D1 (or the K it comes from) is positive. These are basic checks of the input, not a proof
    that the material is stable at every deformation.
    """
    coefficients = {"C10": C10, "C01": C01, "C11": C11}
    for name, value in coefficients.items():
        if value is None or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative number, not {value!r}")
    if C10 + C01 <= 0.0:
        raise ValueError(f"C10 + C01, half the initial shear modulus, must be positive, not {C10 + C01!r}")
    if D1 is None:
        bulk_modulus = lambda_s + 2.0 * mu_s / 3.0
        if not math.isfinite(bulk_modulus) or bulk_modulus <= 0.0:
            raise ValueError(
                f"The bulk modulus lambda_s + 2 mu_s / 3 = {bulk_modulus!r} that D1 = 2 / K is taken from must be "
                "finite and positive"
            )
        D1 = 2.0 / bulk_modulus
    if not math.isfinite(D1) or D1 <= 0.0:
        raise ValueError(f"D1 must be a finite positive number, not {D1!r}")
    return coefficients | {"D1": D1}


def solid_material_coefficients(solid_material, mooney_rivlin_3, mu_s, lambda_s) -> dict | None:
    """The resolved Mooney-Rivlin coefficients of the solver arguments ``solid_material`` and
    ``mooney_rivlin_3``, or ``None`` for the St. Venant-Kirchhoff solid.

    ``solid_material`` is one of ``SOLID_MATERIALS``. With ``"mooney_rivlin_3"``, ``mooney_rivlin_3`` is a
    dict of ``"C10"``, ``"C01"``, ``"C11"`` and optionally ``"D1"``, resolved by ``mooney_rivlin_3_coefficients``
    with the SVK parameters ``mu_s`` and ``lambda_s``; with ``"stvk"`` it must be ``None``. Raises
    ``ValueError`` otherwise.
    """
    if solid_material not in SOLID_MATERIALS:
        raise ValueError(f"solid_material must be one of {SOLID_MATERIALS}, not {solid_material!r}")
    if solid_material == "stvk":
        if mooney_rivlin_3 is not None:
            raise ValueError(f"mooney_rivlin_3 is given, but solid_material is {solid_material!r}")
        return None
    if mooney_rivlin_3 is None or not {"C10", "C01", "C11"} <= set(mooney_rivlin_3) <= {"C10", "C01", "C11", "D1"}:
        raise ValueError(
            "solid_material='mooney_rivlin_3' needs mooney_rivlin_3, a dict of C10, C01, C11 and optionally D1, "
            f"not {mooney_rivlin_3!r}"
        )
    return mooney_rivlin_3_coefficients(
        mooney_rivlin_3["C10"],
        mooney_rivlin_3["C01"],
        mooney_rivlin_3["C11"],
        mooney_rivlin_3.get("D1"),
        mu_s,
        lambda_s,
    )
