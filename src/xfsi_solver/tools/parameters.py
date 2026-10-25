# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Resolve the solver arguments that can be given in more than one, mutually exclusive, way."""


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
