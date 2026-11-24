# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Dimensional analysis of the residual of fsi2_biharmonic_fulljacobian_initialguess.

The residual is built by ``residual_forms`` of the solver on plain UFL meshes and coefficients, and
``ufl_units`` substitutes a unit-carrying reference scale for every unknown, physical constant and
the mesh coordinates. Each equation is then checked for consistent dimensions, and the size of each
of its terms is estimated from the reference scales. Needs the optional dependencies ``ufl-units``
and ``sympy`` (``pip install --no-deps ufl-units sympy mpmath``).

The equations are the rows of each test function, with the rows of u split in two: on the fluid
cells they hold the biharmonic mesh motion, and on the solid cells, the interface included, the
solid kinematic relation, which the solver assembles there in place of the mesh motion. The two
have different dimensions, so the rows of u are not one equation.

Test functions are taken as dimensionless, so the dimension of an equation is that of its
integrand times the measure, an area on the 2D mesh. The size of a term is the product of the
reference scales ``ufl_units`` pulls out of it, which is the size of its leading part: for a term
nonlinear in grad u, the part with grad u = 0. For the solid stress, whose leading part cancels
(E = (F^T F - I) / 2), the size is too large by the strain, about U_ref / L, and for the
differences in time, (v - v_old) / dt and (u - u_old) / dt, the size is that of the whole
velocity over dt, an upper bound of the change in one step.

theta is dimensionless here. In the solver it is 0.5 + ``relative_theta_shift`` * dt, so the
relative shift there carries units of 1 / time.
"""

import math
from fractions import Fraction

import basix.ufl
import numpy as np
import sympy.physics.units as un
import ufl
from sympy.physics.units.systems.si import dimsys_SI
from ufl_units import Quantity, expand, factorize, get_dimension

from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import residual_forms

# Physical constants of FSI2, in SI units, as the solver's defaults.
PHYSICAL = {
    "rho_f": (1.0e3, un.kg / un.m**3),
    "nu_f": (1.0e-3, un.m**2 / un.s),
    "rho_s": (1.0e4, un.kg / un.m**3),
    "mu_s": (5.0e5, un.Pa),
    "lambda_s": (2.0e6, un.Pa),
    "dt": (2.5e-3, un.s),
    "theta": (0.5 + 2.5e-3, None),
}

# Reference scales of FSI2: the cylinder diameter, the amplitude of the vertical tip displacement
# (about 0.08 m), the mean inflow speed, which is also about the largest speed of the solid, the
# dynamic pressure of the inflow, and z = Laplacian of u of the displacement over the length.
REFERENCE = {
    "L": (0.1, un.m),
    "U_ref": (0.08, un.m),
    "V_ref": (1.0, un.m / un.s),
    "P_ref": (1.0e3, un.Pa),
    "Z_ref": (0.08 / 0.1**2, 1 / un.m),
}

# The quantities of the Buckingham Pi analysis: those that define the problem, not P_ref and Z_ref,
# which are derived from them.
PI_QUANTITIES = ("L", "U_ref", "V_ref", "rho_f", "nu_f", "rho_s", "mu_s", "lambda_s", "dt")


def build_equations(physical=PHYSICAL, reference=REFERENCE):
    """The equations of the residual on plain UFL meshes, with the mapping and the quantities.

    Returns ``(equations, mapping, quantities)``: ``equations`` a dict from the name of each
    equation to its UFL form, ``mapping`` the substitution of the unit-carrying scales for the
    unknowns, constants and the mesh scale, and ``quantities`` a dict from name to ``Quantity``.
    """
    cell = "triangle"
    coordinate_element = basix.ufl.element("Lagrange", cell, 1, shape=(2,))
    mesh = ufl.Mesh(coordinate_element)
    fluid_mesh = ufl.Mesh(coordinate_element)
    vector_element = basix.ufl.element("Lagrange", cell, 2, shape=(2,))
    U = ufl.FunctionSpace(mesh, vector_element)
    P = ufl.FunctionSpace(fluid_mesh, basix.ufl.element("Lagrange", cell, 1))
    Z = ufl.FunctionSpace(fluid_mesh, vector_element)
    W = ufl.MixedFunctionSpace(U, U, P, Z)

    u, v, u_old, v_old = (ufl.Coefficient(U) for _ in range(4))
    p, z = ufl.Coefficient(P), ufl.Coefficient(Z)
    du, dv, dp, dz = ufl.TestFunctions(W)
    constants = {name: ufl.Constant(mesh) for name in physical}
    dx = ufl.Measure("dx", domain=mesh)
    residual, kinematic = residual_forms(
        (u, v, p, z), (u_old, v_old), (du, dv, dp, dz), constants, dx_fluid=dx(2), dx_solid=dx(1)
    )
    blocks = ufl.extract_blocks(residual)
    equations = {
        "u (mesh motion, fluid)": blocks[0],
        "u (kinematic, solid)": kinematic,
        "v (momentum)": blocks[1],
        "p (mass)": blocks[2],
        "z (z = Laplacian of u)": blocks[3],
    }

    quantities = {name: Quantity(mesh, scale, unit, name) for name, (scale, unit) in {**reference, **physical}.items()}
    q = quantities
    mapping = {
        mesh: q["L"],
        u: q["U_ref"] * u,
        u_old: q["U_ref"] * u_old,
        v: q["V_ref"] * v,
        v_old: q["V_ref"] * v_old,
        p: q["P_ref"] * p,
        z: q["Z_ref"] * z,
    }
    mapping.update({constants[name]: q[name] for name in physical})
    return equations, mapping, quantities


def base_units(dimension) -> str:
    """A dimension in the SI base dimensions, for example ``mass time^-2``, or ``1`` if dimensionless."""
    dependencies = dimsys_SI.get_dimensional_dependencies(dimension)
    return " ".join(f"{base.name}^{exponent}" for base, exponent in dependencies.items()) or "1"


def analyze(physical=PHYSICAL, reference=REFERENCE):
    """Check each equation for consistent dimensions and estimate the size of each of its terms.

    Raises ``ufl_units.InconsistentDimensions`` if the terms of an equation differ in dimension.
    Returns a dict from the name of each equation to ``(dimension, terms)``, with ``terms`` a list of
    ``(subdomain, factor, size)`` per integral: ``factor`` the product of reference scales, as a
    sympy expression of their symbols, and ``size`` its value in SI base units.
    """
    equations, mapping, quantities = build_equations(physical, reference)
    quantity_list = list(quantities.values())
    symbols = [q.symbol for q in quantity_list]
    factors = np.array([q.factor for q in quantity_list])

    results = {}
    for name, form in equations.items():
        dimension = get_dimension(form, quantity_list, mapping=mapping)
        terms = []
        for integral in form.integrals():
            exponents = factorize(ufl.Form([integral]), quantity_list, mapping=mapping).factor
            size = float(np.prod(factors ** np.array([float(e) for e in exponents])))
            terms.append((integral.subdomain_id(), expand([Fraction(e) for e in exponents], symbols), size))
        results[name] = (base_units(dimension), terms)
    return results


def pi_groups(physical=PHYSICAL, reference=REFERENCE, names=PI_QUANTITIES):
    """The dimensionless groups of the quantities ``names``, as ``(expression, value)`` pairs."""
    from ufl_units import buckingham_pi_analysis

    _, _, quantities = build_equations(physical, reference)
    chosen = [quantities[name] for name in names]
    _, _, groups = buckingham_pi_analysis(chosen)
    result = []
    for group in groups:
        expression = expand(list(group), [q.symbol for q in chosen])
        value = math.prod(q.factor ** float(e) for q, e in zip(chosen, group, strict=True))
        result.append((expression, value))
    return result


SUBDOMAINS = {1: "solid", 2: "fluid"}


def main():
    results = analyze()
    for name, (dimension, terms) in results.items():
        largest = max(size for _, _, size in terms)
        print(f"\n{name}: consistent, dimension {dimension}, largest term {largest:.3g}")
        for subdomain, factor, size in terms:
            print(f"  {SUBDOMAINS.get(subdomain, subdomain):5s}  {size:10.3g}  {size / largest:9.2e}  {factor}")

    print("\nDimensionless groups:")
    for expression, value in pi_groups():
        print(f"  {value:10.3g}  {expression}")


if __name__ == "__main__":
    main()
