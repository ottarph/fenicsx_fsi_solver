import pytest

pytest.importorskip("ufl_units")

import ufl  # noqa: E402
from ufl_units import InconsistentDimensions, get_dimension  # noqa: E402

from xfsi_solver.scripts.dimensional_analysis_fsi2_biharmonic import (  # noqa: E402
    analyze,
    build_equations,
    pi_groups,
)


def test_every_equation_is_consistent():
    results = analyze()
    dimensions = {name: dimension for name, (dimension, _) in results.items()}
    # Dimensionless test functions on the 2D mesh: the integrand times an area.
    assert dimensions == {
        "u (mesh motion, fluid)": "length^-1",
        "u (kinematic, solid)": "mass^1 time^-1",
        "v (momentum)": "mass^1 time^-2",
        "p (mass)": "length^2 time^-1",
        "z (z = Laplacian of u)": "length^1",
    }
    terms = results["v (momentum)"][1]
    assert len(terms) == 10
    assert all(size > 0 for _, _, size in terms)


def test_rows_of_u_are_two_equations():
    equations, mapping, quantities = build_equations()
    both = equations["u (mesh motion, fluid)"] + equations["u (kinematic, solid)"]
    with pytest.raises(InconsistentDimensions):
        get_dimension(both, list(quantities.values()), mapping=mapping)


def test_inconsistent_term_is_found():
    equations, mapping, quantities = build_equations()
    # The mass equation plus the momentum equation, both tested with the same test function.
    momentum = equations["v (momentum)"]
    mass = equations["p (mass)"]
    dv = momentum.arguments()[0]
    dp = mass.arguments()[0]
    mixed = momentum + ufl.replace(mass, {dp: ufl.inner(dv, dv)})
    with pytest.raises(InconsistentDimensions):
        get_dimension(mixed, list(quantities.values()), mapping=mapping)


def test_pi_groups():
    values = sorted(value for _, value in pi_groups())
    # U_ref / L, 1 / Re, rho_s / rho_f, mu_s / (rho_f V^2), lambda_s / (rho_f V^2), V dt / L.
    assert values == pytest.approx(sorted([0.8, 0.01, 10.0, 500.0, 2000.0, 0.025]))
