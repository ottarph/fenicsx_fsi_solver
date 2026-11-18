import numpy as np
import pytest

from xfsi_solver.tools.parameters import lame_lambda, mooney_rivlin_3_coefficients, theta_value


def test_theta_value_defaults_to_relative_shift():
    assert np.isclose(theta_value(0.01, None, None, 1.0), 0.51)
    assert theta_value(0.01, None, None, 0.0) == 0.5


def test_theta_value_absolute_and_relative():
    assert np.isclose(theta_value(0.01, 0.1, None, 1.0), 0.6)
    assert np.isclose(theta_value(0.01, None, 2.0, 1.0), 0.52)


def test_theta_value_rejects_both_shifts():
    with pytest.raises(ValueError):
        theta_value(0.01, 0.1, 2.0, 1.0)


def test_lame_lambda():
    # FSI2: mu_s = 5e5, nu_s = 0.4 gives lambda_s = 2e6
    assert lame_lambda(5.0e5, 0.4, None) == 2.0e6
    assert lame_lambda(5.0e5, None, 3.0e6) == 3.0e6


def test_lame_lambda_rejects_both_or_neither():
    with pytest.raises(ValueError):
        lame_lambda(5.0e5, 0.4, 2.0e6)
    with pytest.raises(ValueError):
        lame_lambda(5.0e5, None, None)


def test_mooney_rivlin_3_d1_defaults_to_the_svk_bulk_modulus():
    # FSI2: K = lambda_s + 2 mu_s / 3 = 2e6 + 1e6 / 3, D1 = 2 / K = 6 / 7e6
    coefficients = mooney_rivlin_3_coefficients(0.0, 2.5e5, 1.8e6, None, 5.0e5, 2.0e6)
    assert coefficients["D1"] == pytest.approx(8.571428571428571e-7, rel=1e-15)
    assert coefficients == {"C10": 0.0, "C01": 2.5e5, "C11": 1.8e6, "D1": coefficients["D1"]}
    # An explicit D1 overrides it
    assert mooney_rivlin_3_coefficients(0.0, 2.5e5, 1.8e6, 1e-6, 5.0e5, 2.0e6)["D1"] == 1e-6


@pytest.mark.parametrize(
    "C10, C01, C11, D1, lambda_s",
    [
        (-1.0, 2.5e5, 0.0, None, 2.0e6),  # negative coefficient
        (0.0, 2.5e5, -1.0, None, 2.0e6),
        (0.0, 0.0, 1.8e6, None, 2.0e6),  # no initial shear stiffness
        (0.0, float("nan"), 0.0, None, 2.0e6),
        (0.0, 2.5e5, float("inf"), None, 2.0e6),
        (0.0, 2.5e5, 0.0, 0.0, 2.0e6),  # D1 must be positive
        (0.0, 2.5e5, 0.0, -1e-6, 2.0e6),
        (0.0, 2.5e5, 0.0, float("inf"), 2.0e6),
        (0.0, 2.5e5, 0.0, None, -1.0e6),  # K = lambda_s + 2 mu_s / 3 < 0
    ],
)
def test_mooney_rivlin_3_rejects_invalid_coefficients(C10, C01, C11, D1, lambda_s):
    with pytest.raises(ValueError):
        mooney_rivlin_3_coefficients(C10, C01, C11, D1, 5.0e5, lambda_s)
