import numpy as np
import pytest

from xfsi_solver.tools.parameters import lame_lambda, theta_value


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
    assert np.isclose(lame_lambda(5.0e5, 0.4, None), 2.0e6)
    assert lame_lambda(5.0e5, None, 3.0e6) == 3.0e6


def test_lame_lambda_rejects_both_or_neither():
    with pytest.raises(ValueError):
        lame_lambda(5.0e5, 0.4, 2.0e6)
    with pytest.raises(ValueError):
        lame_lambda(5.0e5, None, None)
