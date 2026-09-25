import numpy as np
import pytest

from xfsi_solver.scripts.fsi2_qoi_statistics import deviations, periodic_statistics


def test_periodic_statistics_of_noisy_sine():
    t = np.arange(0.0, 5.0, 0.001)
    rng = np.random.default_rng(0)
    y = 3.0 + 2.0 * np.sin(2 * np.pi * 1.9 * t) + 0.3 * np.sin(2 * np.pi * 3.8 * t) + 0.02 * rng.standard_normal(t.size)
    s = periodic_statistics(t, y, (1.0, 4.9))
    assert s["frequency"] == pytest.approx(1.9, rel=2e-3)
    assert s["periods"] >= 6
    # extrema of a + b sin(x) + c sin(2x): the mean shifts by the second harmonic
    x = np.linspace(0, 2 * np.pi, 100001)
    f = 2.0 * np.sin(x) + 0.3 * np.sin(2 * x)
    assert s["amplitude"] == pytest.approx(0.5 * (f.max() - f.min()), rel=0.02)
    assert s["mean"] == pytest.approx(3.0 + 0.5 * (f.max() + f.min()), abs=0.05)

    d = deviations({"A_y": s, **{k: s for k in ("drag", "lift", "A_x")}},
                   {"A_y": s, **{k: s for k in ("drag", "lift", "A_x")}})
    assert all(v == 0.0 for q in d.values() for v in q.values())


def test_non_oscillating_signal():
    t = np.linspace(0, 1, 100)
    s = periodic_statistics(t, t, (0.0, 1.0))
    assert s["periods"] == 0 and np.isnan(s["frequency"])
