import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from xfsi_solver.scripts.plot_fsi3d_time_steps import frequency, main, summary_rows  # noqa: E402

HEADER = "t\tdrag\tlift\tB_x\tB_y\tB_z"


def _write_run(path, dt, f=5.0, phase=0.0):
    """A QoI file from t = 8 to 10 with every QoI oscillating at f Hz around a known mean."""
    t = np.arange(8.0, 10.0 + 0.5 * dt, dt)
    wave = np.sin(2 * np.pi * f * t + phase)
    columns = [t, 180.0 + 4.0 * wave, 40.0 * wave, -2e-3 + 2e-3 * wave, 2e-3 + 25e-3 * wave, 0.5e-3 + 0.8e-3 * wave]
    np.savetxt(path, np.column_stack(columns), delimiter="\t", header=HEADER)
    return path


def test_frequency_of_a_sine():
    t = np.linspace(0.0, 1.0, 4001)
    assert frequency(t, 3.0 + np.sin(2 * np.pi * 5.75 * t)) == pytest.approx(5.75, rel=1e-4)


def test_summary_rows_hold_mean_amplitude_and_frequency():
    t = np.linspace(9.0, 10.0, 4001)
    wave = np.sin(2 * np.pi * 5.0 * t)
    qois = {"t": t, "drag": 180 + 4 * wave, "lift": 40 * wave, "B_x": 1e-3 * wave, "B_y": 2e-3 * wave, "B_z": 0 * t}
    (label, cells, f), *paper = summary_rows({"run": qois})
    assert label == "run"
    drag, lift, b_x, b_y, b_z = cells
    assert drag == pytest.approx((180.0, 4.0), abs=1e-6)
    assert lift == pytest.approx((0.0, 40.0), abs=1e-6)
    assert b_y == pytest.approx((0.0, 2.0), abs=1e-6)  # in mm
    assert f == pytest.approx(5.0, rel=1e-4)
    assert len(paper) == 3


def test_main_writes_the_figures_and_the_table(tmp_path, capsys):
    runs = [
        ("dt = 0.004", _write_run(tmp_path / "dt0.004.txt", 0.004, phase=0.02)),
        ("dt = 0.001", _write_run(tmp_path / "dt0.001.txt", 0.001)),
    ]
    medium = _write_run(tmp_path / "medium.txt", 0.004, f=4.8)
    out = tmp_path / "figures"
    argv = [arg for label, path in runs for arg in ("--run", label, str(path))]
    main([*argv, "--compare", "medium", str(medium), "--output-dir", str(out)])

    for name in ("fsi3d_time_steps", "fsi3d_time_step_differences"):
        for extension in ("pdf", "png"):
            assert (out / f"{name}.{extension}").stat().st_size > 0
    table = capsys.readouterr().out
    assert "| dt = 0.001 | 180 ± 4 |" in table
    assert "| medium |" in table
    assert "Failer & Richter, level 3, k = 0.001" in table
