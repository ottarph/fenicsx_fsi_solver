import numpy as np

from xfsi_solver.tools.newton_log import read_newton_log


def check_solver_newton_log(newton_log_path, qoi_path, blocks, always_assembled=False, atol=1e-7):
    """Check the Newton log a solver wrote alongside its QoI file: one run of iterates per time step
    of the QoI file, each from it = 0 until the residual is below the SNES ``atol``, with the norms of
    ``blocks``. The first step assembles the Jacobian, and with ``always_assembled``, every step does.
    """
    rows = read_newton_log(newton_log_path)
    assert [name for name in rows.dtype.names if name.startswith("residual_")] == [f"residual_{b}" for b in blocks]
    qoi_times = np.loadtxt(qoi_path, ndmin=2)[1:, 0]  # the first row is the initial state
    step_times, starts = np.unique(rows["t"], return_index=True)
    assert np.allclose(step_times, qoi_times, rtol=1e-6)
    for start, end in zip(starts, [*starts[1:], len(rows)], strict=True):
        step = rows[start:end]
        assert list(step["it"]) == list(range(len(step)))
        assert step["residual"][-1] <= atol < step["residual"][0]
        assert np.allclose(step["residual"] ** 2, sum(step[f"residual_{b}"] ** 2 for b in blocks), rtol=1e-5)
    assert rows["assembled"][1] == 1
    if always_assembled:
        assert np.all(rows["assembled"][rows["it"] > 0] == 1)
