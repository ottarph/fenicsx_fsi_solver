import numpy as np
import pytest

from xfsi_solver.scripts.incompressible_pressure_fsi2 import MODES, PressureRecorder, solver_options
from xfsi_solver.solvers.fsi2_incompressible import solve

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"
DT = 0.0025


# From rest the first residual is about 2e-6, so the relative tolerance alone would stop at the round-off
# floor; the absolute tolerance stops above it.


def run(output_dirs, name, mode="J_0", outlet_pressure=0.0, steps=3, sample_steps=(3,)):
    records = []
    recorder = PressureRecorder(records, sample_steps, at_iteration=1)
    newton = output_dirs["qoi"] / f"{name}_newton.txt"
    solve(
        mesh_path=MESH,
        T=steps * DT,
        dt_val=DT,
        output_path=str(output_dirs["pv"] / f"{name}.vtkhdf"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        newton_log_path=str(newton),
        solver_options=solver_options(mode, snes_rtol=1e-10, snes_max_it=30, snes_atol=1e-13),
        gamma_reassemble=None,
        outlet_pressure=outlet_pressure,
        fluid_ale_jacobian=MODES[mode],
        jacobian_callback=recorder,
    )
    state = {name: recorder.data["functions"][name].x.array.copy() for name in ("v", "p", "p_S", "u_s")}
    return records, np.loadtxt(newton), state


def test_outlet_pressure_shifts_both_pressures_and_leaves_the_motion(output_dirs):
    """Both media are incompressible, so the outlet traction -P0 n shifts p and p_S by P0 and changes
    neither v nor the displacement."""
    P0 = 5000.0
    _, _, base = run(output_dirs, "shift_0")
    _, _, shifted = run(output_dirs, "shift_P0", outlet_pressure=P0)
    # Up to the Newton tolerance (the absolute tolerance 1e-13 is about 1e-8 of the first residual)
    for name in ("v", "u_s"):
        assert np.allclose(shifted[name], base[name], rtol=0, atol=1e-6 * np.abs(base[name]).max())
    for name in ("p", "p_S"):
        assert np.allclose(shifted[name] - P0, base[name], rtol=0, atol=1e-6 * P0)


def test_unit_pressure_geometry_terms_cancel_at_the_interface(output_dirs):
    records, _, _ = run(output_dirs, "cancel", outlet_pressure=5000.0)
    unit = records[0]["unit"]
    assert unit["solid"] > 0 and unit["fluid"] > 0
    assert unit["sum"] < 1e-10 * unit["solid"]


@pytest.mark.parametrize("mode", ["interface_ale", "full"])
def test_fluid_ale_jacobians_converge_and_ignore_the_offset(output_dirs, mode):
    """With the fluid's derivative at the interface kept, the offset's geometric derivatives cancel, so
    the Jacobian does not depend on P0. The first residual of the shifted run holds the offset, which the
    first Newton step takes up exactly (the pressures enter linearly); after it the residuals are those
    of the unshifted run, in later steps from the start."""
    _, base, _ = run(output_dirs, f"{mode}_0", mode=mode)
    _, shifted, _ = run(output_dirs, f"{mode}_P0", mode=mode, outlet_pressure=5000.0)
    for i, t in enumerate(np.unique(base[:, 0])):
        r_base, r_shifted = base[base[:, 0] == t, 4], shifted[shifted[:, 0] == t, 4]
        # Later steps start from the shifted pressures, so only the first step's first residual differs
        start = 1 if i == 0 else 0
        if i == 0:
            assert r_shifted[0] > 1e3 * r_base[0]
        n = min(len(r_base), len(r_shifted))
        assert np.allclose(r_shifted[start:n], r_base[start:n], rtol=1e-3, atol=1e-13)


def test_full_jacobian_converges_faster_than_j_0(output_dirs):
    _, j_0, _ = run(output_dirs, "rate_J_0", mode="J_0", outlet_pressure=5000.0)
    _, full, _ = run(output_dirs, "rate_full", mode="full", outlet_pressure=5000.0)
    last = j_0[:, 0] == j_0[-1, 0]
    last_full = full[:, 0] == full[-1, 0]
    # The second contraction of the last step
    rate = lambda rows: rows[2, 4] / rows[1, 4]  # noqa: E731
    assert rate(full[last_full]) < rate(j_0[last])
