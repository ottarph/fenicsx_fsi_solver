import numpy as np
import pytest

from xfsi_solver.scripts.interface_fsi2_biharmonic import InterfaceRecorder
from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import FLUID_ALE_TERMS, solve

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


def run(output_dirs, name, mode, steps=2, light=False, **kwargs):
    records = []
    recorder = InterfaceRecorder(mode, records, light=light)
    dt_val = 0.0025
    solve(
        mesh_path=MESH,
        T=steps * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
        jacobian_callback=recorder,
        snes_atol=0.0,
        snes_rtol=1e-10,
        snes_max_it=30,
        **kwargs,
    )
    recorder.destroy()
    return records


@pytest.mark.parametrize("mode", ["full", "J_0"])
def test_fluid_ale_derivative_is_all_of_it(output_dirs, mode):
    records = run(output_dirs, f"interface_{mode}", mode)
    for record in records:
        # J - C has no fluid ALE derivative left: (p, u) and (v, u_f) are round-off.
        assert max(record["check"].values()) < 1e-12
        assert set(record["blocks"]) == set(FLUID_ALE_TERMS)
    stepped = [r for r in records if "products" in r]
    assert stepped and all(np.isfinite(r["products_total"]["v_other,u_f"]) for r in stepped)
    assert "pressure" in stepped[0]["terms_after"] and "solid stress" in stepped[0]["terms_after"]


def test_j_0_changes_the_newton_step(output_dirs):
    """J_0 drops C from the Jacobian Newton solves with, so its first step in the second time step
    leaves another residual. From rest the mesh hardly moves, so the difference is small there (about
    1% on this mesh); the rates are measured from a moving state by the script."""

    def first_contraction(records):
        second = [r for r in records if r["iteration"] == 0][1]
        after = records[records.index(second) + 1]
        return (
            sum(after["residual"][g]["l2"] ** 2 for g in after["residual"]) ** 0.5
            / sum(second["residual"][g]["l2"] ** 2 for g in second["residual"]) ** 0.5
        )

    full = run(output_dirs, "interface_rate_full", "full")
    j_0 = run(output_dirs, "interface_rate_J_0", "J_0")
    assert abs(first_contraction(j_0) / first_contraction(full) - 1.0) > 1e-3


@pytest.mark.parametrize("mode", ["interface_ale", "J_0"])
def test_light_recorder_takes_the_same_steps(output_dirs, mode):
    """Without the diagnostics, the recorder still solves with the same Jacobian."""
    heavy = run(output_dirs, f"heavy_{mode}", mode)
    light = run(output_dirs, f"light_{mode}", mode, light=True)
    assert len(light) == len(heavy)
    for a, b in zip(light, heavy, strict=True):
        assert set(a) == {"iteration", "residual"}
        for g in a["residual"]:
            assert a["residual"][g]["l2"] == pytest.approx(b["residual"][g]["l2"], rel=1e-9, abs=1e-14)


def test_ramped_outlet_pressure(output_dirs):
    """With a ramp of 4 steps, the first step has a quarter of a quarter-cosine of the pressure, far
    below P0, and after the ramp the pressure is shifted by about P0."""
    records = []
    from xfsi_solver.scripts.block_scales_fsi2_biharmonic import block_scales_hook

    dt_val = 0.0025
    name = "ramp"
    solve(
        mesh_path=MESH,
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        jacobian_callback=block_scales_hook(records),
        outlet_pressure=11.2e3,
        outlet_pressure_ramp=4 * dt_val,
    )
    first_steps = [r for r in records if r["iteration"] == 0]
    # The state at iteration 0 of step k is that of step k - 1 (v and p are not predicted).
    assert first_steps[1]["state"]["p"]["max"] < 0.3 * 11.2e3
    assert first_steps[-1]["state"]["p"]["max"] > 0.9 * 11.2e3
