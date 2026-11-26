import numpy as np
import pytest

from xfsi_solver.scripts.modes_fsi2_biharmonic import X0, X1, Y0, Y1, ModeRecorder, interface_measures
from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import FLUID_ALE_TERMS, solve

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


def outline(n=50):
    """Points along the beam's outline, in no particular order, and their outward normals."""
    bottom = np.column_stack([np.linspace(X0, X1, n), np.full(n, Y0)])
    tip = np.column_stack([np.full(n, X1), np.linspace(Y0, Y1, n)])[1:-1]
    top = np.column_stack([np.linspace(X0, X1, n), np.full(n, Y1)])
    points = np.concatenate([top, bottom, tip])
    normals = np.concatenate(
        [np.tile((0.0, 1.0), (n, 1)), np.tile((0.0, -1.0), (n, 1)), np.tile((1.0, 0.0), (n - 2, 1))]
    )
    return points, normals


def test_area_index_of_uniform_and_tangential_displacements():
    points, normals = outline()
    perimeter = 2 * (X1 - X0) + (Y1 - Y0)
    uniform = interface_measures(points, 1e-3 * normals.astype(complex))
    # The segments next to the two tip corners average the normals of two sides: an error of order 1/n
    assert uniform["area"][0] == pytest.approx(1e-3 * perimeter, rel=1e-2)
    assert uniform["area_index"] == pytest.approx(1.0, abs=0.02)
    # A translation across the beam moves the interface without changing the area. (A translation
    # along it does change the area, since the interface is open at the cylinder.)
    shift = interface_measures(points, np.tile((0j, 1e-3 + 0j), (len(points), 1)))
    assert abs(shift["area"][0]) < 1e-12
    assert shift["area_index"] < 1e-6


def test_modes_on_the_coarse_mesh(output_dirs):
    records = []
    dt_val = 0.0025
    name = "modes"
    solve(
        mesh_path=MESH,
        T=3 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        jacobian_callback=ModeRecorder(records, sample_steps=[3], at_iteration=1, nev=2),
        snes_atol=0.0,
        snes_rtol=1e-10,
        snes_max_it=30,
    )
    assert len(records) == 1 and len(records[0]["modes"]) == 2
    first, second = records[0]["modes"]
    assert first["abs"] >= second["abs"] > 0.0
    assert set(first["term_shares"]) == set(FLUID_ALE_TERMS)
    assert "interface" in first  # serial: rank 0 has it
    assert 0.0 <= first["interface"]["area_index"] <= 1.0 + 1e-9


def test_area_of_a_uniform_displacement_with_rounded_corners():
    r, n = 0.005, 400
    bottom = np.column_stack([np.linspace(X0, X1 - r, n), np.full(n, Y0)])
    top = np.column_stack([np.linspace(X0, X1 - r, n), np.full(n, Y1)])
    tip = np.column_stack([np.full(n, X1), np.linspace(Y0 + r, Y1 - r, n)])
    angles = np.linspace(0.0, 0.5 * np.pi, n)[1:-1]
    lower = np.column_stack([X1 - r + r * np.sin(angles), Y0 + r - r * np.cos(angles)])
    upper = np.column_stack([X1 - r + r * np.cos(angles), Y1 - r + r * np.sin(angles)])
    points = np.concatenate([upper, top, tip, bottom, lower])
    from xfsi_solver.scripts.modes_fsi2_biharmonic import arc_length

    _, normals = arc_length(points, fillet=r)
    perimeter = 2 * (X1 - r - X0) + np.pi * r + (Y1 - Y0 - 2 * r)
    result = interface_measures(points, 1e-3 * normals.astype(complex), fillet=r)
    assert result["area"][0] == pytest.approx(1e-3 * perimeter, rel=1e-3)
    assert result["area_index"] == pytest.approx(1.0, abs=1e-3)
    assert max(result["s"]) == pytest.approx(perimeter, rel=1e-4)
