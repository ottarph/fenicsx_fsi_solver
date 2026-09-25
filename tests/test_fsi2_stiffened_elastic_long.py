"""Opt-in long runs of the stiffened elastic FSI2 solver (``pytest -m long``).

``test_developed_direct_vs_fieldsplit`` needs a restart state written by the
solver on ``XFSI_ELASTIC_MESH`` (default ``data/meshes/fsi2/mesh_sec.xdmf``)
with the default mesh law, given as ``XFSI_ELASTIC_STATE``.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from xfsi_solver.solvers.fsi2_harmonic_diffmesh import SolverConfig
from xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh import MeshMotionConfig, solve

pytestmark = pytest.mark.long

ROOT = Path(__file__).resolve().parent.parent
COARSE = str(ROOT / "data/meshes/fsi2/mesh_sec_coarse.xdmf")
DT = 0.0025
comm = MPI.COMM_WORLD


@pytest.fixture
def shared_path(tmp_path):
    return Path(comm.bcast(str(tmp_path), root=0))


def compare(a, b, amplitude_floor):
    """Maximal QoI difference relative to the signal amplitude (with a floor for near-zero signals)."""
    out = {}
    for name in ("drag", "lift"):
        x, y = np.array([getattr(s, name) for s in a.steps]), np.array([getattr(s, name) for s in b.steps])
        out[name] = np.abs(x - y).max() / max(np.ptp(y), amplitude_floor[name])
    x = np.array([s.tip_displacement for s in a.steps])
    y = np.array([s.tip_displacement for s in b.steps])
    for k, name in enumerate(("A_x", "A_y")):
        out[name] = np.abs(x[:, k] - y[:, k]).max() / max(np.ptp(y[:, k]), amplitude_floor[name])
    return out


FLOORS = {"drag": 1e-3, "lift": 1e-3, "A_x": 1e-8, "A_y": 1e-8}


def run(path, name, config, T, mesh=COARSE, **kwargs):
    return solve(mesh, T=T, dt_val=DT, output_dir=path / name, config=config, output_every=100, **kwargs)


def test_coarse_startup_direct_vs_fieldsplit(shared_path):
    """800 steps (to 2 s) from rest on the coarse mesh, full/direct vs production no_ale/fieldsplit."""
    tight = dict(snes_atol=1e-9, snes_rtol=1e-12, snes_monitor=False)
    reference = run(shared_path, "direct", SolverConfig(**tight), T=2.0)
    result = run(shared_path, "fieldsplit", SolverConfig(jacobian_mode="no_ale", linear_solver="fieldsplit",
                                                         ksp_rtol=1e-8, **tight), T=2.0)
    assert result.steps[-1].t == reference.steps[-1].t == 2.0
    deviation = compare(result, reference, FLOORS)
    assert all(v < 1e-5 for v in deviation.values()), deviation


@pytest.mark.skipif(not os.environ.get("XFSI_ELASTIC_STATE"), reason="needs XFSI_ELASTIC_STATE")
def test_developed_direct_vs_fieldsplit(shared_path):
    """100 steps from a developed state: no_ale/direct vs production no_ale/fieldsplit."""
    mesh = os.environ.get("XFSI_ELASTIC_MESH", str(ROOT / "data/meshes/fsi2/mesh_sec.xdmf"))
    state = os.environ["XFSI_ELASTIC_STATE"]
    mesh_config = MeshMotionConfig(
        mesh_stiffening_exponent=float(os.environ.get("XFSI_ELASTIC_CHI", 2.5)),
        mesh_poisson_ratio=float(os.environ.get("XFSI_ELASTIC_NU", 0.3)))
    t0 = float(np.load(state if comm.size == 1 else
                       state.replace(".npz", f"_rank{comm.rank}of{comm.size}.npz"))["t_next"])
    tight = dict(snes_atol=1e-9, snes_rtol=1e-12, snes_monitor=False)
    kwargs = dict(T=t0 + 100 * DT, mesh=mesh, initial_state=state, mesh_config=mesh_config)
    reference = run(shared_path, "direct", SolverConfig(jacobian_mode="no_ale", **tight), **kwargs)
    result = run(shared_path, "fieldsplit", SolverConfig(jacobian_mode="no_ale", linear_solver="fieldsplit",
                                                         ksp_rtol=1e-8, **tight), **kwargs)
    deviation = compare(result, reference, FLOORS)
    assert all(v < 1e-5 for v in deviation.values()), deviation
