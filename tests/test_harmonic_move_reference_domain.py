"""move_reference_domain of the harmonic FSI solvers: the mesh motion measured against the start state."""

import dolfinx
import numpy as np
import pytest
from mpi4py import MPI
from restart_helpers import check_start_from_initial_state, read_checkpoint_state

from xfsi_solver.solvers import (
    fsi2_harmonic,
    fsi2_harmonic_condensed,
    fsi2_harmonic_fulljacobian,
    fsi2_harmonic_lagrange,
)
from xfsi_solver.tools.checkpoint import Checkpointer

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"
VECTOR = ("Lagrange", 2, (2,))
SCALAR = ("Lagrange", 1)
DT = 0.0025

# For each solver: its fields as checkpointed on the full mesh, the name of the mesh displacement and
# of its reference, keyword arguments for every run, and the tolerances of the restart check, as in
# the solver's own initial-state test
SOLVERS = {
    "harmonic": (fsi2_harmonic.solve, {"u": VECTOR, "v": VECTOR, "p": SCALAR}, "u", "u_ref", {}, {}),
    "fulljacobian": (fsi2_harmonic_fulljacobian.solve, {"u": VECTOR, "v": VECTOR, "p": SCALAR}, "u", "u_ref", {}, {}),
    "condensed": (
        fsi2_harmonic_condensed.solve,
        {"u": VECTOR, "v": VECTOR, "p": SCALAR},
        "u",
        "u_ref",
        {"direct_k_c_solve": True},
        dict(rtol=1e-6, qoi_rtol=1e-6),
    ),
    "lagrange": (
        fsi2_harmonic_lagrange.solve,
        {"u_f": VECTOR, "v_f": VECTOR, "u_s": VECTOR, "v_s": VECTOR, "p": SCALAR},
        "u_f",
        "u_f_ref",
        {},
        dict(rtol=1e-8, field_rtol={"u_f": 1e-5}),
    ),
}
BY_SOLVER = pytest.mark.parametrize("name", list(SOLVERS))


@BY_SOLVER
def test_restart_reads_back_the_reference(name, output_dirs, tmp_path):
    """A run from another run's state, and its restart, which must read the reference from its own checkpoint."""
    solve, elements, _, ref, kwargs, tolerances = SOLVERS[name]
    check_start_from_initial_state(
        solve,
        MESH,
        {**elements, ref: VECTOR},
        output_dirs,
        tmp_path,
        solve_kwargs={"move_reference_domain": True, **kwargs},
        **tolerances,
    )


def bump(x):
    """A smooth mesh displacement in the fluid, zero outside a disc behind the solid, which no harmonic
    mesh motion would give."""
    r2 = ((x[0] - 1.5) ** 2 + (x[1] - 0.2) ** 2) / 0.1**2
    values = np.zeros((2, x.shape[1]))
    values[0] = 1e-3 * np.where(r2 < 1.0, (1.0 - r2) ** 2, 0.0)
    return values


def write_bump_state(directory, elements, u_name):
    """Write a checkpoint at t = 0 of a state at rest whose mesh displacement is ``bump``."""
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, MESH, "r") as infile:
        mesh = infile.read_mesh()
    functions = []
    for field, element in elements.items():
        f = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, element), name=field)
        if field == u_name:
            f.interpolate(bump)
            f.x.scatter_forward()
        functions.append(f)
    Checkpointer(directory, mesh).write(functions, 0.0, 0, DT)
    return functions[list(elements).index(u_name)].x.array.copy()


@BY_SOLVER
@pytest.mark.parametrize("move_reference_domain", [True, False], ids=["moved", "original"])
def test_start_state_keeps_its_mesh_displacement(name, move_reference_domain, output_dirs, tmp_path):
    """From a state at rest whose mesh displacement is not harmonic, one step keeps that displacement
    with move_reference_domain, since the solid barely moves, and smooths it away without it."""
    solve, elements, u_name, _, kwargs, _ = SOLVERS[name]
    u_start = write_bump_state(tmp_path / "start", elements, u_name)
    solve(
        mesh_path=MESH,
        T=DT,
        dt_val=DT,
        output_path=str(output_dirs["pv"] / "run.bp"),
        output_path_p=str(output_dirs["pv"] / "run_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "run_qoi.txt"),
        checkpoint_dir=tmp_path / "run",
        checkpoint_every=1,
        initial_state_dir=tmp_path / "start",
        move_reference_domain=move_reference_domain,
        **kwargs,
    )
    _, step, state = read_checkpoint_state(MESH, tmp_path / "run", DT, {u_name: elements[u_name]})
    assert step == 1
    change = np.max(np.abs(state[u_name].x.array - u_start)) / np.max(np.abs(u_start))
    if move_reference_domain:
        assert change < 1e-3
    else:
        assert change > 0.5
