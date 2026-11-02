import numpy as np
import pytest
from mpi4py import MPI
from restart_helpers import read_checkpoint_state, run_mpi_python

from xfsi_solver.scripts.transfer_fsi2_checkpoint import ELEMENTS, _State, transfer_checkpoint

SOURCE_MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"
# Another mesh of the same domain, not nested in the source
TARGET_MESH = "data/meshes/fsi2/mesh_sec.xdmf"

# Polynomials that the elements of the fields hold exactly
FIELDS = {
    "u": lambda x: np.stack([x[0] ** 2 + 2 * x[0] * x[1], x[1] - x[0] * x[1]]),
    "v": lambda x: np.stack([1 + x[0] - x[1] ** 2, x[0] * x[1]]),
    "p": lambda x: 1 + 2 * x[0] - 3 * x[1],
    "z": lambda x: np.stack([x[0] * x[1], x[0] ** 2 - x[1] ** 2]),
}


def write_source(directory, names):
    """A checkpoint of the polynomial fields ``names`` on the source mesh, at t = 1.5 after step 3."""
    state = _State(SOURCE_MESH, MPI.COMM_WORLD)
    for name in names:
        state.functions[name].interpolate(FIELDS[name])
    state.checkpointer(directory).write([state.functions[name] for name in names], 1.5, 3, 0.01, t0=1.47)


def check_target(directory, names):
    """The transferred fields hold the polynomials at the dofs of the target, away from the curved
    cylinder (the curved source cells reach 0.073 from its centre), where a quadratic cell of one
    mesh is not a polynomial map of the other's, and, for
    the fields on the fluid submesh, outside the solid, where the stored parent function is zero."""
    elements = {name: ELEMENTS[name] for name in names}
    t, step, functions = read_checkpoint_state(TARGET_MESH, directory, None, elements)
    assert t == 1.5
    assert step == 0
    for name, f in functions.items():
        x = f.function_space.tabulate_dof_coordinates().T
        keep = np.hypot(x[0] - 0.2, x[1] - 0.2) > 0.08
        if name in ("p", "z"):
            keep &= ~((x[0] > 0.24) & (x[0] < 0.6) & (np.abs(x[1] - 0.2) < 0.0099))
        bs = f.function_space.dofmap.index_map_bs
        values = f.x.array.reshape(-1, bs)[keep]
        exact = np.atleast_2d(FIELDS[name](x[:, keep])).T
        np.testing.assert_allclose(values, exact, rtol=0, atol=1e-10, err_msg=name)


def test_transfer_fsi2_checkpoint_reproduces_the_fields(tmp_path):
    write_source(tmp_path / "source", list(FIELDS))
    t = transfer_checkpoint(SOURCE_MESH, tmp_path / "source", TARGET_MESH, tmp_path / "target", 0.01)
    assert t == 1.5
    check_target(tmp_path / "target", list(FIELDS))


def test_transfer_fsi2_checkpoint_without_z(tmp_path):
    # The checkpoints of fsi2_simplified hold no z; the transfer leaves it out too.
    write_source(tmp_path / "source", ["u", "v", "p"])
    transfer_checkpoint(SOURCE_MESH, tmp_path / "source", TARGET_MESH, tmp_path / "target", 0.01)
    check_target(tmp_path / "target", ["u", "v", "p"])
    with pytest.raises(Exception):  # noqa: B017, the error io4dolfinx raises for a missing function
        check_target(tmp_path / "target", ["z"])


def test_transfer_fsi2_checkpoint_in_parallel(tmp_path):
    write_source(tmp_path / "source", list(FIELDS))
    script = (
        "from xfsi_solver.scripts.transfer_fsi2_checkpoint import transfer_checkpoint\n"
        f"transfer_checkpoint({SOURCE_MESH!r}, {str(tmp_path / 'source')!r}, {TARGET_MESH!r}, "
        f"{str(tmp_path / 'target')!r}, 0.01)\n"
    )
    run_mpi_python(script, 3, timeout=300)
    check_target(tmp_path / "target", list(FIELDS))


def test_transfer_fsi2_checkpoint_starts_a_solver(tmp_path, output_dirs):
    # A solver on the target mesh starts from the transferred state of a solver run on the source
    # mesh, at its time, and converges.
    from xfsi_solver.solvers.fsi2_biharmonic_mixed import solve

    dt_val = 0.0025

    def run(name, mesh_path, T, **kwargs):
        qoi_path = output_dirs["qoi"] / f"{name}.txt"
        solve(
            mesh_path=mesh_path,
            T=T,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"{name}.bp"),
            output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
            qoi_path=str(qoi_path),
            direct_k_c_solve=True,
            **kwargs,
        )
        return np.loadtxt(qoi_path, ndmin=2)

    run("source", SOURCE_MESH, 3 * dt_val, checkpoint_dir=tmp_path / "source", checkpoint_every=3)
    t = transfer_checkpoint(SOURCE_MESH, tmp_path / "source", TARGET_MESH, tmp_path / "target", dt_val)
    rows = run("started", TARGET_MESH, t + dt_val, initial_state_dir=tmp_path / "target")
    np.testing.assert_allclose(rows[:, 0], [t, t + dt_val])
