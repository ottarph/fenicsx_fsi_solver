import dolfinx
import numpy as np
import pytest
from mpi4py import MPI
from newton_log_helpers import check_solver_newton_log
from restart_helpers import check_restart_reproduces_continuous_run, check_start_from_initial_state

from xfsi_solver.scripts.transfer_fsi2_checkpoint import _State
from xfsi_solver.solvers import fsi2_biharmonic_fulljacobian
from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import solve
from xfsi_solver.tools.newton_log import read_newton_log
from xfsi_solver.tools.submesh import all_cells

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"

# p and z live on the fluid submesh but are checkpointed on the full mesh
ELEMENTS = {"u": ("Lagrange", 2, (2,)), "v": ("Lagrange", 2, (2,)), "p": ("Lagrange", 1), "z": ("Lagrange", 2, (2,))}


@pytest.mark.parametrize("mesh_motion_scaling", [None, 1.0e-9], ids=["unscaled", "scaled"])
def test_fsi2_biharmonic_fulljacobian_initialguess_solve(output_dirs, mesh_motion_scaling):
    dt_val = 0.0025
    name = "fsi2_biharmonic_fulljacobian_initialguess"
    solve(
        mesh_path=MESH,
        T=6 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        vtx_save_every=4,
        mesh_motion_scaling=mesh_motion_scaling,
        newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
    )

    assert (output_dirs["pv"] / f"{name}.bp").exists()
    check_solver_newton_log(
        output_dirs["qoi"] / f"{name}_newton.txt",
        output_dirs["qoi"] / f"{name}.txt",
        ["u", "v", "p", "z"],
        always_assembled=True,
    )


def run(solve_function, output_dirs, name, num_steps, **kwargs):
    """The QoIs and Newton log of ``num_steps`` steps from rest on the coarse mesh."""
    dt_val = 0.0025
    solve_function(
        mesh_path=MESH,
        T=num_steps * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        newton_log_path=str(output_dirs["qoi"] / f"{name}_newton.txt"),
        **kwargs,
    )
    return np.loadtxt(output_dirs["qoi"] / f"{name}.txt", ndmin=2), read_newton_log(
        output_dirs["qoi"] / f"{name}_newton.txt"
    )


def test_fsi2_biharmonic_fulljacobian_initialguess_without_prediction_is_fulljacobian(output_dirs):
    qoi, log = run(fsi2_biharmonic_fulljacobian.solve, output_dirs, "full", 3)
    qoi_off, log_off = run(solve, output_dirs, "off", 3, predict_displacements=False)
    np.testing.assert_array_equal(qoi_off, qoi)
    np.testing.assert_array_equal(log_off["residual"], log["residual"])


def test_fsi2_biharmonic_fulljacobian_initialguess_converges_to_the_same_state(output_dirs):
    # The prediction only changes the initial guess of each Newton solve, so the runs agree within
    # the Newton tolerance. Unlike in fsi2_simplified, the residual of the initial guess is not smaller
    # with the prediction: the solid stress is taken at the predicted displacement, which the old
    # velocity does not balance. The first Newton step from it gets closer, though: on this mesh, from
    # the second step on, the residual after it is 3 to 20 times smaller, more as the structure speeds
    # up. The first step starts from rest, where the prediction changes nothing.
    qoi, log = run(solve, output_dirs, "previous", 4, predict_displacements=False)
    qoi_predicted, log_predicted = run(solve, output_dirs, "predicted", 4)
    for column in range(1, qoi.shape[1]):
        np.testing.assert_allclose(
            qoi_predicted[:, column], qoi[:, column], rtol=0, atol=1e-6 * np.abs(qoi[:, column]).max()
        )
    after_first = log["residual"][log["it"] == 1]
    after_first_predicted = log_predicted["residual"][log_predicted["it"] == 1]
    np.testing.assert_allclose(after_first_predicted[0], after_first[0], rtol=1e-6)
    assert np.all(after_first_predicted[1:] < 0.5 * after_first[1:])
    assert len(log_predicted) <= len(log)


def test_fsi2_biharmonic_fulljacobian_initialguess_restart_reproduces_continuous_run(output_dirs, tmp_path):
    check_restart_reproduces_continuous_run(solve, MESH, ELEMENTS, output_dirs, tmp_path)


@pytest.mark.parametrize("first_ranks,restart_ranks", [(2, 1), (1, 2)], ids=["2to1", "1to2"])
def test_fsi2_biharmonic_fulljacobian_initialguess_restart_on_different_number_of_ranks(
    output_dirs, tmp_path, first_ranks, restart_ranks
):
    # Not bitwise identical across partitions; QoIs are printed to 6 digits, so the last one may flip
    check_restart_reproduces_continuous_run(
        solve,
        MESH,
        ELEMENTS,
        output_dirs,
        tmp_path,
        qoi_rtol=1e-5,
        first_ranks=first_ranks,
        restart_ranks=restart_ranks,
    )


def test_fsi2_biharmonic_fulljacobian_initialguess_starts_from_initial_state(output_dirs, tmp_path):
    check_start_from_initial_state(solve, MESH, ELEMENTS, output_dirs, tmp_path)


def test_fsi2_biharmonic_fulljacobian_initialguess_writes_the_spectrum(output_dirs):
    # The dominant eigenvalues of J_0^-1 N after every step: one row per step, sorted by modulus,
    # and well inside the unit disc, since J_0 converges. From rest the solid hardly moves, so they
    # are small, and grow as it starts to.
    dt_val = 0.0025
    path = output_dirs["qoi"] / "spectrum.txt"
    solve(
        mesh_path=MESH,
        T=3 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "spectrum.bp"),
        output_path_p=str(output_dirs["pv"] / "spectrum_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "spectrum_qoi.txt"),
        spectral_every=1,
        spectral_path=str(path),
    )
    rows = np.genfromtxt(path, names=True)
    np.testing.assert_allclose(rows["t"], dt_val * np.arange(1, 4), rtol=1e-6)
    moduli = np.stack([rows[f"abs_{i}"] for i in (1, 2, 3)], axis=1)
    assert np.all(np.diff(moduli, axis=1) <= 0)
    np.testing.assert_allclose(
        moduli, np.hypot(*(np.stack([rows[f"{c}_{i}"] for i in (1, 2, 3)], axis=1) for c in ("re", "im"))), rtol=1e-5
    )
    assert np.all((0 < moduli) & (moduli < 0.1))
    assert np.all(np.diff(rows["abs_1"]) > 0)


def test_fsi2_biharmonic_fulljacobian_initialguess_spectrum_needs_a_path(output_dirs):
    with pytest.raises(ValueError, match="spectral_path"):
        solve(
            mesh_path=MESH,
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "spectrum.bp"),
            output_path_p=str(output_dirs["pv"] / "spectrum_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "spectrum_qoi.txt"),
            spectral_every=1,
        )


def test_fsi2_biharmonic_fulljacobian_initialguess_spectrum_of_each_part(output_dirs):
    # With only the momentum or only the continuity part of the fluid ALE derivatives left out of
    # J_0, the spectrum is another one, well inside the unit disc too.
    dt_val = 0.0025

    def run(left_out):
        path = output_dirs["qoi"] / f"spectrum_{left_out}.txt"
        solve(
            mesh_path=MESH,
            T=2 * dt_val,
            dt_val=dt_val,
            output_path=str(output_dirs["pv"] / f"spectrum_{left_out}.bp"),
            output_path_p=str(output_dirs["pv"] / f"spectrum_{left_out}_p.bp"),
            qoi_path=str(output_dirs["qoi"] / f"spectrum_{left_out}_qoi.txt"),
            spectral_every=2,
            spectral_path=str(path),
            spectral_left_out=left_out,
        )
        return np.atleast_1d(np.genfromtxt(path, names=True))["abs_1"][0]

    rho = {left_out: run(left_out) for left_out in ("both", "momentum", "continuity")}
    assert all(0 < value < 0.1 for value in rho.values())
    assert len({round(value, 12) for value in rho.values()}) == 3


def test_fsi2_biharmonic_fulljacobian_initialguess_spectrum_rejects_unknown_part(output_dirs):
    with pytest.raises(ValueError, match="spectral_left_out"):
        solve(
            mesh_path=MESH,
            T=0.0025,
            dt_val=0.0025,
            output_path=str(output_dirs["pv"] / "spectrum.bp"),
            output_path_p=str(output_dirs["pv"] / "spectrum_p.bp"),
            qoi_path=str(output_dirs["qoi"] / "spectrum_qoi.txt"),
            spectral_every=1,
            spectral_path=str(output_dirs["qoi"] / "spectrum.txt"),
            spectral_left_out="pressure",
        )


def test_fsi2_biharmonic_fulljacobian_initialguess_writes_the_eigenvectors(output_dirs, tmp_path):
    # Each eigenvector x of J_0^-1 N solves J_0 (lambda x) = N x, and N has no solid rows, so on the
    # solid x keeps the kinematic relation of J_0 there, u = dt theta v, with theta = 0.5 + dt.
    dt_val = 0.0025
    solve(
        mesh_path=MESH,
        T=2 * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / "vectors.bp"),
        output_path_p=str(output_dirs["pv"] / "vectors_p.bp"),
        qoi_path=str(output_dirs["qoi"] / "vectors_qoi.txt"),
        spectral_every=2,
        spectral_path=str(output_dirs["qoi"] / "vectors_spectrum.txt"),
        spectral_nev=2,
        spectral_vectors_dir=str(tmp_path / "vectors"),
    )
    assert [d.name for d in (tmp_path / "vectors").iterdir()] == [f"t_{2 * dt_val:.6f}"]

    state = _State(MESH, MPI.COMM_WORLD)
    functions = {
        f"{name}_{part}_{i}": dolfinx.fem.Function(
            dolfinx.fem.functionspace(state.fluid_mesh if name in ("p", "z") else state.mesh, element),
            name=f"{name}_{part}_{i}",
        )
        for i in (1, 2)
        for part in ("re", "im")
        for name, element in ELEMENTS.items()
    }
    state.checkpointer(tmp_path / "vectors" / f"t_{2 * dt_val:.6f}").read(list(functions.values()), dt_val)

    def owned(f):
        return f.x.array[: f.function_space.dofmap.index_map.size_local * f.function_space.dofmap.bs]

    U = functions["u_re_1"].function_space
    fluid_cells = state.fluid_cell_map.sub_topology_to_topology(all_cells(state.fluid_mesh), inverse=False)
    solid_cells = np.setdiff1d(all_cells(state.mesh), fluid_cells)
    solid = (2 * np.unique(U.dofmap.list[solid_cells])[:, None] + np.arange(2)).ravel()
    theta = 0.5 + dt_val
    for i in (1, 2):
        norm = MPI.COMM_WORLD.allreduce(
            sum(np.sum(owned(functions[f"{name}_{part}_{i}"]) ** 2) for name in ELEMENTS for part in ("re", "im"))
        )
        np.testing.assert_allclose(norm, 1.0, rtol=1e-8)
        # The pressure holds most of the 2-norm, and u on the solid is of order 1e-6 of it here
        u_solid = MPI.COMM_WORLD.allreduce(np.abs(functions[f"u_re_{i}"].x.array[solid]).max(), op=MPI.MAX)
        assert u_solid > 1e-7
        for part in ("re", "im"):
            u_mode, v_mode = functions[f"u_{part}_{i}"].x.array, functions[f"v_{part}_{i}"].x.array
            np.testing.assert_allclose(u_mode[solid], dt_val * theta * v_mode[solid], rtol=0, atol=1e-4 * u_solid)
