import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.scripts.scaling_fsi2_biharmonic import (
    COLUMN_SIZES,
    MUMPS_OPTIONS,
    ScaledLU,
    ScalingRecorder,
    stopping_tests,
    total,
)
from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import solve

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"


def test_scaled_lu_solves_the_unscaled_system():
    """D_c (D_r A D_c)^-1 D_r b = A^-1 b, with rows and columns scaled over 8 decades."""
    comm = MPI.COMM_SELF
    n = 30
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((n, n)) + n * np.eye(n)
    A = PETSc.Mat().createDense((n, n), array=dense, comm=comm)
    A.assemble()
    row, col = A.createVecs()
    row.array[:] = 10.0 ** rng.uniform(-4, 4, n)
    col.array[:] = 10.0 ** rng.uniform(-4, 4, n)
    context = ScaledLU(row, col, "test_scaled_lu_", {"pc_type": "lu"})
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("python")
    ksp.getPC().setPythonContext(context)
    b, x = A.createVecs()
    b.array[:] = rng.standard_normal(n)
    ksp.solve(b, x)
    assert np.allclose(x.array, np.linalg.solve(dense, b.array), rtol=1e-8, atol=1e-12)
    ksp.destroy()
    context.destroy()


def run(output_dirs, name, mode, scaling, steps=3):
    records = []
    recorder = ScalingRecorder(mode, scaling, records, MUMPS_OPTIONS)
    dt_val = 0.0025
    newton_log = output_dirs["qoi"] / f"{name}_newton.txt"
    solve(
        mesh_path=MESH,
        T=steps * dt_val,
        dt_val=dt_val,
        output_path=str(output_dirs["pv"] / f"{name}.bp"),
        output_path_p=str(output_dirs["pv"] / f"{name}_p.bp"),
        qoi_path=str(output_dirs["qoi"] / f"{name}.txt"),
        newton_log_path=str(newton_log),
        jacobian_callback=recorder,
        snes_atol=0.0,
        snes_rtol=1e-10,
        snes_max_it=30,
    )
    recorder.destroy()
    return records, np.loadtxt(newton_log)


@pytest.mark.parametrize("mode", ["full", "J_0"])
def test_newton_does_not_depend_on_the_scaling(output_dirs, mode):
    """The residuals of the Newton iterates are those without scaling, up to round-off, as long as
    they are well above it."""
    _, plain = run(output_dirs, f"scaling_{mode}_none", mode, "none")
    for scaling in ("physical", "random"):
        records, scaled = run(output_dirs, f"scaling_{mode}_{scaling}", mode, scaling)
        assert scaled.shape[0] >= plain.shape[0] - 1
        n = min(plain.shape[0], scaled.shape[0])
        r_plain, r_scaled = plain[:n, 4], scaled[:n, 4]
        above = r_plain > 1e-6 * r_plain.max()
        assert np.allclose(r_scaled[above], r_plain[above], rtol=1e-4)
        # Steps 2 and 3 are recorded, each to its last iterate, where the error is zero.
        assert len(records) == 2
        first_residuals = scaled[scaled[:, 1] == 0, 4]
        for i, step in enumerate(records):
            assert step[-1]["scaled_error"] == {c: 0.0 for c in COLUMN_SIZES}
            assert step[0]["scaled_error"]["u"] > 0.0
            # The groups of rows cover all of the residual of the Newton log, which has 7 digits.
            assert total(step[0]["residual"]) == pytest.approx(first_residuals[i + 1], rel=1e-6)


def test_stopping_tests(output_dirs):
    records, _ = run(output_dirs, "stopping_full", "full", "none")
    result = stopping_tests(
        records,
        {
            "first": lambda step, k: True,
            "never": lambda step, k: False,
        },
    )
    assert result["first"]["iterations"] == 0.0
    assert result["never"]["error"]["v"]["max"] == 0.0
    assert result["first"]["error"]["v"]["max"] > 0.0
