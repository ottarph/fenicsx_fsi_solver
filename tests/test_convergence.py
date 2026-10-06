import dolfinx
import dolfinx.fem.petsc
import pytest
import ufl
from mpi4py import MPI
from restart_helpers import run_mpi_python

from xfsi_solver.tools.convergence import check_converged


def _make_problem(zero_jacobian=False, **petsc_options):
    """Newton for u^3 = 1 (in the L2 sense) from u = 0.5, or with ``zero_jacobian`` a residual with
    a zero Jacobian, which MUMPS fails to factorize."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = dolfinx.fem.Function(V), ufl.TestFunction(V)
    u.x.array[:] = 0.5
    nonlinearity = dolfinx.fem.Constant(mesh, 0.0) * u if zero_jacobian else u**3
    F = nonlinearity * v * ufl.dx - v * ufl.dx
    options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_error_if_not_converged": False,
        "ksp_error_if_not_converged": False,
        **petsc_options,
    }
    return dolfinx.fem.petsc.NonlinearProblem(F, u, petsc_options_prefix="convergence_test_", petsc_options=options)


class _Writer:
    closed = False

    def close(self):
        self.closed = True


def test_check_converged_passes_converged_solve():
    problem = _make_problem()
    problem.solve()
    writer = _Writer()
    check_converged(problem, "the test", writers=[writer])
    assert not writer.closed


def test_check_converged_raises_after_max_iterations_and_closes_writers():
    problem = _make_problem(snes_max_it=1, snes_atol=1e-14, snes_rtol=1e-14)
    problem.solve()
    writers = [_Writer(), _Writer()]
    with pytest.raises(RuntimeError, match="at the test did not converge: SNES DIVERGED_MAX_IT"):
        check_converged(problem, "the test", writers=writers)
    assert all(w.closed for w in writers)


def test_check_converged_raises_on_failed_factorization():
    # Without the *_error_if_not_converged options, solve() itself doesn't raise
    problem = _make_problem(zero_jacobian=True)
    problem.solve()
    with pytest.raises(RuntimeError, match="SNES DIVERGED_LINEAR_SOLVE .* KSP DIVERGED_PCSETUP_FAILED, PC "):
        check_converged(problem, "the test")


def test_check_converged_raises_on_every_rank_on_failed_factorization():
    script = """
import sys
sys.path.insert(0, "tests")
from mpi4py import MPI
from test_convergence import _make_problem
from xfsi_solver.tools.convergence import check_converged

problem = _make_problem(zero_jacobian=True)
problem.solve()
try:
    check_converged(problem, "the test")
except RuntimeError:
    pass
else:
    raise AssertionError(f"rank {MPI.COMM_WORLD.rank} did not raise")
# Every rank has to get here, or the barrier hangs and the run times out
MPI.COMM_WORLD.barrier()
"""
    run_mpi_python(script, ranks=3, timeout=120)
