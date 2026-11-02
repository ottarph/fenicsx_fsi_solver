import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.tools.spectral import dominant_eigenpairs, dominant_eigenvalues


def aij(dense: np.ndarray) -> PETSc.Mat:
    """A serial AIJ matrix with the entries of ``dense``."""
    n = dense.shape[0]
    A = PETSc.Mat().createAIJ([n, n], comm=MPI.COMM_SELF)
    A.setUp()
    rows, cols = np.nonzero(dense)
    for i, j in zip(rows, cols, strict=True):
        A.setValue(i, j, dense[i, j])
    A.assemble()
    return A


def test_dominant_eigenvalues_of_a_known_operator():
    # J_0 lower triangular and N small, so J_0^-1 N is a perturbation with known eigenvalues.
    rng = np.random.default_rng(1)
    n = 40
    J0 = np.tril(rng.uniform(-1, 1, (n, n))) + 4 * np.eye(n)
    N = 0.3 * rng.uniform(-1, 1, (n, n))
    exact = np.linalg.eigvals(np.linalg.solve(J0, N))
    exact = exact[np.argsort(-np.abs(exact))]
    values = dominant_eigenvalues(aij(J0), aij(N), nev=3, tol=1e-10)
    np.testing.assert_allclose(np.abs(values), np.abs(exact[:3]), rtol=1e-8)


def test_dominant_eigenpairs_are_eigenpairs():
    # The real and imaginary parts of each eigenvector x of J_0^-1 N give J_0^-1 N x = lambda x, |x| = 1.
    rng = np.random.default_rng(2)
    n = 40
    J0 = np.tril(rng.uniform(-1, 1, (n, n))) + 4 * np.eye(n)
    N = 0.3 * rng.uniform(-1, 1, (n, n))
    values, vectors = dominant_eigenpairs(aij(J0), aij(N), nev=4, tol=1e-10)
    assert np.all(np.diff(np.abs(values)) <= 1e-12)
    for value, (real, imag) in zip(values, vectors, strict=True):
        x = real.getArray() + 1j * imag.getArray()
        np.testing.assert_allclose(np.linalg.norm(x), 1.0, rtol=1e-10)
        np.testing.assert_allclose(np.linalg.solve(J0, N @ x), value * x, atol=1e-8)


def test_dominant_eigenvalues_leaves_no_options():
    dominant_eigenvalues(aij(np.eye(5) * 2), aij(np.diag([1.0, 0.5, 0.2, 0.1, 0.05])), nev=2)
    assert not [key for key in PETSc.Options().getAll() if key.startswith("spectral_")]
