import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.tools.matrix_nest import NestView


def banded_matrix(n, diagonal_scale=1.0):
    """An n x n AIJ matrix with i + 1 (scaled) on the diagonal and 0.5 two places to the right of it."""
    mat = PETSc.Mat().createAIJ([n, n], nnz=2, comm=MPI.COMM_SELF)
    for i in range(n):
        mat.setValue(i, i, diagonal_scale * (i + 1.0))
        if i + 2 < n:
            mat.setValue(i, i + 2, 0.5)
    mat.assemble()
    return mat


def test_nest_view_keeps_the_numbering_and_follows_updates():
    n = 9
    mat = banded_matrix(n)
    # Interleaved groups, so the nest reorders the rows.
    groups = [np.arange(n)[np.arange(n) % 3 == k] for k in range(3)]
    view = NestView(mat, groups, ["a", "b", "c"])

    # The (a, b) block has no nonzeros: a row i in a has its entries at i and i + 2, in a and c.
    assert view.block("a", "b") is None
    assert view.block("a", "c") is not None
    assert np.allclose(view.block("b", "b").getDiagonal().array, groups[1] + 1.0)

    x = mat.createVecRight()
    x.setRandom()
    expected, result = mat.createVecLeft(), mat.createVecLeft()

    def check():
        mat.mult(x, expected)
        view.nest.mult(x, result)
        assert np.allclose(result.array, expected.array, rtol=1e-14, atol=0)
        converted = view.nest.convert("aij", out=PETSc.Mat())
        converted.axpy(-1.0, mat)
        assert converted.norm() == 0.0
        converted.destroy()

    check()
    new = banded_matrix(n, diagonal_scale=2.0)
    new.copy(mat, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
    view.update()
    check()
    assert np.allclose(view.block("b", "b").getDiagonal().array, 2.0 * (groups[1] + 1.0))

    view.destroy()
    new.destroy()
    mat.destroy()


def test_nest_view_needs_a_partition_of_the_rows():
    mat = banded_matrix(4)
    with pytest.raises(ValueError, match="partition"):
        NestView(mat, [np.arange(2), np.arange(1, 4)], ["a", "b"])
    mat.destroy()
