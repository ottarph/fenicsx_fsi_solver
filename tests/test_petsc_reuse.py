import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.tools.petsc_reuse import IdenticalBlocksPC, ReusePC

COMM = MPI.COMM_WORLD


def test_reuse_pc_applies_one_factorization_to_both_components():
    # Two decoupled components with the same 1d Laplacian, interleaved per node as (x, y), as the
    # components of a vector function space are.
    n = 20
    A = PETSc.Mat().createAIJ(((None, 2 * n), (None, 2 * n)), bsize=2, nnz=3, comm=COMM)
    start, end = A.getOwnershipRange()
    for row in range(start, end):
        node, component = divmod(row, 2)
        A.setValue(row, row, 2.0)
        for neighbour in (node - 1, node + 1):
            if 0 <= neighbour < n:
                A.setValue(row, 2 * neighbour + component, -1.0)
    A.assemble()

    ksp = PETSc.KSP().create(COMM)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    is_x, is_y = (PETSc.IS().createGeneral(np.arange(start + c, end, 2, dtype=np.int32), comm=COMM) for c in (0, 1))
    pc.setFieldSplitIS(("x", is_x), ("y", is_y))
    ksp_x, ksp_y = pc.getFieldSplitSubKSP()
    ksp_x.setType("preonly")
    ksp_x.getPC().setType("lu")
    ksp_x.getPC().setFactorSolverType("mumps")
    ksp_y.setType("preonly")
    ksp_y.getPC().setType(PETSc.PC.Type.PYTHON)
    ksp_y.getPC().setPythonContext(ReusePC(ksp_x.getPC()))

    b, x = A.createVecs()
    b.setArray(np.random.default_rng(COMM.rank).random(end - start))
    ksp.solve(b, x)

    # The fieldsplit is exact, so x solves the coupled system, with the factorization of the x
    # component applied to both.
    r = b.duplicate()
    A.mult(x, r)
    r.axpy(-1.0, b)
    assert r.norm() < 1e-12 * b.norm()
    assert ksp_y.getPC().getType() == PETSc.PC.Type.PYTHON

    ksp.destroy()
    A.destroy()


def test_identical_blocks_pc_solves_all_components_with_one_factorization():
    # Three decoupled components with the same 1d Laplacian, interleaved per node as (x, y, z).
    n, num_components = 20, 3
    size = num_components * n
    A = PETSc.Mat().createAIJ(((None, size), (None, size)), bsize=num_components, nnz=3, comm=COMM)
    start, end = A.getOwnershipRange()
    for row in range(start, end):
        node, component = divmod(row, num_components)
        A.setValue(row, row, 2.0)
        for neighbour in (node - 1, node + 1):
            if 0 <= neighbour < n:
                A.setValue(row, num_components * neighbour + component, -1.0)
    A.assemble()

    block_is = [
        PETSc.IS().createGeneral(np.arange(start + c, end, num_components, dtype=np.int32), comm=COMM)
        for c in range(num_components)
    ]
    context = IdenticalBlocksPC(block_is)
    opts = PETSc.Options()
    opts["identical_blocks_test_block_pc_type"] = "cholesky"
    opts["identical_blocks_test_block_pc_factor_mat_solver_type"] = "mumps"

    ksp = PETSc.KSP().create(COMM)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(context)
    pc.setOptionsPrefix("identical_blocks_test_")

    b, x = A.createVecs()
    b.setArray(np.random.default_rng(COMM.rank).random(end - start))
    ksp.solve(b, x)

    r = b.duplicate()
    A.mult(x, r)
    r.axpy(-1.0, b)
    assert r.norm() < 1e-12 * b.norm()
    assert context.block_pc.getType() == PETSc.PC.Type.CHOLESKY
    assert context.block_matrix.getSize() == (n, n)

    ksp.destroy()
    context.destroy_solvers()
    A.destroy()
    for key in ["identical_blocks_test_block_pc_type", "identical_blocks_test_block_pc_factor_mat_solver_type"]:
        opts.delValue(key)
