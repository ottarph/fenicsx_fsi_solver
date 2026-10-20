import numpy as np
import pytest
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


BLOCK_PC_OPTIONS = {
    # One MatMatSolve on all blocks.
    "cholesky": {"pc_type": "cholesky", "pc_factor_mat_solver_type": "mumps"},
    # A PC that runs a Krylov solve: PCMatApply solves each column with CG.
    "ksp": {"pc_type": "ksp", "ksp_ksp_type": "cg", "ksp_pc_type": "jacobi", "ksp_ksp_rtol": 1e-14},
    # A PC without its own PCMatApply, applied column by column.
    "jacobi": {"pc_type": "jacobi"},
}


@pytest.mark.parametrize("block_pc", BLOCK_PC_OPTIONS)
def test_identical_blocks_pc_applies_the_block_pc_to_all_components(block_pc):
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
    prefix = "identical_blocks_test_"
    opts = PETSc.Options()
    for key, value in BLOCK_PC_OPTIONS[block_pc].items():
        opts[f"{prefix}block_{key}"] = value

    ksp = PETSc.KSP().create(COMM)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(context)
    pc.setOptionsPrefix(prefix)

    b, x = A.createVecs()
    b.setArray(np.random.default_rng(COMM.rank).random(end - start))
    ksp.solve(b, x)

    assert context.block_pc.getType() == block_pc
    assert context.block_matrix.getSize() == (n, n)
    if block_pc == "jacobi":
        # The block PC applied to each block: x = b / diag(A).
        assert np.allclose(x.getArray(readonly=True), b.getArray(readonly=True) / 2.0, rtol=1e-14, atol=0.0)
    else:
        # The block PC solves each block, so x solves the coupled system.
        r = b.duplicate()
        A.mult(x, r)
        r.axpy(-1.0, b)
        assert r.norm() < 1e-12 * b.norm()

    ksp.destroy()
    context.destroy_solvers()
    A.destroy()
    for key in BLOCK_PC_OPTIONS[block_pc]:
        opts.delValue(f"{prefix}block_{key}")


@pytest.mark.parametrize("telescope", [False, True], ids=["direct", "telescope"])
def test_identical_interleaved_blocks_pc_from_options(telescope):
    # Three decoupled components with the same 1d Laplacian, interleaved per node as (x, y, z), with
    # the PC created by PETSc from the options, directly or inside a PCTELESCOPE onto one rank.
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

    prefix = "interleaved_blocks_test_"
    inner = "telescope_" if telescope else ""
    options = {"ksp_type": "preonly"}
    if telescope:
        options |= {"pc_type": "telescope", "pc_telescope_reduction_factor": COMM.size, "telescope_ksp_type": "preonly"}
    options |= {
        f"{inner}pc_type": "python",
        f"{inner}pc_python_type": "xfsi_solver.tools.petsc_reuse.IdenticalInterleavedBlocksPC",
        f"{inner}identical_interleaved_blocks_num": num_components,
        f"{inner}block_pc_type": "cholesky",
        f"{inner}block_pc_factor_mat_solver_type": "mumps",
    }
    opts = PETSc.Options()
    for key, value in options.items():
        opts[f"{prefix}{key}"] = value

    ksp = PETSc.KSP().create(COMM)
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(A)
    ksp.setFromOptions()

    b, x = A.createVecs()
    b.setArray(np.random.default_rng(COMM.rank).random(end - start))
    PETSc.Log.begin()
    mat_mat_solve = PETSc.Log.Event("MatMatSolve")
    count_before = mat_mat_solve.getPerfInfo()["count"]
    ksp.solve(b, x)
    count = COMM.allreduce(mat_mat_solve.getPerfInfo()["count"] - count_before, op=MPI.MAX)

    r = b.duplicate()
    A.mult(x, r)
    r.axpy(-1.0, b)
    assert r.norm() < 1e-12 * b.norm()
    # The components are solved at once, also under the telescope.
    assert count == 1

    ksp.destroy()  # destroys the context's PC through the destroy hook
    A.destroy()
    for key in options:
        opts.delValue(f"{prefix}{key}")
