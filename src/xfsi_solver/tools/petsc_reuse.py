# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Python contexts that reuse one solver for several blocks with the same matrix.

In a fieldsplit over decoupled blocks with the same matrix, such as the displacement components
of the mesh-motion block, one block's solver can be applied to all of them, so that a direct
solver factorizes the matrix only once. Set the context on the split of every other block, not on
the block whose solver it reuses, which would then apply itself forever. The context only holds a
reference to the reused solver, which its own split still owns and destroys.
"""

import numpy as np
from petsc4py import PETSc


class ReuseKSP:
    """KSPPYTHON context that solves with another KSP."""

    def __init__(self, ksp: PETSc.KSP):
        self.ksp = ksp

    def solve(self, ksp: PETSc.KSP, b: PETSc.Vec, x: PETSc.Vec) -> None:
        self.ksp.solve(b, x)


class ReusePC:
    """PCPYTHON context that applies another PC.

    Under a ``preonly`` KSP, this applies the other PC, for example a factorization, to the
    split's right-hand side. The other PC must be set up first, as it is in an additive or
    multiplicative fieldsplit when its split comes before the splits that reuse it.
    """

    def __init__(self, pc: PETSc.PC):
        self.pc = pc

    def apply(self, pc: PETSc.PC, b: PETSc.Vec, x: PETSc.Vec) -> None:
        self.pc.apply(b, x)


class IdenticalBlocksPC:
    """PCPYTHON context for a block diagonal matrix whose diagonal blocks are the same matrix,
    such as the displacement components of the mesh-motion block. It sets up a PC on the first
    block, and applies it to all blocks at once by PCMatApply, with one column per block.

    How much PCMatApply shares between the columns depends on the PC. A factorization applies
    them by one MatMatSolve, which reads the factor once instead of once per block, and takes one
    round of messages between the ranks instead of one per block: the triangular solves are bound
    by memory bandwidth, so this is where most is gained. GAMG runs one V-cycle on all columns,
    with a MatMatSolve on its coarse grid. PCs without their own PCMatApply, such as Jacobi, are
    applied column by column, which gives the same result as applying them to each block.

    ``block_is`` are the index sets of the blocks in the matrix. On each rank, they must hold
    the same number of indices, in matching order, so that the k-th index of every block is the
    same row of the block matrix, as for the components of the owned dofs of a vector function
    space. The PC of the first block is set up from the options with the prefix of the PC
    followed by ``block_``, for example ``<prefix>block_pc_type cholesky``. The context owns
    that PC: call ``destroy_solvers`` on all ranks when done.
    """

    def __init__(self, block_is: list[PETSc.IS]):
        self.block_is = block_is
        self.block_pc = self.block_matrix = self.rhs = self.solution = None

    def setUp(self, pc: PETSc.PC) -> None:
        _, P = pc.getOperators()
        if self.block_pc is not None:
            # The operator has changed: update the block, which sets up its PC again.
            P.createSubMatrix(self.block_is[0], self.block_is[0], submat=self.block_matrix)
            self.block_pc.setUp()
            return
        start = P.getOwnershipRange()[0]
        # The local positions of each block's rows in the vectors of the block matrix.
        self.local_rows = [block.getIndices() - start for block in self.block_is]

        self.block_matrix = P.createSubMatrix(self.block_is[0], self.block_is[0])
        self.block_pc = PETSc.PC().create(pc.getComm())
        self.block_pc.setOperators(self.block_matrix)
        self.block_pc.setOptionsPrefix((pc.getOptionsPrefix() or "") + "block_")
        self.block_pc.setFromOptions()
        self.block_pc.setUp()

        num_local, num_blocks = len(self.local_rows[0]), len(self.block_is)
        self.rhs, self.solution = (
            PETSc.Mat().createDense(((num_local, None), (None, num_blocks)), comm=pc.getComm()) for _ in range(2)
        )
        for dense in (self.rhs, self.solution):
            dense.setUp()
            dense.assemble()

    def apply(self, pc: PETSc.PC, b: PETSc.Vec, x: PETSc.Vec) -> None:
        rhs = self.rhs.getDenseArray()
        b_array = b.getArray(readonly=True)
        for column, rows in enumerate(self.local_rows):
            rhs[:, column] = b_array[rows]
        self.block_pc.matApply(self.rhs, self.solution)
        solution = self.solution.getDenseArray(readonly=True)
        x_array = x.array
        for column, rows in enumerate(self.local_rows):
            x_array[rows] = solution[:, column]

    def destroy_solvers(self) -> None:
        """Destroy the PC of the blocks, collectively. Not named destroy, which petsc4py would call
        as a hook when the PC is destroyed."""
        for obj in (self.block_pc, self.block_matrix, self.rhs, self.solution):
            if obj is not None:
                obj.destroy()
        self.block_pc = self.block_matrix = self.rhs = self.solution = None


class IdenticalInterleavedBlocksPC(IdenticalBlocksPC):
    """IdenticalBlocksPC for blocks interleaved row by row, which PETSc can create from the options,
    for example as the inner PC of a PCTELESCOPE, which petsc4py gives no access to.

    Row i of the matrix belongs to block i % num_blocks, as for the components of the owned dofs
    of a vector function space, and every rank owns whole groups of num_blocks rows. num_blocks
    is the option ``<prefix>identical_interleaved_blocks_num``, by default the block size of the
    matrix. Set the PC by the options ``<prefix>pc_type python`` and
    ``<prefix>pc_python_type xfsi_solver.tools.petsc_reuse.IdenticalInterleavedBlocksPC``. Under a
    PCTELESCOPE, the matrix keeps its global numbering on the smaller communicator, and its rows
    are split over the ranks in whole blocks of the matrix's block size, so the outer matrix
    needs block size num_blocks for every rank to keep whole groups.

    PETSc owns the context, so its PC is destroyed when PETSc destroys the PC holding the context,
    collectively, instead of by ``destroy_solvers``.
    """

    def __init__(self):
        super().__init__([])

    def setUp(self, pc: PETSc.PC) -> None:
        if self.block_pc is None:
            prefix = pc.getOptionsPrefix() or ""
            _, P = pc.getOperators()
            num_blocks = PETSc.Options(prefix).getInt("identical_interleaved_blocks_num", P.getBlockSize())
            start, end = P.getOwnershipRange()
            if start % num_blocks != 0 or end % num_blocks != 0:
                raise ValueError(
                    f"The rows {start} to {end} of a rank are not whole groups of {num_blocks} interleaved blocks"
                )
            self.block_is = [
                PETSc.IS().createGeneral(np.arange(start + b, end, num_blocks, dtype=PETSc.IntType), comm=pc.getComm())
                for b in range(num_blocks)
            ]
        super().setUp(pc)

    def destroy(self, pc: PETSc.PC) -> None:
        """Called by petsc4py when PETSc destroys the PC holding the context."""
        self.destroy_solvers()
        for block in self.block_is:
            block.destroy()
        self.block_is = []
