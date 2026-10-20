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
    such as the displacement components of the mesh-motion block. It factorizes the first block,
    and solves for all blocks at once with MatMatSolve, one right-hand side per block.

    A direct solver's triangular solves are bound by memory bandwidth, so solving with all the
    right-hand sides at once reads the factor once instead of once per block, and takes one
    round of messages between the ranks instead of one per block.

    ``block_is`` are the index sets of the blocks in the matrix. On each rank, they must hold
    the same number of indices, in matching order, so that the k-th index of every block is the
    same row of the block matrix, as for the components of the owned dofs of a vector function
    space. The factorization of the first block is set up from the options with the prefix of
    the PC followed by ``block_``, for example ``<prefix>block_pc_type cholesky``. The context
    owns the factorization: call ``destroy_solvers`` on all ranks when done.
    """

    def __init__(self, block_is: list[PETSc.IS]):
        self.block_is = block_is
        self.block_pc = self.block_matrix = self.rhs = self.solution = None

    def setUp(self, pc: PETSc.PC) -> None:
        if self.block_pc is not None:
            return  # the blocks are only factorized once
        _, P = pc.getOperators()
        start = P.getOwnershipRange()[0]
        # The local positions of each block's rows in the vectors of the block matrix.
        self.local_rows = [block.getIndices() - start for block in self.block_is]

        self.block_matrix = P.createSubMatrix(self.block_is[0], self.block_is[0])
        self.block_pc = PETSc.PC().create(pc.getComm())
        self.block_pc.setOperators(self.block_matrix)
        self.block_pc.setOptionsPrefix((pc.getOptionsPrefix() or "") + "block_")
        self.block_pc.setFromOptions()
        self.block_pc.setUp()
        self.factor = self.block_pc.getFactorMatrix()

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
        self.factor.matSolve(self.rhs, self.solution)
        solution = self.solution.getDenseArray(readonly=True)
        x_array = x.array
        for column, rows in enumerate(self.local_rows):
            x_array[rows] = solution[:, column]

    def destroy_solvers(self) -> None:
        """Destroy the factorization, collectively. Not named destroy, which petsc4py would call
        as a hook when the PC is destroyed."""
        for obj in (self.block_pc, self.block_matrix, self.rhs, self.solution):
            if obj is not None:
                obj.destroy()
        self.block_pc = self.block_matrix = self.rhs = self.solution = None
