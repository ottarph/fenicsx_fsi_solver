# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""A MatNest of the blocks of an assembled matrix, for experiments on block elimination."""

import numpy as np
from petsc4py import PETSc


class NestView:
    """A MatNest ``nest`` whose blocks are copies of the blocks of ``mat`` over a partition of its
    rows into groups, the same partition for the rows and the columns.

    ``groups`` are the global rows of each group that this rank owns, together the owned rows of
    ``mat``. The nest keeps the numbering of ``mat``: its index sets are these rows, so it acts on
    the vectors of ``mat``, ``nest.convert("aij", out=PETSc.Mat())`` gives a copy of ``mat`` back
    (without ``out``, petsc4py converts the nest itself, in place), and ``nest.getNestSubMatrix(i, j)``
    is the block of group ``i`` against group ``j``, with the rows and columns of each group in
    increasing order.

    Blocks without any nonzero in the pattern of ``mat`` are left out (None in the nest), so the
    zero blocks show in ``nest.getNestSubMatrix``. Entries of the pattern that are zero are kept.
    Test for a left-out block with ``is None``, not by truth value: the truth value of a Mat depends
    on its local size, so it can differ between ranks, and a collective call made only where it is
    true hangs.
    Call :meth:`update` after ``mat`` is assembled again with the same pattern, and
    :meth:`destroy` on all ranks when done.
    """

    def __init__(self, mat: PETSc.Mat, groups: list[np.ndarray], names: list[str]):
        if len(groups) != len(names):
            raise ValueError(f"{len(groups)} groups but {len(names)} names")
        comm = mat.getComm()
        start, end = mat.getOwnershipRange()
        rows = np.sort(np.concatenate(groups))
        if not np.array_equal(rows, np.arange(start, end)):
            raise ValueError("The groups must partition the owned rows of the matrix")
        self.mat = mat
        self.names = list(names)
        self.index_sets = [PETSc.IS().createGeneral(np.sort(g).astype(PETSc.IntType), comm=comm) for g in groups]
        self.blocks = []
        for isrow in self.index_sets:
            row = []
            for iscol in self.index_sets:
                block = mat.createSubMatrix(isrow, iscol)
                # A global sum, so the same blocks are left out on all ranks.
                if block.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)["nz_used"] == 0:
                    block.destroy()
                    block = None
                row.append(block)
            self.blocks.append(row)
        self.nest = PETSc.Mat().createNest(self.blocks, isrows=self.index_sets, iscols=self.index_sets, comm=comm)

    def update(self) -> None:
        """Copy the values of ``mat`` into the blocks of the nest, which stay the same objects.
        Collective."""
        for isrow, row in zip(self.index_sets, self.blocks, strict=True):
            for iscol, block in zip(self.index_sets, row, strict=True):
                if block is not None:
                    self.mat.createSubMatrix(isrow, iscol, submat=block)
        self.nest.assemble()

    def block(self, row: str, col: str) -> PETSc.Mat | None:
        """The block of group ``row`` against group ``col``, by name, or None if it has no nonzeros."""
        return self.blocks[self.names.index(row)][self.names.index(col)]

    def destroy(self) -> None:
        """Destroy the nest, its blocks and its index sets. Collective."""
        self.nest.destroy()
        for row in self.blocks:
            for block in row:
                if block is not None:
                    block.destroy()
        for index_set in self.index_sets:
            index_set.destroy()
