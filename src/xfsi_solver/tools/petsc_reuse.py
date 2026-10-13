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
