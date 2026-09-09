# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc

from dolfinx.fem.function import Function as _Function
from typing import Sequence

from dolfinx.fem.bcs import bcs_by_block as _bcs_by_block
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting
from dolfinx.fem.forms import extract_function_spaces as _extract_function_spaces

class MyLinearProblem(dolfinx.fem.petsc.LinearProblem):
    """
    A customized version of :class:`dolfinx.fem.petsc.LinearProblem` that
    exposes the `assemble_matrix` method for assembling the system matrix
    separately from the `solve` method. Saves time when solving multiple
    linear systems with the same system matrix but different right-hand sides.
    """

    def assemble_matrix(self) -> None:
        """Assemble the system matrix and preconditioner matrix (if any)."""
        # Assemble lhs
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a, bcs=self.bcs)  # type: ignore[arg-type, misc]
        self.A.assemble()

        # Assemble preconditioner
        if self.P_mat is not None:
            self.P_mat.zeroEntries()
            assemble_matrix(self.P_mat, self.preconditioner, bcs=self.bcs)  # type: ignore[arg-type, misc]
            self.P_mat.assemble()


    def assemble_rhs(self) -> None:
        """Assemble the right-hand side without solving the system."""
        # Assemble rhs
        dolfinx.la.petsc._zero_vector(self.b)
        assemble_vector(self.b, self.L)  # type: ignore[arg-type]

        # Apply boundary conditions to the rhs
        if self.bcs is not None:
            if isinstance(self.u, Sequence):  # block or nest
                bcs1 = _bcs_by_block(_extract_function_spaces(self.a, 1), self.bcs)  # type: ignore[arg-type]
                apply_lifting(self.b, self.a, bcs=bcs1)  # type: ignore[arg-type]
                dolfinx.la.petsc._ghost_update(
                    self.b,
                    PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
                    PETSc.ScatterMode.REVERSE,  # type: ignore[attr-defined]
                )
                bcs0 = _bcs_by_block(_extract_function_spaces(self.L), self.bcs)  # type: ignore[arg-type]
                dolfinx.fem.petsc.set_bc(self.b, bcs0)
            else:  # single form
                apply_lifting(self.b, [self.a], bcs=[self.bcs])  # type: ignore[arg-type]
                dolfinx.la.petsc._ghost_update(
                    self.b,
                    PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
                    PETSc.ScatterMode.REVERSE,  # type: ignore[attr-defined]
                )
                for bc in self.bcs:
                    bc.set(self.b.array_w)
        else:
            dolfinx.la.petsc._ghost_update(self.b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]

    def solve(self) -> _Function | Sequence[_Function]:
        """Solve the problem using the separately assembled system matrix."""
        self.assemble_rhs()

        # Solve linear system and update ghost values in the solution
        self.solver.solve(self.b, self.x)
        dolfinx.la.petsc._ghost_update(self.x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[attr-defined]
        dolfinx.fem.petsc.assign(self.x, self.u)
        return self.u
