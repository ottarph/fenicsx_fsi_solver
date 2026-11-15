# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Newton iterates that hold their Dirichlet values exactly, and a residual without lifting.

dolfinx's ``assemble_residual`` lifts the residual by every Dirichlet condition: it adds J (g - x) on the
columns of the condition's dofs, which takes the Jacobian kernels on every cell with one of them, so it
costs about as much as assembling the Jacobian on those cells, in every residual evaluation. The lifting
is what makes a Newton step from an iterate with x != g on those dofs move them to g consistently with the
other dofs. Where x = g, it is exactly zero.

``DirichletWithoutLifting`` keeps x = g on the Dirichlet dofs of every iterate, so that the residual
needs no lifting:

- an initial-guess hook of the SNES sets the Dirichlet values in the initial guess of every solve, after
  their values have been updated, for example to the inflow of a new time step;
- a line-search pre-check sets the Newton update Y on the Dirichlet dofs to x - g, which is zero there
  once x holds g, so that the next iterate x - lambda Y holds g for every step length lambda. A linear
  solve whose preconditioner leaves a small nonzero update on those dofs, which can happen with an
  inexact one such as algebraic multigrid, is corrected so;
- the residual is assembled without lifting, with x - g in the Dirichlet rows, as before.

The Newton iterates are then those of Newton's method for the unknowns other than the Dirichlet dofs,
with these fixed at g. They differ from those with the lifting only in the first step of a solve whose
initial guess does not hold g, which starts here from the guess with g set in it, and agree with them
to the tolerance of the Newton solve. The converged solution is the same, since the lifting vanishes
where x = g, and the Dirichlet rows of the residual still hold x - g, so a convergence test on the
residual norm sees any departure from g.

Updating the solution, or anything computed from it, in the Newton update before the line search, as
a post-solve hook of the KSP may do, needs the update held first: call ``hold_update`` there.
"""

from collections.abc import Sequence

import dolfinx
import dolfinx.fem.petsc
from petsc4py import PETSc


def _bcs_for(residual, bcs: Sequence[dolfinx.fem.DirichletBC]):
    """``bcs`` in the form that ``set_bc`` takes for the vector of ``residual``: by block if it is a
    sequence of forms."""
    if isinstance(residual, Sequence):
        return dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(residual), bcs)
    return bcs


def _ghost_update_forward(b: PETSc.Vec) -> None:
    for b_sub in b.getNestSubVecs() if b.getType() == PETSc.Vec.Type.NEST else [b]:
        b_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def assemble_residual_without_lifting(
    snes: PETSc.SNES,
    x: PETSc.Vec,
    b: PETSc.Vec,
    u,
    residual,
    jacobian,
    bcs: Sequence[dolfinx.fem.DirichletBC],
    _blocks=None,
) -> None:
    """dolfinx's ``assemble_residual``, with the same arguments, without the lifting by ``bcs``, and with
    the rows of their dofs set to x - g, as there.

    It gives the same residual where x = g on the dofs of ``bcs``, and otherwise lacks the term
    J (g - x) in the other rows. The Jacobian forms are packed for an empty lifting, but their kernels
    do not run."""
    dolfinx.fem.petsc.assemble_residual(snes, x, b, u=u, residual=residual, jacobian=jacobian, bcs=[], _blocks=_blocks)
    dolfinx.fem.petsc.set_bc(b, _bcs_for(residual, bcs), x0=x, alpha=-1.0)
    _ghost_update_forward(b)


class DirichletWithoutLifting:
    """Hold the Dirichlet values of ``problem`` exactly in every Newton iterate, and assemble its
    residual without lifting (see the module docstring).

    Create it after any initial-guess hook has been set on ``problem.solver``: it calls that hook after
    setting the Dirichlet values in the initial guess, and a hook set later replaces its own. The residual
    of ``problem`` must still be dolfinx's ``assemble_residual``; the replacement takes the same
    arguments, so a wrapper set later, which calls the function and arguments of ``getFunction()``, keeps
    working. The line-search pre-check of the SNES is taken.
    """

    def __init__(self, problem: dolfinx.fem.petsc.NonlinearProblem):
        snes = problem.solver
        b, (function, args, kargs) = snes.getFunction()
        if function is not dolfinx.fem.petsc.assemble_residual:
            raise ValueError(f"The residual of the problem must be dolfinx's assemble_residual, not {function}")
        self._bcs = _bcs_for(kargs["residual"], kargs["bcs"])
        self._blocks = kargs.get("_blocks")
        snes.setFunction(assemble_residual_without_lifting, b, args=args, kargs=kargs)
        self._previous_initial_guess = snes.getInitialGuess()
        snes.setInitialGuess(self._initial_guess)
        snes.setLineSearchPreCheck(self.hold_update)

    def _with_blocks(self, vec: PETSc.Vec) -> PETSc.Vec:
        """``vec``, with the block offsets that ``set_bc`` needs for a block vector. The work vectors of
        the SNES, duplicated from the solution, do not carry them."""
        if self._blocks is not None:
            vec.setAttr("_blocks", self._blocks)
        return vec

    def set_values(self, x: PETSc.Vec) -> None:
        """Set the Dirichlet values g on the owned Dirichlet dofs of ``x``."""
        dolfinx.fem.petsc.set_bc(self._with_blocks(x), self._bcs)

    def hold_update(self, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Set the Newton update ``y`` at the iterate ``x`` to x - g on the owned Dirichlet dofs, so that
        x - y holds g there: exactly where x holds g already, since y is then zero there, and otherwise up to
        rounding. A line-search pre-check of the SNES."""
        dolfinx.fem.petsc.set_bc(self._with_blocks(y), self._bcs, x0=x, alpha=-1.0)

    def _initial_guess(self, snes: PETSc.SNES, x: PETSc.Vec) -> None:
        self.set_values(x)
        if self._previous_initial_guess is not None:
            function, args, kargs = self._previous_initial_guess
            function(snes, x, *args, **kargs)
