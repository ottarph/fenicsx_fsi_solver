# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Skip the lifting of Dirichlet conditions in the residual of a Newton solve, by keeping their values
exactly in every Newton iterate.

dolfinx's ``assemble_residual`` lifts the residual by every Dirichlet condition: it adds J (g - x) on the
columns of the condition's dofs, which takes the Jacobian kernels on every cell with one of them, so it
costs about as much as assembling the Jacobian on those cells, in every residual evaluation. The lifting
is what makes a Newton step from an iterate with x != g on those dofs move them to g consistently with the
other dofs. Where x = g, it is exactly zero.

The three hooks of the SNES that ``skip_dirichlet_lifting`` returns keep x = g on the Dirichlet dofs of
every iterate, so that the residual needs no lifting:

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

The caller sets the hooks, and combines them with any of its own: an initial-guess hook of its own calls
the one here before anything that should see g in the guess, and updating the solution, or anything
computed from it, in the Newton update before the line search, as a post-solve hook of the KSP may do,
needs the pre-check called there first.
"""

from collections.abc import Callable, Sequence

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


def skip_dirichlet_lifting(problem: dolfinx.fem.petsc.NonlinearProblem) -> dict[str, Callable]:
    """The hooks of the SNES of ``problem`` that keep its Dirichlet values exactly in every Newton iterate
    and assemble its residual without lifting them (see the module docstring). Nothing is set here; the
    caller sets all three:

    - ``"residual"``: ``(snes, x, b)``, set with the residual vector of the problem,
      ``snes.setFunction(hooks["residual"], problem.b)``. It is dolfinx's ``assemble_residual``, with the
      arguments that ``problem`` set on the SNES, without the lifting; a wrapper of it, which calls it with
      these, keeps working.
    - ``"set_dirichlet_values"``: ``(snes, x)``, which sets the Dirichlet values g in the initial guess
      ``x``, set with ``snes.setInitialGuess``. A caller with an initial-guess hook of its own calls this
      from that hook, before anything that should see g.
    - ``"set_dirichlet_update"``: ``(x, y)``, which sets the Newton update ``y`` at the iterate ``x`` to
      x - g on the owned Dirichlet dofs, so that x - y holds g there: exactly where x holds g already, as
      every iterate does, since y is then zero there, and otherwise up to rounding. Set with
      ``snes.setLineSearchPreCheck``. A caller that uses the update before the line search, in a
      post-solve hook of the KSP, calls this there first.

    Setting only some of them breaks the solve: without the initial-guess hook or the pre-check, the
    iterates can depart from g, and the residual without lifting is then not that of Newton's method.

    The residual of ``problem`` must still be dolfinx's ``assemble_residual``, whose arguments are taken
    from the SNES.
    """
    b, (function, args, kargs) = problem.solver.getFunction()
    if function is not dolfinx.fem.petsc.assemble_residual:
        raise ValueError(f"The residual of the problem must be dolfinx's assemble_residual, not {function}")
    bcs = _bcs_for(kargs["residual"], kargs["bcs"])
    blocks = kargs.get("_blocks")

    def with_blocks(vec: PETSc.Vec) -> PETSc.Vec:
        # The block offsets that set_bc needs for a block vector. The work vectors of the SNES,
        # duplicated from the solution, do not carry them.
        if blocks is not None:
            vec.setAttr("_blocks", blocks)
        return vec

    def residual(snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        assemble_residual_without_lifting(snes, x, b, *args, **kargs)

    def set_dirichlet_values(snes: PETSc.SNES, x: PETSc.Vec) -> None:
        dolfinx.fem.petsc.set_bc(with_blocks(x), bcs)

    def set_dirichlet_update(x: PETSc.Vec, y: PETSc.Vec) -> None:
        dolfinx.fem.petsc.set_bc(with_blocks(y), bcs, x0=x, alpha=-1.0)

    return {
        "residual": residual,
        "set_dirichlet_values": set_dirichlet_values,
        "set_dirichlet_update": set_dirichlet_update,
    }
