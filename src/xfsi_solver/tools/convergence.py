# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Convergence check after a ``dolfinx.fem.petsc.NonlinearProblem`` solve.

The solvers check the SNES converged reason after every solve instead of
passing ``snes_error_if_not_converged`` / ``ksp_error_if_not_converged``. With
those options PETSc raises from inside the solve, and an error from the
linear solver (e.g. a failed MUMPS factorization) is not always raised on
every rank, which can leave the other ranks waiting in a collective call.
The converged reason is the same on every rank, so checking it fails the run
on all ranks together.
"""

from collections.abc import Iterable

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
from petsc4py import PETSc


def _reason_name(reasons: type, value: int) -> str:
    names = [name for name, v in vars(reasons).items() if not name.startswith("_") and v == value]
    return names[0] if names else str(value)


def check_converged(
    problem: dfx.fem.petsc.NonlinearProblem,
    where: str,
    writers: Iterable[dfx.io.VTXWriter] = (),
) -> None:
    """Raise ``RuntimeError`` if the last ``problem.solve()`` did not converge.

    ``where`` describes the solve in the error message, e.g. ``"t = 0.0100"``.
    ``writers`` are closed before raising, so the output written so far stays
    readable.
    """
    snes = problem.solver
    reason = snes.getConvergedReason()
    if reason > 0:
        return

    for writer in writers:
        writer.close()

    ksp = snes.getKSP()
    message = (
        f"Nonlinear solve at {where} did not converge: "
        f"SNES {_reason_name(PETSc.SNES.ConvergedReason, reason)} after {snes.getIterationNumber()} iterations, "
        f"last linear solve KSP {_reason_name(PETSc.KSP.ConvergedReason, ksp.getConvergedReason())}"
    )
    pc_reason = ksp.getPC().getFailedReason()
    if pc_reason != 0:
        message += f", PC {_reason_name(PETSc.PC.FailedReason, pc_reason)}"
    raise RuntimeError(message)
