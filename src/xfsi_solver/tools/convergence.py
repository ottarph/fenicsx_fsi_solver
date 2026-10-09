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

A failed linear solve nested in a fieldsplit is not caught that way: PETSc
does not raise for ``DIVERGED_ITS`` in a nested KSP, even with
``ksp_error_if_not_converged``, and a failed sub-KSP does not fail its
fieldsplit, so Newton continues with an inaccurate step. ``KSPConvCheck``
records such failures, and ``check_converged`` raises on them.
"""

from collections.abc import Iterable

import dolfinx
import dolfinx.fem.petsc
from petsc4py import PETSc


def _reason_name(reasons: type, value: int) -> str:
    names = [name for name, v in vars(reasons).items() if not name.startswith("_") and v == value]
    return names[0] if names else str(value)


class KSPConvCheck:
    """Record the failed solves of a KSP and of every KSP nested below it in fieldsplits.

    The converged reason is the same on every rank, so every rank records the same
    failures. Nested KSPs only exist once their parent PC is set up, which happens in
    the first solve, so each KSP attaches to its sub-KSPs when it solves. The KSPs
    inside AMG preconditioners (level smoothers) are not checked: they run a fixed
    number of iterations.
    """

    def __init__(self, ksp: PETSc.KSP):
        self.failures: list[tuple[str, int]] = []  # (options prefix, converged reason) since the last clear()
        self._attached: set[int] = set()
        self._attach(ksp)

    def clear(self) -> None:
        """Forget the recorded failures, e.g. before the next nonlinear solve."""
        self.failures.clear()

    def _attach(self, ksp: PETSc.KSP) -> None:
        if ksp.handle in self._attached:
            return
        self._attached.add(ksp.handle)
        ksp.setPreSolve(self._presolve)
        ksp.setPostSolve(self._postsolve)

    def _presolve(self, ksp: PETSc.KSP, b: PETSc.Vec, x: PETSc.Vec) -> None:
        # KSPSolve sets up the PC before calling this, so its sub-KSPs exist now.
        pc = ksp.getPC()
        if pc.getType() == PETSc.PC.Type.FIELDSPLIT:
            for sub_ksp in pc.getFieldSplitSubKSP():
                self._attach(sub_ksp)

    def _postsolve(self, ksp: PETSc.KSP, b: PETSc.Vec, x: PETSc.Vec) -> None:
        reason = ksp.getConvergedReason()
        if reason < 0:
            self.failures.append((ksp.getOptionsPrefix(), reason))


def check_converged(
    problem: dolfinx.fem.petsc.NonlinearProblem,
    where: str,
    writers: Iterable[dolfinx.io.VTXWriter] = (),
    ksp_check: KSPConvCheck | None = None,
) -> None:
    """Raise ``RuntimeError`` if the last ``problem.solve()`` did not converge, or if
    ``ksp_check`` recorded a failed linear solve.

    ``where`` describes the solve in the error message, e.g. ``"t = 0.0100"``.
    ``writers`` are closed before raising, so the output written so far stays
    readable. Clear ``ksp_check`` before each ``problem.solve()``, so that it only
    holds the failures of that solve.
    """
    snes = problem.solver
    reason = snes.getConvergedReason()
    failures = ksp_check.failures if ksp_check is not None else []
    if reason > 0 and not failures:
        return

    for writer in writers:
        writer.close()

    ksp = snes.getKSP()
    if reason > 0:
        message = f"Nonlinear solve at {where} converged, but linear solves failed"
    else:
        message = (
            f"Nonlinear solve at {where} did not converge: "
            f"SNES {_reason_name(PETSc.SNES.ConvergedReason, reason)} after {snes.getIterationNumber()} iterations, "
            f"last linear solve KSP {_reason_name(PETSc.KSP.ConvergedReason, ksp.getConvergedReason())}"
        )
        pc_reason = ksp.getPC().getFailedReason()
        if pc_reason != 0:
            message += f", PC {_reason_name(PETSc.PC.FailedReason, pc_reason)}"
    if failures:
        counts: dict[tuple[str, int], int] = {}
        for failure in failures:
            counts[failure] = counts.get(failure, 0) + 1
        message += "; failed linear solves: " + ", ".join(
            f"{prefix} {_reason_name(PETSc.KSP.ConvergedReason, r)} x{n}" for (prefix, r), n in counts.items()
        )
    raise RuntimeError(message)
