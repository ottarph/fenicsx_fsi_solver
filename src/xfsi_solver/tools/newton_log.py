# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""A record of every Newton iteration of a ``dolfinx.fem.petsc.NonlinearProblem``, for
analysing the nonlinear convergence of the solvers.

``NewtonLog`` is a SNES monitor that appends one row per Newton iterate to a text file,
with the norm of the residual and of the Newton update, in total and per block of
unknowns, whether the Jacobian of the step to the iterate was assembled anew or reused,
and the wall time since the start of the nonlinear solve. PETSc's ``snes_monitor`` prints
only the total residual norm, and has no per-field norms without a DM.
"""

from collections.abc import Mapping
from pathlib import Path
from timeit import default_timer as timer

import dolfinx
import dolfinx.fem.petsc
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.tools.qoi import init_qoi_file


def _owned_array(vec: PETSc.Vec) -> np.ndarray:
    """The owned entries of ``vec``, for a nest vector those of its blocks one after the other."""
    if vec.getType() == PETSc.Vec.Type.NEST:
        return np.concatenate([sub.getArray(readonly=True) for sub in vec.getNestSubVecs()])
    return vec.getArray(readonly=True)


def field_blocks(problem: dolfinx.fem.petsc.NonlinearProblem, names: list[str]) -> dict[str, np.ndarray]:
    """The owned rows of each unknown of ``problem``, as indices into the owned entries of its
    vectors, keyed by ``names``, one per unknown in the order given to the problem.

    The blocks of a blocked vector are stored one after the other in the owned part of each
    rank, as are those of a nest vector in :func:`_owned_array`.
    """
    b = problem.b
    if b.getType() == PETSc.Vec.Type.NEST:
        sizes = [sub.getLocalSize() for sub in b.getNestSubVecs()]
        offsets = np.concatenate([[0], np.cumsum(sizes)])
    else:
        offsets, _ = b.getAttr("_blocks")
    if len(offsets) != len(names) + 1:
        raise ValueError(f"{len(names)} names for {len(offsets) - 1} blocks of unknowns")
    return {name: np.arange(offsets[k], offsets[k + 1]) for k, name in enumerate(names)}


class NewtonLog:
    """Append one row per Newton iterate of ``snes`` to the text file ``path``, on rank 0.

    The columns are

    - ``t``: the time of the time step, set by :meth:`start_step` before each solve,
    - ``it``: the Newton iteration, 0 for the initial guess,
    - ``assembled``: 1 if the Jacobian of the step to this iterate was assembled anew, 0 if
      it was reused from an earlier iterate; 0 at ``it = 0``, where no step was taken,
    - ``time``: the wall time in seconds since the residual of the initial guess was
      evaluated, so the difference between rows is the time of one Newton iteration,
    - ``residual`` and ``residual_<block>``: the l2 norm of the residual at the iterate, as
      the SNES evaluates it, in total and on the rows of each of ``blocks``,
    - ``update`` and ``update_<block>``: the same norms of the Newton update of the step to
      the iterate, ``nan`` at ``it = 0``.

    ``blocks`` maps a name to the owned rows of a block, as indices into the owned entries of
    the residual, see :func:`field_blocks`. A row is written as soon as the iterate is
    reached, so a solve that fails still leaves its iterations in the file. When restarting
    from a state at time ``t``, the rows after ``t`` are dropped from an existing file
    instead of starting a new one, as with the QoI file.

    The monitor is collective: it is called on every rank and reduces the norms over them.
    """

    def __init__(
        self,
        snes: PETSc.SNES,
        path: str | Path,
        blocks: Mapping[str, np.ndarray],
        comm: MPI.Intracomm,
        restart: bool = False,
        t: float = 0.0,
        dt: float = 0.0,
    ):
        self.path = path
        self.comm = comm
        self.blocks = {name: np.asarray(rows, dtype=np.int64) for name, rows in blocks.items()}
        self.t = t
        self._jacobian_state = None
        self._start = None
        names = list(self.blocks)
        self.columns = (
            ["t", "it", "assembled", "time", "residual"]
            + [f"residual_{name}" for name in names]
            + ["update"]
            + [f"update_{name}" for name in names]
        )
        init_qoi_file(path, comm, restart, t, dt, header="\t".join(self.columns))
        snes.setMonitor(self)

    def start_step(self, t: float) -> None:
        """Label the rows of the next nonlinear solve with the time ``t``."""
        self.t = t

    def _norms(self, vec: PETSc.Vec) -> tuple[float, list[float]]:
        """The l2 norm of ``vec`` and of each of its blocks, with one reduction over the ranks."""
        values = _owned_array(vec)
        local = [np.dot(values, values)] + [np.dot(values[rows], values[rows]) for rows in self.blocks.values()]
        squares = self.comm.allreduce(np.array(local), op=MPI.SUM)
        norms = np.sqrt(squares)
        return norms[0], list(norms[1:])

    def __call__(self, snes: PETSc.SNES, it: int, fnorm: float) -> None:
        # The Jacobian matrix changes state whenever it is assembled, and keeps it when the
        # Jacobian callback leaves it to be reused.
        jacobian_state = snes.getJacobian()[0].stateGet()
        if it == 0:
            self._start = timer()
            assembled = 0
            update, update_blocks = np.nan, [np.nan] * len(self.blocks)
        else:
            assembled = int(jacobian_state != self._jacobian_state)
            update, update_blocks = self._norms(snes.getSolutionUpdate())
        self._jacobian_state = jacobian_state
        elapsed = timer() - self._start

        residual, residual_blocks = self._norms(snes.getFunction()[0])
        if self.comm.rank == 0:
            row = [self.t, it, assembled, elapsed, residual, *residual_blocks, update, *update_blocks]
            fmt = ["%.6e", "%d", "%d", "%.4f"] + ["%.6e"] * (len(row) - 4)
            with open(self.path, "ab") as f:
                np.savetxt(f, [row], fmt=fmt, delimiter="\t")


def read_newton_log(path: str | Path) -> np.ndarray:
    """The rows of a :class:`NewtonLog` file, as a structured array with one field per column."""
    return np.genfromtxt(path, names=True, delimiter="\t", ndmin=1)
