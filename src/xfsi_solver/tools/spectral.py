# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""The asymptotic contraction of Newton's method with an approximate Jacobian.

With the exact Jacobian J = J_0 + N, Newton's method with the approximate Jacobian J_0 iterates the
error as e_(k+1) = (I - J_0^-1 J) e_k = -J_0^-1 N e_k near the solution, so its asymptotic
contraction factor is the spectral radius of J_0^-1 N, which :func:`dominant_eigenvalues` computes
with SLEPc, from the action of N and a direct solve with J_0.
"""

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

DEFAULT_SOLVER_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 80,
    "mat_mumps_cntl_1": 1e-4,
}


class _InverseTimes:
    """The shell operator x -> J_0^-1 N x, for SLEPc."""

    def __init__(self, J0: PETSc.Mat, N: PETSc.Mat, ksp: PETSc.KSP):
        self.N, self.ksp = N, ksp
        self.Nx = N.createVecLeft()

    def mult(self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
        self.N.mult(x, self.Nx)
        self.ksp.solve(self.Nx, y)
        if self.ksp.getConvergedReason() < 0:
            raise RuntimeError(f"The solve with J_0 failed: KSP reason {self.ksp.getConvergedReason()}")


def dominant_eigenvalues(
    J0: PETSc.Mat,
    N: PETSc.Mat,
    nev: int = 3,
    tol: float = 1.0e-6,
    max_it: int = 300,
    solver_options: dict | None = None,
    options_prefix: str = "spectral_",
) -> np.ndarray:
    """The ``nev`` eigenvalues of largest magnitude of J_0^-1 N, sorted by decreasing magnitude.

    J_0 is factorized once, by the KSP options ``solver_options`` (by default MUMPS LU) under
    ``options_prefix``, and the eigenvalues are found by Krylov-Schur to the relative tolerance
    ``tol`` in at most ``max_it`` restarts. Raises ``RuntimeError`` if fewer than ``nev`` converge.
    Collective on the communicator of J_0.
    """
    comm = J0.getComm()
    ksp = PETSc.KSP().create(comm)
    ksp.setOptionsPrefix(options_prefix)
    opts = PETSc.Options()
    opts.prefixPush(options_prefix)
    for key, value in (solver_options or DEFAULT_SOLVER_OPTIONS).items():
        opts[key] = value
    opts.prefixPop()
    ksp.setOperators(J0)
    ksp.setFromOptions()
    ksp.setUp()

    operator = PETSc.Mat().createPython(J0.getSizes(), _InverseTimes(J0, N, ksp), comm=comm)
    operator.setUp()
    eps = SLEPc.EPS().create(comm)
    eps.setOperators(operator)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eps.setDimensions(nev=nev)
    eps.setTolerances(tol, max_it)
    try:
        eps.solve()
        converged = eps.getConverged()
        if converged < nev:
            raise RuntimeError(f"Only {converged} of {nev} eigenvalues of J_0^-1 N converged")
        values = np.array([eps.getEigenvalue(i) for i in range(converged)])
    finally:
        eps.destroy()
        operator.destroy()
        ksp.destroy()
        for key in list(opts.getAll()):
            if key.startswith(options_prefix):
                del opts[key]
    return values[np.argsort(-np.abs(values))][:nev]
