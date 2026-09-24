# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""FGMRES / Schur field-split linear solver for the shared-space FSI solver.

The Newton system is ordered ``(u, v, p)`` and split into displacement ``u``
and momentum-pressure ``vp = (v, p)``. The outer preconditioner is a full
Schur factorization of the preconditioning matrix (the no-ALE Jacobian in the
production configuration)::

    FGMRES on J
    +-- Schur fieldsplit: u | (v,p)

Variants:

``"exact"``
    Diagnostic. LU for the displacement block and dense LU on the explicitly
    computed Schur complement (``schur_precondition=full``). Every outer
    iteration is then an exact solve with the preconditioning matrix. Forming
    the Schur complement costs one displacement solve per ``(v, p)`` unknown
    and dense storage, so this is for coarse meshes only.
"""

from contextlib import contextmanager
from timeit import default_timer as timer

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
from petsc4py import PETSc

from xfsi_solver.linalg.fieldsplit import field_index_sets, field_norms, nested_index_sets, union_index_set

FIELDSPLIT_VARIANTS = ("exact",)


@contextmanager
def temporary_options(prefix: str, options: dict):
    """Insert ``options`` under ``prefix`` into the PETSc options database for the duration of the block.

    Full option names are used rather than ``prefixPush``, because clearing an
    option does not apply the pushed prefix.
    """
    opts = PETSc.Options()
    try:
        for key, value in options.items():
            opts[f"{prefix}{key}"] = value
        yield
    finally:
        for key in options:
            del opts[f"{prefix}{key}"]


def _mumps_lu(prefix: str) -> dict:
    return {
        f"{prefix}ksp_type": "preonly",
        f"{prefix}pc_type": "lu",
        f"{prefix}pc_factor_mat_solver_type": "mumps",
        f"{prefix}mat_mumps_icntl_14": 80,
    }


def _dense_lu(prefix: str) -> dict:
    # The explicitly computed Schur complement is a dense matrix. MUMPS does not
    # factor it (INFOG(1)=-3 in the solve phase) and ScaLAPACK does not accept
    # MPIDENSE, so it is gathered onto every rank and factored there.
    return {
        f"{prefix}ksp_type": "preonly",
        f"{prefix}pc_type": "redundant",
        f"{prefix}redundant_pc_type": "lu",
        f"{prefix}redundant_pc_factor_mat_solver_type": "petsc",
    }


def configure_schur_fieldsplit(ksp: PETSc.KSP, is_u: PETSc.IS, is_vp: PETSc.IS, variant: str) -> dict:
    """Configure ``ksp``'s PC as the outer ``u | (v,p)`` full Schur factorization.

    Returns the sub-solver options (relative to the KSP's prefix) that must be
    in the options database when the PC is first set up, see
    :func:`setup_with_options`.
    """
    if variant not in FIELDSPLIT_VARIANTS:
        raise ValueError(f"Unknown fieldsplit variant {variant!r}, expected one of {FIELDSPLIT_VARIANTS}")
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.FIELDSPLIT)
    pc.setFieldSplitIS(("u", is_u), ("vp", is_vp))
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.FieldSplitSchurFactType.FULL)
    if variant == "exact":
        pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.FULL)

    # Every block of the factorization, including the off-diagonal coupling,
    # comes from the preconditioning matrix, never from the Newton operator.
    options = {
        "pc_fieldsplit_diag_use_amat": False,
        "pc_fieldsplit_off_diag_use_amat": False,
    }
    if variant == "exact":
        options |= _mumps_lu("fieldsplit_u_") | _dense_lu("fieldsplit_vp_")
    return options


def setup_with_options(ksp: PETSc.KSP, options: dict) -> None:
    """First setup of ``ksp`` (operators already set) with the sub-solver ``options``."""
    with temporary_options(ksp.getOptionsPrefix() or "", options):
        ksp.getPC().setFromOptions()
        ksp.setUp()


class FieldSplitSolver:
    """Configures and instruments the linear solver of a ``NonlinearProblem``.

    Owns the index sets and any auxiliary operators it creates; the matrices,
    vectors and SNES of the ``NonlinearProblem`` remain owned by it.
    """

    def __init__(self, nonlinear_problem, problem, config):
        self.nonlinear_problem = nonlinear_problem
        self.problem = problem
        self.config = config

        A = nonlinear_problem.A
        self.field_is = field_index_sets(A, [problem.U, problem.V, problem.P])
        self.is_u = self.field_is[0]
        self.is_vp = union_index_set(self.field_is[1:])
        self.vp_field_is = nested_index_sets(self.is_vp, self.field_is[1:])

        self.snes = nonlinear_problem.solver
        self.ksp = self.snes.getKSP()
        self.prefix = self.ksp.getOptionsPrefix()

        self.ksp.setType(PETSc.KSP.Type.FGMRES)
        self.ksp.setTolerances(rtol=config.ksp_rtol, atol=config.ksp_atol, max_it=config.ksp_max_it)
        self.ksp.setGMRESRestart(config.ksp_restart)
        self.ksp.setErrorIfNotConverged(True)

        self._options = configure_schur_fieldsplit(self.ksp, self.is_u, self.is_vp, config.fieldsplit)

        self._jacobian_ctx = {
            "u": problem.solution,
            "jacobian": nonlinear_problem.J,
            "preconditioner": nonlinear_problem.preconditioner,
            "bcs": problem.bcs,
        }
        P_mat = nonlinear_problem.P_mat if nonlinear_problem.P_mat is not None else A
        self.snes.setJacobian(self._assemble_jacobian, A, P_mat)
        self.ksp.setPreSolve(self._pre_solve)
        self.ksp.setPostSolve(self._post_solve)

        self._configured = False
        self.reset_statistics()

    def reset_statistics(self):
        self.linear_solves = []
        self.timings = {"jacobian": 0.0, "setup": 0.0, "linear_solve": 0.0}

    def _assemble_jacobian(self, snes, x, J, P):
        start = timer()
        dfx.fem.petsc.assemble_jacobian(snes, x, J, P, **self._jacobian_ctx)
        self.timings["jacobian"] += timer() - start

        # Set up the preconditioner here, so that its cost is measured apart from
        # the Krylov iterations. SNES passes the same, unchanged matrices to the
        # KSP afterwards, which does not trigger another setup.
        start = timer()
        self.ksp.setOperators(J, P)
        if not self._configured:
            setup_with_options(self.ksp, self._options)
            self._check_hierarchy()
            self._configured = True
        else:
            self.ksp.setUp()
        self.timings["setup"] += timer() - start

    def _check_hierarchy(self):
        """Fail if options from the database replaced the configured outer solver."""
        pc = self.ksp.getPC()
        if self.ksp.getType() != PETSc.KSP.Type.FGMRES or pc.getType() != PETSc.PC.Type.FIELDSPLIT:
            raise RuntimeError(f"Expected FGMRES/fieldsplit, found {self.ksp.getType()}/{pc.getType()}")
        prefixes = [ksp.getOptionsPrefix() for ksp in pc.getFieldSplitSubKSP()]
        if prefixes != [f"{self.prefix}fieldsplit_u_", f"{self.prefix}fieldsplit_vp_"]:
            raise RuntimeError(f"Unexpected field split prefixes {prefixes}")

    def _pre_solve(self, ksp, b, x):
        self._solve_start = timer()

    def _post_solve(self, ksp, b, x):
        self.timings["linear_solve"] += timer() - self._solve_start
        A, _ = ksp.getOperators()
        r = b.duplicate()
        A.mult(x, r)
        r.aypx(-1.0, b)
        b_norm = b.norm()
        self.linear_solves.append({
            "iterations": ksp.getIterationNumber(),
            "reason": ksp.getConvergedReason(),
            "true_relative_residual": r.norm() / b_norm if b_norm > 0 else r.norm(),
            "field_true_residuals": field_norms(r, self.field_is),
        })
        r.destroy()

    def view(self, viewer=None):
        self.ksp.view(viewer)

    def destroy(self):
        for index_set in (*self.field_is, self.is_vp, *self.vp_field_is):
            index_set.destroy()

