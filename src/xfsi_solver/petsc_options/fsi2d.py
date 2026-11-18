# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""PETSc options for the 2d FSI2 benchmark solver, ``xfsi_solver.solvers.fsi2_simplified``."""

DEFAULT_SOLVER_OPTIONS = {
    # SNES options
    "snes_linesearch_type": "none",
    "snes_max_it": 20,
    "snes_atol": 1.0e-7,
    "snes_rtol": 1.0e-12,
    "snes_stol": 0.0,
    # Turn off errors, catch manually instead.
    "snes_error_if_not_converged": False,
    "ksp_error_if_not_converged": False,
    # Print to console.
    # "snes_monitor": None,
    "snes_converged_reason": None,
}
DEFAULT_DIRECT_SOLVER_OPTIONS = DEFAULT_SOLVER_OPTIONS | {
    # KSP options
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_cntl_1": 1e-4,
}
DEFAULT_V_OPTIONS = {
    # One GAMG V-cycle on v, with the near-nullspace and block size attached to is_v.
    "ksp_type": "preonly",
    "pc_type": "gamg",
    "pc_gamg_threshold": 0.01,
    "pc_gamg_reuse_interpolation": True,
    "mg_levels_ksp_type": "richardson",
    "mg_levels_pc_type": "sor",
    # "ksp_converged_reason": None,
}
DEFAULT_P_OPTIONS = {
    # One BoomerAMG V-cycle on \hat S.
    "ksp_type": "preonly",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    # "ksp_converged_reason": None,
}
DEFAULT_ITERATIVE_SOLVER_OPTIONS = (
    DEFAULT_SOLVER_OPTIONS
    | {
        "ksp_converged_reason": None,
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-5,
        "ksp_atol": 1e-8,
        "ksp_max_it": 200,
        #
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "upper",
        # The pressure Schur complement S = -B A^{-1} G is dense and only applied as an operator, so it
        # is preconditioned by the sparse approximation \hat S = -B diag(A)^{-1} G. The pressure term
        # gives G = -B^T (up to quadrature, by the Piola identity), so \hat S = B diag(A)^{-1} B^T is a
        # divergence of a gradient: close to a pressure Laplacian, scaled by about dt / rho_f, since the
        # inertia term rho_f / dt dominates diag(A) at small time steps.
        "pc_fieldsplit_schur_precondition": "selfp",
    }
    | {f"fieldsplit_v_0_{k}": v for k, v in DEFAULT_V_OPTIONS.items()}
    | {f"fieldsplit_p_1_{k}": v for k, v in DEFAULT_P_OPTIONS.items()}
)
# The velocity block of MUMPS_ON_SOLID, split into the dofs on solid cells (with the interface), s, and
# the fluid dofs, f.
DEFAULT_V_S_OPTIONS = {
    # MUMPS LU on v_s.
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
DEFAULT_V_F_OPTIONS = {
    # One GAMG V-cycle on v_f, with the block size attached to is_v_f.
    "ksp_type": "preonly",
    "pc_type": "gamg",
    "pc_gamg_threshold": 0.01,
    "pc_gamg_reuse_interpolation": True,
    "mg_levels_ksp_type": "richardson",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_pc_type": "sor",
    "mg_levels_pc_sor_omega": 0.4,
    # "ksp_converged_reason": None,
}
# The Schur factor of DEFAULT_ITERATIVE_SOLVER_OPTIONS, with the velocity block split again,
# multiplicatively, into the solid dofs, solved exactly, and the fluid dofs, by one GAMG V-cycle. The
# solid rows are dominated by the solid stiffness, where the SOR smoothing of GAMG, which is Jacobi
# between ranks, depends on the partition; an exact solve does not. The fluid rows are dominated by the
# mass. Needs requires_v_sub_IS=True in fsi2_simplified.solve, which sets the inner ISes.
MUMPS_ON_SOLID = (
    DEFAULT_SOLVER_OPTIONS
    | {
        "ksp_converged_reason": None,
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-5,
        "ksp_atol": 1e-8,
        "ksp_max_it": 200,
        #
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "upper",
        # The selfp approximation of the Schur complement, as in DEFAULT_ITERATIVE_SOLVER_OPTIONS.
        "pc_fieldsplit_schur_precondition": "selfp",
        #
        # v split again, multiplicatively, into its solid (with the interface) and fluid dofs.
        "fieldsplit_v_0_ksp_type": "preonly",
        "fieldsplit_v_0_pc_type": "fieldsplit",
        "fieldsplit_v_0_pc_fieldsplit_type": "multiplicative",
    }
    | {f"fieldsplit_v_0_fieldsplit_s_{k}": v for k, v in DEFAULT_V_S_OPTIONS.items()}
    | {f"fieldsplit_v_0_fieldsplit_f_{k}": v for k, v in DEFAULT_V_F_OPTIONS.items()}
    | {f"fieldsplit_p_1_{k}": v for k, v in DEFAULT_P_OPTIONS.items()}
)
