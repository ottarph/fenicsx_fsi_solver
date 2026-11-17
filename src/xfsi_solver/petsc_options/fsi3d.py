# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""PETSc options for the 3d FSI benchmark solver, ``xfsi_solver.solvers.threeDfsi_simplified``."""

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
    # GAMG smoothing with three steps of under-relaxed SOR per level. In parallel, PETSc's SOR is
    # Gauss-Seidel within each rank and Jacobi between ranks, which can diverge where the coupling
    # across a rank boundary is strong, such as in the thin, stiff beam: with omega = 1, GAMG
    # diverged for some partitions of the 3d mesh in threeDfsi_c0ip. There, on the fine mesh with
    # 64 ranks, three steps took a third fewer FGMRES iterations than PETSc's two, and omega = 0.4
    # fewer than 0.3, 0.5 and 0.6 (32bd021:notes/fsi3d/implementation-log.md).
    "mg_levels_ksp_type": "richardson",
    "mg_levels_ksp_max_it": 3,
    "mg_levels_pc_type": "sor",
    "mg_levels_pc_sor_omega": 0.4,
    # "ksp_converged_reason": None,
}
DEFAULT_P_OPTIONS = {
    # One BoomerAMG V-cycle on \hat S.
    "ksp_type": "preonly",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    # hypre's recommendation for 3d problems; its default, 0.25, suits 2d.
    "pc_hypre_boomeramg_strong_threshold": 0.5,
    # "ksp_converged_reason": None,
}
# The K_c options of threeDfsi_c0ip, with a BoomerAMG strong threshold of 0.5.
DEFAULT_ITERATIVE_SOLVER_OPTIONS = (
    DEFAULT_SOLVER_OPTIONS
    | {
        "ksp_converged_reason": None,
        "ksp_type": "fgmres",
        # On the medium mesh, rtol 1e-4 kept the Newton iterations of 1e-5 with a fifth fewer FGMRES
        # iterations (32bd021:notes/fsi3d/implementation-log.md).
        "ksp_rtol": 1e-4,
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
# Block Jacobi on v_f, one block per rank, with ILU(0) on each block. The fluid rows are dominated by
# the mass, rho_f / dt M, and couple only neighbouring dofs, so they need no coarse grid: on the medium
# and the fine mesh with dt = 0.004, block Jacobi took twice the FGMRES iterations of one GAMG V-cycle
# (DEFAULT_V_OPTIONS, with the block size attached to is_v_f), but 17% and 22% less time per step
# (32bd021:notes/fsi3d/implementation-log.md). Where the viscous term, nu dt / h^2, or the convection, U dt / h,
# outgrows the mass, at larger dt or on finer meshes, use DEFAULT_V_OPTIONS here instead.
DEFAULT_V_F_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "bjacobi",
    "sub_pc_type": "ilu",
}
# The Schur factor of DEFAULT_ITERATIVE_SOLVER_OPTIONS, with the velocity block split again,
# multiplicatively, into the solid dofs, solved exactly, and the fluid dofs, by block Jacobi with
# ILU(0). The solid rows are dominated by the solid stiffness, where the SOR smoothing of GAMG, which is
# Jacobi between ranks, depends on the partition; an exact solve does not. The fluid rows are dominated
# by the mass. Needs requires_v_sub_IS=True in threeDfsi_simplified.solve, which sets the inner
# ISes.
MUMPS_ON_SOLID = (
    DEFAULT_SOLVER_OPTIONS
    | {
        "ksp_converged_reason": None,
        "ksp_type": "fgmres",
        # As in DEFAULT_ITERATIVE_SOLVER_OPTIONS, here with a quarter fewer FGMRES iterations than 1e-5;
        # 1e-3 and looser took more Newton iterations.
        "ksp_rtol": 1e-4,
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
