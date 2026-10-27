# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""The linear solve shared by the mesh motion problems."""

import dolfinx
import dolfinx.fem.petsc
from petsc4py import PETSc

# The mesh motion is linear, with a matrix that is the same for every solve: one SNES iteration
# (ksponly) solves it exactly, and the Jacobian is lagged explicitly, so it is assembled, and by a
# direct solver factorized, only in the first solve. Failures are caught by MeshMotion.solve.
SNES_OPTIONS = {
    "snes_type": "ksponly",
    "snes_lag_jacobian": -2,
    "snes_lag_jacobian_persists": True,
    "snes_error_if_not_converged": False,
    "ksp_error_if_not_converged": False,
}


def default_mesh_motion_solver_options(num_components: int) -> dict:
    """MUMPS LU on the first displacement component, applied to all components at once.

    The mesh motion does not couple the displacement components, and the components have the same
    matrix, also in their Dirichlet rows. IdenticalInterleavedBlocksPC factorizes the first
    component only and solves for all ``num_components`` components with that factorization by one
    MatMatSolve, which reads the factor once instead of once per component. It takes the
    components to be interleaved row by row, as the owned dofs of a blocked vector space are, also
    in the block matrix [z | u] of the mixed biharmonic problem.
    """
    return {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "xfsi_solver.tools.petsc_reuse.IdenticalInterleavedBlocksPC",
        "identical_interleaved_blocks_num": num_components,
        "block_pc_type": "lu",
        "block_pc_factor_mat_solver_type": "mumps",
        "block_mat_mumps_cntl_1": 1e-4,
    }


class MeshMotion:
    """A linear mesh motion problem, with residual ``F`` in the ``unknowns`` (the displacement last)
    and a Jacobian that is the same for every solve.

    It is solved by one Newton step from the current values, which is exact since F is linear, with
    the Jacobian assembled, and by a direct solver factorized, only in the first :meth:`solve` and
    reused in all later ones.

    ``solver_options`` are the PETSc options of the KSP, under ``petsc_options_prefix``, which should
    differ from those of other live problems; None gives :func:`default_mesh_motion_solver_options`.
    Options passed replace the defaults rather than add to them, so start from
    ``default_mesh_motion_solver_options(num_components) | {...}`` to change only some of them. The
    SNES options are those of ``SNES_OPTIONS``, which ``solver_options`` can override.
    """

    def __init__(
        self,
        F,
        unknowns: list[dolfinx.fem.Function],
        bcs: list[dolfinx.fem.DirichletBC],
        *,
        solver_options: dict | None,
        petsc_options_prefix: str,
        entity_maps=None,
    ):
        self.unknowns = unknowns
        self.displacement = unknowns[-1]
        if solver_options is None:
            solver_options = default_mesh_motion_solver_options(self.displacement.function_space.dofmap.index_map_bs)
        self.solver_options = solver_options

        # Remove the options of an earlier problem with this prefix, which the persistent options
        # below would otherwise leave behind.
        opts = PETSc.Options()
        for key in list(opts.getAll()):
            if key.startswith(petsc_options_prefix):
                del opts[key]

        self._problem = dolfinx.fem.petsc.NonlinearProblem(
            F,
            unknowns if len(unknowns) > 1 else unknowns[0],
            bcs=bcs,
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=SNES_OPTIONS | solver_options,
            entity_maps=entity_maps,
        )

        # dolfinx means to delete the options in petsc_options once the SNES is set up from them, but
        # PCs set up in the first solve read theirs then: the block PC of IdenticalInterleavedBlocksPC,
        # or the sub-KSPs of a fieldsplit. Its deletion, under a pushed prefix, has no effect with
        # petsc4py 3.25, so they stay; set them again, so that this does not depend on it.
        opts.prefixPush(petsc_options_prefix)
        for key, value in solver_options.items():
            opts[key] = value
        opts.prefixPop()

    @property
    def solver(self) -> PETSc.SNES:
        return self._problem.solver

    def solve(self) -> None:
        """Solve for the current Dirichlet values, and raise ``RuntimeError`` if the solve failed."""
        self._problem.solve()
        self._check()

    def _check(self) -> None:
        """Raise if the solve failed, which the error options in SNES_OPTIONS leave silent. A failed
        factorization in the block PC of IdenticalInterleavedBlocksPC shows only there."""
        snes = self.solver
        ksp = snes.getKSP()
        pc = ksp.getPC()
        pc_reasons = {"PC": pc.getFailedReason()}
        if pc.getType() == PETSc.PC.Type.PYTHON:
            block_pc = getattr(pc.getPythonContext(), "block_pc", None)
            if block_pc is not None:
                pc_reasons["block PC"] = block_pc.getFailedReason()
        if snes.getConvergedReason() < 0 or any(reason != 0 for reason in pc_reasons.values()):
            failed = ", ".join(f"{name} failed reason {reason}" for name, reason in pc_reasons.items())
            raise RuntimeError(
                f"The mesh motion solve {snes.getOptionsPrefix()} failed: SNES reason {snes.getConvergedReason()}, "
                f"KSP reason {ksp.getConvergedReason()}, {failed}"
            )

    def destroy(self) -> None:
        """Destroy the SNES, and with it the factorization, collectively: call on all ranks."""
        self._problem.solver.destroy()
