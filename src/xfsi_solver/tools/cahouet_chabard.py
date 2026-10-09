# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Cahouet-Chabard preconditioner for the pressure Schur complement.

For a velocity block A ~ alpha * M + mu * K, with the velocity mass matrix M and the
viscous operator K, the pressure Schur complement S = -B A^{-1} G is approximated by

    S^{-1} ~ alpha * L_p^{-1} + mu * M_p^{-1}

(Cahouet & Chabard, 1988), with the pressure Laplacian L_p and the pressure mass matrix
M_p. It is a sum of inverse actions, not the inverse of a sum, and is applied as a PETSc
python PC on the Schur complement KSP of a fieldsplit.
"""

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from petsc4py import PETSc

# Approximate solves with L_p and M_p. Only spectral equivalence is needed in a
# preconditioner: Jacobi on the P1 mass matrix is well conditioned, and one AMG V-cycle
# approximates the Laplacian. Both are preonly, so the preconditioner is a fixed linear
# operator, as GMRES requires.
ITERATIVE_MASS_OPTIONS = {"ksp_type": "preonly", "pc_type": "jacobi"}
ITERATIVE_STIFFNESS_OPTIONS = {"ksp_type": "preonly", "pc_type": "hypre", "pc_hypre_type": "boomeramg"}

# Exact solves with L_p and M_p, as a reference.
DIRECT_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_cntl_1": 1e-4,
}


def outflow_pressure_dofs(
    P_space: dolfinx.fem.FunctionSpace,
    parent_mesh: dolfinx.mesh.Mesh,
    outflow_facets: np.ndarray,
    vertex_map: dolfinx.mesh.EntityMap,
) -> np.ndarray:
    """Dofs of the P1 pressure space ``P_space`` on a submesh of ``parent_mesh``, on the
    ``outflow_facets`` of the parent mesh.

    The facets are tagged on the parent mesh, so the dofs are located through their
    vertices, which are the P1 dofs, mapped to the submesh with ``vertex_map``.
    """
    tdim = parent_mesh.topology.dim
    parent_mesh.topology.create_connectivity(tdim - 1, 0)
    outflow_vertices = dolfinx.mesh.compute_incident_entities(parent_mesh.topology, outflow_facets, tdim - 1, 0)
    outflow_vertices_sub = vertex_map.sub_topology_to_topology(outflow_vertices, inverse=True)
    P_space.mesh.topology.create_connectivity(0, P_space.mesh.topology.dim)
    return dolfinx.fem.locate_dofs_topological(P_space, 0, outflow_vertices_sub)


class _PressureOperator:
    """A pressure operator, assembled once, with its KSP set up once, for the whole run."""

    def __init__(self, bilinear_form, P_space, dx, bcs, entity_maps, petsc_options, prefix):
        # The right-hand side is only needed for the setup solve.
        self._solver = dolfinx.fem.petsc.LinearProblem(
            bilinear_form,
            ufl.TestFunction(P_space) * dx,
            bcs=bcs,
            petsc_options_prefix=prefix,
            petsc_options=petsc_options,
            entity_maps=entity_maps,
        )
        # Assembles the matrix and sets up the preconditioner. The LinearProblem destroys its
        # PETSc objects when it is garbage collected, so it is kept alive here.
        self._solver.solve()
        self.ksp = self._solver.solver

    def destroy(self) -> None:
        """Destroy the PETSc objects of the LinearProblem. Destroying a MUMPS factorization or
        an AMG hierarchy is collective, so call this on all ranks at the same point."""
        for obj in (self._solver.solver, self._solver.A, self._solver.b, self._solver.x):
            obj.destroy()


class PressureMass(_PressureOperator):
    """The pressure mass matrix M_p, integrated with ``dx``."""

    def __init__(self, P_space, dx, entity_maps, petsc_options):
        trial_p, test_p = ufl.TrialFunction(P_space), ufl.TestFunction(P_space)
        super().__init__(trial_p * test_p * dx, P_space, dx, [], entity_maps, petsc_options, "pressure_mass_")


class PressureStiffness(_PressureOperator):
    """The pressure Laplacian L_p, integrated with ``dx``, with p = 0 on ``outflow_dofs``.

    p = 0 on the do-nothing outflow fixes the pressure level as it does in the Schur
    complement, with natural (Neumann) conditions elsewhere, where the velocity has
    Dirichlet conditions.
    """

    def __init__(self, P_space, dx, entity_maps, outflow_dofs, petsc_options):
        trial_p, test_p = ufl.TrialFunction(P_space), ufl.TestFunction(P_space)
        bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(0), outflow_dofs, P_space)
        # Owned local indices of the outflow dofs, where L_p^{-1} applied to a residual is zero.
        dofs, num_owned = bc.dof_indices()
        self.bc_dofs = dofs[:num_owned]
        bilinear_form = ufl.inner(ufl.grad(trial_p), ufl.grad(test_p)) * dx
        super().__init__(bilinear_form, P_space, dx, [bc], entity_maps, petsc_options, "pressure_stiffness_")


class CahouetChabard:
    """PCSHELL context approximating the inverse of the pressure Schur complement S by
    y = alpha * L_p^{-1} x + mu * M_p^{-1} x, for the velocity block A ~ alpha * M + mu * K.

    L_p and M_p are assembled with ``dx`` on the space ``P_space`` of the pressure, whose
    owned dofs must have the same order and parallel layout as the Schur complement
    vectors. With ``iterative=True``, L_p^{-1} is approximated by one BoomerAMG V-cycle and
    M_p^{-1} by Jacobi, otherwise both are solved with MUMPS LU. Both keep the
    preconditioner a fixed linear operator. The operators are assembled once, so ``dx``
    should integrate over a fixed (reference) domain.
    """

    def __init__(self, P_space, dx, entity_maps, outflow_dofs, alpha, mu, iterative=True):
        mass_options = ITERATIVE_MASS_OPTIONS if iterative else DIRECT_OPTIONS
        stiffness_options = ITERATIVE_STIFFNESS_OPTIONS if iterative else DIRECT_OPTIONS
        self.mass = PressureMass(P_space, dx, entity_maps, mass_options)
        self.stiffness = PressureStiffness(P_space, dx, entity_maps, outflow_dofs, stiffness_options)
        self.alpha = alpha
        self.mu = mu
        self.iterative = iterative
        self.rhs, self.work = self.stiffness.ksp.getOperators()[0].createVecs()

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        # The Schur complement vectors hold the owned pressure dofs in the same order and with
        # the same parallel layout as the vectors of P_space, so they are passed directly.
        x.copy(self.rhs)
        # Zero the outflow entries, so that L_p^{-1} only acts on the interior pressure. The
        # identity rows of L_p would otherwise return them unchanged, scaled by alpha.
        self.rhs.array[self.stiffness.bc_dofs] = 0.0
        self.stiffness.ksp.solve(self.rhs, self.work)
        self.mass.ksp.solve(x, y)
        y.scale(self.mu)
        y.axpy(self.alpha, self.work)

    def description(self) -> str:
        """The approximate inverse and its pressure solves, for setup printouts."""
        if self.iterative:
            solves = "one BoomerAMG V-cycle on L_p, Jacobi on M_p"
        else:
            solves = "MUMPS LU on L_p and M_p"
        return f"Cahouet-Chabard, {self.alpha:g} * L_p^{{-1}} + {self.mu:g} * M_p^{{-1}}, with {solves}"

    def destroy_solvers(self) -> None:
        """Destroy the pressure solvers. This is collective, so call it on all ranks at the
        same point.

        Not named ``destroy``: petsc4py calls ``destroy(pc)`` on a python PC context when the
        PC is destroyed, which can happen during garbage collection, at different points on
        different ranks.
        """
        self.mass.destroy()
        self.stiffness.destroy()
        self.rhs.destroy()
        self.work.destroy()
