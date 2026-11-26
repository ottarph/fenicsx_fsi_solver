# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""The interface ALE terms of the Jacobian of the solvers on (v, p) only (fsi2_simplified,
threeDfsi_simplified).

These solvers set the displacements from v after each linear solve: the solid displacement and the
interface values of the fluid displacement u_f by the kinematic relation, u_f = u_f_old + dt * (theta * v
+ (1 - theta) * v_old) at the interface dofs, and the interior of u_f by the mesh motion. Their Jacobian
is that of the residual with respect to (v, p) alone, so it lacks the derivatives of the fluid residual
through the mesh. The interface ALE terms add those through the interface dofs of u_f: theta * dt * C K
in the (v, v) and (p, v) blocks, with C the derivative of the momentum and mass residuals with respect to
u_f and K the map from v to the interface dofs of u_f, at the same nodes. Those through the interior of
the fluid mesh, which the mesh motion moves with the interface, are still left out.

For a constant fluid pressure P, the pressure term of the momentum equation is P times the integral of
dv . n over the fluid boundary, so its derivative with respect to the interior mesh nodes vanishes: the
part of the missing derivative that grows with the mean pressure is all at the interface, which these
terms add. On FSI2 they kept Newton's contraction independent of the mean pressure.
"""

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
import ufl.algorithms
from mpi4py import MPI
from petsc4py import PETSc


class InterfaceALE:
    """The interface ALE terms theta * dt * C K, to add to a nest Jacobian of (v, p).

    They are assembled straight into the (v, v) and (p, v) blocks, from the derivative of the residual
    with respect to u_f in the direction theta * dt * w, with w a trial function on a clone of V. U_f and
    V have the same element, so this is C K' with K' the map from v to all dofs of u_f at the same
    nodes. The clone shares the element, dofmap and global numbering of V, so the columns are those of
    the blocks, but it is another space: a Dirichlet condition on it zeroes columns only. One on every
    node of the clone but the free interface nodes keeps the interface columns of C K', which are those
    of C K. The Dirichlet conditions of v zero the rows of their dofs in the (v, v) block; they leave the
    (p, v) block alone, whose rows are in the pressure space. Neither form has the same test and trial
    space, so no diagonal is inserted. The terms are assembled only on the fluid cells with a dof on
    the interface, the only cells with an entry in an interface column. They couple dofs that share a
    cell, so they fit the pattern of the Jacobian.

    Args:
        residual_blocks: the momentum and mass residuals, in this order, with their quadrature degree set
            (``xfsi_solver.tools.quadrature.fix_quadrature_degree``), so that C is their exact derivative.
        u_f: the fluid displacement, on the fluid submesh.
        V: the velocity space, on the parent mesh, with the same element as that of ``u_f``.
        cell_tags, facet_tags: the cell and facet tags of the parent mesh.
        fluid_marker: the cell tag of the fluid cells, the only subdomain of the derivatives of the residual
            with respect to ``u_f``.
        interface_marker: the facet tag of the interface.
        entity_maps: the entity maps of the residual forms.
        bcs: the Dirichlet conditions on V, each on whole nodes (all components).
        theta, dt: the time stepping parameters.
        quadrature_degree: if given, C is integrated with this quadrature degree instead of that of the
            residual, which makes it an approximation of the derivative, cheaper to assemble.
        cell_tag: a cell tag not in ``cell_tags``, for the cells on which C is assembled.
    """

    def __init__(
        self,
        residual_blocks,
        u_f: dolfinx.fem.Function,
        V: dolfinx.fem.FunctionSpace,
        cell_tags: dolfinx.mesh.MeshTags,
        facet_tags: dolfinx.mesh.MeshTags,
        fluid_marker: int,
        interface_marker: int,
        entity_maps,
        bcs,
        theta: float,
        dt: float,
        quadrature_degree: int | None = None,
        cell_tag: int = 1000,
    ):
        mesh = V.mesh
        comm = mesh.comm
        block_size = V.dofmap.index_map_bs
        num_nodes = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts

        # The fluid cells with a node on the interface, owned or ghost, which remote=True finds also where
        # the rank holds the interface only at a vertex.
        interface_nodes = dolfinx.fem.locate_dofs_topological(
            V, mesh.topology.dim - 1, facet_tags.find(interface_marker), remote=True
        )
        on_interface = np.zeros(num_nodes, dtype=bool)
        on_interface[interface_nodes] = True
        fluid_cells = cell_tags.find(fluid_marker)
        ale_cells = np.sort(fluid_cells[on_interface[V.dofmap.list[fluid_cells]].any(axis=1)])
        # Checked on all ranks together, so that either all of them raise or none does.
        if comm.allreduce(bool(np.any(cell_tags.values == cell_tag)), op=MPI.LOR):
            raise ValueError(f"The cell tag {cell_tag}, used for the interface ALE cells, is already in use")
        ale_cell_tags = dolfinx.mesh.meshtags(
            mesh, mesh.topology.dim, ale_cells.astype(np.int32), np.full(len(ale_cells), cell_tag, dtype=np.int32)
        )
        metadata = {} if quadrature_degree is None else {"quadrature_degree": quadrature_degree}

        trial_space = V.clone()
        du_f = ufl.TrialFunction(u_f.function_space)
        direction = dolfinx.fem.Constant(mesh, PETSc.ScalarType(theta * dt)) * ufl.TrialFunction(trial_space)

        def interface_ale_form(form: ufl.Form) -> ufl.Form:
            """The derivative of ``form`` with respect to u_f in the direction theta * dt * w, over ale_cells."""
            # Expanding the derivatives drops the integrals that do not depend on u_f, those of the solid.
            integrals = ufl.algorithms.expand_derivatives(ufl.derivative(form, u_f, du_f)).integrals()
            if any(i.integral_type() != "cell" or i.subdomain_id() != fluid_marker for i in integrals):
                raise ValueError("The derivatives with respect to u_f should only have fluid cell integrals")
            restricted = ufl.Form(
                [
                    i.reconstruct(subdomain_id=cell_tag, subdomain_data=ale_cell_tags, metadata=i.metadata() | metadata)
                    for i in integrals
                ]
            )
            return ufl.replace(restricted, {du_f: direction})

        self._forms = [dolfinx.fem.form(interface_ale_form(form), entity_maps=entity_maps) for form in residual_blocks]

        # The columns kept are those of the interface nodes without a Dirichlet condition on v.
        free = np.ones(num_nodes * block_size, dtype=bool)
        for bc in bcs:
            free[bc.dof_indices()[0]] = False
        free = free.reshape(-1, block_size)
        # Checked on all ranks together, so that either all of them raise or none does.
        if comm.allreduce(bool(np.any(free.any(axis=1) != free.all(axis=1))), op=MPI.LOR):
            raise ValueError("The interface ALE terms need Dirichlet conditions on v that hold on whole nodes")
        dropped = np.flatnonzero(~(on_interface & free.all(axis=1))).astype(np.int32)
        self._bcs = [
            dolfinx.fem.dirichletbc(np.zeros(block_size, dtype=PETSc.ScalarType), dropped, trial_space),
            *bcs,
        ]

    def add(self, J: PETSc.Mat) -> None:
        """Add theta * dt * C K to the (v, v) and (p, v) blocks of the nest Jacobian J, and assemble J."""
        for i, form in enumerate(self._forms):
            # The condition on the clone zeroes the columns of the dropped nodes; those on v zero the rows of
            # their dofs where the rows are in V, in the (v, v) block.
            dolfinx.fem.petsc.assemble_matrix(J.getNestSubMatrix(i, 0), form, bcs=self._bcs)
        J.assemble()


def interface_ale_description(interface_ale: bool, quadrature_degree: int | None) -> str:
    """The interface ALE setting, for the summary at the end of a run."""
    if not interface_ale:
        return "off"
    if quadrature_degree is None:
        return "on, integrated with the quadrature of the residual"
    return f"on, integrated with quadrature degree {quadrature_degree}"
