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

    C is assembled only on the fluid cells with a dof on the interface, the only cells that fill the
    columns of C that K takes. The added entries couple dofs that share a cell, so they fit the pattern
    of the Jacobian. They leave out the rows of the Dirichlet dofs of v in the (v, v) block, and K
    leaves out its columns.

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
        bcs: the Dirichlet conditions on V.
        interface_u_f, interface_v: the unrolled local dofs of u_f on the interface, owned and ghost, and of
            v at the same nodes.
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
        interface_u_f: np.ndarray,
        interface_v: np.ndarray,
        theta: float,
        dt: float,
        quadrature_degree: int | None = None,
        cell_tag: int = 1000,
    ):
        mesh = V.mesh
        self._comm = comm = mesh.comm
        self._scale = theta * dt
        U_f = u_f.function_space
        block_size = V.dofmap.index_map_bs

        # The fluid cells with a dof of v on the interface, owned or ghost, which remote=True finds also
        # where the rank holds the interface only at a vertex.
        interface_nodes = dolfinx.fem.locate_dofs_topological(
            V, mesh.topology.dim - 1, facet_tags.find(interface_marker), remote=True
        )
        on_interface = np.zeros(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, dtype=bool)
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

        def on_interface_cells(form: ufl.Form) -> ufl.Form:
            """``form``, a derivative of the residual with respect to u_f, integrated over ale_cells only."""
            # Expanding the derivatives drops the integrals that do not depend on u_f, those of the solid.
            integrals = ufl.algorithms.expand_derivatives(form).integrals()
            if any(i.integral_type() != "cell" or i.subdomain_id() != fluid_marker for i in integrals):
                raise ValueError("The derivatives with respect to u_f should only have fluid cell integrals")
            return ufl.Form(
                [
                    i.reconstruct(subdomain_id=cell_tag, subdomain_data=ale_cell_tags, metadata=i.metadata() | metadata)
                    for i in integrals
                ]
            )

        du_f = ufl.TrialFunction(U_f)
        self._forms = [
            dolfinx.fem.form(on_interface_cells(ufl.derivative(form, u_f, du_f)), entity_maps=entity_maps)
            for form in residual_blocks
        ]
        self._matrices = [dolfinx.fem.petsc.create_matrix(form) for form in self._forms]

        # The v dofs with Dirichlet conditions, whose rows and columns of the Jacobian stay as they are.
        v_free = np.ones((V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts) * block_size)
        for bc in bcs:
            v_free[bc.dof_indices()[0]] = 0.0
        self._v_free = v_free
        # K: each owned interface dof of u_f to the dof of v at the same node, without the Dirichlet dofs
        # of v. It is used only to find where the entries of C go in J.
        num_owned_u_f = U_f.dofmap.index_map.size_local * block_size
        owned = (interface_u_f < num_owned_u_f) & (v_free[interface_v] == 1.0)
        rows, cols = interface_u_f[owned], interface_v[owned]
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
        cols_global = (
            block_size * V.dofmap.index_map.local_to_global((cols // block_size).astype(np.int32)) + cols % block_size
        ).astype(PETSc.IntType)
        indptr = np.zeros(num_owned_u_f + 1, dtype=PETSc.IntType)
        np.add.at(indptr, rows + 1, 1)
        indptr = np.cumsum(indptr).astype(PETSc.IntType)
        self._K = PETSc.Mat().createAIJ(
            ((num_owned_u_f, None), (V.dofmap.index_map.size_local * block_size, None)),
            csr=(indptr, cols_global, np.ones(len(cols_global))),
            comm=comm,
        )
        # Where theta * dt * C K goes in J, for each of the two blocks: the positions in the CSR values of
        # C of the entries of C K, and C K's CSR row pointer and global columns. Set at the first Jacobian.
        self._targets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._num_entries: list[int] = []  # of C, to check that its pattern stays the same

    def _find_targets(self, C: PETSc.Mat, C_indptr: np.ndarray, C_cols: np.ndarray, i: int) -> None:
        """Find where the entries of C K are in C and in J. C K is C with each interface column moved to the
        column of v at the same node and the other columns dropped, and the pattern of C does not change,
        so this is done once. A copy of C whose values are the positions of its entries, plus 1, is
        multiplied by K: each column of K has at most one entry, a 1, so each entry of the product is
        exactly one entry of the copy. PETSc's product also finds the columns of v of the interface columns
        of C held by other ranks."""
        C_positions = C.duplicate(copy=False)
        C_positions.setValuesCSR(C_indptr, C_cols, np.arange(1, len(C_cols) + 1, dtype=PETSc.ScalarType))
        C_positions.assemble()
        CK = C_positions.matMult(self._K)
        CK_indptr, CK_cols, CK_positions = CK.getValuesCSR()
        CK.destroy()
        C_positions.destroy()
        # Checked on all ranks together, so that either all of them raise or none does.
        if self._comm.allreduce(bool(np.any(CK_positions < 1)), op=MPI.LOR):
            raise RuntimeError("Each entry of C K should be one entry of C")
        num_rows = len(CK_indptr) - 1
        rows = np.repeat(np.arange(num_rows), np.diff(CK_indptr))
        # The rows of the Dirichlet dofs of v in the (v, v) block stay as they are.
        keep = self._v_free[rows] == 1.0 if i == 0 else np.ones(len(rows), dtype=bool)
        indptr = np.zeros(num_rows + 1, dtype=PETSc.IntType)
        indptr[1:] = np.cumsum(np.bincount(rows[keep], minlength=num_rows))
        self._targets.append(
            (np.rint(CK_positions[keep]).astype(np.int64) - 1, indptr, CK_cols[keep].astype(PETSc.IntType))
        )

    def add(self, J: PETSc.Mat) -> None:
        """Add theta * dt * C K to the (v, v) and (p, v) blocks of the nest Jacobian J, as theta * dt times
        the entries of C in the interface columns, and assemble J."""
        for i, (form, C) in enumerate(zip(self._forms, self._matrices, strict=True)):
            C.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(C, form)
            C.assemble()
            C_indptr, C_cols, C_values = C.getValuesCSR()
            if len(self._targets) == i:
                self._find_targets(C, C_indptr, C_cols, i)
                self._num_entries.append(len(C_values))
            elif self._comm.allreduce(len(C_values) != self._num_entries[i], op=MPI.LOR):
                raise RuntimeError("The pattern of C has changed since the first Jacobian")
            positions, indptr, cols = self._targets[i]
            J.getNestSubMatrix(i, 0).setValuesCSR(
                indptr, cols, self._scale * C_values[positions], addv=PETSc.InsertMode.ADD_VALUES
            )
        J.assemble()

    def destroy(self) -> None:
        """Destroy the PETSc matrices, which is collective."""
        for C in self._matrices:
            C.destroy()
        self._K.destroy()


def interface_ale_description(interface_ale: bool, quadrature_degree: int | None) -> str:
    """The interface ALE setting, for the summary at the end of a run."""
    if not interface_ale:
        return "off"
    if quadrature_degree is None:
        return "on, integrated with the quadrature of the residual"
    return f"on, integrated with quadrature degree {quadrature_degree}"
