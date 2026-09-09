# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import ufl

from mpi4py.MPI import COMM_WORLD as comm

from xfsi_solver.tools.custom_linear_problem import MyLinearProblem


def biharmonic(u_bc: dfx.fem.Function):

    mesh = u_bc.function_space.mesh
    U = u_bc.function_space
    V = dfx.fem.functionspace(mesh, U.ufl_element())

    W = ufl.MixedFunctionSpace(U, V)

    u, v = ufl.TrialFunctions(W)
    phi_u, phi_v = ufl.TestFunctions(W)

    bc_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    bc_dofs = dfx.fem.locate_dofs_topological(U, 1, bc_facets)
    bc = dfx.fem.dirichletbc(u_bc, bc_dofs)

    dx = ufl.Measure("dx", domain=mesh)
    f = dfx.fem.Constant(mesh, (0.0, 0.0))

    a  = ufl.inner(ufl.grad(u), ufl.grad(phi_v)) * dx - ufl.inner(v, phi_v) * dx
    a += ufl.inner(ufl.grad(v), ufl.grad(phi_u)) * dx
    a += ufl.inner(dfx.fem.Constant(mesh, 0.0) * u, phi_u) * dx

    L = ufl.inner(f, phi_u) * dx + ufl.inner(f, phi_v) * dx

    a_block = ufl.extract_blocks(a)
    L_block = ufl.extract_blocks(L)

    uh = dfx.fem.Function(U, name="uh")
    vh = dfx.fem.Function(V, name="vh")

    prob = MyLinearProblem(
        a_block, L_block, bcs=[bc], u=[uh, vh],
        petsc_options_prefix="biharmonic_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
            "ksp_error_if_not_converged": True,
        },
    )
    prob.assemble_matrix()
    prob.solve()

    return uh, vh, prob


def solve(N, output_path):

    mesh = dfx.mesh.create_unit_square(comm, N, N, cell_type=dfx.mesh.CellType.triangle)
    mesh.topology.create_connectivity(1, 2)

    V = dfx.fem.functionspace(mesh, ("CG", 2, (2, )))

    def bc_func(x):
        H = 0.2
        values = np.zeros_like(x[:2,:])
        values[0] = 0.0
        values[1] = np.where(np.isclose(x[1], 1.0), 1.0, 0.0) * 4 * x[0] * (1 - x[0]) * H
        return values

    u_bc = dfx.fem.Function(V)
    u_bc.interpolate(bc_func)

    uh, vh, prob = biharmonic(u_bc)

    with dfx.io.VTXWriter(comm, output_path, [uh]) as writer:
        writer.write(0.0)

    return


def main():
    solve(
        N=32,
        output_path="output/pv/biharm.bp",
    )


if __name__ == "__main__":
    main()
