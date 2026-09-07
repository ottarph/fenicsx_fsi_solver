# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import dolfinx as dfx
import dolfinx.fem.petsc as dfpetsc
import numpy as np
import basix.ufl
import ufl

from mpi4py.MPI import COMM_WORLD as comm
from mpi4py import MPI
from petsc4py import PETSc


def biharmonic(u_bc: dfx.fem.Function):

    mesh = u_bc.function_space.mesh
    T = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(2,))
    El = basix.ufl.mixed_element([T, T])

    Fspace = dfx.fem.functionspace(mesh, El)
    fspace, _ = Fspace.sub(0).collapse()


    uv = ufl.TrialFunction(Fspace)
    phi_uv = ufl.TestFunction(Fspace)

    u, v = ufl.split(uv)
    phi_u, phi_v = ufl.split(phi_uv)

    f = dfx.fem.Constant(mesh, (0.0, 0.0))
    a = -ufl.inner(v, phi_v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(phi_v)) * ufl.dx
    a += ufl.inner(ufl.grad(v), ufl.grad(phi_u)) * ufl.dx
    L = ufl.inner(f, phi_u) * ufl.dx + \
        ufl.inner(f, phi_v) * ufl.dx
    

    bc_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dfx.fem.locate_dofs_topological((Fspace.sub(0), fspace), 1, bc_facets)

    u_D = dfx.fem.Function(fspace)
    
    u_D.interpolate(u_bc)
    bc = dfx.fem.dirichletbc(u_D, boundary_dofs, Fspace.sub(0))


    prob = dfpetsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix="biharmonic_",
                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                            "pc_factor_mat_solver_type": "mumps", "ksp_error_if_not_converged": True,
                                                            "mat_mumps_icntl_14": 30})
    prob.solve()

    uh = prob.u.sub(0)
    vh = prob.u.sub(1)

    
    uh_pure = dfx.fem.Function(u_bc.function_space, name="uh")
    vh_pure = dfx.fem.Function(u_bc.function_space, name="vh")
    uh_pure.interpolate(uh)
    vh_pure.interpolate(vh)

    return uh_pure, vh_pure, prob



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


    U = dfx.fem.functionspace(mesh, ("CG", 2, (2,)))
    V = dfx.fem.functionspace(mesh, ("CG", 2, (2,)))
    W = ufl.MixedFunctionSpace(U, V)

    u, v = ufl.TrialFunctions(W)
    phi_u, phi_v = ufl.TestFunctions(W)

    bc_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    bc_dofs = dfx.fem.locate_dofs_topological(U, 1, bc_facets)
    u_bc = dfx.fem.Function(U)
    u_bc.interpolate(bc_func)
    bc = dfx.fem.dirichletbc(u_bc, bc_dofs)

    dx = ufl.Measure("dx", domain=mesh)
    f = dfx.fem.Constant(mesh, (0.0, 0.0))

    a  = ufl.inner(ufl.grad(u), ufl.grad(phi_v)) * dx - ufl.inner(v, phi_v) * dx 
    a += ufl.inner(ufl.grad(v), ufl.grad(phi_u)) * dx
    a += ufl.inner(dfx.fem.Constant(mesh, 0.0) * u, phi_u) * dx
    
    L = ufl.inner(f, phi_u) * dx + ufl.inner(f, phi_v) * dx

    a_block = ufl.extract_blocks(a)
    L_block = ufl.extract_blocks(L)

    uh = dfx.fem.Function(U)
    vh = dfx.fem.Function(V)

    problem = dfx.fem.petsc.LinearProblem(
        a_block, L_block, bcs=[bc], u=[uh, vh],
        petsc_options_prefix="biharm_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
            "ksp_error_if_not_converged": True,
        },
    )
    problem.solve()

    with dfx.io.VTXWriter(comm, output_path, [uh]) as writer:
        writer.write(0.0)

    uh_pure, _, _ = biharmonic(u_bc)

    uh_pure.x.petsc_vec.array[:] -= uh.x.petsc_vec.array
    difference = uh_pure.x.petsc_vec.norm()

    if comm.rank == 0:
        print(f"Norm of differnce in two biharmonic solvers: {difference:.2e}")


    return


def main():
    solve(
        N=32,
        output_path="output/pv/biharm.bp",
    )


if __name__ == "__main__":
    main()
