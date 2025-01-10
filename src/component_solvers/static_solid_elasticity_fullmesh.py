# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import dolfinx as dfx
import dolfinx.fem.petsc as dfpetsc
import numpy as np
import basix.ufl
import ufl
from petsc4py import PETSc

from mpi4py.MPI import COMM_WORLD as comm

PHYSICAL_MARKERS = {
    "solid": 1,
    "ALE_fluid": 2,

    "solid_fluid_interface": 11,

    "obstacle": 21,                 # no-slip for fluid
    "inflow": 22,                   # parabolic inflow for fluid
    "outflow": 23,                  # do-nothing for fluid
    "channel_side": 24,             # no-slip for fluid
    "solid_obstacle_interface": 25, # homogeneous Dirichlet BC for solid
}

def main():


    # load mesh and meshtags

    mesh_path = "data/meshes/fsi2/mesh.xdmf"

    with dfx.io.XDMFFile(comm, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name= "Cell tags")
        mesh.topology.create_connectivity(1, 2)
        facet_tags = infile.read_meshtags(mesh, name= "Facet tags")


    assert len(np.setdiff1d(np.union1d(cell_tags.values, facet_tags.values), [PHYSICAL_MARKERS[i] for i in PHYSICAL_MARKERS])) == 0, "Physical markers and cell tags do not match"
    
    
    # Create submesh for solid

    solid_mesh, solid_cell_map, solid_vertex_map, _ = dfx.mesh.create_submesh(mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"]))
    solid_mesh.topology.create_connectivity(1, 2)

    
    # transfer meshtags to submesh
    
    import scifem
    solid_facet_tags, solid_facet_map = scifem.transfer_meshtags_to_submesh(facet_tags, solid_mesh, solid_vertex_map, solid_cell_map)


    # Create measure with  meshtags

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)(PHYSICAL_MARKERS["solid"])

    
    # Create measure for interface / solid-fluid boundary

    from tools.interior_facet_measure import create_consistent_interior_facet_measure

    new_tag = 100
    new_measure = create_consistent_interior_facet_measure(mesh, facet_tags, cell_tags,
                        PHYSICAL_MARKERS["solid_fluid_interface"], PHYSICAL_MARKERS["solid"], new_tag)
    ds_interface = new_measure(new_tag)


    # create entity maps for mixed mesh integration

    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts
    mesh_to_solid_entity = np.full(num_cells_local, -1, dtype=np.int32)
    mesh_to_solid_entity[solid_cell_map] = np.arange(len(solid_cell_map), dtype=np.int32)

    entity_maps = {solid_mesh: mesh_to_solid_entity}


    # create problem parameters

    rho_s = dfx.fem.Constant(mesh, 0.8e3)
    lambda_s = dfx.fem.Constant(mesh, 1e5)
    mu_s = dfx.fem.Constant(mesh, 2e7)

    g = dfx.fem.Constant(mesh, (0.0, -9.81*4))
    traction = dfx.fem.Constant(mesh, (0.0, 0.0))

    
    # create function space

    U = dfx.fem.functionspace(solid_mesh, ("CG", 2, (2, )))


    # create functions

    u = dfx.fem.Function(U)
    du = ufl.TestFunction(U)
    delta_u = ufl.TrialFunction(U)


    from fsi.materials import Solid

    F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
    J = ufl.det(F)
    n = ufl.FacetNormal(mesh)
    

    # create Dirichlet boundary condition

    bc_func = dfx.fem.Function(U)
    bc_func.x.array[:] = 0.0
    bc_facets = solid_facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"])
    bc_dofs = dfx.fem.locate_dofs_topological(U, mesh.geometry.dim - 1, bc_facets)
    bc = dfx.fem.dirichletbc(bc_func, bc_dofs)

    bcs = [bc]

    # create residual form

    residual = J * ufl.inner(Solid.STVK(u, lambda_s, mu_s) * ufl.inv(F).T, ufl.grad(du)) * dx
    residual -= ufl.inner(rho_s * g, du) * dx
    residual -= ufl.inner(traction, du) * ds_interface

    residual_comp = dfx.fem.form(residual, entity_maps=entity_maps)

    
    # create Jacobian form

    jacobian = ufl.derivative(residual, u, delta_u)
    jacobian_comp = dfx.fem.form(jacobian, entity_maps=entity_maps)


    # vtx writer for output
    writer = dfx.io.VTXWriter(comm, "output/static_solid_elasticity_fm.bp", [u])


    # test with built-in Newton solver

    nlprob = dfpetsc.NonlinearProblem(residual_comp, u, bcs=bcs, J=jacobian_comp)
    
    import dolfinx.nls.petsc as nls
    nlsolv = nls.NewtonSolver(comm, nlprob)


    nlsolv.atol = 1e-8
    nlsolv.rtol = 1e-8
    # nlsolv.convergence_criterion = "incremental"
    nlsolv.convergence_criterion = "residual"
    nlsolv.error_on_nonconvergence = False
    nlsolv.max_it = 20
    
    nlsolv.solve(u)
    writer.write(0)

    u.x.array[:] = 0.0
    u.x.scatter_forward()

    g.value = (0.0, +9.81*4)

    # create matrix and vector for linear algebra

    A = dfpetsc.create_matrix(jacobian_comp)
    b = dfpetsc.create_vector(residual_comp)
    x = dfpetsc.create_vector(residual_comp)


    ksp = PETSc.KSP().create(solid_mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    max_iter = 20
    atol = 1.0e-8
    rtol = 1.0e-8


    n = 0
    res0 = 1.0
    while n < max_iter:


        with b.localForm() as b_loc:
            b_loc.set(0)

        dfpetsc.assemble_vector(b, residual_comp)

        dfpetsc.apply_lifting(b, [jacobian_comp], bcs=[bcs], x0=[u.x.petsc_vec], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dfpetsc.set_bc(b, bcs, x0=u.x.petsc_vec, alpha=-1.0)

        b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

        res = b.norm()
        if n == 0:
            res0 = res
            if comm.rank == 0:
                print(f"{res0 = :.3e}")

        if comm.rank == 0:
            print(f"{n = :2d}:\t\t{res = :.3e}")

        if res < atol or res < rtol * res0:
            break

        A.zeroEntries()
        dfpetsc.assemble_matrix(A, jacobian_comp, bcs=bcs)
        A.assemble()


        ksp.solve(b, x)

        u.x.petsc_vec.axpy(-1.0, x)
        u.x.scatter_forward()

        n += 1


    if n > max_iter:
        writer.close()
        raise RuntimeError("Nonlinear solver did not converge")
    
    writer.write(1)
            


    A.destroy()
    b.destroy()
    x.destroy()
    writer.close()


    return


if __name__ == "__main__":
    main()
