# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import dolfinx as dfx
import dolfinx.fem.petsc as dfpetsc
import numpy as np
import basix.ufl
import ufl
from petsc4py import PETSc

import sys

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
    

    # create submeshes for fluid and solid

    solid_mesh, solid_cell_map, solid_vertex_map, _ = dfx.mesh.create_submesh(mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"]))

    if comm.rank == 0:
        print(f"{solid_mesh.geometry.x.shape = }")
        print(f"{solid_cell_map.shape = }")

    solid_mesh.topology.create_connectivity(1, 2)


    # transfer meshtags to submeshes

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

    dt = dfx.fem.Constant(mesh, 0.0025)
    t0 = 0.0
    T = 0.2

    g = dfx.fem.Constant(mesh, (0.0, -9.81*4))
    traction = dfx.fem.Constant(mesh, (0.0, 0.0))

    
    # create function spaces

    U = dfx.fem.functionspace(solid_mesh, ("CG", 2, (2, )))
    V = dfx.fem.functionspace(solid_mesh, ("CG", 2, (2, )))
    W = ufl.MixedFunctionSpace(U, V)


    # create functions

    u, v = dfx.fem.Function(U), dfx.fem.Function(V)
    u_old, v_old = dfx.fem.Function(U), dfx.fem.Function(V)

    du, dv = ufl.TestFunctions(W)
    delta_u, delta_v = ufl.TrialFunctions(W)


    # Implicit Euler discretization of STVK under influence of gravitational body force and
    # fixed Dirichlet BC on left boundary and otherwise zero traction.
    
    du_dt = (u - u_old) / dt
    dv_dt = (v - v_old) / dt

    from fsi.materials import Solid

    F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
    J = ufl.det(F)
    n = ufl.FacetNormal(mesh)
    

    # create Dirichlet boundary condition

    bc_func = dfx.fem.Function(U)
    bc_func.x.array[:] = 0.0
    bc_facets = solid_facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"])
    bc_dofs = dfx.fem.locate_dofs_topological(U, solid_mesh.geometry.dim - 1, bc_facets)
    bc = dfx.fem.dirichletbc(bc_func, bc_dofs)

    bcs = [bc]

    # create residual form

    residual = rho_s * ufl.inner(du_dt - v, du) * dx

    residual += rho_s * ufl.inner(dv_dt, dv) * dx
    residual += J * ufl.inner(Solid.STVK(u, lambda_s, mu_s) * ufl.inv(F).T, ufl.grad(dv)) * dx
    residual -= ufl.inner(rho_s * g, dv) * dx
    residual -= ufl.inner(traction, dv) * ds_interface

    residual_blocked = ufl.extract_blocks(residual)
    residual_comp = dfx.fem.form(residual_blocked, entity_maps=entity_maps)

    
    # create Jacobian form

    jacobian = ufl.derivative(residual, u, delta_u) + ufl.derivative(residual, v, delta_v)
    jacobian_blocked = ufl.extract_blocks(jacobian)
    jacobian_comp = dfx.fem.form(jacobian_blocked, entity_maps=entity_maps)


    # create matrix and vector for linear algebra

    A = dfpetsc.create_matrix_block(jacobian_comp)
    b = dfpetsc.create_vector_block(residual_comp)
    x = dfpetsc.create_vector_block(residual_comp)
    delta_x = dfpetsc.create_vector_block(residual_comp)

    offset = U.dofmap.index_map.size_local * U.dofmap.index_map_bs


    ksp = PETSc.KSP().create(solid_mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    max_iter = 20
    atol = 1.0e-8
    rtol = 1.0e-8


    writer = dfx.io.VTXWriter(comm, "output/solid_elasticity_fm.bp", [u])

    t = t0

    while t < T:

        t += dt.value

        u_old.x.array[:] = u.x.array
        v_old.x.array[:] = v.x.array
        u_old.x.scatter_forward()
        v_old.x.scatter_forward()

        x.array[:offset] = u.x.array[:offset]
        x.array[offset:] = v.x.array[:(len(x.array_r) - offset)]
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)


        if comm.rank == 0:
            print(f"\n{t = :.3f}", end="\t")

        n = 0
        res0 = 1.0
        while n < max_iter:


            with b.localForm() as b_loc:
                b_loc.set(0)

            dfpetsc.assemble_vector_block(b, residual_comp, jacobian_comp, bcs=bcs, alpha=-1.0, x0=x)
            b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

            res = b.norm()
            if n == 0:
                res0 = res
                if comm.rank == 0:
                    print(f"{res0 = :.3e}")

            if comm.rank == 0:
                print(f"{n = :2d}:\t\t{res  = :.3e}")

            if res < atol or res < rtol * res0:
                break

            A.zeroEntries()
            dfpetsc.assemble_matrix_block(A, jacobian_comp, bcs=bcs)
            A.assemble()


            ksp.solve(b, delta_x)

            x.axpy(-1.0, delta_x)

            u.x.array[:offset] = x.array_r[:offset]
            v.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
            u.x.scatter_forward()
            v.x.scatter_forward()

            n += 1

        if comm.rank == 0:
            sys.stdout.flush()

        if n == max_iter:
            writer.close()
            raise RuntimeError("Nonlinear solver did not converge")
        
        writer.write(t)


    A.destroy()
    b.destroy()
    x.destroy()
    delta_x.destroy()
    writer.close()


    return


if __name__ == "__main__":
    main()
