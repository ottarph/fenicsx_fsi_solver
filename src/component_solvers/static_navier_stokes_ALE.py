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

def solve(mesh_path, output_path, t):


    # load mesh and meshtags

    with dfx.io.XDMFFile(comm, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name= "Cell tags")
        mesh.topology.create_connectivity(1, 2)
        facet_tags = infile.read_meshtags(mesh, name= "Facet tags")

    assert len(np.setdiff1d(np.union1d(cell_tags.values, facet_tags.values), [PHYSICAL_MARKERS[i] for i in PHYSICAL_MARKERS])) == 0, "Physical markers and cell tags do not match"
    

    # create submeshes for fluid and solid

    fluid_mesh, fluid_cell_map, fluid_vertex_map, _ = dfx.mesh.create_submesh(mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"]))

    if comm.rank == 0:
        print(f"{fluid_mesh.geometry.x.shape = }")
        print(f"{fluid_cell_map.shape = }")

    fluid_mesh.topology.create_connectivity(1, 2)


    # transfer meshtags to submeshes

    import scifem
    fluid_facet_tags, fluid_facet_map = scifem.transfer_meshtags_to_submesh(facet_tags, fluid_mesh, fluid_vertex_map, fluid_cell_map)

    if comm.rank == 0:
        print(f"{fluid_facet_tags.indices.shape = }, {np.unique(fluid_facet_tags.values) = }")


    # Create measure with  meshtags

    dx = ufl.Measure("dx", domain=fluid_mesh)
    ds = ufl.Measure("ds", domain=fluid_mesh, subdomain_data=fluid_facet_tags)


    # create problem parameters

    rho_f = dfx.fem.Constant(fluid_mesh, 1.0e3)
    nu_f = dfx.fem.Constant(mesh, 4e-3)

    U_bar = 1.0
    H = 0.41


    # create function spaces

    V = dfx.fem.functionspace(fluid_mesh, ("CG", 2, (2, )))
    P = dfx.fem.functionspace(fluid_mesh, ("CG", 1))
    W = ufl.MixedFunctionSpace(V, P)


    # create functions

    v, p = dfx.fem.Function(V, name="v"), dfx.fem.Function(P, name="p")
    
    dv, dp = ufl.TestFunctions(W)
    delta_v, delta_p = ufl.TrialFunctions(W)

    # Assume static mesh for now
    u = dfx.fem.Function(V, name="u")
    
    def u_func(x):
        values = np.zeros_like(x[:2,:])
        values[0] = -1 * x[1]**2 * x[0] * (2.5 - x[0])**2 / 2.5**3
        values[1] = 0.05 * x[0] * (2.5 - x[0])**2
        return values
    u.interpolate(u_func)

    # ALE formulation of static Navier-Stokes
    # Parabolic inflow on left side, no-slip on top, bottom, obstacle, and flag, do-nothing on right side
    
    from fsi.materials import Fluid

    F = ufl.Identity(fluid_mesh.geometry.dim) + ufl.grad(u)
    J = ufl.det(F)
    n = ufl.FacetNormal(fluid_mesh)
    

    # create Dirichlet boundary condition

    from functools import reduce

    bc_func = dfx.fem.Function(V)
    bc_func.x.array[:] = 0.0
    bc_facets = reduce(np.union1d, [
        fluid_facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
        fluid_facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"]),
        fluid_facet_tags.find(PHYSICAL_MARKERS["inflow"]),
        fluid_facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
        ])
    bc_dofs = dfx.fem.locate_dofs_topological(V, fluid_mesh.geometry.dim - 1, bc_facets)

    class BCFunc:
        def __init__(self, t: float = 0.0):
            self.t = t
        def __call__(self, x: np.ndarray) -> np.ndarray:
            values = np.zeros((2, x.shape[1]), dtype=x.dtype)
            values[0] = np.where(np.isclose(x[0], 0.0), 1.5 * U_bar * 4 * x[1] * (H - x[1]) / H**2, 0.0)
            values[0] *= 0.5 * (1.0 - np.cos(0.5*np.pi * self.t))
            return values
    bc_func.interpolate(BCFunc(t))

    bc = dfx.fem.dirichletbc(bc_func, bc_dofs)

    bcs = [bc]


    # create residual form

    residual = J * rho_f * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx
    residual += J * ufl.inner(Fluid.NS(u, v, p, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx

    residual += ufl.div(J * ufl.inv(F) * v) * dp * dx

    # Do-nothing condition
    residual -= rho_f * nu_f * ufl.inner(ufl.dot(ufl.inv(F).T * ufl.grad(v).T, ufl.inv(F).T * n), dv) * ds(PHYSICAL_MARKERS["outflow"])


    residual_blocked = ufl.extract_blocks(residual)
    residual_comp = dfx.fem.form(residual_blocked)

    
    # create Jacobian form

    jacobian = ufl.derivative(residual, v, delta_v) + ufl.derivative(residual, p, delta_p)
    jacobian_blocked = ufl.extract_blocks(jacobian)
    jacobian_comp = dfx.fem.form(jacobian_blocked)


    # create matrix and vector for linear algebra

    A = dfpetsc.create_matrix_block(jacobian_comp)
    b = dfpetsc.create_vector_block(residual_comp)
    x = dfpetsc.create_vector_block(residual_comp)
    delta_x = dfpetsc.create_vector_block(residual_comp)

    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs


    ksp = PETSc.KSP().create(fluid_mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    max_iter = 20
    atol = 1.0e-8
    rtol = 1.0e-8


    writer = dfx.io.VTXWriter(comm, output_path, [v, u])

    

    x.array[:offset] = v.x.array[:offset]
    x.array[offset:] = p.x.array[:(len(x.array_r) - offset)]
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

        v.x.array[:offset] = x.array_r[:offset]
        p.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
        v.x.scatter_forward()
        p.x.scatter_forward()

        n += 1

    if n == max_iter:
        writer.close()
        raise RuntimeError("Nonlinear solver did not converge")
    
    writer.write(0.0)


    A.destroy()
    b.destroy()
    x.destroy()
    delta_x.destroy()
    writer.close()


    return


def main():
    solve(
        mesh_path="data/meshes/fsi2/mesh.xdmf",
        output_path="output/pv/static_navier_stokes_ale.bp",
        t=2.0,
    )


if __name__ == "__main__":
    main()
