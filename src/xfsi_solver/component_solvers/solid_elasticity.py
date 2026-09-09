# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import ufl

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

def solve(mesh_path, T, dt_val, output_path):


    # load mesh and meshtags

    with dfx.io.XDMFFile(comm, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name= "Cell tags")
        mesh.topology.create_connectivity(1, 2)
        facet_tags = infile.read_meshtags(mesh, name= "Facet tags")

    assert len(np.setdiff1d(np.union1d(cell_tags.values, facet_tags.values), [PHYSICAL_MARKERS[i] for i in PHYSICAL_MARKERS])) == 0, "Physical markers and cell tags do not match"
    

    # create submeshes for fluid and solid

    fluid_mesh, fluid_cell_map, fluid_vertex_map, _ = dfx.mesh.create_submesh(mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"]))
    solid_mesh, solid_cell_map, solid_vertex_map, _ = dfx.mesh.create_submesh(mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"]))

    if comm.rank == 0:
        print(f"{solid_mesh.geometry.x.shape = }")

    solid_mesh.topology.create_connectivity(1, 2)


    # transfer meshtags to submeshes
    fluid_facet_tags = dfx.mesh.transfer_meshtags_to_submesh(facet_tags, fluid_mesh, fluid_vertex_map, fluid_cell_map)
    solid_facet_tags = dfx.mesh.transfer_meshtags_to_submesh(facet_tags, solid_mesh, solid_vertex_map, solid_cell_map)

    if comm.rank == 0:
        print(f"{solid_facet_tags.indices.shape = }, {np.unique(solid_facet_tags.values) = }")

    assert np.all(np.union1d(fluid_facet_tags.values, solid_facet_tags.values) == np.unique(facet_tags.values)), "Transferred facet tags do not match"


    # Create measure with  meshtags

    dx = ufl.Measure("dx", domain=solid_mesh)
    ds = ufl.Measure("ds", domain=solid_mesh, subdomain_data=solid_facet_tags)


    # create problem parameters

    rho_s = dfx.fem.Constant(solid_mesh, 0.8e3)
    lambda_s = dfx.fem.Constant(solid_mesh, 1e5)
    mu_s = dfx.fem.Constant(solid_mesh, 2e7)

    dt = dfx.fem.Constant(solid_mesh, dt_val)
    t0 = 0.0

    g = dfx.fem.Constant(solid_mesh, (0.0, -9.81*4))
    traction = dfx.fem.Constant(solid_mesh, (0.0, 0.0))

    
    # create function spaces

    U = dfx.fem.functionspace(solid_mesh, ("CG", 2, (2, )))
    V = dfx.fem.functionspace(solid_mesh, ("CG", 2, (2, )))
    W = ufl.MixedFunctionSpace(U, V)


    # create functions

    u, v = dfx.fem.Function(U), dfx.fem.Function(V)
    u_old, v_old = dfx.fem.Function(U), dfx.fem.Function(V)

    du, dv = ufl.TestFunctions(W)


    # Implicit Euler discretization of STVK under influence of gravitational body force and
    # fixed Dirichlet BC on left boundary and otherwise zero traction.
    
    du_dt = (u - u_old) / dt
    dv_dt = (v - v_old) / dt

    from xfsi_solver.fsi.materials import Solid

    F = ufl.Identity(solid_mesh.geometry.dim) + ufl.grad(u)
    J = ufl.det(F)
    n = ufl.FacetNormal(solid_mesh)
    

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
    residual -= ufl.inner(traction, dv) * ds(PHYSICAL_MARKERS["solid_fluid_interface"])

    residual_blocked = ufl.extract_blocks(residual)

    max_iter = 20
    atol = 1.0e-8
    rtol = 1.0e-8

    problem = dfx.fem.petsc.NonlinearProblem(
        residual_blocked, [u, v], bcs=bcs,
        petsc_options_prefix="solid_elasticity_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_linesearch_type": "none",
            "snes_max_it": max_iter,
            "snes_atol": atol,
            "snes_rtol": rtol,
            "snes_error_if_not_converged": True,
            "ksp_error_if_not_converged": True,
        },
    )


    writer = dfx.io.VTXWriter(comm, output_path, [u])

    t = t0

    while t < T:

        t += dt.value

        u_old.x.array[:] = u.x.array
        v_old.x.array[:] = v.x.array
        u_old.x.scatter_forward()
        v_old.x.scatter_forward()

        if comm.rank == 0:
            print(f"\n{t = :.3f}")

        try:
            problem.solve()
        except Exception:
            writer.close()
            raise

        if comm.rank == 0:
            sys.stdout.flush()

        writer.write(t)


    writer.close()


    return


def main():
    solve(
        mesh_path="data/meshes/fsi2/mesh.xdmf",
        T=0.2,
        dt_val=0.0025,
        output_path="output/pv/solid_elasticity.bp",
    )


if __name__ == "__main__":
    main()
