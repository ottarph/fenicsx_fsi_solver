# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import basix.ufl
import ufl

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

def solve(mesh_path, output_path):


    # load mesh and meshtags

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
    solid_facet_tags = dfx.mesh.transfer_meshtags_to_submesh(facet_tags, solid_mesh, solid_vertex_map, solid_cell_map)


    # Create measure with  meshtags

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)(PHYSICAL_MARKERS["solid"])

    
    # Create measure for interface / solid-fluid boundary

    import scifem

    new_tag = 100
    interface_facets = facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])
    idata = scifem.compute_interface_data(cell_tags, interface_facets)
    if idata.shape[0] > 0 and cell_tags.values[idata[0, 0]] == PHYSICAL_MARKERS["solid"]:
        integration_entities = idata[:, :2]
    else:
        integration_entities = idata[:, 2:]
    integration_entities = integration_entities.flatten()
    new_measure = ufl.Measure("ds", domain=mesh, subdomain_data=[(new_tag, integration_entities)])
    ds_interface = new_measure(new_tag)


    # create entity maps for mixed mesh integration

    entity_maps = [solid_cell_map]


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


    from xfsi_solver.fsi.materials import Solid

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

    max_iter = 20
    atol = 1.0e-8
    rtol = 1.0e-8

    problem = dfx.fem.petsc.NonlinearProblem(
        residual, u, bcs=bcs,
        petsc_options_prefix="static_solid_elasticity_fullmesh_",
        entity_maps=entity_maps,
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


    # vtx writer for output
    writer = dfx.io.VTXWriter(comm, output_path, [u])

    try:
        problem.solve()
    except Exception:
        writer.close()
        raise
    writer.write(0)

    u.x.array[:] = 0.0
    u.x.scatter_forward()

    g.value = (0.0, +9.81*4)

    try:
        problem.solve()
    except Exception:
        writer.close()
        raise
    writer.write(1)

    writer.close()


    return


def main():
    solve(
        mesh_path="data/meshes/fsi2/mesh.xdmf",
        output_path="output/pv/static_solid_elasticity_fm.bp",
    )


if __name__ == "__main__":
    main()
