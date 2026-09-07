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
from pathlib import Path
from timeit import default_timer as timer

from mpi4py.MPI import COMM_WORLD as comm
from mpi4py import MPI

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

def solve(mesh_path, T, dt_val, output_path, output_path_p, qoi_path):


    # load mesh and meshtags

    # mesh_path alternatives:
    #   "data/meshes/fsi2/mesh_quad.xdmf"
    #   "data/meshes/fsi2/mesh_quad_fine_sec.xdmf"
    #   "data/meshes/fsi2/mesh_sec.xdmf"

    with dfx.io.XDMFFile(comm, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name= "Cell tags")
        mesh.topology.create_connectivity(1, 2)
        facet_tags = infile.read_meshtags(mesh, name= "Facet tags")

    assert len(np.setdiff1d(np.union1d(cell_tags.values, facet_tags.values), [PHYSICAL_MARKERS[i] for i in PHYSICAL_MARKERS])) == 0, "Physical markers and cell tags do not match"
    

    # create submesh for fluid

    fluid_mesh, fluid_cell_map, fluid_vertex_map, _ = dfx.mesh.create_submesh(mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"]))


    # create entity maps for mixed mesh integration

    entity_maps = [fluid_cell_map]

    # Create measure with  meshtags

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    dx_fluid = dx(PHYSICAL_MARKERS["ALE_fluid"])
    dx_solid = dx(PHYSICAL_MARKERS["solid"])

    fluid_volume = comm.reduce(dfx.fem.assemble_scalar(dfx.fem.form(ufl.as_ufl(1.0) * dx_fluid)))
    solid_volume = comm.reduce(dfx.fem.assemble_scalar(dfx.fem.form(ufl.as_ufl(1.0) * dx_solid)))
    if comm.rank == 0:
        print(f"{fluid_volume = }")
        print(f"{solid_volume = }")


    # Create measure for interface / solid-fluid boundary

    from xfsi_solver.tools.interior_facet_measure import create_consistent_interior_facet_measure

    new_tag_fluid = 101
    new_measure_fluid = create_consistent_interior_facet_measure(mesh, facet_tags, cell_tags,
                        PHYSICAL_MARKERS["solid_fluid_interface"], PHYSICAL_MARKERS["ALE_fluid"], new_tag_fluid)
    ds_interface_fluid = new_measure_fluid(new_tag_fluid)


    # create problem parameters

    rho_f = dfx.fem.Constant(mesh, 1.0e3)
    nu_f = dfx.fem.Constant(mesh, 1.0e-3)

    rho_s = dfx.fem.Constant(mesh, 1.0e4)
    mu_s = dfx.fem.Constant(mesh, 5.0e5)
    nu_s = dfx.fem.Constant(mesh, 0.4)
    lambda_s = dfx.fem.Constant(mesh, -mu_s.value / (1 - 0.5 / nu_s.value))
    # lambda_s = dfx.fem.Constant(mesh, 2e6)

    U_bar = 1.0
    H = 0.41

    t0 = 0.0
    dt = dfx.fem.Constant(mesh, dt_val)

    theta = dfx.fem.Constant(mesh, 0.5 + dt.value)

    save_every = 4

    
    # create function spaces

    U = dfx.fem.functionspace(mesh, ("CG", 2, (2, )))
    V = dfx.fem.functionspace(mesh, ("CG", 2, (2, )))
    P = dfx.fem.functionspace(fluid_mesh, ("DG", 1))
    W = ufl.MixedFunctionSpace(U, V, P)


    # create functions

    u, v, p = dfx.fem.Function(U, name="u"), dfx.fem.Function(V, name="v"), dfx.fem.Function(P, name="p")
    u_old, v_old = dfx.fem.Function(U), dfx.fem.Function(V)

    
    du, dv, dp = ufl.TestFunctions(W)


    # create Dirichlet boundary condition

    from functools import reduce

    inflow_bc_func = dfx.fem.Function(V)
    inflow_bc_func.x.array[:] = 0.0
    inflow_bc_facets = reduce(np.union1d, [
        facet_tags.find(PHYSICAL_MARKERS["inflow"]),
        ])
    inflow_bc_dofs = dfx.fem.locate_dofs_topological(V, mesh.geometry.dim - 1, inflow_bc_facets)

    inflow_bc = dfx.fem.dirichletbc(inflow_bc_func, inflow_bc_dofs)

    class InflowFunc:
        def __init__(self, t: float = 0.0):
            self.t = t
        def __call__(self, x: np.ndarray) -> np.ndarray:
            values = np.zeros((2, x.shape[1]), dtype=x.dtype)
            values[0] = 1.5 * U_bar * 4 * x[1] * (H - x[1]) / H**2
            if self.t < 2.0:
                values[0] *= 0.5 * (1.0 - np.cos(0.5*np.pi * self.t))
            return values

    noslip_bc_func = dfx.fem.Function(V)
    noslip_bc_func.x.array[:] = 0.0
    noslip_bc_facets = reduce(np.union1d, [
        facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
        facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
        facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
        ])
    noslip_bc_dofs = dfx.fem.locate_dofs_topological(V, mesh.geometry.dim - 1, noslip_bc_facets)
    
    noslip_bc = dfx.fem.dirichletbc(noslip_bc_func, noslip_bc_dofs)

    
    # Create ALE Dirichlet boundary condition
    
    u_bc_func = dfx.fem.Function(U)
    u_bc_func.x.array[:] = 0.0
    u_bc_facets = reduce(np.union1d, [
        facet_tags.find(PHYSICAL_MARKERS["inflow"]),
        facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
        facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
        facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
        facet_tags.find(PHYSICAL_MARKERS["outflow"]),
    ])
    u_bc_dofs = dfx.fem.locate_dofs_topological(U, mesh.geometry.dim - 1, u_bc_facets)
    u_bc = dfx.fem.dirichletbc(u_bc_func, u_bc_dofs)


    # Collect Dirichlet boundary conditions

    bcs = [u_bc, inflow_bc, noslip_bc]


    # DESCRIBE FSI PROBLEM
    # FLUID: Parabolic inflow on left side, no-slip on top, bottom, and obstacle, do-nothing on right side
    # SOLID: Homogeneous Dirichlet on left side
    
    from xfsi_solver.fsi.materials import Fluid, Solid

    # create residual form

    def A_T(u, u_old, v, v_old):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)
        F_old = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_old)
        J_old = ufl.det(F_old)
        J_mid = 0.5 * (J + J_old)

        residual  = rho_f * J_mid * ufl.inner((v - v_old) / dt, dv) * dx_fluid

        residual -= rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * ((u - u_old) / dt), dv) * dx_fluid

        residual += rho_s * ufl.inner((v - v_old) / dt, dv) * dx_solid

        residual += rho_s * ufl.inner((u - u_old) / dt, du) * dx_solid

        return residual
    
    def A_I(u, v, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)
        normal = ufl.FacetNormal(mesh)

        alpha_u0 = dfx.fem.Constant(mesh, 1e-9)
        alpha_u = alpha_u0

        residual  = ufl.inner(alpha_u * ufl.grad(u), ufl.grad(du)) * dx_fluid
        residual -= ufl.inner(alpha_u * ufl.grad(u) * normal, du) * ds_interface_fluid
        
        residual += ufl.div(J * ufl.inv(F) * v) * dp * dx_fluid

        return residual

    def A_E(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual  = rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx_fluid

        residual += ufl.inner(J * Fluid.NS_velocity(u, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        residual += ufl.inner(J * Solid.STVK(u, lambda_s, mu_s) * ufl.inv(F).T, ufl.grad(dv)) * dx_solid

        residual -= rho_s * ufl.inner(v, du) * dx_solid

        return residual
    
    def A_P(u, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual  = J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    residual  = A_T(u, u_old, v, v_old)
    residual += A_I(u, v, p)
    residual += A_P(u, p)
    residual += theta * A_E(u, v)
    residual += (1.0 - theta) * A_E(u_old, v_old)


    #--------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])


    residual_blocked = ufl.extract_blocks(residual)

    max_iter = 20
    atol = 1.0e-7
    rtol = 1.0e-12

    problem = dfx.fem.petsc.NonlinearProblem(
        residual_blocked, [u, v, p], bcs=bcs,
        petsc_options_prefix="fsi2_harmonic_diffmesh_",
        entity_maps=entity_maps,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 80,
            "snes_linesearch_type": "none",
            "snes_max_it": max_iter,
            "snes_atol": atol,
            "snes_rtol": rtol,
            "snes_error_if_not_converged": True,
            "ksp_error_if_not_converged": True,
        },
    )


    policy = dfx.io.VTXMeshPolicy.reuse
    writer = dfx.io.VTXWriter(comm, output_path, [u,v], mesh_policy=policy)
    writer_p = dfx.io.VTXWriter(comm, output_path_p, [p], mesh_policy=policy)

    dm_loc_size = U.dofmap.index_map.size_local
    spot = np.array([0.6, 0.2, 0.0], dtype=np.float64)
    spot_dof_cand = np.flatnonzero(np.all(np.isclose(U.tabulate_dof_coordinates()[:dm_loc_size,:], spot, atol=1e-6), axis=1))
    spot_dof = spot_dof_cand[0] if len(spot_dof_cand) > 0 else None
    assert comm.allreduce(len(spot_dof_cand), op=MPI.SUM) == 1, "None or multiple dofs found for measurement point"

    loc_u_spot = np.zeros(2, dtype=np.float64)

    normal = ufl.FacetNormal(mesh)
    e_x = dfx.fem.Constant(mesh, (-1.0, 0.0))
    e_y = dfx.fem.Constant(mesh, (0.0, 1.0))
    F = ufl.Identity(2) + ufl.grad(u)
    transformed_normal = ufl.dot(ufl.inv(F.T), normal)

    drag_form_obstacle = ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_x) * ufl.det(F) * ds(PHYSICAL_MARKERS["obstacle"])
    lift_form_obstacle = ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_y) * ufl.det(F) * ds(PHYSICAL_MARKERS["obstacle"])

    drag_form_interface = ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_x) * ufl.det(F) * ds_interface_fluid
    lift_form_interface = ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_y) * ufl.det(F) * ds_interface_fluid

    drag_form_obstacle = dfx.fem.form(drag_form_obstacle, entity_maps=entity_maps)
    drag_form_interface = dfx.fem.form(drag_form_interface, entity_maps=entity_maps)
    lift_form_obstacle = dfx.fem.form(lift_form_obstacle, entity_maps=entity_maps)
    lift_form_interface = dfx.fem.form(lift_form_interface, entity_maps=entity_maps)


    if comm.rank == 0:
        Path(qoi_path).parent.mkdir(parents=True, exist_ok=True)
        with open(qoi_path, "wb") as f:
            np.savetxt(f, [], fmt="%.6e", delimiter="\t", header="t\tdrag\tlift\tA_x\tA_y")


    t = t0
    step = -1
    max_steps = np.inf
    start = timer()
    while step < max_steps and t < T:

        inflow_bc_func.interpolate(InflowFunc(t))
        inflow_bc_func.x.scatter_forward()

        u_old.x.array[:] = u.x.array[:]
        v_old.x.array[:] = v.x.array[:]

        if comm.rank == 0:
            print(f"\n{t = :.3f}")

        try:
            problem.solve()
        except Exception:
            writer.close()
            writer_p.close()
            raise

        if comm.rank == 0:
            sys.stdout.flush()
        
        if step % save_every == 0:
            writer.write(t)
            writer_p.write(t)

        loc_u_spot[:] = u.x.array[2*spot_dof:2*(spot_dof+1)] if spot_dof is not None else 0.0
        u_spot = comm.reduce(loc_u_spot, op=MPI.SUM, root=0)
        drag = comm.reduce(dfx.fem.assemble_scalar(drag_form_obstacle) + dfx.fem.assemble_scalar(drag_form_interface))
        lift = comm.reduce(dfx.fem.assemble_scalar(lift_form_obstacle) + dfx.fem.assemble_scalar(lift_form_interface))
        if comm.rank == 0:
            with open(qoi_path, "ab") as f:
                np.savetxt(f, [[t, drag, lift, *u_spot]], fmt="%.6e", delimiter="\t")


        step += 1
        t += dt.value

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / (step+1):.3f} s")


    writer.close()


    return


def main():
    solve(
        mesh_path="data/meshes/fsi2/mesh_quad_fine_sec.xdmf",
        T=12.0,
        dt_val=0.0025,
        output_path="output/pv/fsi2_harm_dm.bp",
        output_path_p="output/pv/fsi2_harm_p_dm.bp",
        qoi_path="output/qoi/fsi2_harm_dm_qoi.txt",
    )


if __name__ == "__main__":
    main()
