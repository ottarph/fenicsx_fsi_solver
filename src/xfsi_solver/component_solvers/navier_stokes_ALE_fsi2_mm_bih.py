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

def solve(mesh_path, dt_val, bd_dset_path, output_path, max_steps):

    assert comm.size == 1, "This example only works in serial"

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

    t0 = 0.0
    dt = dfx.fem.Constant(mesh, dt_val)

    theta = dfx.fem.Constant(mesh, 0.5)


    # create function spaces

    U = dfx.fem.functionspace(fluid_mesh, ("CG", 2, (2, )))
    Z = dfx.fem.functionspace(fluid_mesh, ("CG", 2, (2, )))
    V = dfx.fem.functionspace(fluid_mesh, ("CG", 2, (2, )))
    P = dfx.fem.functionspace(fluid_mesh, ("CG", 1))
    W = ufl.MixedFunctionSpace(U, Z, V, P)


    # create functions

    u, z, v, p = dfx.fem.Function(U, name="u"), dfx.fem.Function(Z, name="z"), dfx.fem.Function(V, name="v"), dfx.fem.Function(P, name="p")
    u_old, v_old = dfx.fem.Function(U), dfx.fem.Function(V)


    # Prepare boundary deformations for ale fields for all time steps

    msh_x = np.load(bd_dset_path + "msh_x.npy")
    msh_conn = np.load(bd_dset_path + "msh_conn.npy")
    uh_bd_fsi2 = np.load(bd_dset_path + "uh.npy")
    # uh_bd_fsi2 = uh_bd_fsi2[:20,:]


    c_el = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(msh_x.shape[1],)))
    bd_from_mesh = dfx.mesh.create_mesh(comm, msh_conn, msh_x, c_el)

    bd_to_mesh, *_ = dfx.mesh.create_submesh(fluid_mesh, 1, dfx.mesh.locate_entities_boundary(fluid_mesh, 1, lambda x: np.full(x.shape[1], True)))
    bd_to_cells = bd_to_mesh.topology.index_map(1)
    bd_cells_on_proc = bd_to_cells.size_local + bd_to_cells.num_ghosts
    bd_interp_cells = np.arange(bd_cells_on_proc, dtype=np.int32)

    V_from = dfx.fem.functionspace(bd_from_mesh, ("CG", 2, (2, )))
    V_to = dfx.fem.functionspace(bd_to_mesh, ("CG", 2, (2, )))

    bd_interp_data = dfx.fem.create_interpolation_data(V_to, V_from,
                            cells=bd_interp_cells, padding=1e-6)

    u_from = dfx.fem.Function(V_from, name="u_from")
    u_to = dfx.fem.Function(V_to, name="u_to")

    whole_cells = fluid_mesh.topology.index_map(2)
    whole_cells_on_proc = whole_cells.size_local + whole_cells.num_ghosts
    whole_interp_cells = np.arange(whole_cells_on_proc, dtype=np.int32)
    whole_interp_data = dfx.fem.create_interpolation_data(V, V_to, whole_interp_cells, padding=1e-8)

    u_bc = dfx.fem.Function(V, name="u_whole")

    from tqdm import tqdm
    u_bc_arr = np.zeros((uh_bd_fsi2.shape[0], u_bc.x.array.shape[0]), dtype=u_bc.x.array.dtype)
    for t in tqdm(range(uh_bd_fsi2.shape[0]), desc="Preparing boundary deformations..."):
        u_from.x.array[:] = uh_bd_fsi2[t,:]
        u_to.interpolate_nonmatching(u_from, bd_interp_cells, bd_interp_data)
        u_bc.interpolate_nonmatching(u_to, whole_interp_cells, whole_interp_data)
        u_bc_arr[t,:] = u_bc.x.array

    from xfsi_solver.component_solvers.biharm import biharmonic
    uh_pure, *_ = biharmonic(u_bc)
    u_old.interpolate(uh_pure)
    u_bc.x.array[:] = u_bc_arr[0,:]
    uh_pure, *_ = biharmonic(u_bc)
    u.interpolate(uh_pure)


    num_steps = u_bc_arr.shape[0]



    du_dt = (u - u_old) / dt
    dv_dt = (v - v_old) / dt

    u_theta = theta * u + (1.0 - theta) * u_old
    v_theta = theta * v + (1.0 - theta) * v_old

    du, dz, dv, dp = ufl.TestFunctions(W)


    # ALE formulation of transient Navier-Stokes
    # Parabolic inflow on left side, no-slip on top, bottom, obstacle, and flag, do-nothing on right side

    from xfsi_solver.fsi.materials import Fluid

    n = ufl.FacetNormal(fluid_mesh)

    F = ufl.Identity(fluid_mesh.geometry.dim) + ufl.grad(u)
    J = ufl.det(F)
    F_old = ufl.Identity(fluid_mesh.geometry.dim) + ufl.grad(u_old)
    J_old = ufl.det(F_old)
    J_mid = 0.5 * (ufl.det(F) + ufl.det(F_old))
    F_mid = 0.5 * (F + F_old)

    # create Dirichlet boundary condition

    from functools import reduce

    v_bc_func = dfx.fem.Function(V)
    v_bc_func.x.array[:] = 0.0
    v_bc_facets = reduce(np.union1d, [
        fluid_facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
        fluid_facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"]),
        fluid_facet_tags.find(PHYSICAL_MARKERS["inflow"]),
        fluid_facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
        ])
    v_bc_dofs = dfx.fem.locate_dofs_topological(V, fluid_mesh.geometry.dim - 1, v_bc_facets)

    class BCFunc:
        def __init__(self, t: float = 0.0):
            self.t = t
        def __call__(self, x: np.ndarray) -> np.ndarray:
            values = np.zeros((2, x.shape[1]), dtype=x.dtype)
            values[0] = np.where(np.isclose(x[0], 0.0), 1.5 * U_bar * 4 * x[1] * (H - x[1]) / H**2, 0.0)
            values[0] *= 0.5 * (1.0 - np.cos(2.0*np.pi * min(self.t, 0.5)))
            return values
    v_bc_func.interpolate(BCFunc(t0))

    v_bc = dfx.fem.dirichletbc(v_bc_func, v_bc_dofs)


    # Create ALE Dirichlet boundary condition

    u_bc_func = dfx.fem.Function(U)
    u_bc_func.x.array[:] = 0.0
    u_bc_facets = dfx.mesh.exterior_facet_indices(fluid_mesh.topology)
    u_bc_dofs = dfx.fem.locate_dofs_topological(U, fluid_mesh.geometry.dim - 1, u_bc_facets)
    u_bc = dfx.fem.dirichletbc(u_bc_func, u_bc_dofs)


    # Collect Dirichlet boundary conditions

    bcs = [u_bc, v_bc]

    # create residual form

    # Crank-Nicolson discretization of ALE_formulation Navier-Stokes,
    # with pressure treated fully implicitly and secant evaluation rule
    # for cross temporal-spatial differential terms.

    #--------------------------------------------

    # v time derivative term
    residual  = rho_f * J_mid * ufl.inner(dv_dt, dv) * dx

    # v-contribution convective term
    residual  += theta * rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx
    residual  += (1.0 - theta) * rho_f * J_old * ufl.inner(ufl.grad(v_old) * ufl.inv(F_old) * v_old, dv) * dx

    # du_dt-contribution convective term
    residual -= rho_f * J_mid * ufl.inner(ufl.grad(v_theta) * ufl.inv(F_mid) * ((u - u_old) / dt), dv) * dx

    # stress pressure-component term done implicitly
    residual += J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv)) * dx

    # stress velocity-component term
    residual += theta * J * ufl.inner(Fluid.NS_velocity(u, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx
    residual += (1.0 - theta) * J_old * ufl.inner(Fluid.NS_velocity(u_old, v_old, nu_f, rho_f) * ufl.inv(F_old).T, ufl.grad(dv)) * dx

    # incompressibility constraint done implicitly
    residual += ufl.div(J * ufl.inv(F) * v) * dp * dx


    # ale deformation
    # Use a biharmonic mesh motion

    residual += ufl.inner(ufl.grad(u), ufl.grad(dz)) * dx
    residual -= ufl.inner(z, dz) * dx
    residual += ufl.inner(ufl.grad(z), ufl.grad(du)) * dx
    residual += ufl.inner(dfx.fem.Constant(fluid_mesh, 0.0) * u, du) * dx

    #--------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])


    residual_blocked = ufl.extract_blocks(residual)

    max_iter = 20
    atol = 1.0e-7
    rtol = 1.0e-16

    problem = dfx.fem.petsc.NonlinearProblem(
        residual_blocked, [u, z, v, p], bcs=bcs,
        petsc_options_prefix="navier_stokes_ale_fsi2_mm_bih_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
            "snes_linesearch_type": "none",
            "snes_max_it": max_iter,
            "snes_atol": atol,
            "snes_rtol": rtol,
            "snes_error_if_not_converged": True,
            "ksp_error_if_not_converged": True,
        },
    )


    writer = dfx.io.VTXWriter(comm, output_path, [u,v])

    t = t0
    step = -1
    # max_steps = 4 * num_steps
    while step < max_steps:

        step += 1
        t += dt.value
        u_bc_func.x.array[:] = u_bc_arr[step % num_steps]
        v_bc_func.interpolate(BCFunc(t))

        u_old.x.array[:] = u.x.array[:]
        u_old.x.scatter_forward()

        v_old.x.array[:] = v.x.array
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
        dt_val=0.0025,
        bd_dset_path="data/fsi2_boundary/",
        output_path="output/pv/navier_stokes_ale_fsi2_mm_bih.bp",
        max_steps=100,
    )


if __name__ == "__main__":
    main()
