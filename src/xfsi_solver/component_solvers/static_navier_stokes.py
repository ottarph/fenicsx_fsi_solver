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


    # Eulerian formulation of static Navier-Stokes
    # Parabolic inflow on left side, no-slip on top, bottom, obstacle, and flag, do-nothing on right side
    
    from xfsi_solver.fsi.materials import Fluid

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

    residual = rho_f * ufl.inner(ufl.dot(v, ufl.nabla_grad(v)), dv) * dx
    residual += ufl.inner(Fluid.NS_eulerian(v, p, nu_f, rho_f), ufl.grad(dv)) * dx

    residual += ufl.div(v) * dp * dx

    # Do-nothing condition
    residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])


    residual_blocked = ufl.extract_blocks(residual)

    max_iter = 20
    atol = 1.0e-8
    rtol = 1.0e-8

    problem = dfx.fem.petsc.NonlinearProblem(
        residual_blocked, [v, p], bcs=bcs,
        petsc_options_prefix="static_navier_stokes_",
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


    writer = dfx.io.VTXWriter(comm, output_path, [v])


    if comm.rank == 0:
        print(f"\n{t = :.3f}", end="\t")

    try:
        problem.solve()
    except Exception:
        writer.close()
        raise

    writer.write(0.0)


    writer.close()


    return


def main():
    solve(
        mesh_path="data/meshes/fsi2/mesh.xdmf",
        output_path="output/pv/static_navier_stokes.bp",
        t=2.0,
    )


if __name__ == "__main__":
    main()
