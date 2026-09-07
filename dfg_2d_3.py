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
from os import PathLike
from pathlib import Path

from matplotlib import pyplot as plt

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

PHYSICAL_MARKERS = {
    "solid": 1,
    "ALE_fluid": 2,

    "solid_fluid_interface": 11,

    "obstacle": 21,                 # no-slip for fluid
    "inflow": 22,                   # parabolic inflow for fluid
    "outflow": 23,                  # do-nothing for fluid
    "channel_side": 24,             # no-slip for fluid
}

def solve(mesh_path, T, dt_val, output_path, output_path_p, drag_path, lift_path, drag_plot_path, lift_plot_path, drag_coeff_plot_path, lift_coeff_plot_path):

    log = PETSc.Log()
    log.begin()

    # load mesh and meshtags

    # other mesh options:
    #   "data/meshes/dfg2d/mesh.xdmf"
    #   "data/meshes/dfg2d_alt/mesh_tri.xdmf"

    with dfx.io.XDMFFile(comm, mesh_path, "r") as infile:
        fluid_mesh = infile.read_mesh()
        fluid_mesh.topology.create_connectivity(1, 2)
        fluid_facet_tags = infile.read_meshtags(fluid_mesh, name= "Facet tags")

    
    if comm.rank == 0:
        print(f"{fluid_mesh.geometry.x.shape = }")
        print(f"{fluid_facet_tags.indices.shape = }, {np.unique(fluid_facet_tags.values) = }")


    # Create measure with  meshtags

    dx = ufl.Measure("dx", domain=fluid_mesh)
    ds = ufl.Measure("ds", domain=fluid_mesh, subdomain_data=fluid_facet_tags)


    # create problem parameters

    rho_f = dfx.fem.Constant(fluid_mesh, 1.0)
    mu_val = 1e-3
    nu_f = dfx.fem.Constant(fluid_mesh, mu_val / rho_f.value)

    U_bar = 1.0
    H = 0.41

    t0 = 0.0
    dt = dfx.fem.Constant(fluid_mesh, dt_val)

    theta = dfx.fem.Constant(fluid_mesh, 0.5)

    save_every = 10

    
    # create function spaces

    V = dfx.fem.functionspace(fluid_mesh, ("CG", 2, (2, )))
    P = dfx.fem.functionspace(fluid_mesh, ("CG", 1))
    W = ufl.MixedFunctionSpace(V, P)


    # create functions

    v, p = dfx.fem.Function(V, name="v"), dfx.fem.Function(P, name="p")
    v_old = dfx.fem.Function(V)

    
    # Crank-Nicolson discretization of Navier-Stokes, with pressure treated fully implicitly
    # Parabolic inflow on left side, no-slip on top, bottom, obstacle, and flag, do-nothing on right side
    
    dv_dt = (v - v_old) / dt

    v_theta = theta * v + (1.0 - theta) * v_old
    
    dv, dp = ufl.TestFunctions(W)


    # Eulerian formulation of transient Navier-Stokes
    # Parabolic inflow on left side, no-slip on top, bottom, obstacle, and flag, do-nothing on right side
    
    from xfsi_solver.fsi.materials import Fluid

    n = ufl.FacetNormal(fluid_mesh)
    

    # create Dirichlet boundary condition

    from functools import reduce

    bc_func = dfx.fem.Function(V)
    bc_func.x.array[:] = 0.0
    bc_facets = reduce(np.union1d, [
        fluid_facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
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
            values[0] *= np.sin(self.t * np.pi / 8)
            return values
    bc_func.interpolate(BCFunc(t0))

    bc = dfx.fem.dirichletbc(bc_func, bc_dofs)

    bcs = [bc]


    # create residual form

    residual  = rho_f * ufl.inner(dv_dt, dv) * dx

    residual += theta * rho_f * ufl.inner(ufl.dot(v, ufl.nabla_grad(v)), dv) * dx
    residual += (1.0 - theta) * rho_f * ufl.inner(ufl.dot(v_old, ufl.nabla_grad(v_old)), dv) * dx

    residual += theta * ufl.inner(Fluid.NS_velocity_eulerian(v, nu_f, rho_f), ufl.grad(dv)) * dx
    residual += (1.0 - theta) * ufl.inner(Fluid.NS_velocity_eulerian(v_old, nu_f, rho_f), ufl.grad(dv)) * dx

    residual += ufl.inner(Fluid.NS_pressure(p), ufl.grad(dv)) * dx

    residual += ufl.div(v) * dp * dx

    # Do-nothing condition
    # Effect on measured drag and lift is negligible compared to the 
    # do-nothing condition where the term is dropped.
    residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])


    residual_blocked = ufl.extract_blocks(residual)

    max_iter = 20
    atol = 1.0e-8
    rtol = 1.0e-8

    problem = dfpetsc.NonlinearProblem(
        residual_blocked, [v, p], bcs=bcs,
        petsc_options_prefix="dfg_2d_3_",
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
    writer_p = dfx.io.VTXWriter(comm, output_path_p, [p])

    from timeit import default_timer as timer


    # Set up computing drag and lift

    class Drag:
        def __init__(self, save_to: PathLike | list, mesh: dfx.mesh.Mesh, tags: int | tuple[int]):

            self.save_to = save_to
            self.comm = mesh.comm

            self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=fluid_facet_tags)(tags)
            n = -ufl.FacetNormal(mesh)

            form = ufl.inner(Fluid.NS_eulerian(v, p, nu_f, rho_f) * n, ufl.as_vector([-1.0, 0.0])) * self.ds
            self.form = dfx.fem.form(form)

            if not isinstance(self.save_to, list):
                Path(self.save_to).parent.mkdir(parents=True, exist_ok=True)
                with open(self.save_to, "wb") as f:
                    np.savetxt(f, [], header="time drag", fmt='%.4e', delimiter=' ')

            return
        
        def __call__(self, t: float):
            
            drag = self.comm.reduce(dfx.fem.assemble_scalar(self.form), op=MPI.SUM, root=0)
            if comm.rank == 0:
                if isinstance(self.save_to, list):
                    self.save_to.append([t, drag])
                else:
                    with open(self.save_to, "ab") as f:
                        np.savetxt(f, [[t, drag]], fmt='%.4e', delimiter=' ')

            return
        
    class Lift:
        def __init__(self, save_to: PathLike | list, mesh: dfx.mesh.Mesh, tags: int | tuple[int]):

            self.save_to = save_to
            self.comm = mesh.comm

            self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=fluid_facet_tags)(tags)
            n = -ufl.FacetNormal(mesh)

            form = ufl.inner(Fluid.NS_eulerian(v, p, nu_f, rho_f) * n, ufl.as_vector([0.0, 1.0])) * self.ds
            self.form = dfx.fem.form(form)

            if comm.rank == 0 and not isinstance(self.save_to, list):
                Path(self.save_to).parent.mkdir(parents=True, exist_ok=True)
                with open(self.save_to, "wb") as f:
                    np.savetxt(f, [], header="time lift", fmt='%.4e', delimiter=' ')

            return
        
        def __call__(self, t: float):
            
            lift = self.comm.reduce(dfx.fem.assemble_scalar(self.form), op=MPI.SUM, root=0)
            if comm.rank == 0:
                if isinstance(self.save_to, list):
                    self.save_to.append([t, lift])
                else:
                    with open(self.save_to, "ab") as f:
                        np.savetxt(f, [[t, lift]], fmt='%.4e', delimiter=' ')

            return
        
    drag_hook = Drag(drag_path, fluid_mesh, PHYSICAL_MARKERS["obstacle"])
    lift_hook = Lift(lift_path, fluid_mesh, PHYSICAL_MARKERS["obstacle"])
    # drag_hook = Drag([], fluid_mesh, PHYSICAL_MARKERS["obstacle"])
    # lift_hook = Lift([], fluid_mesh, PHYSICAL_MARKERS["obstacle"])

    hooks = [drag_hook, lift_hook]

    start = timer()

    i = 0
    t = t0
    while t < T:

        i += 1
        t += dt.value
        bc_func.interpolate(BCFunc(t))

        v_old.x.array[:] = v.x.array
        v_old.x.scatter_forward()

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

        for hook in hooks:
            hook(t)

        if i % save_every == 0:
            writer.write(t)
            writer_p.write(t)

    end = timer()
    if comm.rank == 0:
        print(f"\n# processes: {comm.size}")
        print(f"Elapsed time: {end - start:.2f} s")
        print(f"Time per step: {(end - start) / i:.3e} s")

    writer.close()


    # dfx.common.list_timings(comm, [dfx.common.TimingType.wall])
    # log.view()

    if comm.rank == 0:
        if isinstance(drag_hook.save_to, list):
            drag_arr = np.array(drag_hook.save_to)
        else:
            drag_arr = np.loadtxt(drag_hook.save_to)
        if isinstance(lift_hook.save_to, list):
            lift_arr = np.array(lift_hook.save_to)
        else:
            lift_arr = np.loadtxt(lift_hook.save_to)
        
        Path(drag_plot_path).parent.mkdir(parents=True, exist_ok=True)
        Path(lift_plot_path).parent.mkdir(parents=True, exist_ok=True)
        Path(drag_coeff_plot_path).parent.mkdir(parents=True, exist_ok=True)
        Path(lift_coeff_plot_path).parent.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(drag_arr[:, 0], drag_arr[:, 1], 'k-')
        plt.xlabel("Time")
        plt.ylabel("Drag")
        plt.savefig(drag_plot_path)

        plt.figure()
        plt.plot(lift_arr[:, 0], lift_arr[:, 1], 'k-')
        plt.xlabel("Time")
        plt.ylabel("Lift")
        plt.savefig(lift_plot_path)

        
        # Compare with values at https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html

        drag_coeff = -2 / 0.1 * drag_arr[:, 1]
        lift_coeff = 2 / 0.1 * lift_arr[:, 1]

        plt.figure(figsize=(25,8))
        plt.plot(drag_arr[:, 0], drag_coeff, 'k-', label="drag coefficient")
        plt.grid()
        plt.legend()
        plt.savefig(drag_coeff_plot_path)

        plt.figure(figsize=(25,8))
        plt.plot(lift_arr[:, 0], lift_coeff, 'k-', label="lift coefficient")
        plt.grid()
        plt.legend()
        plt.savefig(lift_coeff_plot_path)



    return


def main():
    solve(
        mesh_path="data/meshes/dfg2d_alt/mesh_quad.xdmf",
        T=8.0,
        dt_val=1 / 400,
        output_path="output/pv/dfg_2d_3.bp",
        output_path_p="output/pv/dfg_2d_3_p.bp",
        drag_path="output/qoi/dfg_2d_3_drag.txt",
        lift_path="output/qoi/dfg_2d_3_lift.txt",
        drag_plot_path="output/figures/dfg_2d_3_drag.png",
        lift_plot_path="output/figures/dfg_2d_3_lift.png",
        drag_coeff_plot_path="output/figures/dfg_2d_3_drag_coeff.png",
        lift_coeff_plot_path="output/figures/dfg_2d_3_lift_coeff.png",
    )


if __name__ == "__main__":
    main()
