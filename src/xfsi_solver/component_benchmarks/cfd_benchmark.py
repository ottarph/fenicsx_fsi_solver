# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import sys
import warnings
from timeit import default_timer as timer

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py.MPI import COMM_WORLD as comm
from petsc4py import PETSc

from xfsi_solver.component_benchmarks.benchmark_parameters import CFD_parameters
from xfsi_solver.tools.convergence import check_converged
from xfsi_solver.tools.qoi import (
    append_qoi_row,
    assemble_force,
    drag_lift_forms,
    init_qoi_file,
)

PHYSICAL_MARKERS = {
    "solid": 1,
    "ALE_fluid": 2,
    #
    "solid_fluid_interface": 11,
    #
    "obstacle": 21,  # no-slip for fluid
    "inflow": 22,  # parabolic inflow for fluid
    "outflow": 23,  # do-nothing for fluid
    "channel_side": 24,  # no-slip for fluid
    "solid_obstacle_interface": 25,  # homogeneous Dirichlet BC for solid
}


def solve(
    mesh_path,
    T,
    dt_val,
    cfd_parameters,
    solver_prefix,
    output_path,
    output_path_p,
    qoi_path,
    checkpoint_dir=None,
    checkpoint_every=None,
    vtx_save_every=None,
    relative_theta_shift=1,
    theta_value=None,
    max_steps=None,
    pseudo_transient=False,
    restart=False,
    snes_converged_reason_on=True,
):

    if restart and checkpoint_dir is None:
        raise ValueError("restart=True requires checkpoint_dir")
    if checkpoint_every is not None and checkpoint_dir is None:
        raise ValueError("checkpoint_every requires checkpoint_dir")

    # load mesh and meshtags

    with dolfinx.io.XDMFFile(comm, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name="Cell tags")
        mesh.topology.create_connectivity(1, 2)
        facet_tags = infile.read_meshtags(mesh, name="Facet tags")

    assert (
        len(
            np.setdiff1d(
                np.union1d(cell_tags.values, facet_tags.values), [PHYSICAL_MARKERS[i] for i in PHYSICAL_MARKERS]
            )
        )
        == 0
    ), "Physical markers and cell tags do not match"

    # create submesh for fluid

    fluid_mesh, fluid_cell_map, fluid_vertex_map, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"])
    )
    fluid_mesh.topology.create_connectivity(fluid_mesh.topology.dim - 1, fluid_mesh.topology.dim)

    # create entity maps for mixed mesh integration

    entity_maps = [fluid_cell_map]

    # meshtags on fluid submesh

    facet_tags_fluid = dolfinx.mesh.transfer_meshtags_to_submesh(
        facet_tags, fluid_mesh, fluid_vertex_map, fluid_cell_map
    )

    # Create measure with  meshtags

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    dx_fluid = dx(PHYSICAL_MARKERS["ALE_fluid"])

    # Create measure for interface / solid-fluid boundary

    import scifem

    new_tag_fluid = 101
    interface_facets = facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])
    idata = scifem.compute_interface_data(cell_tags, interface_facets)
    if idata.shape[0] > 0 and cell_tags.values[idata[0, 0]] == PHYSICAL_MARKERS["ALE_fluid"]:
        fluid_entities = idata[:, :2]
    else:
        fluid_entities = idata[:, 2:]
    new_measure_fluid = ufl.Measure("ds", domain=mesh, subdomain_data=[(new_tag_fluid, fluid_entities.flatten())])
    ds_interface_fluid = new_measure_fluid(new_tag_fluid)

    # create problem parameters

    rho_f = dolfinx.fem.Constant(mesh, cfd_parameters["rho_f"])
    nu_f = dolfinx.fem.Constant(mesh, cfd_parameters["nu_f"])

    U_bar = cfd_parameters["U_bar"]
    H = 0.41

    t0 = 0.0
    dt = dolfinx.fem.Constant(mesh, dt_val)

    theta = dolfinx.fem.Constant(mesh, 0.5 + dt.value * relative_theta_shift)
    if theta_value is not None:
        theta.value = theta_value

    num_steps = round((T - t0) / dt_val)
    if vtx_save_every is not None and num_steps < vtx_save_every:
        warnings.warn(
            f"vtx_save_every ({vtx_save_every}) is larger than the total number of time "
            f"steps ({num_steps}); at most one VTX snapshot will be written to "
            f"{output_path!r} or {output_path_p!r}, which is not a usable time "
            f"series in ParaView.",
            stacklevel=2,
        )

    # create function spaces

    V = dolfinx.fem.functionspace(fluid_mesh, ("CG", 2, (2,)))
    P = dolfinx.fem.functionspace(fluid_mesh, ("CG", 1))
    W = ufl.MixedFunctionSpace(V, P)

    # create functions

    v, p = (
        dolfinx.fem.Function(V, name="v"),
        dolfinx.fem.Function(P, name="p"),
    )
    v_old = dolfinx.fem.Function(V)

    dv, dp = ufl.TestFunctions(W)

    # create Dirichlet boundary condition

    from functools import reduce

    inflow_bc_func = dolfinx.fem.Function(V)
    inflow_bc_func.x.array[:] = 0.0
    inflow_bc_facets = reduce(
        np.union1d,
        [
            facet_tags_fluid.find(PHYSICAL_MARKERS["inflow"]),
        ],
    )
    inflow_bc_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.geometry.dim - 1, inflow_bc_facets)

    inflow_bc = dolfinx.fem.dirichletbc(inflow_bc_func, inflow_bc_dofs)

    class InflowFunc:
        def __init__(self, t: float = 0.0):
            self.t = t

        def __call__(self, x: np.ndarray) -> np.ndarray:
            values = np.zeros((2, x.shape[1]), dtype=x.dtype)
            values[0] = 1.5 * U_bar * 4 * x[1] * (H - x[1]) / H**2
            if self.t < 2.0:
                values[0] *= 0.5 * (1.0 - np.cos(0.5 * np.pi * self.t))
            return values

    noslip_bc_func = dolfinx.fem.Function(V)
    noslip_bc_func.x.array[:] = 0.0
    noslip_bc_facets = reduce(
        np.union1d,
        [
            facet_tags_fluid.find(PHYSICAL_MARKERS["obstacle"]),
            facet_tags_fluid.find(PHYSICAL_MARKERS["solid_fluid_interface"]),
            facet_tags_fluid.find(PHYSICAL_MARKERS["channel_side"]),
        ],
    )
    noslip_bc_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.geometry.dim - 1, noslip_bc_facets)

    noslip_bc = dolfinx.fem.dirichletbc(noslip_bc_func, noslip_bc_dofs)

    bcs = [inflow_bc, noslip_bc]

    # DESCRIBE CFD PROBLEM
    # FLUID: Parabolic inflow on left side, no-slip on top, bottom, and obstacle, do-nothing on right side

    from xfsi_solver.fsi.materials import Fluid

    # create residual form

    def A_T(v, v_old):

        residual = rho_f * ufl.inner((v - v_old) / dt, dv) * dx_fluid

        return residual

    def A_I(v):

        residual = ufl.div(v) * dp * dx_fluid

        return residual

    def A_E(v):

        residual = rho_f * ufl.inner(ufl.grad(v) * v, dv) * dx_fluid

        residual += ufl.inner(Fluid.NS_velocity_eulerian(v, nu_f, rho_f), ufl.grad(dv)) * dx_fluid

        # Do-nothing condition
        residual -= (
            rho_f * nu_f * ufl.inner(ufl.grad(v).T * ufl.FacetNormal(mesh), dv) * ds(PHYSICAL_MARKERS["outflow"])
        )

        return residual

    def A_P(p):

        residual = ufl.inner(Fluid.NS_pressure(p), ufl.grad(dv)) * dx_fluid

        return residual

    residual = A_T(v, v_old)
    residual += A_I(v)
    residual += A_P(p)
    residual += theta * A_E(v)
    residual += (1.0 - theta) * A_E(v_old)

    # --------------------------------------------

    residual_blocked = ufl.extract_blocks(residual)

    max_iter = 20
    atol = 1.0e-7
    rtol = 1.0e-12

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocked,
        [v, p],
        bcs=bcs,
        petsc_options_prefix=solver_prefix,
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
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
            # "snes_monitor": None,
            **({"snes_converged_reason": None} if snes_converged_reason_on else {}),
        },
    )

    solver = problem.solver

    from xfsi_solver.tools.checkpoint import load_start_state

    # (t, step) are the time and number of completed steps of the current state
    t, step, checkpointer, (output_path, output_path_p) = load_start_state(
        mesh,
        [v, p],
        t0=t0,
        T=T,
        dt=dt_val,
        num_steps=num_steps,
        restart=restart,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        initial_state_dir=None,
        submeshes=[(fluid_mesh, fluid_cell_map)],
        output_paths=[output_path, output_path_p],
        write_vtx=vtx_save_every is not None,
    )

    writers = []
    if vtx_save_every is not None:
        writers = [
            dolfinx.io.VTXWriter(comm, output_path, [v], mesh_policy=dolfinx.io.VTXMeshPolicy.reuse, engine="BP4"),
            dolfinx.io.VTXWriter(comm, output_path_p, [p]),
        ]

    # Quantities of interest: the displacement at the tip of the structure, and the drag and
    # lift on the obstacle and on the fluid side of the solid-fluid interface.
    drag_forms, lift_forms = drag_lift_forms(
        mesh,
        dolfinx.fem.Function(V, name="zero_deformation"),
        v,
        p,
        nu_f,
        rho_f,
        [ds(PHYSICAL_MARKERS["obstacle"]), ds_interface_fluid],
        entity_maps,
    )

    def write_output(t, step):
        """Write the QoI row, and every vtx_save_every steps the VTX snapshot, of the state at time t."""
        if vtx_save_every is not None and step % vtx_save_every == 0:
            for writer in writers:
                writer.write(t)

        u_spot = np.zeros(2, dtype=np.float64)
        drag = assemble_force(drag_forms, comm)
        lift = assemble_force(lift_forms, comm)
        append_qoi_row(qoi_path, comm, t, drag, lift, u_spot)

    init_qoi_file(qoi_path, comm, restart, t, dt_val, header="t\tdrag\tlift\tdummy\tdummy")

    if not restart:
        write_output(t, step)

    PETSc.Sys.Print("")
    mesh.comm.barrier()

    solver.setConvergenceHistory(reset=True)
    reference_set = False
    reference = 0.0

    first_step = step
    num_newton_iterations = 0
    start = timer()
    while step < num_steps and (max_steps is None or step < max_steps):
        t = t + dt.value
        inflow_bc_func.interpolate(InflowFunc(t))
        inflow_bc_func.x.scatter_forward()

        v_old.x.array[:] = v.x.array[:]

        PETSc.Sys.Print(f"\n{t = :.3f}")

        problem.solve()
        check_converged(problem, f"t = {t:.4f}", writers=writers)
        num_newton_iterations += solver.getIterationNumber()

        steady_residual = solver.getConvergenceHistory()[0][0]

        if not reference_set and t >= 2.0:
            reference = steady_residual * 1.0
            reference_set = True

        if comm.rank == 0:
            sys.stdout.flush()

        step += 1
        write_output(t, step)

        if checkpoint_every is not None and step % checkpoint_every == 0:
            checkpointer.write([v, p], t, step, dt_val)

        if pseudo_transient:
            dt.value *= 2.0
            theta.value = 0.5 + dt.value * relative_theta_shift
            if theta_value is not None:
                theta.value = theta_value

        PETSc.Sys.Print(f"\t{steady_residual = :.2e}")
        if pseudo_transient and reference_set and steady_residual < max(atol, 1e-10 * reference):
            PETSc.Sys.Print("Pseudo-transient break")
            break

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s", flush=True)

    for writer in writers:
        writer.close()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    args = parser.parse_args()

    solve(
        mesh_path="data/meshes/fsi2/mesh_sec.xdmf",
        T=100.0,
        dt_val=0.1,
        cfd_parameters=CFD_parameters["CFD1"],
        solver_prefix="cfd1_",
        output_path="output/pv/component_benchmarks/cfd_1.bp",
        output_path_p="output/pv/component_benchmarks/cfd_1_p.bp",
        qoi_path="output/qoi/component_benchmarks/cfd_1.txt",
        checkpoint_dir="output/checkpoints/component_benchmarks/cfd_1",
        checkpoint_every=None,
        vtx_save_every=5,
        theta_value=1.0,
        max_steps=40,
        pseudo_transient=True,
        restart=args.restart,
    )
    solve(
        mesh_path="data/meshes/fsi2/mesh_sec.xdmf",
        T=100.0,
        dt_val=0.1,
        cfd_parameters=CFD_parameters["CFD2"],
        solver_prefix="cfd2_",
        output_path="output/pv/component_benchmarks/cfd_2.bp",
        output_path_p="output/pv/component_benchmarks/cfd_2_p.bp",
        qoi_path="output/qoi/component_benchmarks/cfd_2.txt",
        checkpoint_dir="output/checkpoints/component_benchmarks/cfd_2",
        checkpoint_every=None,
        vtx_save_every=5,
        theta_value=1.0,
        max_steps=40,
        pseudo_transient=True,
        restart=args.restart,
    )
    solve(
        mesh_path="data/meshes/fsi2/mesh_sec.xdmf",
        T=10.0,
        dt_val=0.005,
        cfd_parameters=CFD_parameters["CFD3"],
        solver_prefix="cfd3_",
        output_path="output/pv/component_benchmarks/cfd_3.bp",
        output_path_p="output/pv/component_benchmarks/cfd_3_p.bp",
        qoi_path="output/qoi/component_benchmarks/cfd_3.txt",
        checkpoint_dir="output/checkpoints/component_benchmarks/cfd_3",
        checkpoint_every=400,
        vtx_save_every=5,
        relative_theta_shift=1.0,
        max_steps=3000,
        pseudo_transient=False,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
