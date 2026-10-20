# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import warnings
from timeit import default_timer as timer

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py.MPI import COMM_WORLD as comm
from petsc4py import PETSc

from xfsi_solver.component_benchmarks.benchmark_parameters import CSM_parameters
from xfsi_solver.tools.convergence import KSPConvCheck, check_converged
from xfsi_solver.tools.qoi import (
    append_qoi_row,
    assemble_force,
    find_point_dof,
    init_qoi_file,
    point_value,
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
    csm_parameters,
    solver_prefix,
    output_path,
    qoi_path,
    checkpoint_dir=None,
    checkpoint_every=None,
    vtx_save_every=None,
    relative_theta_shift=1,
    theta_value=None,
    max_steps=None,
    pseudo_transient=False,
    static_rtol=1e-7,
    restart=False,
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

    # create submesh for fluid, solid

    solid_mesh, solid_cell_map, solid_vertex_map, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"])
    )
    solid_mesh.topology.create_connectivity(solid_mesh.topology.dim - 1, solid_mesh.topology.dim)

    # create entity maps for mixed mesh integration

    entity_maps = [solid_cell_map]

    # meshtags on solid submesh

    facet_tags_solid = dolfinx.mesh.transfer_meshtags_to_submesh(
        facet_tags, solid_mesh, solid_vertex_map, solid_cell_map
    )

    # Create measure with  meshtags

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

    dx_solid = dx(PHYSICAL_MARKERS["solid"])

    # create problem parameters

    rho_s = dolfinx.fem.Constant(mesh, csm_parameters["rho_s"])
    mu_s = dolfinx.fem.Constant(mesh, csm_parameters["mu_s"])
    nu_s = dolfinx.fem.Constant(mesh, csm_parameters["nu_s"])
    lambda_s = dolfinx.fem.Constant(mesh, -mu_s.value / (1 - 0.5 / nu_s.value))

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
            f"{output_path!r}, which is not a usable time "
            f"series in ParaView.",
            stacklevel=2,
        )

    # create function spaces

    U = dolfinx.fem.functionspace(solid_mesh, ("CG", 2, (2,)))
    V = dolfinx.fem.functionspace(solid_mesh, ("CG", 2, (2,)))
    W = ufl.MixedFunctionSpace(U, V)

    # create functions

    u, v = (
        dolfinx.fem.Function(U, name="u"),
        dolfinx.fem.Function(V, name="v"),
    )
    u_old, v_old = dolfinx.fem.Function(U), dolfinx.fem.Function(V)

    du, dv = ufl.TestFunctions(W)

    from functools import reduce

    # create Dirichlet boundary conditions

    u_s_bc_func = dolfinx.fem.Function(U)
    u_s_bc_func.x.array[:] = 0.0
    u_s_bc_facets = reduce(
        np.union1d,
        [
            facet_tags_solid.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
        ],
    )
    u_s_bc_dofs = dolfinx.fem.locate_dofs_topological(U, mesh.geometry.dim - 1, u_s_bc_facets)
    u_s_bc = dolfinx.fem.dirichletbc(u_s_bc_func, u_s_bc_dofs)

    v_s_bc_func = dolfinx.fem.Function(V)
    v_s_bc_func.x.array[:] = 0.0
    v_s_bc_facets = reduce(
        np.union1d,
        [
            facet_tags_solid.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
        ],
    )
    v_s_bc_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.geometry.dim - 1, v_s_bc_facets)
    v_s_bc = dolfinx.fem.dirichletbc(v_s_bc_func, v_s_bc_dofs)

    # Collect Dirichlet boundary conditions to pass to NonlinearProblem

    bcs = [u_s_bc, v_s_bc]

    # DESCRIBE CSM PROBLEM
    # SOLID: Homogeneous Dirichlet on left side

    from xfsi_solver.fsi.materials import Solid

    # create residual form with u condensed out.

    def A_T(u, u_old, v, v_old):

        residual = rho_s * ufl.inner((u - u_old) / dt, du) * dx_solid
        residual += rho_s * ufl.inner((v - v_old) / dt, dv) * dx_solid

        return residual

    def A_E(u, v, u_condensed):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_condensed)
        J = ufl.det(F)

        residual = -rho_s * ufl.inner(v, du) * dx_solid
        residual += ufl.inner(J * Solid.STVK(u_condensed, lambda_s, mu_s) * ufl.inv(F).T, ufl.grad(dv)) * dx_solid

        return residual

    residual = A_T(u, u_old, v, v_old)

    u_condensed = u_old + dt * ((1.0 - theta) * v_old + theta * v)
    residual += theta * A_E(u, v, u_condensed)
    residual += (1.0 - theta) * A_E(u_old, v_old, u_old)

    # Add in u_s with zero constants for correct sparsity patterns
    residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(u, du) * dx_solid
    residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(v, du) * dx_solid

    # Add body force
    gravity = dolfinx.fem.Constant(mesh, (0.0, -csm_parameters["g"]))
    residual -= ufl.inner(rho_s * gravity, dv) * dx_solid

    # Residual of the static problem at the current displacement, for the stopping test of the pseudo-transient
    # runs, relative to that at zero displacement (the gravity load). The residual of the time step is not used for
    # this: at its start, u_condensed is u_old + dt * v_old, not u_old.
    w = ufl.TestFunction(U)
    F_static = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
    load_form = ufl.inner(rho_s * gravity, w) * dx_solid
    stress_form = ufl.inner(ufl.det(F_static) * Solid.STVK(u, lambda_s, mu_s) * ufl.inv(F_static).T, ufl.grad(w))
    static_residual_form = dolfinx.fem.form(stress_form * dx_solid - load_form, entity_maps=entity_maps)
    load_form = dolfinx.fem.form(load_form, entity_maps=entity_maps)

    def residual_norm(form):
        """Norm of the vector assembled from form, with the rows of the clamped dofs set to zero."""
        b = dolfinx.fem.petsc.assemble_vector(form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, [u_s_bc])
        norm = b.norm()
        b.destroy()
        return norm

    load_norm = residual_norm(load_form)

    # --------------------------------------------

    # Set up output, checkpointing, and qoi tracking.

    from xfsi_solver.tools.checkpoint import Checkpointer, restart_output_path

    # (t, step) are the time and number of completed steps of the current state
    t = t0
    step = 0
    checkpointer = None
    if checkpoint_dir is not None:
        checkpointer = Checkpointer(checkpoint_dir, mesh, submeshes=[(solid_mesh, solid_cell_map)])
    if restart:
        t, step = checkpointer.read([u, v], dt_val)
        if step >= num_steps:
            raise ValueError(
                f"Cannot restart from the checkpoint at {t = :.4f} ({step = }): no time step is left to solve "
                f"up to {T = } ({num_steps = })"
            )
        output_path = restart_output_path(output_path, t)
        vtx_message = "" if vtx_save_every is None else f", writing VTX output to {output_path}"
        PETSc.Sys.Print(f"Restarting from {checkpoint_dir} at {t = :.4f} ({step = }){vtx_message}")
    elif checkpoint_every is not None:
        checkpointer.clear()

    writers = []
    if vtx_save_every is not None:
        writers = [dolfinx.io.VTXWriter(comm, output_path, [u, v])]

    # Quantities of interest: the displacement at the tip of the structure, and the drag and
    # lift on the obstacle and on the fluid side of the solid-fluid interface.
    spot_dof = find_point_dof(U, np.array([0.6, 0.2, 0.0], dtype=np.float64))
    dummy_qoi = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 0.0) * dx_solid)
    drag_forms, lift_forms = (dummy_qoi, dummy_qoi), (dummy_qoi, dummy_qoi)

    def write_output(t, step):
        """Write the QoI row, and every vtx_save_every steps the VTX snapshot, of the state at time t."""
        if vtx_save_every is not None and step % vtx_save_every == 0:
            for writer in writers:
                writer.write(t)

        u_spot = point_value(u, spot_dof)
        drag = assemble_force(drag_forms, comm)
        lift = assemble_force(lift_forms, comm)
        append_qoi_row(qoi_path, comm, t, drag, lift, u_spot)

    problem = dolfinx.fem.petsc.NonlinearProblem(
        ufl.extract_blocks(residual),
        [u, v],
        bcs=bcs,
        petsc_options_prefix=solver_prefix,
        entity_maps=entity_maps,
        petsc_options={
            # SNES options
            "snes_linesearch_type": "none",
            "snes_max_it": 20,
            "snes_atol": 1.0e-7,
            "snes_rtol": 1.0e-12,
            "snes_stol": 0.0,
            # KSP options
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_cntl_1": 1e-4,
            # Turn off errors, catch manually instead.
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
            # Print to console.
            # "snes_monitor": None,
            "snes_converged_reason": None,
        },
    )

    solver = problem.solver

    # Records failed linear solves nested in the fieldsplits, such as a K_c FGMRES that reaches its
    # iteration limit, which neither PETSc nor the SNES converged reason report.
    ksp_check = KSPConvCheck(solver.getKSP())

    # Open qoi file on fresh runs, discard entries after the restart time if a restarted run.
    init_qoi_file(qoi_path, comm, restart, t, dt_val, header="t\tdummy\tdummy\tA_x\tA_y")

    if not restart:
        write_output(t, step)

    PETSc.Sys.Print()
    mesh.comm.barrier()

    first_step = step
    num_newton_iterations = 0
    start = timer()
    while step < num_steps and (max_steps is None or step < max_steps):
        t = t + dt.value

        u_old.x.array[:] = u.x.array[:]
        v_old.x.array[:] = v.x.array[:]

        PETSc.Sys.Print(f"\n\n{t = :.3f}")

        ksp_check.clear()
        problem.solve()
        num_newton_iterations += solver.getIterationNumber()
        check_converged(problem, f"t = {t:.4f}", writers=writers, ksp_check=ksp_check)

        step += 1
        write_output(t, step)

        if checkpoint_every is not None and step % checkpoint_every == 0:
            checkpointer.write([u, v], t, step, dt_val)

        u_spot = comm.bcast(point_value(u, spot_dof), root=0)
        PETSc.Sys.Print(f"\tu_x = {u_spot[0]:3e}, u_y = {u_spot[1]:3e}")

        if pseudo_transient:
            relative_static_residual = residual_norm(static_residual_form) / load_norm
            PETSc.Sys.Print(f"\trelative static residual = {relative_static_residual:.3e}")
            if relative_static_residual < static_rtol:
                PETSc.Sys.Print(f"Static residual below {static_rtol = :.1e}, stopping")
                break

            dt.value *= 2.0
            theta.value = 0.5 + dt.value * relative_theta_shift
            if theta_value is not None:
                theta.value = theta_value

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")

    for writer in writers:
        writer.close()

    # Destroying the MUMPS factorization and the AMG hierarchies is collective, so destroy them on
    # all ranks here instead of leaving it to garbage collection, which can run at different points
    # on different ranks.
    solver.destroy()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    args = parser.parse_args()

    solve(
        mesh_path="data/meshes/fsi2/mesh_fine_sec.xdmf",
        T=100.0,
        dt_val=0.1,
        csm_parameters=CSM_parameters["CSM1"],
        solver_prefix="csm1_",
        output_path="output/pv/component_benchmarks/csm_1.bp",
        qoi_path="output/qoi/component_benchmarks/csm_1.txt",
        checkpoint_dir="output/checkpoints/component_benchmarks/csm_1",
        checkpoint_every=None,
        vtx_save_every=1,
        theta_value=1.0,
        max_steps=1000,
        pseudo_transient=True,
        restart=args.restart,
    )
    solve(
        mesh_path="data/meshes/fsi2/mesh_fine_sec.xdmf",
        T=100.0,
        dt_val=0.1,
        csm_parameters=CSM_parameters["CSM2"],
        solver_prefix="csm2_",
        output_path="output/pv/component_benchmarks/csm_2.bp",
        qoi_path="output/qoi/component_benchmarks/csm_2.txt",
        checkpoint_dir="output/checkpoints/component_benchmarks/csm_2",
        checkpoint_every=None,
        vtx_save_every=1,
        theta_value=1.0,
        max_steps=1000,
        pseudo_transient=True,
        restart=args.restart,
    )
    solve(
        mesh_path="data/meshes/fsi2/mesh_fine_sec.xdmf",
        T=10.0,
        dt_val=0.005,
        csm_parameters=CSM_parameters["CSM3"],
        solver_prefix="csm3_",
        output_path="output/pv/component_benchmarks/csm_3.bp",
        qoi_path="output/qoi/component_benchmarks/csm_3.txt",
        checkpoint_dir="output/checkpoints/component_benchmarks/csm_3",
        checkpoint_every=None,
        vtx_save_every=5,
        relative_theta_shift=0.0,
        max_steps=2000,
        pseudo_transient=False,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
