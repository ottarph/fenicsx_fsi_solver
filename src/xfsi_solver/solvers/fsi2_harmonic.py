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

from xfsi_solver.scripts.plot_qois import plot_run_qois
from xfsi_solver.tools.checkpoint import Checkpointer, check_start_options, restart_output_path, start_time
from xfsi_solver.tools.convergence import check_converged
from xfsi_solver.tools.parameters import lame_lambda, theta_value
from xfsi_solver.tools.qoi import (
    append_qoi_row,
    assemble_force,
    drag_lift_forms,
    find_point_dof,
    init_qoi_file,
    point_value,
)
from xfsi_solver.tools.quadrature import cap_quadrature_degree, quadrature_description

PHYSICAL_MARKERS = {
    "solid": 1,
    "ALE_fluid": 2,
    "solid_fluid_interface": 11,
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
    output_path,
    output_path_p,
    qoi_path,
    elements={
        "U": ("P", 2, (2,)),
        "V": ("P", 2, (2,)),
        "P": ("P", 1),
    },
    rho_f=1.0e3,
    nu_f=1.0e-3,
    rho_s=1.0e4,
    mu_s=5.0e5,
    nu_s=None,
    lambda_s=None,
    U_bar=1.0,
    H=0.41,
    absolute_theta_shift=None,
    relative_theta_shift=None,
    checkpoint_dir=None,
    checkpoint_every=None,
    vtx_save_every=None,
    restart=False,
    initial_state_dir=None,
    max_quadrature_degree=None,
):
    """Solve the FSI2 benchmark up to time ``T``, in ``round((T - t0) / dt_val)`` time
    steps, writing the initial state at ``t0`` as the first QoI row and, with
    ``vtx_save_every``, as the first VTX snapshot. With ``vtx_save_every=None``, no VTX
    output is written; otherwise a snapshot is written every ``vtx_save_every`` steps.

    With ``checkpoint_dir`` and ``checkpoint_every``, a restart checkpoint of
    the state is saved in ``checkpoint_dir`` every ``checkpoint_every`` time
    steps (replacing checkpoints there from earlier runs). With
    ``restart=True``, the run instead continues from the latest checkpoint in
    ``checkpoint_dir``, on any number of MPI ranks: QoI rows after the
    checkpoint time are dropped from ``qoi_path`` before appending, and the
    VTX output is written to new files with a ``_from_t<time>`` suffix, since
    ``VTXWriter`` cannot append.

    With ``initial_state_dir``, the run instead starts from the state in the latest checkpoint
    in ``initial_state_dir``, written by another run, possibly with another ``dt_val``. ``t0`` is
    the time of that checkpoint, the run writes its own QoI file, VTX output and checkpoints,
    and it only reads from ``initial_state_dir``. A restart of such a run continues it from
    the same ``t0``.

    The function spaces are given by ``elements``, a dict from the name of each function space in the
    solver (U, V, P) to the element tuple passed to ``dolfinx.fem.functionspace``; the default is
    piecewise quadratic vector fields and piecewise linear pressure.

    The physical constants are ``rho_f``, ``nu_f`` (fluid density and kinematic viscosity), ``rho_s``, ``mu_s`` (solid
    density and shear modulus), ``U_bar`` (mean inflow velocity) and ``H`` (channel height), by default the FSI2
    values. The first Lamé parameter of the solid is given either directly as ``lambda_s`` or through the Poisson
    ratio ``nu_s``, as 2 mu_s nu_s / (1 - 2 nu_s); with neither given, ``nu_s = 0.4``. Giving both raises
    ``ValueError``. The time stepping parameter is ``theta = 0.5 + absolute_theta_shift`` or
    ``theta = 0.5 + relative_theta_shift * dt_val``; with neither given, ``relative_theta_shift = 1.0``. Giving
    both raises ``ValueError``.

    If ``max_quadrature_degree`` is given, the quadrature degree of the residual, and so of its
    Jacobian, and of the drag and lift is capped at it: integrals for which UFL estimates a higher
    degree are integrated inexactly at that degree, and the others keep their estimate. With the
    default None, every integral keeps the degree UFL estimates.
    """
    check_start_options(restart, checkpoint_dir, checkpoint_every, initial_state_dir)
    theta_val = theta_value(dt_val, absolute_theta_shift, relative_theta_shift, 1.0)
    if nu_s is None and lambda_s is None:
        nu_s = 0.4
    lambda_s_val = lame_lambda(mu_s, nu_s, lambda_s)

    # load mesh and meshtags

    # mesh_path alternatives:
    #   "data/meshes/fsi2/mesh_quad.xdmf"
    #   "data/meshes/fsi2/mesh_quad_fine_sec.xdmf"
    #   "data/meshes/fsi2/mesh_sec.xdmf"

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

    # create entity maps for mixed mesh integration

    entity_maps = [fluid_cell_map]

    # Create measure with  meshtags

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    dx_fluid = dx(PHYSICAL_MARKERS["ALE_fluid"])
    dx_solid = dx(PHYSICAL_MARKERS["solid"])

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

    rho_f = dolfinx.fem.Constant(mesh, rho_f)
    nu_f = dolfinx.fem.Constant(mesh, nu_f)

    rho_s = dolfinx.fem.Constant(mesh, rho_s)
    mu_s = dolfinx.fem.Constant(mesh, mu_s)
    lambda_s = dolfinx.fem.Constant(mesh, lambda_s_val)

    t0 = start_time(mesh, restart, checkpoint_dir, initial_state_dir)
    dt = dolfinx.fem.Constant(mesh, dt_val)

    theta = dolfinx.fem.Constant(mesh, theta_val)

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

    U = dolfinx.fem.functionspace(mesh, elements["U"])
    V = dolfinx.fem.functionspace(mesh, elements["V"])
    P = dolfinx.fem.functionspace(fluid_mesh, elements["P"])
    W = ufl.MixedFunctionSpace(U, V, P)

    # create functions

    u, v, p = dolfinx.fem.Function(U, name="u"), dolfinx.fem.Function(V, name="v"), dolfinx.fem.Function(P, name="p")
    u_old, v_old = dolfinx.fem.Function(U), dolfinx.fem.Function(V)

    du, dv, dp = ufl.TestFunctions(W)

    # create Dirichlet boundary condition

    from functools import reduce

    inflow_bc_func = dolfinx.fem.Function(V)
    inflow_bc_func.x.array[:] = 0.0
    inflow_bc_facets = reduce(
        np.union1d,
        [
            facet_tags.find(PHYSICAL_MARKERS["inflow"]),
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
            facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
            facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
            facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
        ],
    )
    noslip_bc_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.geometry.dim - 1, noslip_bc_facets)

    noslip_bc = dolfinx.fem.dirichletbc(noslip_bc_func, noslip_bc_dofs)

    # Create ALE Dirichlet boundary condition

    u_bc_func = dolfinx.fem.Function(U)
    u_bc_func.x.array[:] = 0.0
    u_bc_facets = reduce(
        np.union1d,
        [
            facet_tags.find(PHYSICAL_MARKERS["inflow"]),
            facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
            facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
            facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
            facet_tags.find(PHYSICAL_MARKERS["outflow"]),
        ],
    )
    u_bc_dofs = dolfinx.fem.locate_dofs_topological(U, mesh.geometry.dim - 1, u_bc_facets)
    u_bc = dolfinx.fem.dirichletbc(u_bc_func, u_bc_dofs)

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

        residual = rho_f * J_mid * ufl.inner((v - v_old) / dt, dv) * dx_fluid

        residual -= rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * ((u - u_old) / dt), dv) * dx_fluid

        residual += rho_s * ufl.inner((v - v_old) / dt, dv) * dx_solid

        residual += rho_s * ufl.inner((u - u_old) / dt, du) * dx_solid

        return residual

    def A_I(u, v, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)
        normal = ufl.FacetNormal(mesh)

        alpha_u0 = dolfinx.fem.Constant(mesh, 1e-9)
        alpha_u = alpha_u0

        residual = ufl.inner(alpha_u * ufl.grad(u), ufl.grad(du)) * dx_fluid
        residual -= ufl.inner(alpha_u * ufl.grad(u) * normal, du) * ds_interface_fluid

        residual += ufl.div(J * ufl.inv(F) * v) * dp * dx_fluid

        return residual

    def A_E(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx_fluid

        residual += ufl.inner(J * Fluid.NS_velocity(u, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        residual += ufl.inner(J * Solid.STVK(u, lambda_s, mu_s) * ufl.inv(F).T, ufl.grad(dv)) * dx_solid

        residual -= rho_s * ufl.inner(v, du) * dx_solid

        return residual

    def A_P(u, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    residual = A_T(u, u_old, v, v_old)
    residual += A_I(u, v, p)
    residual += A_P(u, p)
    residual += theta * A_E(u, v)
    residual += (1.0 - theta) * A_E(u_old, v_old)

    # --------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])

    residual_blocked = ufl.extract_blocks(residual)
    # Cap the quadrature degree before the Jacobian is derived from the residual, so that it keeps the cap.
    residual_blocked = [cap_quadrature_degree(form, max_quadrature_degree) for form in residual_blocked]

    max_iter = 20
    atol = 1.0e-7
    rtol = 1.0e-12

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocked,
        [u, v, p],
        bcs=bcs,
        petsc_options_prefix="fsi2_harmonic_",
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
            # "snes_monitor": "ascii:output/logs/fsi2_harmonic_snes_log.txt",
            "snes_converged_reason": None,
        },
    )

    # (t, step) are the time and number of completed steps of the current state
    t = t0
    step = 0
    checkpointer = None
    if checkpoint_dir is not None:
        checkpointer = Checkpointer(checkpoint_dir, mesh, submeshes=[(fluid_mesh, fluid_cell_map)])
    if restart:
        t, step = checkpointer.read([u, v, p], dt_val)
        if step >= num_steps:
            raise ValueError(
                f"Cannot restart from the checkpoint at {t = :.4f} ({step = }): no time step is left to solve "
                f"up to {T = } ({num_steps = })"
            )
        output_path = restart_output_path(output_path, t)
        output_path_p = restart_output_path(output_path_p, t)
        if comm.rank == 0:
            vtx_message = "" if vtx_save_every is None else f", writing VTX output to {output_path} and {output_path_p}"
            print(f"Restarting from {checkpoint_dir} at {t = :.4f} ({step = }){vtx_message}")
    elif checkpoint_every is not None:
        checkpointer.clear()
    if initial_state_dir is not None:
        t, _ = Checkpointer(initial_state_dir, mesh, submeshes=[(fluid_mesh, fluid_cell_map)]).read([u, v, p], None)
        if t != t0:
            raise RuntimeError(
                f"The latest checkpoint in {initial_state_dir} changed from t = {t0} to {t} while it was read; "
                "start from a copy of a run that is still writing checkpoints"
            )
        if num_steps <= 0:
            raise ValueError(
                f"Cannot start from the state at {t = :.4f} in {initial_state_dir}: "
                f"no time step is left to solve up to {T = }"
            )
        if mesh.comm.rank == 0:
            print(f"Starting from the state in {initial_state_dir} at {t = :.4f}")

    writers = []
    if vtx_save_every is not None:
        writers = [dolfinx.io.VTXWriter(comm, output_path, [u, v]), dolfinx.io.VTXWriter(comm, output_path_p, [p])]

    # Quantities of interest: the displacement at the tip of the structure, and the drag and
    # lift on the obstacle and on the fluid side of the solid-fluid interface.
    spot_dof = find_point_dof(U, np.array([0.6, 0.2, 0.0], dtype=np.float64))
    drag_forms, lift_forms = drag_lift_forms(
        mesh,
        u,
        v,
        p,
        nu_f,
        rho_f,
        [ds(PHYSICAL_MARKERS["obstacle"]), ds_interface_fluid],
        entity_maps,
        max_quadrature_degree=max_quadrature_degree,
    )

    def write_output(t, step):
        """Write the QoI row, and every vtx_save_every steps the VTX snapshot, of the state at time t."""
        if vtx_save_every is not None and step % vtx_save_every == 0:
            for writer in writers:
                writer.write(t)

        u_spot = point_value(u, spot_dof)
        drag = assemble_force(drag_forms, comm)
        lift = assemble_force(lift_forms, comm)
        append_qoi_row(qoi_path, comm, t, drag, lift, u_spot)

    init_qoi_file(qoi_path, comm, restart, t, dt_val)

    if not restart:
        write_output(t, step)

    if mesh.comm.rank == 0:
        print("", flush=True)
    mesh.comm.barrier()

    first_step = step
    start = timer()
    while step < num_steps:
        t = t0 + (step + 1) * dt_val
        inflow_bc_func.interpolate(InflowFunc(t))
        inflow_bc_func.x.scatter_forward()

        u_old.x.array[:] = u.x.array[:]
        v_old.x.array[:] = v.x.array[:]

        if comm.rank == 0:
            print(f"\n{t = :.3f}")

        problem.solve()
        check_converged(problem, f"t = {t:.4f}", writers=writers)

        if comm.rank == 0:
            sys.stdout.flush()

        step += 1
        write_output(t, step)

        if checkpoint_every is not None and step % checkpoint_every == 0:
            checkpointer.write([u, v, p], t, step, dt_val, t0)

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")
        print(f"\nSolver setup:\n  Quadrature degree: {quadrature_description(max_quadrature_degree)}")

    for writer in writers:
        writer.close()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument("--initial-state-dir", help="start from the latest checkpoint of another run in this directory")
    args = parser.parse_args()

    mesh_path = "data/meshes/fsi2/mesh_sec.xdmf"
    T = 15.0  # Will fail at around t=7s
    dt_val = 0.0025
    output_path = "output/pv/fsi2_harmonic.bp"
    output_path_p = "output/pv/fsi2_harmonic_p.bp"
    qoi_path = "output/qoi/fsi2_harmonic.txt"
    checkpoint_dir = "output/checkpoints/fsi2_harmonic"
    checkpoint_every = 100
    vtx_save_every = 4
    max_quadrature_degree = None  # Default None, no cap.
    restart = args.restart
    initial_state_dir = args.initial_state_dir
    solve(
        mesh_path=mesh_path,
        T=T,
        dt_val=dt_val,
        output_path=output_path,
        output_path_p=output_path_p,
        qoi_path=qoi_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        vtx_save_every=vtx_save_every,
        max_quadrature_degree=max_quadrature_degree,
        restart=restart,
        initial_state_dir=initial_state_dir,
    )

    if comm.rank == 0:
        plot_run_qois(qoi_path, T)


if __name__ == "__main__":
    main()
