# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import warnings
from timeit import default_timer as timer

import dolfinx
import dolfinx.fem.forms
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from petsc4py import PETSc

from xfsi_solver.mesh_motion import biharmonic_mixed_mesh_motion
from xfsi_solver.scripts.plot_qois import plot_run_qois
from xfsi_solver.tools.checkpoint import check_start_options, load_start_state, start_time
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
from xfsi_solver.tools.rigid_body_modes import rigid_body_modes
from xfsi_solver.tools.submesh import SubmeshComposite, all_cells

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

DEFAULT_SOLVER_OPTIONS = {
    # SNES options
    "snes_linesearch_type": "none",
    "snes_max_it": 20,
    "snes_atol": 1.0e-7,
    "snes_rtol": 1.0e-12,
    "snes_stol": 0.0,
    # Turn off errors, catch manually instead.
    "snes_error_if_not_converged": False,
    "ksp_error_if_not_converged": False,
    # Print to console.
    # "snes_monitor": None,
    "snes_converged_reason": None,
}
DEFAULT_DIRECT_SOLVER_OPTIONS = DEFAULT_SOLVER_OPTIONS | {
    # KSP options
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_cntl_1": 1e-4,
}
DEFAULT_ITERATIVE_SOLVER_OPTIONS = DEFAULT_SOLVER_OPTIONS | {
    "ksp_converged_reason": None,
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-5,
    "ksp_max_it": 200,
    #
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "upper",
    # The pressure Schur complement S = -B A^{-1} G is dense and only applied as an operator, so it
    # is preconditioned by the sparse approximation \hat S = -B diag(A)^{-1} G. The pressure term
    # gives G = -B^T (up to quadrature, by the Piola identity), so \hat S = B diag(A)^{-1} B^T is a
    # divergence of a gradient: close to a pressure Laplacian, scaled by about dt / rho_f, since the
    # inertia term rho_f / dt dominates diag(A) at small time steps.
    "pc_fieldsplit_schur_precondition": "selfp",
    #
    # One GAMG V-cycle on v, with the near-nullspace and block size attached to is_v.
    "fieldsplit_v_ksp_type": "preonly",
    "fieldsplit_v_pc_type": "gamg",
    "fieldsplit_v_pc_gamg_threshold": 0.01,
    "fieldsplit_v_pc_gamg_reuse_interpolation": True,
    "fieldsplit_v_mg_levels_ksp_type": "richardson",
    "fieldsplit_v_mg_levels_pc_type": "sor",
    # "fieldsplit_v_ksp_converged_reason": None,
    #
    # One BoomerAMG V-cycle on \hat S.
    "fieldsplit_p_ksp_type": "preonly",
    "fieldsplit_p_pc_type": "hypre",
    "fieldsplit_p_pc_hypre_type": "boomeramg",
    # "fieldsplit_p_ksp_converged_reason": None,
}


def solve(
    mesh_path,
    T,
    dt_val,
    output_path,
    output_path_p,
    qoi_path,
    elements={  # noqa: B006
        "U": ("P", 2, (2,)),
        "V": ("P", 2, (2,)),
        "P": ("P", 1),
    },
    solver_options=None,
    mesh_motion_type=None,
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
    gamma_reassemble=0.2,
    max_quadrature_degree=None,
    mesh_motion_options=None,
):
    """Solve the FSI2 benchmark with Newton's method on the velocity and pressure only.

    The solid stress is written in terms of v through the kinematic relation, as in
    fsi2_biharmonic_mixed, and the Jacobian is that of the momentum and continuity equations with
    respect to (v, p), solved as ``solver_options`` sets: by MUMPS LU by default, or, with
    ``DEFAULT_ITERATIVE_SOLVER_OPTIONS``, by FGMRES with a Schur complement fieldsplit on a nest
    Jacobian. The displacements are not unknowns of the Newton system:
    after each linear solve, a post-solve hook of the KSP updates v to the next iterate, sets the solid
    displacement u_s on the solid mesh from the kinematic relation, node by node, and extends it into
    the fluid as u_f on the fluid mesh with the mesh motion ``mesh_motion_type``, one of the functions
    of ``xfsi_solver.mesh_motion``, by default ``biharmonic_mixed_mesh_motion``, before SNES evaluates
    the residual there. By default, the mesh motion is solved as there: it does not couple the x and y
    components, which have the same matrix, so only the x component is factorized, by MUMPS once for
    the whole run, and both components are solved with that factorization at once, by one
    MatMatSolve. ``mesh_motion_options``, the PETSc options of its KSP, replace that default. With the
    mixed biharmonic mesh motion and exact solves, this gives the same iterates as
    fsi2_biharmonic_mixed.

    Checkpoints store the displacement as one function u on the parent mesh, with v and p, as the
    other FSI2 solvers do, so runs can restart from or start from each other's states. They hold no
    field of the mesh motion, which each solve computes again, so a run can also start from the state
    of a run with another mesh motion.

    If ``max_quadrature_degree`` is given, the quadrature degree of the residual, and so of its
    Jacobian, and of the drag and lift is capped at it: integrals for which UFL estimates a higher
    degree are integrated inexactly at that degree, and the others keep their estimate. With the
    default None, every integral keeps the degree UFL estimates. The other arguments are those of
    ``fsi2_biharmonic_mixed.solve``.
    """
    check_start_options(restart, checkpoint_dir, checkpoint_every, initial_state_dir)
    theta_val = theta_value(dt_val, absolute_theta_shift, relative_theta_shift, 1.0)
    if nu_s is None and lambda_s is None:
        nu_s = 0.4
    lambda_s_val = lame_lambda(mu_s, nu_s, lambda_s)

    if solver_options is None:
        solver_options = DEFAULT_DIRECT_SOLVER_OPTIONS
    if mesh_motion_type is None:
        mesh_motion_type = biharmonic_mixed_mesh_motion

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

    # create submesh for solid

    solid_mesh, solid_cell_map, solid_vertex_map, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"])
    )

    # create entity maps for mixed mesh integration

    entity_maps = [fluid_cell_map, solid_cell_map]

    # Need facet tags on the solid mesh, for its Dirichlet condition
    solid_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
        facet_tags, solid_mesh, solid_vertex_map, solid_cell_map
    )

    fluid_mesh.topology.create_connectivity(fluid_mesh.topology.dim - 1, fluid_mesh.topology.dim)
    solid_mesh.topology.create_connectivity(solid_mesh.topology.dim - 1, solid_mesh.topology.dim)

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
    V = dolfinx.fem.functionspace(mesh, elements["V"])
    P = dolfinx.fem.functionspace(fluid_mesh, elements["P"])
    U_s = dolfinx.fem.functionspace(solid_mesh, elements["U"])
    U_f = dolfinx.fem.functionspace(fluid_mesh, elements["U"])
    W = ufl.MixedFunctionSpace(V, P)

    # create functions

    v, p, u_s, u_f = (
        dolfinx.fem.Function(V, name="v"),
        dolfinx.fem.Function(P, name="p"),
        dolfinx.fem.Function(U_s, name="u_s"),
        dolfinx.fem.Function(U_f, name="u_f"),
    )
    v_old, u_s_old, u_f_old = dolfinx.fem.Function(V), dolfinx.fem.Function(U_s), dolfinx.fem.Function(U_f)

    dv, dp = ufl.TestFunctions(W)

    # The displacement on the parent mesh, u_s on the solid cells and u_f on the fluid cells, which
    # the checkpoints and the VTX output hold in place of u_s and u_f. The checkpoints hold no other
    # field of the mesh motion, such as z of the mixed form: every mesh motion solve computes those
    # again from u_s, so the state is (u, v, p) whichever mesh motion is used.
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, elements["U"]), name="u")
    checkpoint_fields = (u, v, p)

    displacement = SubmeshComposite(u, [(u_s, solid_cell_map), (u_f, fluid_cell_map)])

    # Set up output, checkpointing, and qoi tracking.

    # (t, step) are the time and number of completed steps of the current state
    t, step, checkpointer, (output_path, output_path_p) = load_start_state(
        mesh,
        checkpoint_fields,
        t0=t0,
        T=T,
        dt=dt_val,
        num_steps=num_steps,
        restart=restart,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        initial_state_dir=initial_state_dir,
        submeshes=[(fluid_mesh, fluid_cell_map)],
        output_paths=[output_path, output_path_p],
        write_vtx=vtx_save_every is not None,
    )
    displacement.scatter()  # u_s, u_f <- u, which is 0 on a new run from rest

    writers = []
    if vtx_save_every is not None:
        writers = [dolfinx.io.VTXWriter(comm, output_path, [u, v]), dolfinx.io.VTXWriter(comm, output_path_p, [p])]

    # Quantities of interest: the displacement at the tip of the structure, and the drag and
    # lift on the obstacle and on the fluid side of the solid-fluid interface.
    spot_dof = find_point_dof(U_s, np.array([0.6, 0.2, 0.0], dtype=np.float64))
    drag_forms, lift_forms = drag_lift_forms(
        mesh,
        u_f,
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
            displacement.gather()  # u is written with v
            for writer in writers:
                writer.write(t)

        u_spot = point_value(u_s, spot_dof)
        drag = assemble_force(drag_forms, comm)
        lift = assemble_force(lift_forms, comm)
        append_qoi_row(qoi_path, comm, t, drag, lift, u_spot)

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

    # Create fluid ALE Dirichlet boundary condition

    u_f_bc_func = dolfinx.fem.Function(U_f)
    u_f_bc_func.x.array[:] = 0.0
    u_f_bc_facets = dolfinx.mesh.exterior_facet_indices(fluid_mesh.topology)
    u_f_bc_dofs = dolfinx.fem.locate_dofs_topological(U_f, fluid_mesh.geometry.dim - 1, u_f_bc_facets)
    u_f_bc = dolfinx.fem.dirichletbc(u_f_bc_func, u_f_bc_dofs)

    # Create solid Dirichlet boundary condition

    u_s_bc_func = dolfinx.fem.Function(U_s)
    u_s_bc_func.x.array[:] = 0.0
    u_s_bc_facets = reduce(
        np.union1d,
        [
            solid_facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
        ],
    )
    u_s_bc_dofs = dolfinx.fem.locate_dofs_topological(U_s, solid_mesh.geometry.dim - 1, u_s_bc_facets)
    u_s_bc = dolfinx.fem.dirichletbc(u_s_bc_func, u_s_bc_dofs)

    # Create Dirichlet boundary condition for restricting test functions.

    total_interface_facets_found = mesh.comm.allreduce(interface_facets.size, op=MPI.SUM)
    assert total_interface_facets_found > 0, "Interface not found."

    # Collect Dirichlet boundary conditions to pass to NonlinearProblem

    bcs = [inflow_bc, noslip_bc]

    # DESCRIBE FSI PROBLEM
    # FLUID: Parabolic inflow on left side, no-slip on top, bottom, and obstacle, do-nothing on right side
    # SOLID: Homogeneous Dirichlet on left side

    from xfsi_solver.fsi.materials import Fluid, Solid

    # create residual form with u condensed out.

    def A_T_mesh_transport(v, v_old, u_f, u_f_old):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = -rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * ((u_f - u_f_old) / dt), dv) * dx_fluid

        return residual

    def A_T_mass(v, v_old, u_f, u_f_old):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)
        F_old = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f_old)
        J_old = ufl.det(F_old)
        J_mid = 0.5 * (J + J_old)

        residual = rho_f * J_mid * ufl.inner((v - v_old) / dt, dv) * dx_fluid

        residual += rho_s * ufl.inner((v - v_old) / dt, dv) * dx_solid

        return residual

    def A_T(v, v_old, u_f, u_f_old):
        return A_T_mass(v, v_old, u_f, u_f_old) + A_T_mesh_transport(v, v_old, u_f, u_f_old)

    def A_I(v, u_f):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = ufl.div(J * ufl.inv(F) * v) * dp * dx_fluid

        return residual

    def A_E_fluid_viscous(v, u_f):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = ufl.inner(J * Fluid.NS_velocity(u_f, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    def A_E_fluid_transport(v, u_f):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx_fluid

        return residual

    def A_E_fluid(v, u_f):

        return A_E_fluid_viscous(v, u_f) + A_E_fluid_transport(v, u_f)

    def A_E_solid(u_s):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_s)
        J = ufl.det(F)

        residual = ufl.inner(J * Solid.STVK(u_s, lambda_s, mu_s) * ufl.inv(F).T, ufl.grad(dv)) * dx_solid

        return residual

    def A_P(p, u_f):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    residual = A_T(v, v_old, u_f, u_f_old)
    residual += A_I(v, u_f)
    residual += A_P(p, u_f)
    residual += theta * A_E_fluid(v, u_f)
    residual += (1.0 - theta) * A_E_fluid(v_old, u_f_old)

    u_s_condensed = u_s_old + dt * ((1.0 - theta) * v_old + theta * v)
    residual += theta * A_E_solid(u_s_condensed)
    residual += (1.0 - theta) * A_E_solid(u_s_old)

    residual_blocks = ufl.extract_blocks(residual)
    # Cap the quadrature degree before taking derivatives, so that the Jacobian blocks keep it.
    residual_blocks = [cap_quadrature_degree(form, max_quadrature_degree) for form in residual_blocks]

    jacobian = dolfinx.fem.forms.derivative_block(residual_blocks, (v, p))

    # --------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])

    # The fieldsplit preconditioner takes the (v, v) and (p, p) blocks of the Jacobian, which a nest
    # matrix holds as separate matrices; LU needs the Jacobian as one matrix.
    use_fieldsplit = solver_options.get("pc_type") == "fieldsplit"
    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocks,
        (v, p),
        J=jacobian,
        bcs=bcs,
        petsc_options_prefix="solver_",
        entity_maps=entity_maps,
        kind="nest" if use_fieldsplit else None,
        petsc_options=solver_options,
    )

    if use_fieldsplit:
        # NonlinearProblem gives the PC the nest index sets only if it has a preconditioner form, and
        # then names the splits v_0 and p_1, so set them here with the names of the fieldsplit_v_ and
        # fieldsplit_p_ options.
        is_v, is_p = problem.A.getNestISs()[0]
        # The owned v dofs are stored interleaved per node, (x, y), so A has block size 2, which GAMG
        # uses for its aggregates. PCFIELDSPLIT attaches the near-nullspace to A when it extracts it.
        is_v.setBlockSize(V.dofmap.index_map_bs)
        is_v.compose("nearnullspace", rigid_body_modes(V))
        problem.solver.getKSP().getPC().setFieldSplitIS(("v", is_v), ("p", is_p))

        # The fieldsplit calls setFromOptions on its sub-KSPs in PCSetUp, in the first solve, after
        # dolfinx means to have deleted the options in petsc_options. Its deletion, under a pushed
        # prefix, has no effect with petsc4py 3.25, so they stay; set those of the sub-KSPs again, so
        # that this does not depend on it.
        opts = PETSc.Options()
        opts.prefixPush(problem.solver.getOptionsPrefix())
        for key, value in solver_options.items():
            if key.startswith("fieldsplit_"):
                opts[key] = value
        opts.prefixPop()

    # The linear solve for the end-of-run summary, built here so that options without the keys it
    # reads fail before the time loop, not after it.
    options = solver_options
    if use_fieldsplit:
        gamg_description = (
            "GAMG built from A (rigid body modes as near-nullspace),\n"
            f"      {options['fieldsplit_v_mg_levels_ksp_type']} with "
            f"{options['fieldsplit_v_mg_levels_pc_type']} smoothing"
        )
        if "fieldsplit_v_mg_levels_pc_sor_omega" in options:
            gamg_description += f", omega = {options['fieldsplit_v_mg_levels_pc_sor_omega']}"
        p_pc_description = options["fieldsplit_p_pc_type"]
        if p_pc_description == "hypre":
            p_pc_description += f" {options.get('fieldsplit_p_pc_hypre_type', 'boomeramg')}"  # the default type
        linear_solve_description = (
            f"{options['ksp_type'].upper()} (rtol {options['ksp_rtol']:g}) on the nest Jacobian, "
            f"preconditioned by the {options['pc_fieldsplit_schur_fact_type']} Schur\n"
            f"    factor, one V-cycle of {gamg_description}, for A^{{-1}},\n"
            f"    one V-cycle of {p_pc_description} on -B diag(A)^{{-1}} G "
            f"({options['pc_fieldsplit_schur_precondition']}), for S^{{-1}}\n"
        )
    else:
        linear_solve_description = f"{options['pc_factor_mat_solver_type'].upper()} {options['pc_type'].upper()}\n"

    # Add extra callbacks to change how residual and jacobian is assembled,
    # to account for test function restriction on the interface.

    solver = problem.solver
    b_vec, (fem_residual, res_args, res_kargs) = solver.getFunction()
    J_mat, P_mat, (fem_jacobian, jac_args, jac_kargs) = solver.getJacobian()

    # State of the Jacobian reuse in wrapped_jacobian_approximate. previous_norm is the
    # residual norm at the previous Newton iterate of the current time step.
    jacobian_state = {
        "assembled": False,
        "previous_norm": None,
        "num_assemblies": 0,
    }

    def wrapped_jacobian(snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat) -> None:
        norm = snes.getFunctionNorm()  # residual norm at the current iterate x
        previous_norm = jacobian_state["previous_norm"] if snes.getIterationNumber() > 0 else None
        jacobian_state["previous_norm"] = norm
        # Reuse J if the last Newton iteration converged fast enough, or at the first iteration
        # of a time step, where there is no rate yet. Leaving J unchanged makes PETSc skip
        # PCSetUp, so the fieldsplit submatrices and the factorization of K_c are reused too.
        if jacobian_state["assembled"] and (previous_norm is None or norm <= gamma_reassemble * previous_norm):
            return
        fem_jacobian(snes, x, J, P, *jac_args, **jac_kargs)
        jacobian_state["assembled"] = True
        jacobian_state["num_assemblies"] += 1

    solver.setJacobian(wrapped_jacobian, J_mat, P_mat)

    # The solid displacement is updated node by node by the kinematic relation. U_s and V have the same
    # element, so the dofs of a solid cell are in the same order in both, and the dofmaps of the solid
    # cells give the dof of v at each local dof of u_s.
    if elements["U"] != elements["V"]:
        raise ValueError(
            f"The nodal update needs the same element for U and V, not {elements['U']} and {elements['V']}"
        )
    solid_cells, fluid_cells = all_cells(solid_mesh), all_cells(fluid_mesh)
    solid_parent_cells = solid_cell_map.sub_topology_to_topology(solid_cells, inverse=False)
    num_u_s_dofs = U_s.dofmap.index_map.size_local + U_s.dofmap.index_map.num_ghosts
    v_dof_of_u_s = np.empty(num_u_s_dofs, dtype=np.int32)
    v_dof_of_u_s[U_s.dofmap.list[solid_cells]] = V.dofmap.list[solid_parent_cells]
    block_size_u = U_s.dofmap.index_map_bs
    v_dof_of_u_s = (block_size_u * v_dof_of_u_s[:, None] + np.arange(block_size_u)).ravel()  # unrolled

    # The mesh motion is linear, with a matrix that is the same for the whole run, so it is assembled
    # once, and by default factorized once, in the first solve.
    mesh_motion = mesh_motion_type(u_f, [u_f_bc], solver_options=mesh_motion_options)
    mesh_motion_name = mesh_motion_type.__name__.removesuffix("_mesh_motion")
    if mesh_motion_options is None:
        mesh_motion_description = (
            f"{mesh_motion_name}, MUMPS LU on the x component, once for the whole run, for both\n"
            "    components at once by MatMatSolve"
        )
    else:
        mesh_motion_description = f"{mesh_motion_name}, with the KSP options {mesh_motion_options}"

    num_v = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    # The Dirichlet values of u_f are u_s on the interface and 0 on the rest of the fluid boundary.
    # interpolate_nonmatching locates the points across the ranks, so it does not depend on which
    # solid cells a rank holds, and gives 0 at the points outside the solid. The meshes do not move,
    # so the search is done once. It writes only the dofs of the cells it is given, so these are all
    # local fluid cells with an interface dof, including cells that touch the interface only at a
    # vertex: locate_dofs_topological finds the interface dofs also on ranks that hold them only
    # through such cells (remote=True), which no interface facet of the rank would reach.
    fluid_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
        facet_tags, fluid_mesh, fluid_vertex_map, fluid_cell_map
    )
    fluid_interface_dofs = dolfinx.fem.locate_dofs_topological(
        U_f, fluid_mesh.topology.dim - 1, fluid_facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"]), remote=True
    )
    interface_cells = fluid_cells[np.isin(U_f.dofmap.list[fluid_cells], fluid_interface_dofs).any(axis=1)]
    interface_data = dolfinx.fem.create_interpolation_data(U_f, U_s, interface_cells, padding=1e-8)

    def post_velocity_solve_update(ksp: PETSc.KSP, b: PETSc.Vec, x: PETSc.Vec) -> None:
        """Update the displacements to the next Newton iterate. Called at the end of the linear solve,
        where x is the Newton update Y of (v, p) and v still holds the current iterate, before SNES
        sets the next iterate to v - Y and evaluates the residual there."""
        # The owned v rows come first in x, or are its first block if x is a nest vector.
        x_v = x.getNestSubVecs()[0] if x.getType() == PETSc.Vec.Type.NEST else x
        v.x.array[:num_v] -= x_v.getArray(readonly=True)[:num_v]
        v.x.scatter_forward()

        # u_s = u_s_old + dt * (theta * v + (1 - theta) * v_old), at the owned and ghost dofs. The
        # Dirichlet values hold already, since v and v_old vanish there, and are set again exactly.
        u_s.x.array[:] = u_s_old.x.array + dt_val * (
            theta_val * v.x.array[v_dof_of_u_s] + (1.0 - theta_val) * v_old.x.array[v_dof_of_u_s]
        )
        u_s_bc.set(u_s.x.array)

        u_f_bc_func.interpolate_nonmatching(u_s, interface_cells, interface_data)
        u_f_bc_func.x.scatter_forward()

        mesh_motion.solve()

    # A KSP has one post-solve hook, so KSPConvCheck, which sets its own, cannot be attached to this KSP.
    # The K_c solve is a direct one, and check_converged reports a failure of it through the SNES.
    solver.getKSP().setPostSolve(post_velocity_solve_update)

    # Open qoi file on fresh runs, discard entries after the restart time if a restarted run.
    init_qoi_file(qoi_path, comm, restart, t, dt_val)

    if not restart:
        write_output(t, step)

    PETSc.Sys.Print()
    mesh.comm.barrier()

    first_step = step
    num_newton_iterations = 0
    start = timer()
    while step < num_steps:
        t = t0 + (step + 1) * dt_val
        inflow_bc_func.interpolate(InflowFunc(t))
        inflow_bc_func.x.scatter_forward()

        v_old.x.array[:] = v.x.array[:]
        u_s_old.x.array[:] = u_s.x.array[:]
        u_f_old.x.array[:] = u_f.x.array[:]

        PETSc.Sys.Print(f"\n\n{t = :.3f}")

        problem.solve()
        num_newton_iterations += solver.getIterationNumber()
        check_converged(problem, f"t = {t:.4f}", writers=writers, ksp_check=None)

        step += 1
        write_output(t, step)

        if checkpoint_every is not None and step % checkpoint_every == 0:
            displacement.gather()
            checkpointer.write(checkpoint_fields, t, step, dt_val, t0)

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")
        print(
            "\nSolver setup:\n"
            f"  Quadrature degree: {quadrature_description(max_quadrature_degree)}\n"
            "  Solid stress: in terms of v, via u_s = u_s_old + dt * (theta * v + (1 - theta) * v_old)\n"
            "  Newton system: (v, p) only; u_s set node by node and u_f by the mesh motion after each\n"
            "    linear solve\n"
            f"  Linear solve: {linear_solve_description}"
            f"  Mesh motion: {mesh_motion_description}\n"
            f"  Jacobian reuse: reassembled when ||R_l|| > gamma_reassemble * ||R_(l-1)||, "
            f"with {gamma_reassemble = }\n"
            f"  Jacobian assemblies: {jacobian_state['num_assemblies']} in {num_newton_iterations} Newton iterations "
            f"over {step - first_step} time steps\n"
        )

    for writer in writers:
        writer.close()

    # Destroying the MUMPS factorization and the AMG hierarchies is collective, so destroy them on
    # all ranks here instead of leaving it to garbage collection, which can run at different points
    # on different ranks.
    solver.destroy()
    mesh_motion.destroy()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument("--initial-state-dir", help="start from the latest checkpoint of another run in this directory")
    args = parser.parse_args()

    mesh_path = "data/meshes/fsi2/mesh_sec.xdmf"
    run_name = "fsi2_biharmonic_mixed_twice_simplified"
    dt_val = 0.0025
    T = 16.0 + 10 * dt_val
    output_path = f"output/pv/{run_name}.bp"
    output_path_p = f"output/pv/{run_name}_p.bp"
    qoi_path = f"output/qoi/{run_name}.txt"
    checkpoint_dir = f"output/checkpoints/{run_name}"
    checkpoint_every = 400
    vtx_save_every = None
    gamma_reassemble = 0.2  # Default 0.2.
    max_quadrature_degree = 5  # Default None, no cap. 6 gives the same QoIs, 20% faster.
    solver_options = DEFAULT_DIRECT_SOLVER_OPTIONS
    initial_state_dir = args.initial_state_dir
    restart = args.restart
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
        gamma_reassemble=gamma_reassemble,  # Default 0.2.
        max_quadrature_degree=max_quadrature_degree,  # Default None, no cap. 6 gives the same QoIs, 20% faster.
        solver_options=solver_options,
        restart=restart,
        initial_state_dir=initial_state_dir,
    )

    if MPI.COMM_WORLD.rank == 0:
        plot_run_qois(qoi_path, T)


if __name__ == "__main__":
    main()
