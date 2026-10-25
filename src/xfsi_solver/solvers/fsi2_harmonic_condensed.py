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

from xfsi_solver.tools.checkpoint import Checkpointer, check_start_options, restart_output_path, start_time
from xfsi_solver.tools.convergence import KSPConvCheck, check_converged
from xfsi_solver.tools.parameters import lame_lambda, theta_value
from xfsi_solver.tools.petsc_reuse import IdenticalBlocksPC
from xfsi_solver.tools.qoi import (
    append_qoi_row,
    assemble_force,
    drag_lift_forms,
    find_point_dof,
    init_qoi_file,
    point_value,
)

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
    plot_matrices=False,
    restart=False,
    initial_state_dir=None,
    gamma_reassemble=0.2,
    direct_k_c_solve=False,
):
    """Solve the FSI2 benchmark up to time ``T``, in ``round((T - t0) / dt_val)`` time
    steps, writing the initial state at ``t0`` as the first QoI row and, with
    ``vtx_save_every``, as the first VTX snapshot. With ``vtx_save_every=None``, no VTX
    output is written; otherwise a snapshot is written every ``vtx_save_every`` steps.

    The solid stress in the momentum equation is written in terms of the velocity, with
    the solid displacement given by the kinematic relation u = u_old + dt * (theta * v
    + (1 - theta) * v_old) (Failer & Richter, J. Sci. Comput. 82:28, 2020, eq. (5)). The
    momentum equation then no longer depends on the solid displacement u_S, and its
    derivative with respect to v contains the solid stiffness. The approximate Jacobian
    J_0 drops the remaining derivatives of the momentum and continuity equations with
    respect to u, which are the fluid ALE derivatives, and is block lower triangular in
    (v, p), u_S, u_I. Each Newton step is solved by a multiplicative fieldsplit in that
    order, with MUMPS Cholesky on K_m and CG with Jacobi on A_S. A_S (the solid mass) and
    K_m (the mesh motion) are constant, so their preconditioners are set up once for the
    whole run. K_m does not couple the x and y components, and the components have the
    same matrix, so only the x component is factorized, and both components are solved with
    that factorization at once, by one MatMatSolve with two right-hand sides.

    This is fsi2_biharmonic_mixed.py with harmonic mesh motion instead of biharmonic: the
    mesh equation is (grad u, grad psi)_f on the reference fluid domain, without a scaling or
    an interface flux term, and with the displacement test functions restricted on the
    interface as in the biharmonic solvers. The fluid displacement u_I is the only mesh-motion
    unknown, and K_m = A_II, the vector Laplacian on the fluid-interior dofs, is symmetric
    positive definite (the harmonic case of docs/restricted-iterative-solver.md).

    With ``direct_k_c_solve=True``, the velocity-pressure block K_c is solved with MUMPS
    LU. Otherwise, it is solved by FGMRES, preconditioned by the block upper triangular
    factor of the Schur complement factorization over v and p: one GAMG V-cycle on the
    velocity block A, with the rigid body modes as near-nullspace and SOR smoothing, for
    A^{-1}, and one BoomerAMG V-cycle on the approximation -B diag(A)^{-1} G (selfp) of
    the pressure Schur complement S = -B A^{-1} G, for S^{-1}.

    The approximate Jacobian is only reassembled, and the physical block K_c only
    refactorized, when the Newton residual norm decreased by less than a factor
    ``gamma_reassemble`` in the last iteration (Failer & Richter, J. Sci. Comput. 82:28,
    2020, Sec. 5.3), and is otherwise reused, also across time steps. With
    ``gamma_reassemble=0``, it is reassembled in every Newton iteration except the
    first of each time step.

    ``elements`` gives the finite element of each function space, keyed by the name of the
    space in this solver, as the tuple passed to ``dolfinx.fem.functionspace``: ``"U"`` and
    ``"V"`` for the displacement and the velocity, on the full mesh, and ``"P"`` for the
    pressure, on the fluid submesh. The default is quadratic vector Lagrange elements for ``U``
    and ``V`` and linear Lagrange elements for ``P``.

    The physical constants are the fluid density ``rho_f`` (default 1.0e3), the fluid kinematic
    viscosity ``nu_f`` (1.0e-3), the solid density ``rho_s`` (1.0e4), the solid shear modulus ``mu_s``
    (5.0e5), the mean inflow velocity ``U_bar`` (1.0) and the channel height ``H`` (0.41). The first
    Lamé parameter of the solid is given either directly as ``lambda_s`` or by the Poisson ratio
    ``nu_s``, as 2 mu_s nu_s / (1 - 2 nu_s); with neither given, ``nu_s = 0.4``, and giving both
    raises ``ValueError``. The time stepping parameter is theta = 0.5 + ``absolute_theta_shift``
    or theta = 0.5 + ``relative_theta_shift`` * ``dt_val``; with neither given, the relative
    shift is 1.0, and giving both raises ``ValueError``.

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

    With ``plot_matrices=True``, spy plots and block norm plots of the condensed Jacobian
    (the full Jacobian of this formulation, with u_S condensed out of the solid stress) and
    the approximate Jacobian J_0 at the final state are saved in
    ``output/figures/matrix_blocks/fsi2_harmonic_condensed/``, with the dofs ordered by the
    storage blocks [u, v, p], by the blocks (u_S, v, p, u_I) of the harmonic version of
    eq. (1) of docs/restricted-iterative-solver.md, and in the order (v, p, u_S, u_I) in which the
    fieldsplit solves them. The matrices are gathered on rank 0.
    """
    check_start_options(restart, checkpoint_dir, checkpoint_every, initial_state_dir)
    theta_val = theta_value(dt_val, absolute_theta_shift, relative_theta_shift, 1.0)
    if nu_s is None and lambda_s is None:
        nu_s = 0.4
    lambda_s_val = lame_lambda(mu_s, nu_s, lambda_s)

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

    u, v, p = (
        dolfinx.fem.Function(U, name="u"),
        dolfinx.fem.Function(V, name="v"),
        dolfinx.fem.Function(P, name="p"),
    )
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

    # Create fluid ALE Dirichlet boundary condition

    u_f_bc_func = dolfinx.fem.Function(U)
    u_f_bc_func.x.array[:] = 0.0
    u_f_bc_facets = reduce(
        np.union1d,
        [
            facet_tags.find(PHYSICAL_MARKERS["inflow"]),
            facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
            facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
            facet_tags.find(PHYSICAL_MARKERS["outflow"]),
        ],
    )
    u_f_bc_dofs = dolfinx.fem.locate_dofs_topological(U, mesh.geometry.dim - 1, u_f_bc_facets)
    u_f_bc = dolfinx.fem.dirichletbc(u_f_bc_func, u_f_bc_dofs)

    # Create solid Dirichlet boundary condition

    u_s_bc_func = dolfinx.fem.Function(U)
    u_s_bc_func.x.array[:] = 0.0
    u_s_bc_facets = reduce(
        np.union1d,
        [
            facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
        ],
    )
    u_s_bc_dofs = dolfinx.fem.locate_dofs_topological(U, mesh.geometry.dim - 1, u_s_bc_facets)
    u_s_bc = dolfinx.fem.dirichletbc(u_s_bc_func, u_s_bc_dofs)

    # Create Dirichlet boundary condition for restricting test functions.

    total_interface_facets_found = mesh.comm.allreduce(interface_facets.size, op=MPI.SUM)
    assert total_interface_facets_found > 0, "Interface not found."
    dofs_interface = dolfinx.fem.locate_dofs_topological(U, mesh.topology.dim - 1, interface_facets)
    bc_deactivate = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, (0.0, 0.0)), dofs_interface, U)

    # Collect Dirichlet boundary conditions to pass to NonlinearProblem

    bcs = [u_f_bc, inflow_bc, noslip_bc]

    # DESCRIBE FSI PROBLEM
    # FLUID: Parabolic inflow on left side, no-slip on top, bottom, and obstacle, do-nothing on right side
    # SOLID: Homogeneous Dirichlet on left side

    from xfsi_solver.fsi.materials import Fluid, Solid

    # create residual form with u condensed out.

    def A_T_mesh_transport(u, u_old, v, v_old):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = -rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * ((u - u_old) / dt), dv) * dx_fluid

        return residual

    def A_T_mass(u, u_old, v, v_old):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)
        F_old = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_old)
        J_old = ufl.det(F_old)
        J_mid = 0.5 * (J + J_old)

        residual = rho_f * J_mid * ufl.inner((v - v_old) / dt, dv) * dx_fluid

        residual += rho_s * ufl.inner((v - v_old) / dt, dv) * dx_solid

        return residual

    def A_T(u, u_old, v, v_old):
        return A_T_mass(u, u_old, v, v_old) + A_T_mesh_transport(u, u_old, v, v_old)

    def A_I(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        # Harmonic mesh motion on the reference fluid domain. The interface rows of du are
        # restricted, so no interface flux term is needed.
        residual = ufl.inner(ufl.grad(u), ufl.grad(du)) * dx_fluid

        residual += ufl.div(J * ufl.inv(F) * v) * dp * dx_fluid

        return residual

    def A_E_fluid_viscous(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = ufl.inner(J * Fluid.NS_velocity(u, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    def A_E_fluid_transport(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx_fluid

        return residual

    def A_E_fluid(u, v):

        return A_E_fluid_viscous(u, v) + A_E_fluid_transport(u, v)

    def A_E_solid(u):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = ufl.inner(J * Solid.STVK(u, lambda_s, mu_s) * ufl.inv(F).T, ufl.grad(dv)) * dx_solid

        return residual

    def A_P(u, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    residual = A_T(u, u_old, v, v_old)
    residual += A_I(u, v)
    residual += A_P(u, p)
    residual += theta * A_E_fluid(u, v)
    residual += (1.0 - theta) * A_E_fluid(u_old, v_old)

    u_condensed = u_old + dt * ((1.0 - theta) * v_old + theta * v)
    residual += theta * A_E_solid(u_condensed)
    residual += (1.0 - theta) * A_E_solid(u_old)

    # Add in u_s with zero constants for correct sparsity patterns
    residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(u, du) * dx_solid
    residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(v, du) * dx_solid

    # --------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])

    residual_base_ufl = ufl.extract_blocks(residual)
    jacobian_full_base_ufl = dolfinx.fem.forms.derivative_block(residual_base_ufl, [u, v, p])

    # One block per unknown in [u, v, p], since blocked assembly places forms by
    # position. Not built with ufl.extract_blocks, which drops the zero blocks. The
    # zero Jacobian rows are built directly, since ufl.derivative does not accept a
    # ufl.ZeroBaseForm, with only their diagonal blocks as zero forms, since a zero
    # form cannot couple the parent mesh and the fluid submesh.
    residual_post_u = rho_s * ufl.inner((u - u_old) / dt - theta * v - (1 - theta) * v_old, du) * dx_solid
    trials = [ufl.TrialFunction(w.function_space) for w in [u, v, p]]

    residual_post_ufl = [residual_post_u, ufl.ZeroBaseForm((dv,)), ufl.ZeroBaseForm((dp,))]

    jacobian_post_ufl = [[None for _ in range(len([u, v, p]))] for __ in range(len([u, v, p]))]
    for j in range(len([u, v, p])):
        # Derivatives w.r.t. missing residual ufl expression.
        jacobian_post_ufl[0][j] = ufl.derivative(residual_post_u, [u, v, p][j], trials[j])
    for i in range(1, len([u, v, p])):
        # Zero blocks on the diagonal elsewhere
        jacobian_post_ufl[i][i] = ufl.ZeroBaseForm(([du, dv, dp][i], trials[i]))

    residual_post = dolfinx.fem.form(residual_post_ufl, entity_maps=entity_maps)
    jacobian_post = dolfinx.fem.form(jacobian_post_ufl, entity_maps=entity_maps)

    jacobian_approximate_base_ufl = [
        [jacobian_full_base_ufl[row][column] for column in range(len([u, v, p]))] for row in range(len([u, v, p]))
    ]

    # Neglect the terms depending on u_f in v and p.
    # The neglecting is really in terms of fluid interior
    # and fluid interface dofs of u, but we don't have that distinction here,
    # so we do it by hand.

    # Neglect the terms for v w.r.t. u. E_S is no longer dependent on u because of the condensation.
    jacobian_approximate_base_ufl[1][0] = None

    # Neglect terms for p w.r.t. u.
    jacobian_approximate_base_ufl[2][0] = None

    # The approximate jacobian forms for the second assembly are the same as for non-approximate case.
    # Therefore, we make only the one jacobian_post-form.

    # Set up output, checkpointing, and qoi tracking.

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
        vtx_message = "" if vtx_save_every is None else f", writing VTX output to {output_path} and {output_path_p}"
        PETSc.Sys.Print(f"Restarting from {checkpoint_dir} at {t = :.4f} ({step = }){vtx_message}")
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

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_base_ufl,
        [u, v, p],
        J=jacobian_approximate_base_ufl,
        bcs=bcs,
        petsc_options_prefix="solver_",
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
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative",
            # Run the fieldsplit on the approximated Jacobian instead of the supplied preconditioner.
            "pc_fieldsplit_diag_use_amat": True,
            "pc_fieldsplit_off_diag_use_amat": True,
            # Turn off errors, catch manually instead.
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
            # Print to console.
            # "snes_monitor": None,
            "snes_converged_reason": None,
            # "snes_monitor": "ascii:output/logs/fsi2_harmonic_condensed_snes_log.txt",
        },
    )

    # Add extra callbacks to change how residual and jacobian is assembled,
    # to account for test function restriction on the interface.

    solver = problem.solver
    b_vec, (fem_residual, res_args, res_kargs) = solver.getFunction()
    J_mat, P_mat, (fem_jacobian, jac_args, jac_kargs) = solver.getJacobian()

    # Prevent possibly overwriting the sparsity pattern on zeroRows.
    problem.A.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)

    # Global rows of the interface u-dofs owned by this rank, for zeroing with zeroRows.
    # Each rank stores its owned dofs as [u | v | p], starting at its first global row.
    # offsets_owned[k] is where field k starts within that local part (so offsets_owned[0] = 0
    # for u). Every interface dof is owned by a rank that also locates it, so zeroing the owned
    # rows on each rank covers all of them.
    interface_dofs, num_owned = bc_deactivate.dof_indices()  # unrolled local indices, owned first
    offsets_owned, _ = problem.b.getAttr("_blocks")
    first_row = J_mat.getOwnershipRange()[0]
    u_block = 0
    rows_d = (first_row + offsets_owned[u_block] + interface_dofs[:num_owned]).astype(PETSc.IntType)

    bcs_post = [u_s_bc, *bcs]
    bcs_rows = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(residual_post), bcs_post)
    bcs_cols = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(jacobian_post, 1), bcs_post)

    def pre_jacobian(x: PETSc.Vec, J: PETSc.Mat) -> None:
        pass

    def post_jacobian(x: PETSc.Vec, J: PETSc.Mat) -> None:
        J.zeroRows(rows_d, diag=0.0)
        dolfinx.fem.petsc.assemble_matrix(J, jacobian_post, bcs=[u_s_bc, *bcs])
        J.assemble()

    def pre_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:
        pass

    def post_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:

        dolfinx.fem.petsc.set_bc(b, [[bc_deactivate], [], []], alpha=0.0)  # zero restricted rows

        with b.localForm() as bl:
            # Remove ghost-entries.
            bl.array[b.getLocalSize() :] = 0.0

        dolfinx.fem.petsc.assemble_vector(
            b,
            residual_post,
        )
        dolfinx.fem.petsc.apply_lifting(b, jacobian_post, bcs=bcs_cols, x0=x, alpha=-1.0)

        b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)

        dolfinx.fem.petsc.set_bc(b, bcs_rows, x0=x, alpha=-1.0)

        b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        pass

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
        pre_jacobian(x, J)
        fem_jacobian(snes, x, J, P, *jac_args, **jac_kargs)
        post_jacobian(x, J)
        jacobian_state["assembled"] = True
        jacobian_state["num_assemblies"] += 1

    def wrapped_residual(snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        pre_residual(x, b)  # x is not yet assigned to u here
        fem_residual(snes, x, b, *res_args, **res_kargs)
        post_residual(x, b)  # u is now updated and b assembled (BCs applied)

    solver.setJacobian(wrapped_jacobian, J_mat, P_mat)
    solver.setFunction(wrapped_residual, b_vec)

    # Helper function to retrieve the dofs that are supported on solid cells. Don't know
    # what is the most appropriate location of this function for legibility.
    def solid_supported(space, num_dofs):
        """Mask of the owned unrolled dofs of space that are supported on solid cells."""
        bs = space.dofmap.index_map_bs
        dofs = dolfinx.fem.locate_dofs_topological(space, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"]))
        return np.isin(np.arange(num_dofs), (bs * dofs[:, None] + np.arange(bs)).ravel())

    def owned_rows(k):
        """Global rows of the owned dofs of block k in [u, v, p]."""
        return first_row + np.arange(offsets_owned[k], offsets_owned[k + 1])

    rows_u, rows_v, rows_p = (owned_rows(k) for k in range(3))
    u_solid = solid_supported(U, len(rows_u))  # mask over owned unrolled u dofs
    rows_q_c = np.concatenate([rows_v, rows_p])
    rows_u_S = rows_u[u_solid]
    rows_m = rows_u[~u_solid]

    if not len(rows_q_c) + len(rows_u_S) + len(rows_m) == J_mat.getLocalSize()[0]:
        raise RuntimeError()

    def make_is(rows):
        return PETSc.IS().createGeneral(np.sort(rows).astype(PETSc.IntType), comm=comm)

    start = comm.exscan(len(rows_q_c)) or 0  # first row of this rank in the q_c submatrix
    is_v = make_is(start + np.arange(len(rows_v)))
    is_p = make_is(start + len(rows_v) + np.arange(len(rows_p)))

    # Rows of each displacement component in the m submatrix, for solving K_m component by
    # component. The owned u dofs are stored interleaved per node, (x, y), so the
    # component of a dof is its position in its block modulo block_size. The m submatrix holds the
    # rows of rows_m in sorted order.
    block_size = U.dofmap.bs
    component_u_I = (np.arange(len(rows_u)) % block_size)[~u_solid]
    components_m = component_u_I[np.argsort(rows_m)]
    start_m = comm.exscan(len(rows_m)) or 0  # first row of this rank in the m submatrix
    is_m_components = [make_is(start_m + np.flatnonzero(components_m == c)) for c in range(block_size)]

    # The owned v dofs are stored interleaved per node, (x, y), so A has block size 2, which
    # BoomerAMG uses for its systems coarsening and GAMG for its aggregates.
    is_v.setBlockSize(V.dofmap.index_map_bs)

    def rigid_body_modes(space):
        """Orthonormal rigid body modes of a 2D vector space: two translations and a rotation."""
        index_map, bs = space.dofmap.index_map, space.dofmap.index_map_bs
        num_dofs = index_map.size_local + index_map.num_ghosts
        x = space.tabulate_dof_coordinates()[:num_dofs]
        basis = [dolfinx.la.vector(index_map, bs=bs, dtype=PETSc.ScalarType) for _ in range(3)]
        b = [w.array.reshape(-1, bs) for w in basis]
        b[0][:, 0] = 1.0
        b[1][:, 1] = 1.0
        b[2][:, 0], b[2][:, 1] = -x[:, 1], x[:, 0]
        dolfinx.la.orthonormalize(basis)
        return PETSc.NullSpace().create(vectors=[dolfinx.la.petsc.create_vector_wrap(w) for w in basis])

    # PCFIELDSPLIT attaches this to the velocity block A when it extracts it.
    is_v.compose("nearnullspace", rigid_body_modes(V))

    # The multiplicative fieldsplit solves the splits in the order they are set here,
    # independent of the storage order [u | v | p]: first K_c for (v, p), then A_S for
    # u_S with the right-hand side updated by C delta v, then K_m for u_I with the
    # right-hand side updated by L delta u_S.
    pc = solver.getKSP().getPC()
    pc.setFieldSplitIS(("q_c", make_is(rows_q_c)), ("u_S", make_is(rows_u_S)), ("m", make_is(rows_m)))

    if direct_k_c_solve:
        velocity_pressure_solve_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_cntl_1": 1e-4,
        }

    else:
        # FGMRES on K_c, preconditioned by the block upper triangular [[A, G], [0, S]], with one GAMG
        # V-cycle for A^{-1} and one BoomerAMG V-cycle on \hat S for S^{-1}. Each iteration costs two
        # V-cycles, instead of solving with A in every application of S. Only the tolerance of the outer
        # FGMRES matters.
        velocity_pressure_solve_options = {
            "ksp_converged_reason": None,
            "ksp_type": "fgmres",
            "ksp_rtol": 1e-5,
            "ksp_max_it": 200,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "upper",
            # The pressure Schur complement S = -B A^{-1} G is dense and only applied as an operator, so it
            # is preconditioned by the sparse approximation \hat S = -B diag(A)^{-1} G. The pressure term
            # gives G = -B^T (up to quadrature, by the Piola identity), so \hat S = B diag(A)^{-1} B^T is a
            # divergence of a gradient: close to a pressure Laplacian, scaled by about dt / rho_f, since the
            # inertia term rho_f / dt dominates diag(A) at small time steps.
            "pc_fieldsplit_schur_precondition": "selfp",
            # One GAMG V-cycle on v, with the near-nullspace and block size attached to is_v.
            "fieldsplit_v_ksp_type": "preonly",
            "fieldsplit_v_pc_type": "gamg",
            "fieldsplit_v_pc_gamg_threshold": 0.01,
            "fieldsplit_v_pc_gamg_reuse_interpolation": True,
            "fieldsplit_v_mg_levels_ksp_type": "richardson",
            "fieldsplit_v_mg_levels_pc_type": "sor",
            # "fieldsplit_v_ksp_converged_reason": None,
            # One BoomerAMG V-cycle on \hat S.
            "fieldsplit_p_ksp_type": "preonly",
            "fieldsplit_p_pc_type": "hypre",
            "fieldsplit_p_pc_hypre_type": "boomeramg",
            # "fieldsplit_p_ksp_converged_reason": None,
        }

    # A_S is a symmetric positive definite solid mass matrix, which CG with Jacobi solves to
    # round-off in about 30 iterations. That is cheaper than a MUMPS solve on many ranks.
    # A_S is the same matrix in every Newton iteration and time step, so its preconditioner
    # is set up only once.
    mass_solve_options = {
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "ksp_rtol": 1e-12,
        "ksp_atol": 1e-50,
        "ksp_reuse_preconditioner": True,
    }

    # K_m is assembled from linear forms on the reference fluid mesh, with fixed Dirichlet
    # rows, so it is the same matrix in every Newton iteration and time step: factorize it
    # only once. K_m does not couple the x and y components, and the components have the same
    # matrix, so only the x component is factorized. IdenticalBlocksPC solves for both
    # components at once with that factorization, by MatMatSolve, which reads the factor once
    # instead of twice.
    # The harmonic K_m = A_II is symmetric positive definite, so it is factorized by Cholesky, as
    # the C0IP K_m of fsi2_biharmonic_c0ip.py.
    mesh_motion_solve_options = {
        "ksp_type": "preonly",
        "ksp_reuse_preconditioner": True,
        "pc_type": "python",  # IdenticalBlocksPC
        "block_pc_type": "cholesky",
        "block_pc_factor_mat_solver_type": "mumps",
        # "block_mat_mumps_icntl_14": 80,  # if the factorization runs out of MUMPS workspace
        "block_mat_mumps_cntl_1": 1e-4,
    }

    # Set through the options database, since the fieldsplit calls setFromOptions on its
    # sub-KSPs in PCSetUp, which would override types set directly on them.
    split_options = {"q_c": velocity_pressure_solve_options, "u_S": mass_solve_options, "m": mesh_motion_solve_options}
    opts = PETSc.Options()
    # List the options that were set but never used when PETSc finalizes, which catches misspelled
    # option names. A global option without a prefix, so not in a petsc_options dict, whose options
    # dolfinx prefixes and deletes after setting up the solver.
    opts["options_left"] = None
    for options, sub_ksp in zip(split_options.values(), pc.getFieldSplitSubKSP(), strict=True):
        opts.prefixPush(sub_ksp.getOptionsPrefix())  # <prefix>fieldsplit_q_c_ / _u_S_ / _m_
        for key, value in options.items():
            opts[key] = value
        opts.prefixPop()
        sub_ksp.setFromOptions()

    ksp_q_c, ksp_u_S, ksp_m = pc.getFieldSplitSubKSP()
    if not direct_k_c_solve:
        ksp_q_c.getPC().setFieldSplitIS(("v", is_v), ("p", is_p))

    mesh_motion_pc = IdenticalBlocksPC(is_m_components)
    ksp_m.getPC().setPythonContext(mesh_motion_pc)

    # Records failed linear solves nested in the fieldsplits, such as a K_c FGMRES that reaches its
    # iteration limit, which neither PETSc nor the SNES converged reason report.
    ksp_check = KSPConvCheck(solver.getKSP())

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
            checkpointer.write([u, v, p], t, step, dt_val, t0)

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")
        if direct_k_c_solve:
            k_c_solve_description = "MUMPS LU\n"
        else:
            options = velocity_pressure_solve_options
            gamg_description = (
                "GAMG built from A (rigid body modes as near-nullspace),\n"
                f"      {options['fieldsplit_v_mg_levels_ksp_type']} with "
                f"{options['fieldsplit_v_mg_levels_pc_type']} smoothing"
            )
            if "fieldsplit_v_mg_levels_pc_sor_omega" in options:
                gamg_description += f", omega = {options['fieldsplit_v_mg_levels_pc_sor_omega']}"
            k_c_solve_description = (
                f"FGMRES (rtol {options['ksp_rtol']:g}), preconditioned by the block upper triangular Schur\n"
                f"    factor, one V-cycle of {gamg_description}, for A^{{-1}},\n"
                "    one V-cycle of BoomerAMG on -B diag(A)^{-1} G (selfp), for S^{-1}\n"
            )
        print(
            "\nSolver setup:\n"
            "  Solid stress: in terms of v, via u = u_old + dt * (theta * v + (1 - theta) * v_old)\n"
            "  Jacobian: approximate J_0, without the fluid ALE derivatives (eq. (3) of\n"
            "    docs/restricted-iterative-solver.md)\n"
            "  Linear solve: multiplicative fieldsplit over q_c = (v, p), u_S and m = u_I,\n"
            "    MUMPS Cholesky on the x component of the harmonic K_m = A_II, for the whole run and for both\n"
            "    components at once by MatMatSolve,\n"
            "    CG with Jacobi on A_S\n"
            f"  K_c solve: {k_c_solve_description}"
            f"  Jacobian reuse: reassembled when ||R_l|| > gamma_reassemble * ||R_(l-1)||, "
            f"with {gamma_reassemble = }\n"
            f"  Jacobian assemblies: {jacobian_state['num_assemblies']} in {num_newton_iterations} Newton iterations "
            f"over {step - first_step} time steps\n"
        )

    for writer in writers:
        writer.close()

    if plot_matrices:
        from xfsi_solver.tools.matrix_blocks import matrix_blocks_dir, plot_matrix_blocks

        # The full Jacobian, and J_0, at the final state, assembled into new matrices with the
        # sparsity pattern of the full Jacobian, which contains that of J_0. J_mat has no room for
        # the full Jacobian, since it is created from the forms of J_0.
        jacobian_full_base = dolfinx.fem.form(jacobian_full_base_ufl, entity_maps=entity_maps)
        dolfinx.fem.petsc.assign([u, v, p], problem.x)
        partitions = {
            "storage": ([rows_u, rows_v, rows_p], ["$u$", "$v$", "$p$"]),
            "eq1": ([rows_u_S, rows_v, rows_p, rows_u[~u_solid]], ["$u_S$", "$v$", "$p$", "$u_I$"]),
            "split": ([rows_v, rows_p, rows_u_S, rows_u[~u_solid]], ["$v$", "$p$", "$u_S$", "$u_I$"]),
        }
        for label, title, jacobian_forms in [
            ("", "Condensed Jacobian", jacobian_full_base),
            ("approximate_", "Approximate Jacobian", jac_kargs["jacobian"]),
        ]:
            J = dolfinx.fem.petsc.create_matrix(jacobian_full_base)
            J.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
            fem_jacobian(solver, problem.x, J, P_mat, *jac_args, **(jac_kargs | {"jacobian": jacobian_forms}))
            post_jacobian(problem.x, J)
            plot_matrix_blocks(
                J,
                {label + name: partition for name, partition in partitions.items()},
                title,
                matrix_blocks_dir(__file__),
                t,
            )
            J.destroy()

    # Destroying the MUMPS factorization and the AMG hierarchies is collective, so destroy them on
    # all ranks here instead of leaving it to garbage collection, which can run at different points
    # on different ranks.
    solver.destroy()
    mesh_motion_pc.destroy_solvers()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument("--initial-state-dir", help="start from the latest checkpoint of another run in this directory")
    args = parser.parse_args()

    solve(
        mesh_path="data/meshes/fsi2/mesh_sec.xdmf",
        T=7.0,
        dt_val=0.0025,
        output_path="output/pv/fsi2_harmonic_condensed.bp",
        output_path_p="output/pv/fsi2_harmonic_condensed_p.bp",
        qoi_path="output/qoi/fsi2_harmonic_condensed.txt",
        checkpoint_dir="output/checkpoints/fsi2_harmonic_condensed",
        checkpoint_every=400,
        vtx_save_every=None,
        plot_matrices=False,
        gamma_reassemble=0.2,  # Default 0.2.
        direct_k_c_solve=True,
        restart=args.restart,
        initial_state_dir=args.initial_state_dir,
    )


if __name__ == "__main__":
    main()
