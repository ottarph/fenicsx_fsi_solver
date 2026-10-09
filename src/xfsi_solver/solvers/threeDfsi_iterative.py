# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import sys
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

from xfsi_solver.tools.cahouet_chabard import CahouetChabard, outflow_pressure_dofs
from xfsi_solver.tools.convergence import KSPConvCheck, check_converged
from xfsi_solver.tools.qoi import (
    QOI_HEADER_3D,
    append_qoi_row,
    assemble_force,
    drag_lift_forms,
    find_point_dof,
    init_qoi_file,
    point_value,
)
from xfsi_solver.tools.quadrature import cap_quadrature_degree

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
    checkpoint_dir=None,
    checkpoint_every=None,
    vtx_save_every=None,
    restart=False,
    gamma_reassemble=0.2,
    direct_k_c_solve=False,
    block_preconditioned_k_c_solve=False,
    auxiliary_vv_preconditioner=False,
    cahouet_chabard_schur_preconditioner=True,
    max_quadrature_degree=8,
):
    """Solve the 3d FSI benchmark of Failer & Richter (J. Sci. Comput. 82:28, 2020,
    Sect. 5.1.2), an elastic beam attached to a cylinder in a channel of square cross
    section, up to time ``T``, in ``round((T - t0) / dt_val)`` time steps, writing the
    initial state at ``t0`` as the first QoI row and, with ``vtx_save_every``, as the
    first VTX snapshot. With ``vtx_save_every=None``, no VTX output is written; otherwise
    a snapshot is written every ``vtx_save_every`` steps.

    This is the 3d version of the solver in fsi2_biharmonic_diffmesh_restricted_split_condensed.py,
    with the same formulation and the same solver options. The QoIs are the drag and the lift
    on the cylinder and the beam, and the displacement of the point B = (0.9, 0.2, 0.3) on the
    back face of the beam.

    The solid stress in the momentum equation is written in terms of the velocity, with
    the solid displacement given by the kinematic relation u = u_old + dt * (theta * v
    + (1 - theta) * v_old) (Failer & Richter, J. Sci. Comput. 82:28, 2020, eq. (5)). The
    momentum equation then no longer depends on the solid displacement u_S, and its
    derivative with respect to v contains the solid stiffness. The approximate Jacobian
    J_0 drops the remaining derivatives of the momentum and continuity equations with
    respect to u, which are the fluid ALE derivatives, and is block lower triangular in
    (v, p), u_S, (z, u_I). Each Newton step is solved by a multiplicative fieldsplit in
    that order, with MUMPS LU on K_m and CG with Jacobi on A_S. A_S (the solid mass) and
    K_m (the mesh motion) are constant, so their preconditioners are set up once for the
    whole run. K_m does not couple the x, y and z components, so it is solved by an
    additive fieldsplit over the components, with one MUMPS factorization each, which
    leaves out the zero couplings between the components that the blocked matrix stores.

    With ``direct_k_c_solve=True``, the velocity-pressure block K_c is solved with MUMPS
    LU. Otherwise, it is solved with a full Schur complement factorization over v and p:
    GMRES with GAMG on the velocity block A, with the rigid body modes as near-nullspace
    and SOR smoothing, and GMRES on the pressure Schur complement S = -B A^{-1} G,
    preconditioned by BoomerAMG on its approximation -B diag(A)^{-1} G. A is solved to a
    tighter tolerance than S, since every application of S solves with A. With
    ``block_preconditioned_k_c_solve=True``, K_c is instead solved by FGMRES, preconditioned
    by the block upper triangular factor of the Schur factorization, with one GAMG V-cycle
    for A^{-1} and one BoomerAMG V-cycle on -B diag(A)^{-1} G for S^{-1}.

    With ``auxiliary_vv_preconditioner=True``, which requires an iterative K_c solve, the
    GAMG hierarchy for A is built from the symmetric positive definite auxiliary operator
    A_0 of docs/restricted-iterative-solver.md instead of from A: A without fluid
    transport, and with a linear elastic solid. A_0 is the velocity block of a separate
    preconditioning matrix, assembled in addition to the Jacobian at every reassembly.
    GAMG's finest smoother iterates with the velocity KSP's operator. In the
    block-preconditioned solve, the velocity KSP only applies the V-cycle, so its operator
    is A_0 as well, smoothed by Chebyshev with Jacobi. In the full Schur solve, the
    velocity GMRES must solve with A, so the finest smoother iterates with the
    nonsymmetric A, and SOR smoothing is used. Without the auxiliary operator, GAMG is
    built from A with SOR smoothing in both solves. SOR smoothing is damped by 0.5: in
    parallel, PETSc's SOR is Gauss-Seidel within each rank and Jacobi between ranks, and
    undamped it made GAMG diverge for some partitions of the 3d mesh.

    The quadrature degree is capped at ``max_quadrature_degree``: UFL estimates degrees up to
    19 for the nonlinear ALE and St. Venant-Kirchhoff terms in the residual (25 in the
    Jacobian), with 1000 quadrature points per tetrahedron, against 45 points at degree 8.
    Integrals with a lower estimate keep it.

    With ``cahouet_chabard_schur_preconditioner=True``, which requires an iterative K_c
    solve, S^{-1} is approximated by Cahouet-Chabard instead of BoomerAMG on
    -B diag(A)^{-1} G: rho_f / dt * L_p^{-1} + theta * rho_f * nu_f * M_p^{-1}, with the
    pressure Laplacian L_p (p = 0 on the outflow) and the pressure mass matrix M_p on the
    reference fluid mesh, applied with one BoomerAMG V-cycle on L_p and Jacobi on M_p. It
    accounts for both the inertia and the viscous term of A, while -B diag(A)^{-1} G only
    approximates S well while inertia dominates.

    The approximate Jacobian is only reassembled, and the physical block K_c only
    refactorized, when the Newton residual norm decreased by less than a factor
    ``gamma_reassemble`` in the last iteration (Failer & Richter, J. Sci. Comput. 82:28,
    2020, Sec. 5.3), and is otherwise reused, also across time steps. With
    ``gamma_reassemble=0``, it is reassembled in every Newton iteration except the
    first of each time step.

    With ``checkpoint_dir`` and ``checkpoint_every``, a restart checkpoint of
    the state is saved in ``checkpoint_dir`` every ``checkpoint_every`` time
    steps (replacing checkpoints there from earlier runs). With
    ``restart=True``, the run instead continues from the latest checkpoint in
    ``checkpoint_dir``, on any number of MPI ranks: QoI rows after the
    checkpoint time are dropped from ``qoi_path`` before appending, and the
    VTX output is written to new files with a ``_from_t<time>`` suffix, since
    ``VTXWriter`` cannot append.
    """
    if restart and checkpoint_dir is None:
        raise ValueError("restart=True requires checkpoint_dir")
    if auxiliary_vv_preconditioner and direct_k_c_solve:
        # MUMPS would factorize the preconditioning matrix, a copy of the Jacobian here.
        raise ValueError("auxiliary_vv_preconditioner=True requires an iterative K_c solve")
    if cahouet_chabard_schur_preconditioner and direct_k_c_solve:
        raise ValueError("cahouet_chabard_schur_preconditioner=True requires an iterative K_c solve")
    if checkpoint_every is not None and checkpoint_dir is None:
        raise ValueError("checkpoint_every requires checkpoint_dir")

    # load mesh and meshtags

    with dolfinx.io.XDMFFile(comm, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name="Cell tags")
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
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

    # create problem parameters (Table 1 of Failer & Richter, 3d configuration)

    gdim = mesh.geometry.dim

    rho_f = dolfinx.fem.Constant(mesh, 1.0e3)
    nu_f = dolfinx.fem.Constant(mesh, 1.0e-3)

    rho_s = dolfinx.fem.Constant(mesh, 1.0e3)
    mu_s = dolfinx.fem.Constant(mesh, 2.0e6)
    lambda_s = dolfinx.fem.Constant(mesh, 8.0e6)

    U_bar = 1.75
    H = 0.41  # channel height (y) and width (z)

    t0 = 0.0
    dt = dolfinx.fem.Constant(mesh, dt_val)

    theta = dolfinx.fem.Constant(mesh, 0.5 + dt.value)

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

    U = dolfinx.fem.functionspace(mesh, ("CG", 2, (gdim,)))
    V = dolfinx.fem.functionspace(mesh, ("CG", 2, (gdim,)))
    P = dolfinx.fem.functionspace(fluid_mesh, ("CG", 1))
    Z = dolfinx.fem.functionspace(fluid_mesh, ("CG", 2, (gdim,)))
    W = ufl.MixedFunctionSpace(U, V, P, Z)

    # create functions

    u, v, p, z = (
        dolfinx.fem.Function(U, name="u"),
        dolfinx.fem.Function(V, name="v"),
        dolfinx.fem.Function(P, name="p"),
        dolfinx.fem.Function(Z, name="z"),
    )
    u_old, v_old = dolfinx.fem.Function(U), dolfinx.fem.Function(V)

    du, dv, dp, dz = ufl.TestFunctions(W)

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
            # Bi-parabolic profile with mean U_bar, ramped up over 0 <= t < 1.
            values = np.zeros((gdim, x.shape[1]), dtype=x.dtype)
            values[0] = U_bar * 36 * x[1] * (H - x[1]) * x[2] * (H - x[2]) / H**4
            if self.t < 1.0:
                values[0] *= 0.5 * (1.0 - np.cos(np.pi * self.t))
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
    bc_deactivate = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, np.zeros(gdim)), dofs_interface, U)

    # Collect Dirichlet boundary conditions to pass to NonlinearProblem

    bcs = [u_f_bc, inflow_bc, noslip_bc]

    # DESCRIBE FSI PROBLEM
    # FLUID: Bi-parabolic inflow at x = 0, no-slip on the channel walls and the cylinder, do-nothing at x = 2.8
    # SOLID: Homogeneous Dirichlet where the beam is attached to the cylinder

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

    def A_I(u, v, z):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = ufl.inner(z, dz) * dx_fluid
        residual -= ufl.inner(ufl.grad(u), ufl.grad(dz)) * dx_fluid

        residual += ufl.inner(ufl.grad(z), ufl.grad(du)) * dx_fluid
        residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(u, du) * dx_fluid

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
    residual += A_I(u, v, z)
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

    # Cap the quadrature degree before taking derivatives, so that the Jacobian blocks keep it.
    residual_base_ufl = [cap_quadrature_degree(form, max_quadrature_degree) for form in ufl.extract_blocks(residual)]
    jacobian_full_base_ufl = dolfinx.fem.forms.derivative_block(residual_base_ufl, [u, v, p, z])

    # One block per unknown in [u, v, p, z], since blocked assembly places forms by
    # position. Not built with ufl.extract_blocks, which drops the zero blocks. The
    # zero Jacobian rows are built directly, since ufl.derivative does not accept a
    # ufl.ZeroBaseForm, with only their diagonal blocks as zero forms, since a zero
    # form cannot couple the parent mesh and the fluid submesh.
    residual_post_u = rho_s * ufl.inner((u - u_old) / dt - theta * v - (1 - theta) * v_old, du) * dx_solid
    residual_post_u = cap_quadrature_degree(residual_post_u, max_quadrature_degree)
    trials = [ufl.TrialFunction(w.function_space) for w in [u, v, p, z]]

    residual_post_ufl = [residual_post_u, ufl.ZeroBaseForm((dv,)), ufl.ZeroBaseForm((dp,)), ufl.ZeroBaseForm((dz,))]

    jacobian_post_ufl = [[None for _ in range(len([u, v, p, z]))] for __ in range(len([u, v, p, z]))]
    for j in range(len([u, v, p, z])):
        # Derivatives w.r.t. missing residual ufl expression.
        jacobian_post_ufl[0][j] = ufl.derivative(residual_post_u, [u, v, p, z][j], trials[j])
    for i in range(1, len([u, v, p, z])):
        # Zero blocks on the diagonal elsewhere
        jacobian_post_ufl[i][i] = ufl.ZeroBaseForm(([du, dv, dp, dz][i], trials[i]))

    residual_post = dolfinx.fem.form(residual_post_ufl, entity_maps=entity_maps)
    jacobian_post = dolfinx.fem.form(jacobian_post_ufl, entity_maps=entity_maps)

    jacobian_approximate_base_ufl = [
        [jacobian_full_base_ufl[row][column] for column in range(len([u, v, p, z]))] for row in range(len([u, v, p, z]))
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

    jacobian_preconditioner_base_ufl = [
        [jacobian_full_base_ufl[row][column] for column in range(len([u, v, p, z]))] for row in range(len([u, v, p, z]))
    ]

    # Neglect the terms depending on u_f in v and p.
    # The neglecting is really in terms of fluid interior
    # and fluid interface dofs of u, but we don't have that distinction here,
    # so we do it by hand.

    # Neglect the terms for v w.r.t. u. E_S is no longer dependent on u because of the condensation.
    jacobian_preconditioner_base_ufl[1][0] = None

    # Neglect terms for p w.r.t. u.
    jacobian_preconditioner_base_ufl[2][0] = None

    # Switch the preconditioner's v-v block to A_0 as in docs/restricted-iterative-solver.md.
    # A_0 is designed to contain enough terms to be a good preconditioner while still being SPD.

    w = ufl.TrialFunction(V)

    a0_mass = ufl.derivative(A_T_mass(u, u_old, v, v_old), v, w)

    a0_viscous = theta * ufl.derivative(A_E_fluid_viscous(u, v), v, w)

    # Linear elasticity term
    a0_solid = (
        theta**2
        * dt
        * (2 * mu_s * ufl.inner(ufl.sym(ufl.grad(w)), ufl.sym(ufl.grad(dv))) + lambda_s * ufl.div(w) * ufl.div(dv))
        * dx_solid
    )

    a0 = cap_quadrature_degree(a0_mass + a0_viscous + a0_solid, max_quadrature_degree)
    jacobian_preconditioner_base_ufl[1][1] = a0

    # Set up output, checkpointing, and qoi tracking.

    from xfsi_solver.tools.checkpoint import Checkpointer, restart_output_path

    # (t, step) are the time and number of completed steps of the current state
    t = t0
    step = 0
    checkpointer = None
    if checkpoint_dir is not None:
        checkpointer = Checkpointer(checkpoint_dir, mesh, submeshes=[(fluid_mesh, fluid_cell_map)])
    if restart:
        t, step = checkpointer.read([u, v, p, z], dt_val)
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

    writers = []
    if vtx_save_every is not None:
        writers = [dolfinx.io.VTXWriter(comm, output_path, [u, v]), dolfinx.io.VTXWriter(comm, output_path_p, [p])]

    # Quantities of interest: the displacement at the point B on the back face of the beam, and
    # the drag and lift on the obstacle and on the fluid side of the solid-fluid interface.
    spot_dof = find_point_dof(U, np.array([0.9, 0.2, 0.3], dtype=np.float64))
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

    # Without a separate preconditioning matrix, PETSc uses the Jacobian for both.
    preconditioner = jacobian_preconditioner_base_ufl if auxiliary_vv_preconditioner else None

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_base_ufl,
        [u, v, p, z],
        J=jacobian_approximate_base_ufl,
        P=preconditioner,
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
            # "snes_monitor": "ascii:output/logs/fsi3d_iterative_snes_log.txt",
        },
    )

    # Add extra callbacks to change how residual and jacobian is assembled,
    # to account for test function restriction on the interface.

    solver = problem.solver
    b_vec, (fem_residual, res_args, res_kargs) = solver.getFunction()
    J_mat, P_mat, (fem_jacobian, jac_args, jac_kargs) = solver.getJacobian()

    # Prevent possibly overwriting the sparsity pattern on zeroRows.
    problem.A.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
    if auxiliary_vv_preconditioner:
        problem.P_mat.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)

    # Global rows of the interface u-dofs owned by this rank, for zeroing with zeroRows.
    # Each rank stores its owned dofs as [u | v | p | z], starting at its first global row.
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

    def post_jacobian_preconditioned(x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat) -> None:
        J.zeroRows(rows_d, diag=0.0)
        dolfinx.fem.petsc.assemble_matrix(J, jacobian_post, bcs=[u_s_bc, *bcs])
        J.assemble()

        P.zeroRows(rows_d, diag=0.0)
        dolfinx.fem.petsc.assemble_matrix(P, jacobian_post, bcs=[u_s_bc, *bcs])
        P.assemble()

    def pre_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:
        pass

    def post_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:

        dolfinx.fem.petsc.set_bc(b, [[bc_deactivate], [], [], []], alpha=0.0)  # zero restricted rows

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
        "num_preconditioner_assemblies": 0,
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

    def wrapped_jacobian_preconditioned(snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat) -> None:
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
        post_jacobian_preconditioned(x, J, P)
        jacobian_state["assembled"] = True
        jacobian_state["num_assemblies"] += 1
        jacobian_state["num_preconditioner_assemblies"] += 1

    def wrapped_residual(snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        pre_residual(x, b)  # x is not yet assigned to u here
        fem_residual(snes, x, b, *res_args, **res_kargs)
        post_residual(x, b)  # u is now updated and b assembled (BCs applied)

    jacobian_callback = wrapped_jacobian_preconditioned if auxiliary_vv_preconditioner else wrapped_jacobian
    solver.setJacobian(jacobian_callback, J_mat, P_mat)
    solver.setFunction(wrapped_residual, b_vec)

    # Helper function to retrieve the dofs that are supported on solid cells. Don't know
    # what is the most appropriate location of this function for legibility.
    def solid_supported(space, num_dofs):
        """Mask of the owned unrolled dofs of space that are supported on solid cells."""
        bs = space.dofmap.index_map_bs
        dofs = dolfinx.fem.locate_dofs_topological(space, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"]))
        return np.isin(np.arange(num_dofs), (bs * dofs[:, None] + np.arange(bs)).ravel())

    first_row = J_mat.getOwnershipRange()[0]
    offsets_owned, _ = problem.b.getAttr("_blocks")

    def owned_rows(k):
        """Global rows of the owned dofs of block k in [u, v, p, z]."""
        return first_row + np.arange(offsets_owned[k], offsets_owned[k + 1])

    rows_u, rows_v, rows_p, rows_z = (owned_rows(k) for k in range(4))
    u_solid = solid_supported(U, len(rows_u))  # mask over owned unrolled u dofs
    rows_q_c = np.concatenate([rows_v, rows_p])
    rows_u_S = rows_u[u_solid]
    rows_m = np.concatenate([rows_z, rows_u[~u_solid]])

    if not len(rows_q_c) + len(rows_u_S) + len(rows_m) == J_mat.getLocalSize()[0]:
        raise RuntimeError()

    def make_is(rows):
        return PETSc.IS().createGeneral(np.sort(rows).astype(PETSc.IntType), comm=comm)

    # Rows of each displacement component in the m submatrix, for solving K_m component by
    # component. The owned z and u dofs are stored interleaved per node, (x, y, z), so the
    # component of a dof is its position in its block modulo gdim. The m submatrix holds the
    # rows of rows_m in sorted order.
    component_names = ["x", "y", "z"]
    component_z = np.arange(len(rows_z)) % gdim
    component_u_I = (np.arange(len(rows_u)) % gdim)[~u_solid]
    components_m = np.concatenate([component_z, component_u_I])[np.argsort(rows_m)]
    start_m = comm.exscan(len(rows_m)) or 0  # first row of this rank in the m submatrix
    is_m_components = [make_is(start_m + np.flatnonzero(components_m == c)) for c in range(gdim)]

    start = comm.exscan(len(rows_q_c)) or 0  # first row of this rank in the q_c submatrix
    is_v = make_is(start + np.arange(len(rows_v)))
    is_p = make_is(start + len(rows_v) + np.arange(len(rows_p)))

    # The owned v dofs are stored interleaved per node, (x, y, z), so A has block size 3, which
    # BoomerAMG uses for its systems coarsening and GAMG for its aggregates.
    is_v.setBlockSize(V.dofmap.index_map_bs)

    def rigid_body_modes(space):
        """Orthonormal rigid body modes of a 3D vector space: three translations and three rotations."""
        index_map, bs = space.dofmap.index_map, space.dofmap.index_map_bs
        num_dofs = index_map.size_local + index_map.num_ghosts
        x = space.tabulate_dof_coordinates()[:num_dofs]
        basis = [dolfinx.la.vector(index_map, bs=bs, dtype=PETSc.ScalarType) for _ in range(6)]
        b = [w.array.reshape(-1, bs) for w in basis]
        b[0][:, 0] = 1.0
        b[1][:, 1] = 1.0
        b[2][:, 2] = 1.0
        b[3][:, 0], b[3][:, 1] = -x[:, 1], x[:, 0]  # rotation about the z-axis
        b[4][:, 1], b[4][:, 2] = -x[:, 2], x[:, 1]  # rotation about the x-axis
        b[5][:, 2], b[5][:, 0] = -x[:, 0], x[:, 2]  # rotation about the y-axis
        dolfinx.la.orthonormalize(basis)
        return PETSc.NullSpace().create(vectors=[dolfinx.la.petsc.create_vector_wrap(w) for w in basis])

    # PCFIELDSPLIT attaches this to the velocity block A when it extracts it.
    is_v.compose("nearnullspace", rigid_body_modes(V))

    # The multiplicative fieldsplit solves the splits in the order they are set here,
    # independent of the storage order [u | v | p | z]: first K_c for (v, p), then A_S for
    # u_S with the right-hand side updated by C delta v, then K_m for (z, u_I) with the
    # right-hand side updated by L delta u_S.
    pc = solver.getKSP().getPC()
    pc.setFieldSplitIS(("q_c", make_is(rows_q_c)), ("u_S", make_is(rows_u_S)), ("m", make_is(rows_m)))

    # GAMG smoothing with SOR. In parallel, PETSc's SOR is Gauss-Seidel within each rank and
    # Jacobi between ranks, which can diverge where the coupling across a rank boundary is strong,
    # such as in the thin, stiff beam. Undamped, GAMG diverged on 26 to 30 ranks on the coarse mesh
    # and on 30 ranks on the medium mesh; damped by 0.5, it converged on all ranks tried (1 to 30)
    # (notes/fsi3d/implementation-log.md).
    sor_smoothing_options = {
        "fieldsplit_v_mg_levels_ksp_type": "richardson",
        "fieldsplit_v_mg_levels_ksp_richardson_scale": 0.5,
        "fieldsplit_v_mg_levels_pc_type": "sor",
    }

    if direct_k_c_solve:
        velocity_pressure_solve_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_cntl_1": 1e-4,
        }

    else:
        velocity_pressure_solve_options = {
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            # GAMG on v, with the near-nullspace and block size attached to is_v.
            "fieldsplit_v_pc_type": "gamg",
            "fieldsplit_v_pc_gamg_threshold": 0.01,
            "fieldsplit_v_pc_gamg_reuse_interpolation": True,
        }
        if cahouet_chabard_schur_preconditioner:
            velocity_pressure_solve_options |= {
                # The Cahouet-Chabard python PC set up below approximates S^{-1} without a preconditioning
                # matrix, so take the (empty) A11 block instead of assembling selfp.
                "pc_fieldsplit_schur_precondition": "a11",
                "fieldsplit_p_pc_type": "python",
            }
        else:
            velocity_pressure_solve_options |= {
                # The pressure Schur complement S = -B A^{-1} G is dense and only applied as an operator, so it
                # is preconditioned by the sparse approximation \hat S = -B diag(A)^{-1} G. The pressure term
                # gives G = -B^T (up to quadrature, by the Piola identity), so \hat S = B diag(A)^{-1} B^T is a
                # divergence of a gradient: close to a pressure Laplacian, scaled by about dt / rho_f, since the
                # inertia term rho_f / dt dominates diag(A) at small time steps.
                "pc_fieldsplit_schur_precondition": "selfp",
                # BoomerAMG on \hat S.
                "fieldsplit_p_pc_type": "hypre",
                "fieldsplit_p_pc_hypre_type": "boomeramg",
            }
        if block_preconditioned_k_c_solve:
            # FGMRES on K_c, preconditioned by the block upper triangular [[A, G], [0, S]], with one GAMG
            # V-cycle for A^{-1} and one BoomerAMG V-cycle on \hat S for S^{-1}. Each iteration costs two
            # V-cycles, instead of solving with A in every application of S. Only the tolerance of the outer
            # FGMRES matters.
            velocity_pressure_solve_options |= {
                "ksp_converged_reason": None,
                "ksp_type": "fgmres",
                "ksp_rtol": 1e-5,
                "ksp_max_it": 200,
                "pc_fieldsplit_schur_fact_type": "upper",
                # The velocity KSP is preonly, so its operator is only used by the finest GAMG smoother.
                # Use the SPD auxiliary operator A_0 there too, so that the whole V-cycle approximates
                # A_0^{-1}, which Chebyshev with Jacobi requires. FGMRES on K_c still uses the Jacobian.
                # This applies only if auxiliary_vv_preconditioner == True.
                "pc_fieldsplit_diag_use_amat": False,
                "pc_fieldsplit_off_diag_use_amat": True,
                "fieldsplit_v_ksp_type": "preonly",
                # "fieldsplit_v_ksp_converged_reason": None,
                "fieldsplit_p_ksp_type": "preonly",
                # "fieldsplit_p_ksp_converged_reason": None,
            }
            if auxiliary_vv_preconditioner:
                velocity_pressure_solve_options |= {
                    "fieldsplit_v_mg_levels_ksp_type": "chebyshev",
                    "fieldsplit_v_mg_levels_pc_type": "jacobi",
                }
            else:
                velocity_pressure_solve_options |= sor_smoothing_options
        else:
            # Full Schur complement factorization, applied once: A is solved by GMRES in each application of
            # S, so it is solved to a tighter tolerance than S, which is solved by GMRES.
            velocity_pressure_solve_options |= {
                "ksp_converged_reason": None,
                "ksp_type": "preonly",
                "pc_fieldsplit_schur_fact_type": "full",
                # The velocity GMRES solves with its operator inside every application of S, so the
                # operator must be the Jacobian block A. A_0 remains the preconditioning matrix, from
                # which GAMG builds its hierarchy. The finest smoother then iterates with the
                # nonsymmetric A, which Richardson with SOR tolerates and Chebyshev does not.
                # This applies only if auxiliary_vv_preconditioner == True.
                "pc_fieldsplit_diag_use_amat": True,
                "pc_fieldsplit_off_diag_use_amat": True,
                "fieldsplit_v_ksp_type": "gmres",
                "fieldsplit_v_ksp_rtol": 1e-7,
                "fieldsplit_v_ksp_max_it": 100,
                **sor_smoothing_options,
                "fieldsplit_v_ksp_converged_reason": None,
                "fieldsplit_p_ksp_type": "gmres",
                "fieldsplit_p_ksp_rtol": 1e-5,
                "fieldsplit_p_ksp_max_it": 100,
                "fieldsplit_p_ksp_converged_reason": None,
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
    # only once. K_m does not couple the x, y and z components, so the additive fieldsplit over
    # the components solves it exactly, with one MUMPS factorization per component. That leaves
    # out the zero couplings between the components, which the blocked matrix stores: on the
    # medium mesh, MUMPS needs 42 GB for K_m as a whole and 7 GB for one component.
    mesh_motion_solve_options = {
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "ksp_reuse_preconditioner": True,
    }
    for name in component_names:
        mesh_motion_solve_options |= {
            f"fieldsplit_{name}_ksp_type": "preonly",
            f"fieldsplit_{name}_pc_type": "lu",
            f"fieldsplit_{name}_pc_factor_mat_solver_type": "mumps",
            f"fieldsplit_{name}_mat_mumps_icntl_14": 80,
            f"fieldsplit_{name}_mat_mumps_cntl_1": 1e-4,
        }

    # Set through the options database, since the fieldsplit calls setFromOptions on its
    # sub-KSPs in PCSetUp, which would override types set directly on them.
    split_options = {"q_c": velocity_pressure_solve_options, "u_S": mass_solve_options, "m": mesh_motion_solve_options}
    opts = PETSc.Options()
    # List the options that were set but never used when PETSc finalizes, which catches misspelled
    # option names. A global option without a prefix, so not in a petsc_options dict, whose options
    # dolfinx prefixes and deletes after setting up the solver.
    opts["options_left"] = None
    opts["log_view"] = None
    for options, sub_ksp in zip(split_options.values(), pc.getFieldSplitSubKSP(), strict=True):
        opts.prefixPush(sub_ksp.getOptionsPrefix())  # <prefix>fieldsplit_q_c_ / _u_S_ / _m_
        for key, value in options.items():
            opts[key] = value
        opts.prefixPop()
        sub_ksp.setFromOptions()

    ksp_q_c, ksp_u_S, ksp_m = pc.getFieldSplitSubKSP()
    if not direct_k_c_solve:
        ksp_q_c.getPC().setFieldSplitIS(("v", is_v), ("p", is_p))
    ksp_m.getPC().setFieldSplitIS(*zip(component_names, is_m_components, strict=True))

    if cahouet_chabard_schur_preconditioner:
        # The pressure operators are assembled on the reference fluid mesh, so they, and their
        # preconditioners, are constant for the whole run. The velocity block is
        # A ~ rho_f / dt * M + theta * rho_f * nu_f * K: the inertia term, and the viscous term,
        # which enters the residual with the factor theta.
        cahouet_chabard = CahouetChabard(
            P,
            dx_fluid,
            entity_maps,
            outflow_pressure_dofs(P, mesh, facet_tags.find(PHYSICAL_MARKERS["outflow"]), fluid_vertex_map),
            alpha=rho_f.value / dt_val,
            mu=theta.value * rho_f.value * nu_f.value,
            iterative=True,
        )

        # The Schur complement KSP is only created when the q_c fieldsplit is set up, which needs
        # an assembled Jacobian. Assemble it once here and set up the KSPs, then set the context
        # on the Schur complement PC. The assembly copies problem.x into u, v, p, z, so problem.x
        # must hold the current state first. The context persists when the fieldsplit is set up
        # again after reassembly, since the options database keeps the PC type python.
        dolfinx.fem.petsc.assign([u, v, p, z], problem.x)
        jacobian_callback(solver, problem.x, J_mat, P_mat)
        solver.getKSP().setUp()
        ksp_q_c.setUp()
        _, ksp_schur = ksp_q_c.getPC().getFieldSplitSubKSP()
        ksp_schur.getPC().setPythonContext(cahouet_chabard)

        # The Jacobian assembled above used u_old and v_old before the time loop sets them, so
        # make the first Newton iteration assemble it again instead of reusing it.
        jacobian_state["assembled"] = False
        jacobian_state["num_assemblies"] = 0
        jacobian_state["num_preconditioner_assemblies"] = 0

    # Records failed linear solves nested in the fieldsplits, such as a K_c FGMRES that reaches its
    # iteration limit, which neither PETSc nor the SNES converged reason report.
    ksp_check = KSPConvCheck(solver.getKSP())

    # Open qoi file on fresh runs, discard entries after the restart time if a restarted run.
    init_qoi_file(qoi_path, comm, restart, t, dt_val, header=QOI_HEADER_3D)

    if not restart:
        write_output(t, step)

    if mesh.comm.rank == 0:
        print("", flush=True)
    mesh.comm.barrier()

    PETSc.Log.begin()

    first_step = step
    num_newton_iterations = 0
    start = timer()
    while step < num_steps:
        t = t0 + (step + 1) * dt_val
        inflow_bc_func.interpolate(InflowFunc(t))
        inflow_bc_func.x.scatter_forward()

        u_old.x.array[:] = u.x.array[:]
        v_old.x.array[:] = v.x.array[:]

        if comm.rank == 0:
            print(f"\n{t = :.3f}")

        ksp_check.clear()
        problem.solve()
        num_newton_iterations += solver.getIterationNumber()
        check_converged(problem, f"t = {t:.4f}", writers=writers, ksp_check=ksp_check)

        if comm.rank == 0:
            sys.stdout.flush()

        step += 1
        write_output(t, step)

        if checkpoint_every is not None and step % checkpoint_every == 0:
            checkpointer.write([u, v, p, z], t, step, dt_val)

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")
        if direct_k_c_solve:
            k_c_solve_description = "MUMPS LU\n"
        else:
            options = velocity_pressure_solve_options
            if cahouet_chabard_schur_preconditioner:
                schur_description = cahouet_chabard.description()
            else:
                schur_description = "BoomerAMG on -B diag(A)^{-1} G (selfp)"
            hierarchy = "the auxiliary SPD operator A_0" if auxiliary_vv_preconditioner else "A"
            gamg_description = (
                f"GAMG built from {hierarchy} (rigid body modes as near-nullspace),\n"
                f"      {options['fieldsplit_v_mg_levels_ksp_type']} with "
                f"{options['fieldsplit_v_mg_levels_pc_type']} smoothing"
            )
            if "fieldsplit_v_mg_levels_ksp_richardson_scale" in options:
                gamg_description += f", damped by {options['fieldsplit_v_mg_levels_ksp_richardson_scale']}"
            if block_preconditioned_k_c_solve:
                k_c_solve_description = (
                    f"FGMRES (rtol {options['ksp_rtol']:g}), preconditioned by the block upper triangular Schur\n"
                    f"    factor, one V-cycle of {gamg_description}, for A^{{-1}},\n"
                    f"    one application of {schur_description}, for S^{{-1}}\n"
                )
            else:
                k_c_solve_description = (
                    "full Schur complement factorization over v and p,\n"
                    f"    GMRES on A (rtol {options['fieldsplit_v_ksp_rtol']:g}) with {gamg_description},\n"
                    f"    GMRES on S (rtol {options['fieldsplit_p_ksp_rtol']:g}) preconditioned by\n"
                    f"      {schur_description}\n"
                )
        print(
            "\nSolver setup:\n"
            "  Solid stress: in terms of v, via u = u_old + dt * (theta * v + (1 - theta) * v_old)\n"
            "  Jacobian: approximate J_0, without the fluid ALE derivatives (eq. (3) of\n"
            "    docs/restricted-iterative-solver.md)\n"
            "  Linear solve: multiplicative fieldsplit over q_c = (v, p), u_S and m = (z, u_I),\n"
            "    MUMPS LU on each component of K_m, CG with Jacobi on A_S, with the factorizations of K_m reused\n"
            "    for the whole run\n"
            f"  K_c solve: {k_c_solve_description}"
            f"  Jacobian reuse: reassembled when ||R_l|| > gamma_reassemble * ||R_(l-1)||, "
            f"with {gamma_reassemble = }\n"
            f"  Jacobian assemblies: {jacobian_state['num_assemblies']} in {num_newton_iterations} Newton iterations "
            f"over {step - first_step} time steps\n"
            f"  Separate preconditioning matrix assemblies (A_0): {jacobian_state['num_preconditioner_assemblies']}\n"
            f"  Quadrature degree: at most {max_quadrature_degree}\n"
        )

    for writer in writers:
        writer.close()

    # Destroying the MUMPS factorization and the AMG hierarchies is collective, so destroy them on
    # all ranks here instead of leaving it to garbage collection, which can run at different points
    # on different ranks.
    if cahouet_chabard_schur_preconditioner:
        cahouet_chabard.destroy_solvers()
    solver.destroy()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    args = parser.parse_args()

    solve(
        mesh_path="data/meshes/fsi3d/mesh_sec_coarse.xdmf",
        T=10.0,
        dt_val=0.004,
        output_path="output/pv/fsi3d_iterative_coarse.bp",
        output_path_p="output/pv/fsi3d_iterative_p_coarse.bp",
        qoi_path="output/qoi/fsi3d_iterative_qoi_coarse.txt",
        checkpoint_dir="output/checkpoints/fsi3d_iterative_coarse",
        checkpoint_every=250,
        vtx_save_every=None,
        gamma_reassemble=0.2,  # Default 0.2.
        direct_k_c_solve=True,
        block_preconditioned_k_c_solve=True,
        auxiliary_vv_preconditioner=False,
        cahouet_chabard_schur_preconditioner=False,
        max_quadrature_degree=8,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
