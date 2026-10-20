# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import sys
import warnings
from timeit import default_timer as timer

import basix
import dolfinx
import dolfinx.fem.forms
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from petsc4py import PETSc

from xfsi_solver.tools.convergence import KSPConvCheck, check_converged
from xfsi_solver.tools.petsc_reuse import IdenticalBlocksPC
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
    max_quadrature_degree=8,
    c0ip_a_val=2.0,
):
    """Solve the 3d FSI benchmark of Failer & Richter (J. Sci. Comput. 82:28, 2020,
    Sect. 5.1.2), an elastic beam attached to a cylinder in a channel of square cross
    section, up to time ``T``, in ``round((T - t0) / dt_val)`` time steps, writing the
    initial state at ``t0`` as the first QoI row and, with ``vtx_save_every``, as the
    first VTX snapshot. With ``vtx_save_every=None``, no VTX output is written; otherwise
    a snapshot is written every ``vtx_save_every`` steps.

    This is the 3d version of the solver in fsi2_biharmonic_c0ip.py: threeDfsi_mixed.py with the
    biharmonic mesh motion in C0 interior penalty (C0IP) form instead of the mixed form. The
    fluid displacement is the only mesh-motion unknown, without the auxiliary field z, and K_m is
    symmetric positive definite. The zero normal derivative on the fluid boundary is imposed by
    Nitsche's method. The penalty parameters are those of
    tools/dg_optpar.py:get_dg_penalty_parameters_tetrahedron with ``a_val = c0ip_a_val``: the
    triangle parameters carried over to tetrahedra, neither tuned nor proven coercive. The mesh
    motion forms are linear in u and integrated on the reference mesh, so they are assembled once,
    as a matrix B_m, which the residual (as B_m u) and the Jacobian reassemblies add instead of
    integrating them again. The QoIs
    are the drag and the lift on the cylinder and the beam, and the displacement of the point
    B = (0.9, 0.2, 0.3) on the back face of the beam.

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
    whole run. K_m does not couple the x, y and z components, and the components have the
    same matrix, so only the x component is factorized, which leaves out the zero couplings
    between the components that the blocked matrix stores. The three components are solved
    with that factorization at once, by one MatMatSolve with three right-hand sides.

    With ``direct_k_c_solve=True``, the velocity-pressure block K_c is solved with MUMPS
    LU. Otherwise, it is solved by FGMRES, preconditioned by the block upper triangular
    factor of the Schur complement factorization over v and p: one GAMG V-cycle on the
    velocity block A, with the rigid body modes as near-nullspace and SOR smoothing, for
    A^{-1}, and one BoomerAMG V-cycle on the approximation -B diag(A)^{-1} G (selfp) of
    the pressure Schur complement S = -B A^{-1} G, for S^{-1}. The SOR is under-relaxed,
    with omega = 0.5: in parallel, PETSc's SOR is Gauss-Seidel within each rank and
    Jacobi between ranks, and with omega = 1 it made GAMG diverge for some partitions of
    the 3d mesh.

    The quadrature degree is capped at ``max_quadrature_degree``: UFL estimates degrees up to
    19 for the nonlinear ALE and St. Venant-Kirchhoff terms in the residual (25 in the
    Jacobian), with 1000 quadrature points per tetrahedron, against 45 points at degree 8.
    Integrals with a lower estimate keep it.

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

    # Create measures for the C0IP mesh motion: the interior facets with fluid cells on
    # both sides, and the fluid boundary including the fluid side of the interface, on
    # which a zero normal derivative is weakly imposed. They get their own integration
    # entities, since all integrals of one type in a form must share subdomain data.

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim - 1)
    cell_index_map = mesh.topology.index_map(tdim)
    is_fluid_cell = np.zeros(cell_index_map.size_local + cell_index_map.num_ghosts, dtype=bool)
    is_fluid_cell[cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"])] = True
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
    interior_facets = np.flatnonzero(np.diff(facet_to_cell.offsets) == 2).astype(np.int32)
    first_cells = facet_to_cell.array[facet_to_cell.offsets[interior_facets]]
    second_cells = facet_to_cell.array[facet_to_cell.offsets[interior_facets] + 1]
    fluid_interior_facets = interior_facets[is_fluid_cell[first_cells] & is_fluid_cell[second_cells]]
    fluid_interior_entities = dolfinx.fem.compute_integration_domains(
        dolfinx.fem.IntegralType.interior_facet, mesh.topology, fluid_interior_facets
    )
    tag_c0ip = 102
    dS_fluid = ufl.Measure("dS", domain=mesh, subdomain_data=[(tag_c0ip, fluid_interior_entities)])(tag_c0ip)

    fluid_exterior_facets = np.unique(
        np.concatenate(
            [facet_tags.find(PHYSICAL_MARKERS[name]) for name in ["inflow", "outflow", "channel_side", "obstacle"]]
        )
    ).astype(np.int32)
    fluid_exterior_entities = dolfinx.fem.compute_integration_domains(
        dolfinx.fem.IntegralType.exterior_facet, mesh.topology, fluid_exterior_facets
    )
    fluid_boundary_entities = np.concatenate([np.ravel(fluid_exterior_entities), fluid_entities.flatten()])
    ds_boundary_fluid = ufl.Measure("ds", domain=mesh, subdomain_data=[(tag_c0ip, fluid_boundary_entities)])(tag_c0ip)

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
    W = ufl.MixedFunctionSpace(U, V, P)

    # create functions

    u, v, p = (
        dolfinx.fem.Function(U, name="u"),
        dolfinx.fem.Function(V, name="v"),
        dolfinx.fem.Function(P, name="p"),
    )
    u_old, v_old = dolfinx.fem.Function(U), dolfinx.fem.Function(V)

    du, dv, dp = ufl.TestFunctions(W)

    # C0IP penalty parameters for the mesh motion

    from xfsi_solver.tools.dg_optpar import (
        compute_cell_volumes,
        compute_facet_diameters_and_areas,
        get_dg_penalty_parameters_tetrahedron,
    )

    # The cell volumes and the facet diameters and areas are computed as DG0 functions, since
    # ufl.CellVolume and ufl.FacetArea are not supported on the curved mesh. The facet sizes live
    # on a submesh of the facets, so its entity map is needed in the forms.
    if mesh.basix_cell() != basix.CellType.tetrahedron:
        raise NotImplementedError(
            f"C0IP penalty parameters are only implemented for tetrahedra, not {mesh.basix_cell()}"
        )
    mm_element_order = U.element.basix_element.degree
    cell_volume = compute_cell_volumes(mesh)
    h_facet, area_facet, facet_entity_map = compute_facet_diameters_and_areas(mesh)
    entity_maps.append(facet_entity_map)
    beta, beta_boundary = get_dg_penalty_parameters_tetrahedron(
        mesh, a_val=c0ip_a_val, p=mm_element_order, vol=cell_volume, h=h_facet, area=area_facet
    )

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

    def A_mesh_motion(u, du):
        n = ufl.FacetNormal(mesh)

        # C0IP biharmonic mesh motion, with the zero normal derivative on the fluid
        # boundary imposed by Nitsche's method, as the natural boundary condition of the
        # mixed formulation does. The interface rows of du are restricted, so the term
        # with the normal derivative of the Laplacian of u on the interface is not needed.
        residual = ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(du))) * dx_fluid

        residual -= ufl.inner(ufl.avg(ufl.div(ufl.grad(u))), ufl.jump(ufl.grad(du), n)) * dS_fluid
        residual -= ufl.inner(ufl.avg(ufl.div(ufl.grad(du))), ufl.jump(ufl.grad(u), n)) * dS_fluid
        residual += beta / h_facet("+") * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(du), n)) * dS_fluid

        residual -= ufl.inner(ufl.div(ufl.grad(u)), ufl.grad(du) * n) * ds_boundary_fluid
        residual -= ufl.inner(ufl.div(ufl.grad(du)), ufl.grad(u) * n) * ds_boundary_fluid
        residual += beta_boundary / h_facet * ufl.inner(ufl.grad(u) * n, ufl.grad(du) * n) * ds_boundary_fluid

        return residual

    def A_I(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = ufl.div(J * ufl.inv(F) * v) * dp * dx_fluid

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

    # The mesh motion is linear in u and integrated on the reference mesh, so it is assembled
    # only once, as the matrix B_m below, which the residual and Jacobian callbacks add. The
    # Jacobian forms keep it for the first assembly, which gives the matrix the sparsity of its
    # facet couplings.
    residual_mesh_motion = A_mesh_motion(u, du)

    # Cap the quadrature degree before taking derivatives, so that the Jacobian blocks keep it.
    residual_base_ufl = [cap_quadrature_degree(form, max_quadrature_degree) for form in ufl.extract_blocks(residual)]
    jacobian_full_base_ufl = dolfinx.fem.forms.derivative_block(
        [
            cap_quadrature_degree(form, max_quadrature_degree)
            for form in ufl.extract_blocks(residual + residual_mesh_motion)
        ],
        [u, v, p],
    )

    # One block per unknown in [u, v, p], since blocked assembly places forms by
    # position. Not built with ufl.extract_blocks, which drops the zero blocks. The
    # zero Jacobian rows are built directly, since ufl.derivative does not accept a
    # ufl.ZeroBaseForm, with only their diagonal blocks as zero forms, since a zero
    # form cannot couple the parent mesh and the fluid submesh.
    residual_post_u = rho_s * ufl.inner((u - u_old) / dt - theta * v - (1 - theta) * v_old, du) * dx_solid
    residual_post_u = cap_quadrature_degree(residual_post_u, max_quadrature_degree)
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

    # B_m: the mesh motion as the u-u block of a matrix with the layout of the Jacobian, with
    # the rows and columns of u_f_bc zeroed as in the Jacobian.
    mesh_motion_ufl = [[None for _ in range(len([u, v, p]))] for __ in range(len([u, v, p]))]
    mesh_motion_ufl[0][0] = cap_quadrature_degree(A_mesh_motion(trials[0], du), max_quadrature_degree)
    for i in range(1, len([u, v, p])):
        mesh_motion_ufl[i][i] = ufl.ZeroBaseForm(([du, dv, dp][i], trials[i]))
    mesh_motion_forms = dolfinx.fem.form(mesh_motion_ufl, entity_maps=entity_maps)
    mesh_motion_matrix = dolfinx.fem.petsc.create_matrix(mesh_motion_forms)
    # assemble_matrix puts diag on the diagonal of the rows of bcs, 1 by default. The Jacobian
    # reassembly already puts 1 there, and adds B_m to it, so B_m gets 0: with 1, those rows
    # would have 2 on the diagonal, and Newton would only take half the step that sets u_f_bc.
    dolfinx.fem.petsc.assemble_matrix(mesh_motion_matrix, mesh_motion_forms, bcs=[u_f_bc], diag=0.0)
    mesh_motion_matrix.assemble()

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

    from xfsi_solver.tools.checkpoint import Checkpointer, restart_output_path

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
            # "snes_monitor": "ascii:output/logs/threeDfsi_c0ip_snes_log.txt",
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

    # The residual and the Jacobian reassemblies get the u-u block as a zero form on the same
    # spaces instead of the mesh motion forms, and the callbacks add B_m. dolfinx also lifts the
    # residual by the boundary conditions with the Jacobian forms, which integrated the mesh
    # motion forms in every residual evaluation: 0.88 of 0.92 s per residual on the coarse mesh.
    # The first Jacobian assembly integrates them, since the matrix only keeps the entries
    # inserted in its first assembly, and the facet couplings are in no other block.
    assembly_forms = [list(row) for row in jac_kargs["jacobian"]]
    assembly_forms[0][0] = dolfinx.fem.form(
        dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(trials[0], du) * dx_solid, entity_maps=entity_maps
    )
    jac_kargs_reassembly = jac_kargs | {"jacobian": assembly_forms}
    res_kargs = res_kargs | {"jacobian": assembly_forms}

    def pre_jacobian(x: PETSc.Vec, J: PETSc.Mat) -> None:
        pass

    def post_jacobian(x: PETSc.Vec, J: PETSc.Mat) -> None:
        J.zeroRows(rows_d, diag=0.0)
        dolfinx.fem.petsc.assemble_matrix(J, jacobian_post, bcs=[u_s_bc, *bcs])
        J.assemble()

    # B_m x is the mesh motion residual, with the lifting by u_f_bc, which is homogeneous. Its rows
    # of the boundary conditions and the test function restriction are left out, as assembling
    # the mesh motion forms into the residual would leave them.
    mesh_motion_product = mesh_motion_matrix.createVecLeft()
    u_rows = slice(offsets_owned[u_block], offsets_owned[u_block + 1])
    mesh_motion_bc_rows = offsets_owned[u_block] + np.concatenate(
        [indices[:num_owned] for indices, num_owned in (bc.dof_indices() for bc in [u_f_bc, u_s_bc, bc_deactivate])]
    )

    def pre_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:
        pass

    def post_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:

        mesh_motion_matrix.mult(x, mesh_motion_product)
        mesh_motion_product.array[mesh_motion_bc_rows] = 0.0
        with b.localForm() as bl:
            bl.array[u_rows] += mesh_motion_product.array[u_rows]

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
        if jacobian_state["assembled"]:
            fem_jacobian(snes, x, J, P, *jac_args, **jac_kargs_reassembly)
            J.axpy(1.0, mesh_motion_matrix, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        else:
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

    # Rows of each displacement component in the m submatrix, for solving K_m component by
    # component. The owned u dofs are stored interleaved per node, (x, y, z), so the
    # component of a dof is its position in its block modulo gdim, and the k-th rows of the
    # components on a rank are the same node. The m submatrix holds the rows of rows_m in
    # sorted order.
    component_u_I = (np.arange(len(rows_u)) % gdim)[~u_solid]
    components_m = component_u_I[np.argsort(rows_m)]
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
    # independent of the storage order [u | v | p]: first K_c for (v, p), then A_S for
    # u_S with the right-hand side updated by C delta v, then K_m for u_I with the
    # right-hand side updated by L delta u_S.
    pc = solver.getKSP().getPC()
    pc.setFieldSplitIS(("q_c", make_is(rows_q_c)), ("u_S", make_is(rows_u_S)), ("m", make_is(rows_m)))

    # GAMG smoothing with SOR. In parallel, PETSc's SOR is Gauss-Seidel within each rank and
    # Jacobi between ranks, which can diverge where the coupling across a rank boundary is strong,
    # such as in the thin, stiff beam. With omega = 1, GAMG diverged on 26 to 30 ranks on the coarse
    # mesh and on 30 ranks on the medium mesh; with omega = 0.7, on 30 ranks on the coarse mesh.
    # Under-relaxed with omega = 0.5, it converged on all ranks tried, also with the beam
    # oscillating. Damping the Richardson iteration instead (scale 0.5) converged too, but costs
    # a separate MatMult for the residual in every smoothing step, since PETSc only applies SOR
    # directly as the smoother when the Richardson scale is 1, and needed about 20% more FGMRES
    # iterations: omega = 0.5 takes 14 to 16% less time per step on the medium mesh
    # (notes/fsi3d/implementation-log.md).
    sor_smoothing_options = {
        "fieldsplit_v_mg_levels_ksp_type": "richardson",
        "fieldsplit_v_mg_levels_pc_type": "sor",
        "fieldsplit_v_mg_levels_pc_sor_omega": 0.5,
    }

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
            **sor_smoothing_options,
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
    # only once. K_m does not couple the x, y and z components, and every component has the
    # same matrix, since every boundary condition on the mesh motion constrains all components.
    # So only the x component is factorized, which leaves out the zero couplings between the
    # components that the blocked matrix stores: on the medium mesh, MUMPS needed 42 GB for the
    # mixed K_m as a whole and 7 GB for one component. IdenticalBlocksPC solves for the three
    # components at once with MatMatSolve, which reads the factor once instead of three times.
    # The C0IP K_m is symmetric positive definite, so it is factorized by Cholesky
    # (notes/c0ip-mesh-motion/implementation-log.md). The PC type is set below, with the context.
    mesh_motion_solve_options = {
        "ksp_type": "preonly",
        "ksp_reuse_preconditioner": True,
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
    # opts["log_view"] = None  # PETSc performance summary at the end, with PETSc.Log.begin() below
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
    ksp_m.getPC().setType(PETSc.PC.Type.PYTHON)
    ksp_m.getPC().setPythonContext(mesh_motion_pc)

    # Records failed linear solves nested in the fieldsplits, such as a K_c FGMRES that reaches its
    # iteration limit, which neither PETSc nor the SNES converged reason report.
    ksp_check = KSPConvCheck(solver.getKSP())

    # Open qoi file on fresh runs, discard entries after the restart time if a restarted run.
    init_qoi_file(qoi_path, comm, restart, t, dt_val, header=QOI_HEADER_3D)

    if not restart:
        write_output(t, step)

    PETSc.Sys.Print("")
    mesh.comm.barrier()

    # PETSc.Log.begin()

    first_step = step
    num_newton_iterations = 0
    start = timer()
    while step < num_steps:
        t = t0 + (step + 1) * dt_val
        inflow_bc_func.interpolate(InflowFunc(t))
        inflow_bc_func.x.scatter_forward()

        u_old.x.array[:] = u.x.array[:]
        v_old.x.array[:] = v.x.array[:]

        PETSc.Sys.Print(f"\n{t = :.3f}")

        ksp_check.clear()
        problem.solve()
        num_newton_iterations += solver.getIterationNumber()
        check_converged(problem, f"t = {t:.4f}", writers=writers, ksp_check=ksp_check)

        if comm.rank == 0:
            sys.stdout.flush()

        step += 1
        write_output(t, step)

        if checkpoint_every is not None and step % checkpoint_every == 0:
            checkpointer.write([u, v, p], t, step, dt_val)

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
            "    MUMPS Cholesky on the x component of the C0IP K_m, for the whole run and for all three\n"
            "    components at once by MatMatSolve, CG with Jacobi on A_S\n"
            f"  C0IP penalty: a_val = {c0ip_a_val}\n"
            "  C0IP mesh motion: B_m assembled once, added to the residual and the Jacobian reassemblies\n"
            f"  K_c solve: {k_c_solve_description}"
            f"  Jacobian reuse: reassembled when ||R_l|| > gamma_reassemble * ||R_(l-1)||, "
            f"with {gamma_reassemble = }\n"
            f"  Jacobian assemblies: {jacobian_state['num_assemblies']} in {num_newton_iterations} Newton iterations "
            f"over {step - first_step} time steps\n"
            f"  Quadrature degree: at most {max_quadrature_degree}\n"
        )

    for writer in writers:
        writer.close()

    # Destroying the MUMPS factorization and the AMG hierarchies is collective, so destroy them on
    # all ranks here instead of leaving it to garbage collection, which can run at different points
    # on different ranks.
    solver.destroy()
    mesh_motion_pc.destroy_solvers()
    mesh_motion_matrix.destroy()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    args = parser.parse_args()

    solve(
        mesh_path="data/meshes/fsi3d/mesh_sec_medium.xdmf",
        T=10.0,
        dt_val=0.004,
        output_path="output/pv/threeDfsi_c0ip_medium.bp",
        output_path_p="output/pv/threeDfsi_c0ip_medium_p.bp",
        qoi_path="output/qoi/threeDfsi_c0ip_medium.txt",
        checkpoint_dir="output/checkpoints/threeDfsi_c0ip_medium",
        checkpoint_every=250,
        vtx_save_every=None,
        gamma_reassemble=0.2,  # Default 0.2.
        direct_k_c_solve=False,
        max_quadrature_degree=8,
        c0ip_a_val=2.0,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
