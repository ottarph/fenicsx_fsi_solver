import sys
import warnings
from timeit import default_timer as timer

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py.MPI import COMM_WORLD as comm

from xfsi_solver.scripts.plot_qois import LAGRANGE_QOI_LABELS, plot_run_qois
from xfsi_solver.tools.checkpoint import Checkpointer, check_start_options, restart_output_path, start_time
from xfsi_solver.tools.convergence import check_converged
from xfsi_solver.tools.parameters import lame_lambda, theta_value
from xfsi_solver.tools.qoi import (
    LAGRANGE_QOI_HEADER,
    append_qoi_row,
    assemble_force,
    assemble_gap_norm,
    drag_lift_forms,
    find_point_dof,
    init_qoi_file,
    interface_gap_forms,
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
        "U_f": ("P", 2, (2,)),
        "V_f": ("P", 2, (2,)),
        "U_s": ("P", 2, (2,)),
        "V_s": ("P", 2, (2,)),
        "P": ("P", 1),
        "Lu": ("P", 1, (2,)),
        "Lv": ("P", 1, (2,)),
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

    The Lagrange multipliers ``lambda_u`` and ``lambda_v`` are not
    checkpointed, so a restart, or a run from ``initial_state_dir``, starts
    them from zero; see the comment where the checkpointer is set up.

    ``elements`` maps the name of each function space (``U_f``, ``V_f``, ``U_s``, ``V_s``, ``P``,
    ``Lu``, ``Lv``) to its element, the tuple passed to ``dolfinx.fem.functionspace``; the default
    is P2 for the displacements and velocities, P1 for the pressure and P1 for the multipliers.

    The physical parameters default to the FSI2 benchmark: ``rho_f = 1e3``, ``nu_f = 1e-3``,
    ``rho_s = 1e4``, ``mu_s = 5e5``, mean inflow velocity ``U_bar = 1`` and channel height
    ``H = 0.41``. The first Lamé parameter of the solid is given either as ``lambda_s`` or through
    the Poisson ratio ``nu_s`` (default ``nu_s = 0.4``); giving both raises ``ValueError``. The
    time stepping uses theta = 0.5 + ``absolute_theta_shift``, or theta = 0.5 +
    ``relative_theta_shift * dt_val``; the default is ``relative_theta_shift = 1.0``, and giving
    both shifts raises ``ValueError``.

    If ``max_quadrature_degree`` is given, the quadrature degree of the residual, and so of its
    Jacobian, and of the drag and lift is capped at it: integrals for which UFL estimates a higher
    degree are integrated inexactly at that degree, and the others keep their estimate. With the
    default None, every integral keeps the degree UFL estimates. The interface gap diagnostics keep the
    degree UFL estimates.
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

    # create submeshes for fluid and solid

    fluid_mesh, fluid_cell_map, fluid_vertex_map, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"])
    )
    solid_mesh, solid_cell_map, solid_vertex_map, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"])
    )

    # Create measure with meshtags

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    dx_fluid = dx(PHYSICAL_MARKERS["ALE_fluid"])
    dx_solid = dx(PHYSICAL_MARKERS["solid"])

    # Create measure for interface / solid-fluid boundary (both sides), and a
    # submesh of the interface itself to host the Lagrange multiplier fields
    # that will enforce continuity of u and v across it.

    import scifem

    new_tag_fluid = 101
    new_tag_solid = 102
    interface_facets = facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])
    idata = scifem.compute_interface_data(cell_tags, interface_facets)
    fluid_is_first_side = idata.shape[0] > 0 and cell_tags.values[idata[0, 0]] == PHYSICAL_MARKERS["ALE_fluid"]
    if fluid_is_first_side:
        fluid_entities, solid_entities = idata[:, :2], idata[:, 2:]
    else:
        fluid_entities, solid_entities = idata[:, 2:], idata[:, :2]
    # Both tags must share one Measure's subdomain_data object: dolfinx requires
    # every exterior-facet integral within a single (per-testfunction) form to
    # reference the *same* subdomain_data object, and the multiplier's own
    # equations below integrate against both the fluid and the solid side.
    new_measure_interface = ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=[(new_tag_fluid, fluid_entities.flatten()), (new_tag_solid, solid_entities.flatten())],
    )
    ds_interface_fluid = new_measure_interface(new_tag_fluid)
    ds_interface_solid = new_measure_interface(new_tag_solid)

    # Interior-facet measure over the same interface, keeping the fluid/solid
    # cell pairing from `idata` on both sides at once (via +/- restriction).
    # Used only for the continuity-violation QOI below, which needs u_f and
    # u_s (or v_f and v_s) evaluated at the same point in a single integral.
    new_tag_dS = 103
    dS_interface = ufl.Measure("dS", domain=mesh, subdomain_data=[(new_tag_dS, idata.flatten())])(new_tag_dS)
    fluid_side, solid_side = ("+", "-") if fluid_is_first_side else ("-", "+")

    gamma_mesh, gamma_facet_map, gamma_vertex_map, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim - 1, interface_facets
    )

    # create entity maps for mixed mesh integration

    entity_maps = [fluid_cell_map, solid_cell_map, gamma_facet_map]

    # Transfer facet tags onto the fluid and solid submeshes so boundary
    # conditions can be located directly on them.

    fluid_mesh.topology.create_connectivity(1, 2)
    solid_mesh.topology.create_connectivity(1, 2)
    fluid_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
        facet_tags, fluid_mesh, fluid_vertex_map, fluid_cell_map
    )
    solid_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
        facet_tags, solid_mesh, solid_vertex_map, solid_cell_map
    )

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
    #
    # u/v are no longer shared between the solid and fluid subdomains: each
    # subdomain gets its own displacement and velocity space, and continuity
    # across the interface is enforced weakly through the Lagrange
    # multiplier spaces Lu, Lv (defined on the interface submesh gamma_mesh)
    # instead of through shared degrees of freedom.

    U_f = dolfinx.fem.functionspace(fluid_mesh, elements["U_f"])
    V_f = dolfinx.fem.functionspace(fluid_mesh, elements["V_f"])
    U_s = dolfinx.fem.functionspace(solid_mesh, elements["U_s"])
    V_s = dolfinx.fem.functionspace(solid_mesh, elements["V_s"])
    P = dolfinx.fem.functionspace(fluid_mesh, elements["P"])
    Lu = dolfinx.fem.functionspace(gamma_mesh, elements["Lu"])
    Lv = dolfinx.fem.functionspace(gamma_mesh, elements["Lv"])
    W = ufl.MixedFunctionSpace(U_f, V_f, U_s, V_s, P, Lu, Lv)

    # create functions

    u_f, v_f = dolfinx.fem.Function(U_f, name="u_f"), dolfinx.fem.Function(V_f, name="v_f")
    u_s, v_s = dolfinx.fem.Function(U_s, name="u_s"), dolfinx.fem.Function(V_s, name="v_s")
    p = dolfinx.fem.Function(P, name="p")
    lambda_u, lambda_v = dolfinx.fem.Function(Lu, name="lambda_u"), dolfinx.fem.Function(Lv, name="lambda_v")

    u_f_old, v_f_old = dolfinx.fem.Function(U_f), dolfinx.fem.Function(V_f)
    u_s_old, v_s_old = dolfinx.fem.Function(U_s), dolfinx.fem.Function(V_s)

    du_f, dv_f, du_s, dv_s, dp, dlambda_u, dlambda_v = ufl.TestFunctions(W)

    # create Dirichlet boundary condition

    from functools import reduce

    inflow_bc_func = dolfinx.fem.Function(V_f)
    inflow_bc_func.x.array[:] = 0.0
    inflow_bc_facets = reduce(
        np.union1d,
        [
            fluid_facet_tags.find(PHYSICAL_MARKERS["inflow"]),
        ],
    )
    inflow_bc_dofs = dolfinx.fem.locate_dofs_topological(V_f, fluid_mesh.geometry.dim - 1, inflow_bc_facets)

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

    noslip_bc_func = dolfinx.fem.Function(V_f)
    noslip_bc_func.x.array[:] = 0.0
    noslip_bc_facets = reduce(
        np.union1d,
        [
            fluid_facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
            fluid_facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
        ],
    )
    noslip_bc_dofs = dolfinx.fem.locate_dofs_topological(V_f, fluid_mesh.geometry.dim - 1, noslip_bc_facets)

    noslip_bc = dolfinx.fem.dirichletbc(noslip_bc_func, noslip_bc_dofs)

    # Create ALE Dirichlet boundary condition (mesh motion, fluid domain only)

    u_f_bc_func = dolfinx.fem.Function(U_f)
    u_f_bc_func.x.array[:] = 0.0
    u_f_bc_facets = reduce(
        np.union1d,
        [
            fluid_facet_tags.find(PHYSICAL_MARKERS["inflow"]),
            fluid_facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
            fluid_facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
            fluid_facet_tags.find(PHYSICAL_MARKERS["outflow"]),
        ],
    )
    u_f_bc_dofs = dolfinx.fem.locate_dofs_topological(U_f, fluid_mesh.geometry.dim - 1, u_f_bc_facets)
    u_f_bc = dolfinx.fem.dirichletbc(u_f_bc_func, u_f_bc_dofs)

    # Clamp the solid where it meets the fixed obstacle (homogeneous
    # Dirichlet BC for solid displacement and velocity)

    u_s_bc_func = dolfinx.fem.Function(U_s)
    u_s_bc_func.x.array[:] = 0.0
    v_s_bc_func = dolfinx.fem.Function(V_s)
    v_s_bc_func.x.array[:] = 0.0
    solid_obstacle_facets = solid_facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"])
    u_s_bc_dofs = dolfinx.fem.locate_dofs_topological(U_s, solid_mesh.geometry.dim - 1, solid_obstacle_facets)
    v_s_bc_dofs = dolfinx.fem.locate_dofs_topological(V_s, solid_mesh.geometry.dim - 1, solid_obstacle_facets)
    u_s_bc = dolfinx.fem.dirichletbc(u_s_bc_func, u_s_bc_dofs)
    v_s_bc = dolfinx.fem.dirichletbc(v_s_bc_func, v_s_bc_dofs)

    # Collect Dirichlet boundary conditions

    bcs = [u_f_bc, inflow_bc, noslip_bc, u_s_bc, v_s_bc]

    # DESCRIBE FSI PROBLEM
    # FLUID: Parabolic inflow on left side, no-slip on top, bottom, and obstacle, do-nothing on right side
    # SOLID: Homogeneous Dirichlet on left side

    from xfsi_solver.fsi.materials import Fluid, Solid

    # create residual form

    def A_T(u_f, u_f_old, v_f, v_f_old, u_s, u_s_old, v_s, v_s_old):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)
        F_old = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f_old)
        J_old = ufl.det(F_old)
        J_mid = 0.5 * (J + J_old)

        residual = rho_f * J_mid * ufl.inner((v_f - v_f_old) / dt, dv_f) * dx_fluid

        residual -= rho_f * J * ufl.inner(ufl.grad(v_f) * ufl.inv(F) * ((u_f - u_f_old) / dt), dv_f) * dx_fluid

        residual += rho_s * ufl.inner((v_s - v_s_old) / dt, dv_s) * dx_solid

        residual += rho_s * ufl.inner((u_s - u_s_old) / dt, du_s) * dx_solid

        return residual

    def A_I(u_f, v_f, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        alpha_u0 = dolfinx.fem.Constant(mesh, 1e-9)
        alpha_u = alpha_u0

        residual = ufl.inner(alpha_u * ufl.grad(u_f), ufl.grad(du_f)) * dx_fluid

        residual += ufl.div(J * ufl.inv(F) * v_f) * dp * dx_fluid

        return residual

    def A_E(u_f, v_f, u_s, v_s):
        F_f = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J_f = ufl.det(F_f)

        residual = rho_f * J_f * ufl.inner(ufl.grad(v_f) * ufl.inv(F_f) * v_f, dv_f) * dx_fluid

        residual += (
            ufl.inner(J_f * Fluid.NS_velocity(u_f, v_f, nu_f, rho_f) * ufl.inv(F_f).T, ufl.grad(dv_f)) * dx_fluid
        )

        F_s = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_s)
        J_s = ufl.det(F_s)

        residual += ufl.inner(J_s * Solid.STVK(u_s, lambda_s, mu_s) * ufl.inv(F_s).T, ufl.grad(dv_s)) * dx_solid

        residual -= rho_s * ufl.inner(v_s, du_s) * dx_solid

        return residual

    def A_P(u_f, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv_f)) * dx_fluid

        return residual

    residual = A_T(u_f, u_f_old, v_f, v_f_old, u_s, u_s_old, v_s, v_s_old)
    residual += A_I(u_f, v_f, p)
    residual += A_P(u_f, p)
    residual += theta * A_E(u_f, v_f, u_s, v_s)
    residual += (1.0 - theta) * A_E(u_f_old, v_f_old, u_s_old, v_s_old)

    # --------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])

    # Lagrange-multiplier interface coupling: enforce u_f = u_s and v_f = v_s
    # weakly on the solid-fluid interface, replacing the continuity that the
    # old shared function spaces gave for free.

    residual += ufl.inner(lambda_u, du_f) * ds_interface_fluid
    residual -= ufl.inner(lambda_u, du_s) * ds_interface_solid
    residual += ufl.inner(dlambda_u, u_f) * ds_interface_fluid
    residual -= ufl.inner(dlambda_u, u_s) * ds_interface_solid

    residual += ufl.inner(lambda_v, dv_f) * ds_interface_fluid
    residual -= ufl.inner(lambda_v, dv_s) * ds_interface_solid
    residual += ufl.inner(dlambda_v, v_f) * ds_interface_fluid
    residual -= ufl.inner(dlambda_v, v_s) * ds_interface_solid

    residual_blocked = ufl.extract_blocks(residual)
    # Cap the quadrature degree before the Jacobian is derived from the residual, so that it keeps the cap.
    residual_blocked = [cap_quadrature_degree(form, max_quadrature_degree) for form in residual_blocked]

    # Set up checkpointing
    #
    # The Lagrange multipliers lambda_u, lambda_v are deliberately left out of
    # the checkpoint, and a restart starts them from zero, as at t = t0. They
    # live on the interface submesh gamma_mesh, a facet (codimension 1)
    # submesh, which Checkpointer doesn't support, and they have no _old
    # values, so they only serve as the Newton initial guess. They enter the
    # residual linearly with constant coefficients, so in exact arithmetic
    # the iterates after the first Newton update don't depend on them.
    #
    # In practice, restarting them from zero changes the fluid mesh displacement
    # u_f by ~1e-6 relative -- as much as tightening the Newton atol does,
    # since the u_f equation is scaled by alpha_u = 1e-9 -- and the other
    # fields by < 1e-9 relative. That is at the level the solver resolves
    # u_f anyway, and its effect on the flow through the mesh velocity
    # (u_f - u_f_old) / dt is included in those < 1e-9 changes in v_f and p;
    # drag, lift and tip displacement QoIs are unchanged
    # (notes/checkpointing/implementation-log.md).

    checkpointed = [u_f, v_f, u_s, v_s, p]
    output_path_solid = output_path.replace(".bp", "_solid.bp")

    # (t, step) are the time and number of completed steps of the current state
    t = t0
    step = 0
    checkpointer = None
    if checkpoint_dir is not None:
        checkpointer = Checkpointer(
            checkpoint_dir, mesh, submeshes=[(fluid_mesh, fluid_cell_map), (solid_mesh, solid_cell_map)]
        )
    if restart:
        t, step = checkpointer.read(checkpointed, dt_val)
        if step >= num_steps:
            raise ValueError(
                f"Cannot restart from the checkpoint at {t = :.4f} ({step = }): no time step is left to solve "
                f"up to {T = } ({num_steps = })"
            )
        output_path = restart_output_path(output_path, t)
        output_path_solid = restart_output_path(output_path_solid, t)
        output_path_p = restart_output_path(output_path_p, t)
        if comm.rank == 0:
            vtx_message = (
                ""
                if vtx_save_every is None
                else f", writing VTX output to {output_path}, {output_path_solid} and {output_path_p}"
            )
            print(f"Restarting from {checkpoint_dir} at {t = :.4f} ({step = }){vtx_message}")
    elif checkpoint_every is not None:
        checkpointer.clear()
    if initial_state_dir is not None:
        t, _ = Checkpointer(
            initial_state_dir, mesh, submeshes=[(fluid_mesh, fluid_cell_map), (solid_mesh, solid_cell_map)]
        ).read(checkpointed, None)
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

    # Set up output

    writers = []
    if vtx_save_every is not None:
        policy = dolfinx.io.VTXMeshPolicy.reuse
        writers = [
            dolfinx.io.VTXWriter(comm, output_path, [u_f, v_f], mesh_policy=policy, engine="BP4"),
            dolfinx.io.VTXWriter(comm, output_path_solid, [u_s, v_s], mesh_policy=policy, engine="BP4"),
            dolfinx.io.VTXWriter(comm, output_path_p, [p], mesh_policy=policy, engine="BP4"),
        ]

    # Quantities of interest: the displacement at the tip of the structure, the drag and lift on
    # the obstacle and on the fluid side of the solid-fluid interface, and the violation of the
    # interface continuity, which the Lagrange multipliers should keep close to zero.
    spot_dof = find_point_dof(U_s, np.array([0.6, 0.2, 0.0], dtype=np.float64))
    drag_forms, lift_forms = drag_lift_forms(
        mesh,
        u_f,
        v_f,
        p,
        nu_f,
        rho_f,
        [ds(PHYSICAL_MARKERS["obstacle"]), ds_interface_fluid],
        entity_maps,
        max_quadrature_degree=max_quadrature_degree,
    )
    continuity_u_form, continuity_v_form = interface_gap_forms(
        u_f, u_s, v_f, v_s, fluid_side, solid_side, dS_interface, entity_maps
    )

    def write_output(t, step):
        """Write the QoI row, and every vtx_save_every steps the VTX snapshot, of the state at time t."""
        if vtx_save_every is not None and step % vtx_save_every == 0:
            for writer in writers:
                writer.write(t)

        u_spot = point_value(u_s, spot_dof)
        drag = assemble_force(drag_forms, comm)
        lift = assemble_force(lift_forms, comm)
        interface_u_gap = assemble_gap_norm(continuity_u_form, comm)
        interface_v_gap = assemble_gap_norm(continuity_v_form, comm)
        append_qoi_row(qoi_path, comm, t, drag, lift, u_spot, interface_u_gap, interface_v_gap)

    # Set up solver

    max_iter = 20
    atol = 1.0e-7
    rtol = 1.0e-12

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocked,
        [u_f, v_f, u_s, v_s, p, lambda_u, lambda_v],
        bcs=bcs,
        entity_maps=entity_maps,
        petsc_options_prefix="fsi2_harmonic_lagrange_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_linesearch_type": "none",
            "snes_max_it": max_iter,
            "snes_atol": atol,
            "snes_rtol": rtol,
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
            "snes_converged_reason": None,
        },
    )

    b_vec, *_ = problem.solver.getFunction()

    init_qoi_file(qoi_path, comm, restart, t, dt_val, header=LAGRANGE_QOI_HEADER)

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

        u_f_old.x.array[:] = u_f.x.array[:]
        v_f_old.x.array[:] = v_f.x.array[:]
        u_s_old.x.array[:] = u_s.x.array[:]
        v_s_old.x.array[:] = v_s.x.array[:]

        problem.solve()

        converged_reason = problem.solver.getConvergedReason()
        converged = converged_reason > 0
        n = problem.solver.getIterationNumber()
        res = b_vec.norm()
        if comm.rank == 0:
            print(f"{t = :.3f}", end="\t")
            print(f"{n = :02d}", end="\t")
            print(f"{res = :.3e}", end="\t")
            print(f"{converged = }", end="\n")
            sys.stdout.flush()

        check_converged(problem, f"t = {t:.4f}", writers=writers)

        step += 1
        write_output(t, step)

        if checkpoint_every is not None and step % checkpoint_every == 0:
            checkpointer.write(checkpointed, t, step, dt_val, t0)

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
    T = 15.0
    dt_val = 0.0025
    output_path = "output/pv/fsi2_harmonic_lagrange.bp"
    output_path_p = "output/pv/fsi2_harmonic_lagrange_p.bp"
    qoi_path = "output/qoi/fsi2_harmonic_lagrange.txt"
    checkpoint_dir = "output/checkpoints/fsi2_harmonic_lagrange"
    checkpoint_every = 100
    vtx_save_every = 8
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
        plot_run_qois(qoi_path, T, qoi_labels=LAGRANGE_QOI_LABELS)


if __name__ == "__main__":
    main()
