import sys
from pathlib import Path
from timeit import default_timer as timer

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from xfsi_solver.tools.convergence import check_converged

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
    restart=False,
):
    """Solve the FSI2 benchmark up to time ``T``, in ``round((T - t0) / dt_val)`` time
    steps, writing the initial state at ``t0`` as the first QoI row and VTX snapshot.

    With ``checkpoint_dir`` and ``checkpoint_every``, a restart checkpoint of
    the state is saved in ``checkpoint_dir`` every ``checkpoint_every`` time
    steps (replacing checkpoints there from earlier runs). With
    ``restart=True``, the run instead continues from the latest checkpoint in
    ``checkpoint_dir``, on any number of MPI ranks: QoI rows after the
    checkpoint time are dropped from ``qoi_path`` before appending, and the
    VTX output is written to new files with a ``_from_t<time>`` suffix, since
    ``VTXWriter`` cannot append.

    The Lagrange multipliers ``lambda_u`` and ``lambda_v`` are not
    checkpointed, so a restart starts them from zero; see the comment where
    the checkpointer is set up.
    """
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

    rho_f = dolfinx.fem.Constant(mesh, 1.0e3)
    nu_f = dolfinx.fem.Constant(mesh, 1.0e-3)

    rho_s = dolfinx.fem.Constant(mesh, 1.0e4)
    mu_s = dolfinx.fem.Constant(mesh, 5.0e5)
    nu_s = dolfinx.fem.Constant(mesh, 0.4)
    lambda_s = dolfinx.fem.Constant(mesh, -mu_s.value / (1 - 0.5 / nu_s.value))
    # lambda_s = dolfinx.fem.Constant(mesh, 2e6)

    U_bar = 1.0
    H = 0.41

    t0 = 0.0
    dt = dolfinx.fem.Constant(mesh, dt_val)

    theta = dolfinx.fem.Constant(mesh, 0.5 + dt.value)

    save_every = 8
    num_steps = round((T - t0) / dt_val)

    # create function spaces
    #
    # u/v are no longer shared between the solid and fluid subdomains: each
    # subdomain gets its own displacement and velocity space, and continuity
    # across the interface is enforced weakly through the Lagrange
    # multiplier spaces Lu, Lv (defined on the interface submesh gamma_mesh)
    # instead of through shared degrees of freedom.

    U_f = dolfinx.fem.functionspace(fluid_mesh, ("CG", 2, (2,)))
    V_f = dolfinx.fem.functionspace(fluid_mesh, ("CG", 2, (2,)))
    U_s = dolfinx.fem.functionspace(solid_mesh, ("CG", 2, (2,)))
    V_s = dolfinx.fem.functionspace(solid_mesh, ("CG", 2, (2,)))
    P = dolfinx.fem.functionspace(fluid_mesh, ("CG", 1))
    Lu = dolfinx.fem.functionspace(gamma_mesh, ("CG", 1, (2,)))
    Lv = dolfinx.fem.functionspace(gamma_mesh, ("CG", 1, (2,)))
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

    from xfsi_solver.tools.checkpoint import Checkpointer, restart_output_path, truncate_qoi_file

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
        output_path = restart_output_path(output_path, t)
        output_path_solid = restart_output_path(output_path_solid, t)
        output_path_p = restart_output_path(output_path_p, t)
        if comm.rank == 0:
            print(
                f"Restarting from {checkpoint_dir} at {t = :.4f} ({step = }), "
                f"writing VTX output to {output_path}, {output_path_solid} and {output_path_p}"
            )
    elif checkpoint_every is not None:
        checkpointer.clear()

    # Set up output

    policy = dolfinx.io.VTXMeshPolicy.reuse
    writer = dolfinx.io.VTXWriter(comm, output_path, [u_f, v_f], mesh_policy=policy, engine="BP4")
    writer_solid = dolfinx.io.VTXWriter(comm, output_path_solid, [u_s, v_s], mesh_policy=policy, engine="BP4")
    writer_p = dolfinx.io.VTXWriter(comm, output_path_p, [p], mesh_policy=policy, engine="BP4")

    dm_loc_size = U_s.dofmap.index_map.size_local
    spot = np.array([0.6, 0.2, 0.0], dtype=np.float64)
    spot_dof_cand = np.flatnonzero(
        np.all(np.isclose(U_s.tabulate_dof_coordinates()[:dm_loc_size, :], spot, atol=1e-6), axis=1)
    )
    spot_dof = spot_dof_cand[0] if len(spot_dof_cand) > 0 else None
    assert comm.allreduce(len(spot_dof_cand), op=MPI.SUM) == 1, "None or multiple dofs found for measurement point"

    loc_u_spot = np.zeros(2, dtype=np.float64)

    normal = ufl.FacetNormal(mesh)
    e_x = dolfinx.fem.Constant(mesh, (-1.0, 0.0))
    e_y = dolfinx.fem.Constant(mesh, (0.0, 1.0))
    F = ufl.Identity(2) + ufl.grad(u_f)
    transformed_normal = ufl.dot(ufl.inv(F.T), normal)

    drag_form_obstacle = (
        ufl.dot(ufl.dot(Fluid.NS(u_f, v_f, p, nu_f, rho_f), transformed_normal), e_x)
        * ufl.det(F)
        * ds(PHYSICAL_MARKERS["obstacle"])
    )
    lift_form_obstacle = (
        ufl.dot(ufl.dot(Fluid.NS(u_f, v_f, p, nu_f, rho_f), transformed_normal), e_y)
        * ufl.det(F)
        * ds(PHYSICAL_MARKERS["obstacle"])
    )

    drag_form_interface = (
        ufl.dot(ufl.dot(Fluid.NS(u_f, v_f, p, nu_f, rho_f), transformed_normal), e_x) * ufl.det(F) * ds_interface_fluid
    )
    lift_form_interface = (
        ufl.dot(ufl.dot(Fluid.NS(u_f, v_f, p, nu_f, rho_f), transformed_normal), e_y) * ufl.det(F) * ds_interface_fluid
    )

    drag_form_obstacle = dolfinx.fem.form(drag_form_obstacle, entity_maps=entity_maps)
    drag_form_interface = dolfinx.fem.form(drag_form_interface, entity_maps=entity_maps)
    lift_form_obstacle = dolfinx.fem.form(lift_form_obstacle, entity_maps=entity_maps)
    lift_form_interface = dolfinx.fem.form(lift_form_interface, entity_maps=entity_maps)

    # QOI: L2 norm of the interface continuity violation u_f - u_s / v_f - v_s.
    # Should stay close to zero (up to solver tolerance) if the Lagrange
    # multiplier coupling is enforcing continuity correctly.

    u_gap = u_f(fluid_side) - u_s(solid_side)
    v_gap = v_f(fluid_side) - v_s(solid_side)
    continuity_u_form = dolfinx.fem.form(ufl.inner(u_gap, u_gap) * dS_interface, entity_maps=entity_maps)
    continuity_v_form = dolfinx.fem.form(ufl.inner(v_gap, v_gap) * dS_interface, entity_maps=entity_maps)

    if comm.rank == 0:
        Path(qoi_path).parent.mkdir(parents=True, exist_ok=True)
        if restart and Path(qoi_path).exists():
            truncate_qoi_file(qoi_path, t, dt_val)
        else:
            with open(qoi_path, "wb") as f:
                np.savetxt(
                    f,
                    [],
                    fmt="%.6e",
                    delimiter="\t",
                    header="t\tdrag\tlift\tA_x\tA_y\tinterface_u_gap\tinterface_v_gap",
                )

    # Set up solver

    max_iter = 20
    atol = 1.0e-7
    rtol = 1.0e-12

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocked,
        [u_f, v_f, u_s, v_s, p, lambda_u, lambda_v],
        bcs=bcs,
        entity_maps=entity_maps,
        petsc_options_prefix="fsi2_harm_lg_",
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
            "snes_monitor": None,
        },
    )

    b_vec, *_ = problem.solver.getFunction()

    def write_output(t, step):
        """Write the QoI row, and every save_every steps the VTX snapshot, of the state at time t."""
        if step % save_every == 0:
            writer.write(t)
            writer_solid.write(t)
            writer_p.write(t)

        loc_u_spot[:] = u_s.x.array[2 * spot_dof : 2 * (spot_dof + 1)] if spot_dof is not None else 0.0
        u_spot = comm.reduce(loc_u_spot, op=MPI.SUM, root=0)
        drag = comm.reduce(
            dolfinx.fem.assemble_scalar(drag_form_obstacle) + dolfinx.fem.assemble_scalar(drag_form_interface)
        )
        lift = comm.reduce(
            dolfinx.fem.assemble_scalar(lift_form_obstacle) + dolfinx.fem.assemble_scalar(lift_form_interface)
        )
        interface_u_gap = comm.reduce(dolfinx.fem.assemble_scalar(continuity_u_form))
        interface_v_gap = comm.reduce(dolfinx.fem.assemble_scalar(continuity_v_form))
        if comm.rank == 0:
            interface_u_gap = np.sqrt(max(interface_u_gap, 0.0))
            interface_v_gap = np.sqrt(max(interface_v_gap, 0.0))
            with open(qoi_path, "ab") as f:
                np.savetxt(f, [[t, drag, lift, *u_spot, interface_u_gap, interface_v_gap]], fmt="%.6e", delimiter="\t")

    if not restart:
        write_output(t, step)

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

        check_converged(problem, f"t = {t:.4f}", writers=[writer, writer_solid, writer_p])

        step += 1
        write_output(t, step)

        if checkpoint_every is not None and step % checkpoint_every == 0:
            checkpointer.write(checkpointed, t, step, dt_val)

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")

    writer.close()
    writer_solid.close()
    writer_p.close()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    args = parser.parse_args()

    solve(
        mesh_path="data/meshes/fsi2/mesh_sec.xdmf",
        T=15.0,
        dt_val=0.0025,
        output_path="output/pv/fsi2_harm_lg.bp",
        output_path_p="output/pv/fsi2_harm_lg_p.bp",
        qoi_path="output/qoi/fsi2_harm_lg_qoi.txt",
        checkpoint_dir="output/checkpoints/fsi2_harm_lg",
        checkpoint_every=100,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
