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

from xfsi_solver.scripts.plot_qois import plot_run_qois
from xfsi_solver.tools.checkpoint import check_start_options, load_start_state, start_time
from xfsi_solver.tools.convergence import KSPConvCheck, check_converged
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
    elements={  # noqa: B006
        "U": ("P", 2, (2,)),
        "V": ("P", 2, (2,)),
        "P": ("P", 1),
        "Z": ("P", 2, (2,)),
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
    max_quadrature_degree=None,
):
    """A version of the solver where all the terms with test function `u` (and `z`) are removed from the residual.

    The arguments are those of ``fsi2_biharmonic_mixed.solve``, and the Newton system is solved by the
    same multiplicative fieldsplit over q_c = (v, p), u_S and m = (z, u_I). Without the rows of u,
    the u_S and m splits are solved by Python PCs that set the displacement of the next Newton
    iterate directly: ``NodalUpdatePC`` from the kinematic relation, node by node, at the new
    velocity, and ``SeparateSolverPC`` by extending it into the fluid with the separate biharmonic
    solver of ``xfsi_solver.component_solvers.biharm`` on the fluid mesh. With exact solves in these
    splits, as in fsi2_biharmonic_mixed, the Newton iterates are the same as in that solver.

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

    empty_tag = 999
    if empty_tag in cell_tags.values:
        raise ValueError()
    dx_empty = dx(empty_tag)

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
    # breakpoint()
    V = dolfinx.fem.functionspace(mesh, elements["V"])
    P = dolfinx.fem.functionspace(fluid_mesh, elements["P"])
    U = dolfinx.fem.functionspace(mesh, elements["U"])
    Z = dolfinx.fem.functionspace(fluid_mesh, elements["Z"])
    W = ufl.MixedFunctionSpace(V, P, U, Z)

    # create functions

    v, p, u, z = (
        dolfinx.fem.Function(V, name="v"),
        dolfinx.fem.Function(P, name="p"),
        dolfinx.fem.Function(U, name="u"),
        dolfinx.fem.Function(Z, name="z"),
    )
    v_old, u_old = dolfinx.fem.Function(V), dolfinx.fem.Function(U)

    ordering = {"V": 0, "P": 1, "U": 2, "Z": 3}

    fields = (v, p, u, z)
    trials = ufl.TrialFunctions(W)
    tests = ufl.TestFunctions(W)

    dv, dp, du, dz = tests
    trial_v, trial_p, trial_u, trial_z = trials

    # Set up output, checkpointing, and qoi tracking.

    # (t, step) are the time and number of completed steps of the current state
    t, step, checkpointer, (output_path, output_path_p) = load_start_state(
        mesh,
        fields,
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

    # Collect Dirichlet boundary conditions to pass to NonlinearProblem

    bcs = [u_s_bc, u_f_bc, inflow_bc, noslip_bc]

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

    def A_I(u, v, z):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = ufl.inner(z, dz) * dx_fluid
        residual -= ufl.inner(ufl.grad(u), ufl.grad(dz)) * dx_fluid

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
    # residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(u, du) * dx_solid
    # residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(v, du) * dx_solid

    # The term we do separately in the other solvers
    # residual_post_u = rho_s * ufl.inner((u - u_old) / dt - theta * v - (1 - theta) * v_old, du) * dx_solid

    residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(u, du) * dx_empty
    residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(z, dz) * dx_empty

    # Add this for now to ensure that bcs work properly.
    residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(u, du) * ds

    residual_blocks = ufl.extract_blocks(residual)
    # Cap the quadrature degree before taking derivatives, so that the Jacobian blocks keep it.
    residual_blocks = [cap_quadrature_degree(form, max_quadrature_degree) for form in residual_blocks]

    jacobian = dolfinx.fem.forms.derivative_block(residual_blocks, fields)
    jacobian[ordering["V"]][ordering["U"]] = None
    jacobian[ordering["P"]][ordering["U"]] = None

    # --------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocks,
        fields,
        J=jacobian,
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
        },
    )

    # Add extra callbacks to change how residual and jacobian is assembled,
    # to account for test function restriction on the interface.

    solver = problem.solver
    b_vec, (fem_residual, res_args, res_kargs) = solver.getFunction()
    J_mat, P_mat, (fem_jacobian, jac_args, jac_kargs) = solver.getJacobian()

    def pre_jacobian(x: PETSc.Vec, J: PETSc.Mat) -> None:
        pass

    def post_jacobian(x: PETSc.Vec, J: PETSc.Mat) -> None:
        pass

    def pre_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:
        pass

    def post_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:
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

    offsets_owned, _ = problem.b.getAttr("_blocks")
    first_row = J_mat.getOwnershipRange()[0]

    def owned_rows(k):
        """Global rows of the owned dofs of block k in fields."""
        return first_row + np.arange(offsets_owned[k], offsets_owned[k + 1])

    rows_v, rows_p, rows_u, rows_z = (owned_rows(k) for k in range(len(fields)))
    rows_q_c = np.concatenate([rows_v, rows_p])
    u_solid = solid_supported(U, len(rows_u))  # mask over owned unrolled u dofs
    rows_u_S = rows_u[u_solid]
    rows_m = np.concatenate([rows_z, rows_u[~u_solid]])

    if not len(rows_q_c) + len(rows_u_S) + len(rows_m) == J_mat.getLocalSize()[0]:
        raise RuntimeError()

    def make_is(rows):
        return PETSc.IS().createGeneral(np.sort(rows).astype(PETSc.IntType), comm=comm)

    start = comm.exscan(len(rows_q_c)) or 0  # first row of this rank in the q_c submatrix
    is_v = make_is(start + np.arange(len(rows_v)))
    is_p = make_is(start + len(rows_v) + np.arange(len(rows_p)))

    def split_positions(rows):
        """Local positions of rows in the vectors of the split whose submatrix holds rows, which
        stores them in sorted order."""
        positions = np.empty(len(rows), dtype=np.int64)
        positions[np.argsort(rows)] = np.arange(len(rows))
        return positions

    # Owned unrolled u dofs of the solid (u_S) and of the rest (u_I), each in the order of its split.
    dofs_u_S, dofs_u_I = np.flatnonzero(u_solid), np.flatnonzero(~u_solid)
    positions_v = split_positions(rows_q_c)[: len(rows_v)]  # of the owned v dofs in q_c
    positions_m = split_positions(rows_m)
    positions_z, positions_u_I = positions_m[: len(rows_z)], positions_m[len(rows_z) :]  # in m
    if not np.array_equal(split_positions(rows_u_S), np.arange(len(rows_u_S))):
        raise RuntimeError()

    # Mask of the owned u dofs with a Dirichlet condition, at whose rows the PCs below return the
    # right-hand side x - g, as dolfinx's Newton system does with its unit diagonal there.
    u_dirichlet = np.zeros(len(rows_u), dtype=bool)
    for bc in (u_s_bc, u_f_bc):
        bc_dofs, num_owned = bc.dof_indices()  # unrolled local indices, owned first
        u_dirichlet[bc_dofs[:num_owned]] = True

    # The owned v dofs are stored interleaved per node, (x, y), so A has block size 2, which
    # BoomerAMG uses for its systems coarsening and GAMG for its aggregates.
    is_v.setBlockSize(V.dofmap.index_map_bs)

    from xfsi_solver.tools.rigid_body_modes import rigid_body_modes

    # PCFIELDSPLIT attaches this to the velocity block A when it extracts it.
    is_v.compose("nearnullspace", rigid_body_modes(V))

    # The multiplicative fieldsplit solves the splits in the order they are set here,
    # independent of the storage order [v | p | u | z]: first K_c for (v, p), then u_S by the
    # nodal update from delta v, then m = (z, u_I) by the separate mesh motion solver from u_S.
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
        raise NotImplementedError()
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

    # Solve mass update using nodal update
    mass_solve_options = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "ksp_reuse_preconditioner": True,
    }

    # Solve mesh motion using separate mesh motion solver and python pc-type
    mesh_motion_solve_options = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "ksp_reuse_preconditioner": True,
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

    # The residual has no rows for u, except the Dirichlet rows, so the u_S and m splits get no
    # information from their right-hand sides. Instead, their PCs compute the Newton correction y
    # (which SNES subtracts from the iterate) of u and z from the current iterate, held by the
    # fields since the residual was assembled at it, and from the corrections of the splits solved
    # before them. They are not linear preconditioners, so the outer KSP must be preonly.

    # The displacement of the next Newton iterate, from NodalUpdatePC for SeparateSolverPC.
    u_next = dolfinx.fem.Function(U)

    class NodalUpdatePC:
        """PCPYTHON context on the u_S split, which updates the solid displacement node by node by the
        kinematic relation u = u_old + dt * (theta * v + (1 - theta) * v_old), at the velocity
        v - y_v of the next Newton iterate. y_v is the solution of the q_c split, which the
        multiplicative fieldsplit solves first. The correction is

            y_S = u - u_old - dt * (theta * v + (1 - theta) * v_old) + theta * dt * y_v,

        as in eq. (5) of docs/restricted-iterative-solver.md, with the nodal kinematic defect g_S
        taken from the iterate instead of the residual. u and v have the same element, so their
        dofs match one to one. At the Dirichlet rows, the correction is the right-hand side.
        The displacement of the next iterate is written to u_next.
        """

        def __init__(self, ksp_q_c: PETSc.KSP, dofs_u_S, positions_v_S, positions_dirichlet):
            self.ksp_q_c = ksp_q_c
            self.dofs_u_S = dofs_u_S  # owned unrolled u (and v) dofs, in the order of the split
            self.positions_v_S = positions_v_S  # of the same v dofs in the q_c split
            self.positions_dirichlet = positions_dirichlet  # of the Dirichlet rows in the split

        def setUp(self, pc: PETSc.PC) -> None:
            pass

        def apply(self, pc: PETSc.PC, b: PETSc.Vec, x: PETSc.Vec) -> None:
            dofs = self.dofs_u_S
            y_v = self.ksp_q_c.getSolution().getArray(readonly=True)[self.positions_v_S]
            defect = u.x.array[dofs] - u_old.x.array[dofs]
            defect -= dt_val * (theta_val * v.x.array[dofs] + (1.0 - theta_val) * v_old.x.array[dofs])
            y = defect + theta_val * dt_val * y_v
            y[self.positions_dirichlet] = b.getArray(readonly=True)[self.positions_dirichlet]
            x.array[:] = y

            u_next.x.array[:] = u.x.array
            u_next.x.array[dofs] -= y
            u_next.x.scatter_forward()
            u_f_bc.set(u_next.x.array)  # the fluid boundary values of the next iterate

        def destroy_solvers(self) -> None:
            pass

    class SeparateSolverPC:
        """PCPYTHON context on the m split, which solves the mesh motion with a separate solver on
        the fluid mesh: it extends the displacement u_next of the next Newton iterate from the
        boundary of the fluid mesh, which includes the interface, into the fluid, and returns the
        corrections of u_I and z from the current iterate to that extension. The mesh motion
        equations are linear, so the iterate then satisfies them exactly, as with an exact solve of
        K_m in fsi2_biharmonic_mixed. At the Dirichlet rows, the correction is the right-hand side.

        ``solver`` is the tuple (u_h, z_h, problem) of the separate solver, with the displacement
        u_h and the auxiliary field z_h on the fluid mesh, in which ``problem.solve()`` extends the
        boundary values of ``u_bc``. Its matrix is assembled once, on the reference fluid mesh.
        """

        def __init__(self, solver, u_bc, cells, positions_u_I, positions_z, positions_dirichlet):
            self.u_h, self.z_h, self.problem = solver
            self.u_bc = u_bc
            self.parent_cells, self.sub_cells = cells  # the fluid cells on the mesh and the fluid mesh
            self.positions_u_I = positions_u_I  # of the owned u_I dofs in the split
            self.positions_z = positions_z  # of the owned z dofs in the split
            self.positions_dirichlet = positions_dirichlet  # of the Dirichlet rows in the split
            self.u_new = dolfinx.fem.Function(U)
            self.z_new = dolfinx.fem.Function(Z)

        def setUp(self, pc: PETSc.PC) -> None:
            pass

        def apply(self, pc: PETSc.PC, b: PETSc.Vec, x: PETSc.Vec) -> None:
            self.u_bc.interpolate(u_next, cells0=self.parent_cells, cells1=self.sub_cells)
            self.problem.solve()
            self.u_new.interpolate(self.u_h, cells0=self.sub_cells, cells1=self.parent_cells)
            self.z_new.interpolate(self.z_h)

            x_array = x.array
            x_array[self.positions_u_I] = u.x.array[dofs_u_I] - self.u_new.x.array[dofs_u_I]
            num_z = len(self.positions_z)
            x_array[self.positions_z] = z.x.array[:num_z] - self.z_new.x.array[:num_z]
            x_array[self.positions_dirichlet] = b.getArray(readonly=True)[self.positions_dirichlet]

        def destroy_solvers(self) -> None:
            """Destroy the separate solver, collectively."""
            for obj in (self.problem.solver, self.problem.A, self.problem.b, self.problem.x):
                obj.destroy()

    if not (U.dofmap.index_map_bs == V.dofmap.index_map_bs and np.array_equal(U.dofmap.list, V.dofmap.list)):
        raise ValueError(
            f"The nodal update needs the same element for U and V, not {elements['U']} and {elements['V']}"
        )
    mass_pc = NodalUpdatePC(ksp_q_c, dofs_u_S, positions_v[dofs_u_S], np.flatnonzero(u_dirichlet[dofs_u_S]))
    ksp_u_S.getPC().setPythonContext(mass_pc)

    from xfsi_solver.component_solvers.biharm import biharmonic

    # The separate solver applies its Dirichlet condition on all exterior facets of the fluid mesh.
    fluid_mesh.topology.create_connectivity(fluid_mesh.topology.dim - 1, fluid_mesh.topology.dim)
    U_fluid = dolfinx.fem.functionspace(fluid_mesh, elements["U"])
    u_mesh_bc = dolfinx.fem.Function(U_fluid)
    fluid_cell_index_map = fluid_mesh.topology.index_map(fluid_mesh.topology.dim)
    sub_cells = np.arange(fluid_cell_index_map.size_local + fluid_cell_index_map.num_ghosts, dtype=np.int32)
    parent_cells = fluid_cell_map.sub_topology_to_topology(sub_cells, inverse=False)

    mesh_motion_pc = SeparateSolverPC(
        biharmonic(u_mesh_bc),
        u_mesh_bc,
        (parent_cells, sub_cells),
        positions_u_I,
        positions_z,
        positions_u_I[u_dirichlet[dofs_u_I]],
    )
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
            checkpointer.write(fields, t, step, dt_val, t0)

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")
        print(f"\nSolver setup:\n  Quadrature degree: {quadrature_description(max_quadrature_degree)}\n")

    for writer in writers:
        writer.close()

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

    mesh_path = "data/meshes/fsi2/mesh_sec.xdmf"
    T = 16.0 + 10 * 0.0025
    dt_val = 0.0025
    output_path = "output/pv/fsi2_biharmonic_mixed_simple.bp"
    output_path_p = "output/pv/fsi2_biharmonic_mixed_simple_p.bp"
    qoi_path = "output/qoi/fsi2_biharmonic_mixed_simple.txt"
    checkpoint_dir = "output/checkpoints/fsi2_biharmonic_mixed_simple"
    checkpoint_every = 400
    vtx_save_every = None
    plot_matrices = False
    gamma_reassemble = 0.2  # Default 0.2.
    direct_k_c_solve = True
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
        plot_matrices=plot_matrices,
        gamma_reassemble=gamma_reassemble,  # Default 0.2.
        direct_k_c_solve=direct_k_c_solve,
        max_quadrature_degree=max_quadrature_degree,  # Default None, no cap.
        restart=restart,
        initial_state_dir=initial_state_dir,
    )

    if MPI.COMM_WORLD.rank == 0:
        plot_run_qois(qoi_path, T)


if __name__ == "__main__":
    main()
