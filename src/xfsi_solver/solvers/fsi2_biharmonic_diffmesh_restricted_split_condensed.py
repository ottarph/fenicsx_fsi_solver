# Copyright (C) 2025 Ottar Hellan
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

from xfsi_solver.tools.convergence import check_converged
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
    checkpoint_dir=None,
    checkpoint_every=None,
    restart=False,
    gamma_reassemble=0.2,
):
    """Solve the FSI2 benchmark up to time ``T``, in ``round((T - t0) / dt_val)`` time
    steps, writing the initial state at ``t0`` as the first QoI row and VTX snapshot.

    The solid stress in the momentum equation is written in terms of the velocity, with
    the solid displacement given by the kinematic relation u = u_old + dt * (theta * v
    + (1 - theta) * v_old) (Failer & Richter, J. Sci. Comput. 82:28, 2020, eq. (5)). The
    momentum equation then no longer depends on the solid displacement u_S, and its
    derivative with respect to v contains the solid stiffness. The approximate Jacobian
    J_0 drops the remaining derivatives of the momentum and continuity equations with
    respect to u, which are the fluid ALE derivatives, and is block lower triangular in
    (v, p), u_S, (z, u_I). Each Newton step is solved by a multiplicative fieldsplit in
    that order, with MUMPS LU on K_c and K_m, and CG with Jacobi on A_S. A_S (the solid
    mass) and K_m (the mesh motion) are constant, so their preconditioners are set up
    once for the whole run.

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

    rho_f = dolfinx.fem.Constant(mesh, 1.0e3)
    nu_f = dolfinx.fem.Constant(mesh, 1.0e-3)

    rho_s = dolfinx.fem.Constant(mesh, 1.0e4)
    mu_s = dolfinx.fem.Constant(mesh, 5.0e5)
    nu_s = dolfinx.fem.Constant(mesh, 0.4)
    lambda_s = dolfinx.fem.Constant(mesh, -mu_s.value / (1 - 0.5 / nu_s.value))
    assert np.isclose(lambda_s.value, 2e6), "Lambda value is not as expected"
    # lambda_s = dolfinx.fem.Constant(mesh, 2e6)

    U_bar = 1.0
    H = 0.41

    t0 = 0.0
    dt = dolfinx.fem.Constant(mesh, dt_val)

    theta = dolfinx.fem.Constant(mesh, 0.5 + dt.value)

    save_every = 4

    num_steps = round((T - t0) / dt_val)
    if num_steps < save_every:
        warnings.warn(
            f"save_every ({save_every}) is larger than the total number of time "
            f"steps ({num_steps}); at most one VTX snapshot will be written to "
            f"{output_path!r} or {output_path_p!r}, which is not a usable time "
            f"series in ParaView.",
            stacklevel=2,
        )

    # create function spaces

    U = dolfinx.fem.functionspace(mesh, ("CG", 2, (2,)))
    V = dolfinx.fem.functionspace(mesh, ("CG", 2, (2,)))
    P = dolfinx.fem.functionspace(fluid_mesh, ("CG", 1))
    Z = dolfinx.fem.functionspace(fluid_mesh, ("CG", 2, (2,)))
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

    def A_T(u, u_old, v, v_old):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)
        F_old = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_old)
        J_old = ufl.det(F_old)
        J_mid = 0.5 * (J + J_old)

        residual = rho_f * J_mid * ufl.inner((v - v_old) / dt, dv) * dx_fluid

        residual -= rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * ((u - u_old) / dt), dv) * dx_fluid

        residual += rho_s * ufl.inner((v - v_old) / dt, dv) * dx_solid

        return residual

    def A_I(u, v, z):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = ufl.inner(z, dz) * dx_fluid
        residual -= ufl.inner(ufl.grad(u), ufl.grad(dz)) * dx_fluid

        residual += ufl.inner(ufl.grad(z), ufl.grad(du)) * dx_fluid
        residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(u, du) * dx_fluid

        residual += ufl.div(J * ufl.inv(F) * v) * dp * dx_fluid

        return residual

    def A_E_fluid(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx_fluid

        residual += ufl.inner(J * Fluid.NS_velocity(u, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

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

    residual_base_ufl = ufl.extract_blocks(residual)
    jacobian_full_base_ufl = dolfinx.fem.forms.derivative_block(residual_base_ufl, [u, v, p, z])

    jacobian_full_base = dolfinx.fem.form(jacobian_full_base_ufl, entity_maps=entity_maps)

    # One block per unknown in [u, v, p, z], since blocked assembly places forms by
    # position. Not built with ufl.extract_blocks, which drops the zero blocks. The
    # zero Jacobian rows are built directly, since ufl.derivative does not accept a
    # ufl.ZeroBaseForm, with only their diagonal blocks as zero forms, since a zero
    # form cannot couple the parent mesh and the fluid submesh.
    residual_post_u = rho_s * ufl.inner((u - u_old) / dt - theta * v - (1 - theta) * v_old, du) * dx_solid
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

    jacobian_approximate_base = dolfinx.fem.form(jacobian_approximate_base_ufl, entity_maps=entity_maps)

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
            print(
                f"Restarting from {checkpoint_dir} at {t = :.4f} ({step = }), "
                f"writing VTX output to {output_path} and {output_path_p}"
            )
    elif checkpoint_every is not None:
        checkpointer.clear()

    writer = dolfinx.io.VTXWriter(comm, output_path, [u, v])
    writer_p = dolfinx.io.VTXWriter(comm, output_path_p, [p])

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
        """Write the QoI row, and every save_every steps the VTX snapshot, of the state at time t."""
        if step % save_every == 0:
            writer.write(t)
            writer_p.write(t)

        u_spot = point_value(u, spot_dof)
        drag = assemble_force(drag_forms, comm)
        lift = assemble_force(lift_forms, comm)
        append_qoi_row(qoi_path, comm, t, drag, lift, u_spot)

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_base_ufl,
        [u, v, p, z],
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
            # KSP options
            "ksp_type": "preonly",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative",
            # Turn off errors, catch manually instead.
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
            # Print to console.
            "snes_monitor": None,
            # "snes_monitor": "ascii:output/logs/fsi2_biharm_restr_split_snes_log.txt",
        },
    )

    # Add extra callbacks to change how residual and jacobian is assembled,
    # to account for test function restriction on the interface.

    solver = problem.solver
    b_vec, (fem_residual, res_args, res_kargs) = solver.getFunction()
    J_mat, P_mat, (fem_jacobian, jac_args, jac_kargs) = solver.getJacobian()

    # Use this to assemble the approximate jacobian using fem_jacobian.
    jac_kargs_approximate = {key: jac_kargs[key] for key in jac_kargs}
    jac_kargs_approximate["jacobian"] = jacobian_approximate_base

    # Use this to assemble the full jacobian using fem_jacobian.
    jac_kargs_full = {key: jac_kargs[key] for key in jac_kargs}
    jac_kargs_full["jacobian"] = jacobian_full_base

    # Prevent possibly overwriting the sparsity pattern on zeroRows.
    problem.A.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)

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

    def wrapped_jacobian_full(snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat) -> None:
        pre_jacobian(x, J)
        fem_jacobian(snes, x, J, P, *jac_args, **jac_kargs_full)
        post_jacobian(x, J)

    # State of the Jacobian reuse in wrapped_jacobian_approximate. previous_norm is the
    # residual norm at the previous Newton iterate of the current time step.
    jacobian_state = {"assembled": False, "previous_norm": None, "num_assemblies": 0}

    def wrapped_jacobian_approximate(snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat) -> None:
        norm = snes.getFunctionNorm()  # residual norm at the current iterate x
        previous_norm = jacobian_state["previous_norm"] if snes.getIterationNumber() > 0 else None
        jacobian_state["previous_norm"] = norm
        # Reuse J if the last Newton iteration converged fast enough, or at the first iteration
        # of a time step, where there is no rate yet. Leaving J unchanged makes PETSc skip
        # PCSetUp, so the fieldsplit submatrices and the factorization of K_c are reused too.
        if jacobian_state["assembled"] and (previous_norm is None or norm <= gamma_reassemble * previous_norm):
            return
        pre_jacobian(x, J)
        fem_jacobian(snes, x, J, P, *jac_args, **jac_kargs_approximate)
        post_jacobian(x, J)
        jacobian_state["assembled"] = True
        jacobian_state["num_assemblies"] += 1

    def wrapped_residual(snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        pre_residual(x, b)  # x is not yet assigned to u here
        fem_residual(snes, x, b, *res_args, **res_kargs)
        post_residual(x, b)  # u is now updated and b assembled (BCs applied)

    solver.setJacobian(wrapped_jacobian_approximate, J_mat, P_mat)
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

    start = comm.exscan(len(rows_q_c)) or 0  # first row of this rank in the q_c submatrix
    is_v = make_is(start + np.arange(len(rows_v)))
    is_p = make_is(start + len(rows_v) + np.arange(len(rows_p)))

    # The multiplicative fieldsplit solves the splits in the order they are set here,
    # independent of the storage order [u | v | p | z]: first K_c for (v, p), then A_S for
    # u_S with the right-hand side updated by C delta v, then K_m for (z, u_I) with the
    # right-hand side updated by L delta u_S.
    pc = solver.getKSP().getPC()
    pc.setFieldSplitIS(("q_c", make_is(rows_q_c)), ("u_S", make_is(rows_u_S)), ("m", make_is(rows_m)))

    DIRECT_K_C = False

    if DIRECT_K_C:
        velocity_pressure_solve_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_cntl_1": 1e-4,
        }

    else:
        velocity_pressure_solve_options = {
            "ksp_type": "preonly",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            # Since the pressure schur complement, S = -B A^{-1} G, is dense, use gmres with the sparse matrix
            # \hat S = - B diag(A)^{-1} G. B is a divergence and G is close to B^T, a gradient, so this is close
            # to a laplacian.
            "pc_fieldsplit_schur_precondition": "selfp",
            # Use a direct solver on v.
            "fieldsplit_v_ksp_type": "preonly",
            "fieldsplit_v_pc_type": "lu",
            "fieldsplit_v_pc_factor_mat_solver_type": "mumps",
            # Use gmres on p.
            "fieldsplit_p_ksp_type": "gmres",
            "fieldsplit_p_ksp_rtol": 1e-10,
            # Use a mumps LU-factorization on the preconditioner.
            "fieldsplit_p_pc_type": "lu",
            "fieldsplit_p_pc_factor_mat_solver_type": "mumps",
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
    # only once.
    mesh_motion_solve_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": 80,
        "mat_mumps_cntl_1": 1e-4,
        "ksp_reuse_preconditioner": True,
    }

    # Set through the options database, since the fieldsplit calls setFromOptions on its
    # sub-KSPs in PCSetUp, which would override types set directly on them.
    split_options = {"q_c": velocity_pressure_solve_options, "u_S": mass_solve_options, "m": mesh_motion_solve_options}
    opts = PETSc.Options()
    for options, sub_ksp in zip(split_options.values(), pc.getFieldSplitSubKSP(), strict=True):
        opts.prefixPush(sub_ksp.getOptionsPrefix())  # <prefix>fieldsplit_q_c_ / _u_S_ / _m_
        for key, value in options.items():
            opts[key] = value
        opts.prefixPop()
        sub_ksp.setFromOptions()

    if not DIRECT_K_C:
        ksp_q_c, ksp_u_S, ksp_m = pc.getFieldSplitSubKSP()
        ksp_q_c.getPC().setFieldSplitIS(("v", is_v), ("p", is_p))

    # Open qoi file on fresh runs, discard entries after the restart time if a restarted run.
    init_qoi_file(qoi_path, comm, restart, t, dt_val)

    if not restart:
        write_output(t, step)

    if mesh.comm.rank == 0:
        print("", flush=True)
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

        if comm.rank == 0:
            print(f"\n{t = :.3f}")

        problem.solve()
        num_newton_iterations += solver.getIterationNumber()
        check_converged(problem, f"t = {t:.4f}", writers=[writer, writer_p])

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
        print(
            "\nSolver setup:\n"
            "  Solid stress: in terms of v, via u = u_old + dt * (theta * v + (1 - theta) * v_old)\n"
            "  Jacobian: approximate J_0, without the fluid ALE derivatives (eq. (3) of\n"
            "    docs/restricted-iterative-solver.md)\n"
            "  Linear solve: multiplicative fieldsplit over q_c = (v, p), u_S and m = (z, u_I),\n"
            "    MUMPS LU on K_c and K_m, CG with Jacobi on A_S, with the factorization of K_m\n"
            "    reused for the whole run\n"
            f"  Jacobian reuse: reassembled when ||R_l|| > gamma_reassemble * ||R_(l-1)||, "
            f"with {gamma_reassemble = }\n"
            f"  Jacobian assemblies: {jacobian_state['num_assemblies']} in {num_newton_iterations} Newton iterations "
            f"over {step - first_step} time steps"
            "\n"
        )

    writer.close()
    writer_p.close()

    import matplotlib.pyplot as plt

    # Spy plots and block norms of the Jacobian and the approximate Jacobian at the
    # current state, with the dofs reordered block by block (the global numbering is
    # [u, v, p, z] per rank).
    dolfinx.fem.petsc.assign([u, v, p, z], problem.x)

    # gather_jacobian assembles into new matrices, so make wrapped_jacobian_approximate
    # assemble instead of reusing J_mat.
    jacobian_state["assembled"] = False

    def gather_dofs(dofs):
        """Global dofs of all ranks on rank 0, in rank order."""
        dofs = comm.gather(dofs, root=0)
        return np.concatenate(dofs) if comm.rank == 0 else None

    # Owned global dofs of the storage blocks [u, v, p, z]. As in eq. (1) of
    # docs/restricted-iterative-solver.md, u is split into the solid-supported dofs
    # u_S (including the interface) and the remaining fluid-interior dofs u_I.
    row_start = J_mat.getOwnershipRange()[0]
    u_dofs, v_dofs, p_dofs, z_dofs = (
        row_start + np.arange(offsets_owned[k], offsets_owned[k + 1]) for k in range(len(offsets_owned) - 1)
    )
    u_solid = solid_supported(U, len(u_dofs))
    dofs_u_S = gather_dofs(u_dofs[u_solid])
    dofs_u_I = gather_dofs(u_dofs[~u_solid])
    dofs_u = gather_dofs(u_dofs)
    dofs_v = gather_dofs(v_dofs)
    dofs_p = gather_dofs(p_dofs)
    dofs_z = gather_dofs(z_dofs)

    # Make spy plots and block norm plots with the different partitionings.

    def gather_jacobian(assemble_jacobian):
        """Assemble a Jacobian with assemble_jacobian, and return its nonzero global rows,
        columns and values on rank 0.

        Assembles into a new matrix with the full Jacobian sparsity pattern, which contains
        the approximate one. J_mat cannot be used: it is created from the approximate
        Jacobian forms, and PETSc also drops the preallocated entries that its first
        assembly does not set, so it has no room for the full Jacobian.
        """
        J = dolfinx.fem.petsc.create_matrix(jacobian_full_base)
        J.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        assemble_jacobian(solver, problem.x, J, P_mat)
        indptr, cols, vals = J.getValuesCSR()
        J.destroy()
        rows = row_start + np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        nonzero = vals != 0.0
        gathered = [comm.gather(a[nonzero], root=0) for a in (rows, cols, vals)]
        return [np.concatenate(a) for a in gathered] if comm.rank == 0 else (None, None, None)

    rows_full, cols_full, vals_full = gather_jacobian(wrapped_jacobian_full)
    rows_approximate, cols_approximate, vals_approximate = gather_jacobian(wrapped_jacobian_approximate)

    def plot_blocks(J, blocks, names, title, label):
        """Save spy_{label}.png and block_norms_{label}.png of the scipy sparse matrix J,
        reordered by the global dofs in blocks."""
        import matplotlib.colors

        perm = np.concatenate(blocks)
        J_spy = J.tocsr()[perm][:, perm].tocoo()
        J_spy.eliminate_zeros()
        n = J_spy.shape[0]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.spy(J_spy, markersize=0.05, color="black")
        bounds = np.cumsum([0] + [len(b) for b in blocks])
        for b in bounds[1:-1]:
            ax.axhline(b - 0.5, color="tab:red", linewidth=0.8)
            ax.axvline(b - 0.5, color="tab:red", linewidth=0.8)
        centers = 0.5 * (bounds[:-1] + bounds[1:])
        ax.set_xticks(centers, names)
        ax.set_yticks(centers, names)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xlabel(f"{title} nonzeros at t = {t:.4f} ({J_spy.nnz} entries, {n} dofs)", fontsize="large")
        fig.savefig(f"spy_{label}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved spy plot to spy_{label}.png ({J_spy.nnz} nonzeros, {n} dofs)")

        # Frobenius norms of the blocks, with blocks without nonzeros left blank
        row_blocks = np.searchsorted(bounds, J_spy.row, side="right") - 1
        col_blocks = np.searchsorted(bounds, J_spy.col, side="right") - 1
        sq_norms = np.zeros((len(blocks), len(blocks)))
        np.add.at(sq_norms, (row_blocks, col_blocks), J_spy.data**2)
        block_norms = np.ma.masked_equal(np.sqrt(sq_norms), 0.0)

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")
        fig, ax = plt.subplots()
        im = ax.matshow(block_norms, cmap=cmap, norm=matplotlib.colors.LogNorm())
        fig.colorbar(im, ax=ax, label="Frobenius norm")
        for (i, j), norm in np.ndenumerate(block_norms):
            if norm is not np.ma.masked:
                ax.text(j, i, f"{norm:.2e}", ha="center", va="center", color="white", fontsize="small")
        ax.set_xticks(range(len(blocks)), names)
        ax.set_yticks(range(len(blocks)), names)
        ax.set_xlabel(f"{title} block norms at t = {t:.4f}", fontsize="large")
        fig.savefig(f"block_norms_{label}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved block norm plot to block_norms_{label}.png")

    if comm.rank == 0:
        import scipy.sparse

        n = J_mat.getSize()[0]
        J_full = scipy.sparse.coo_matrix((vals_full, (rows_full, cols_full)), shape=(n, n))

        names_eq1 = ["$u_S$", "$v$", "$p$", "$z$", "$u_I$"]
        blocks_eq1 = [dofs_u_S, dofs_v, dofs_p, dofs_z, dofs_u_I]
        plot_blocks(
            J_full,
            [dofs_u, dofs_v, dofs_p, dofs_z],
            ["u", "v", "p", "z"],
            "Condensed Jacobian",
            "condensed_full_storage",
        )
        plot_blocks(J_full, blocks_eq1, names_eq1, "Condensed Jacobian", "condensed_full_eq1")

        J_approximate = scipy.sparse.coo_matrix((vals_approximate, (rows_approximate, cols_approximate)), shape=(n, n))
        plot_blocks(
            J_approximate,
            [dofs_u, dofs_v, dofs_p, dofs_z],
            ["u", "v", "p", "z"],
            "Condensed Approximate Jacobian",
            "condensed_storage",
        )
        plot_blocks(J_approximate, blocks_eq1, names_eq1, "Condensed Approximate Jacobian", "condensed_eq1")

    # Destroying the MUMPS factorization is collective, so destroy the solver on all
    # ranks here instead of leaving it to garbage collection, which can run at
    # different points on rank 0 after the plotting.
    solver.destroy()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    args = parser.parse_args()

    solve(
        mesh_path="data/meshes/fsi2/mesh_sec.xdmf",
        T=16.0 + 10 * 0.0025,
        dt_val=0.0025,
        output_path="output/pv/fsi2_biharm_dm_restr_split_condensed.bp",
        output_path_p="output/pv/fsi2_biharm_p_dm_restr_split_condensed.bp",
        qoi_path="output/qoi/fsi2_biharm_qoi_restr_split_condensed.txt",
        checkpoint_dir="output/checkpoints/fsi2_biharm_dm_restr_split_approx",
        checkpoint_every=None,
        gamma_reassemble=0.2,  # Default 0.2.
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
