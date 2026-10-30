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
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from petsc4py import PETSc

from xfsi_solver.scripts.plot_qois import plot_run_qois
from xfsi_solver.tools.checkpoint import check_start_options, load_start_state, start_time
from xfsi_solver.tools.convergence import check_converged
from xfsi_solver.tools.newton_log import NewtonLog, field_blocks
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
    mesh_motion_scaling=None,
    max_quadrature_degree=None,
    newton_log_path=None,
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

    With ``mesh_motion_scaling`` (e.g. ``1e-9``), the mesh-motion rows (the z block
    and the fluid-only u rows) of the Jacobian are scaled by this factor before
    MUMPS factorizes it. With restricted test functions these rows are decoupled
    from the interface, so the scaling does not change the Newton step, but it
    reduces the pivoting in MUMPS, see
    notes/restricted-test-functions/solver-time.md. The residual and Jacobian
    seen by SNES stay unscaled, so the convergence tests see the unscaled
    residual. With ``None``, the Jacobian is factorized as it is.

    The function spaces are given by ``elements``, a dict from the name of each space ("U", "V", "P", "Z") to
    the element passed to ``dolfinx.fem.functionspace``. The default is quadratic vector Lagrange for
    u, v, z and linear Lagrange for p.

    The physical constants are ``rho_f``, ``nu_f`` (fluid density and kinematic viscosity), ``rho_s``,
    ``mu_s`` (solid density and shear modulus), ``U_bar`` (mean inflow speed scale) and ``H`` (channel
    height), with defaults 1e3, 1e-3, 1e4, 5e5, 1.0 and 0.41. The first Lamé parameter of the solid is
    ``lambda_s``, or computed from the Poisson ratio ``nu_s``; giving both raises ``ValueError``, and
    with neither, ``nu_s = 0.4``. Theta of the time stepping is 0.5 + ``absolute_theta_shift``, or
    0.5 + ``relative_theta_shift`` * ``dt_val``; giving both raises ``ValueError``, and with neither,
    ``relative_theta_shift = 1.0``.

    With ``plot_matrices=True``, a spy plot and a block norm plot of the Jacobian at the
    final state, without the row scaling, are saved in
    ``output/figures/matrix_blocks/fsi2_biharmonic_fulljacobian/``, with the dofs ordered by
    the storage blocks [u, v, p, z], by the blocks of eq. (1) of
    docs/restricted-iterative-solver.md, and in the order (v, p, u_S, z, u_I) in which the
    fieldsplit of fsi2_biharmonic_mixed.py solves them. The matrix is gathered on rank 0.

    If ``max_quadrature_degree`` is given, the quadrature degree of the residual, and so of its
    Jacobian, and of the drag and lift is capped at it: integrals for which UFL estimates a higher
    degree are integrated inexactly at that degree, and the others keep their estimate. With the
    default None, every integral keeps the degree UFL estimates.

    With ``newton_log_path``, every Newton iterate is recorded in that file by
    ``xfsi_solver.tools.newton_log.NewtonLog``: the norms of the residual and of the Newton update,
    in total and on each block u, v, p and z, whether the Jacobian of the step was assembled anew,
    and the wall time of each iteration. A restart drops the rows after the checkpoint time, as it
    does from the QoI file.
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
    Z = dolfinx.fem.functionspace(fluid_mesh, elements["Z"])
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

    def A_E(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx_fluid

        residual += ufl.inner(J * Fluid.NS_velocity(u, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        # This one stays because the test function is dv, not du.
        residual += ufl.inner(J * Solid.STVK(u, lambda_s, mu_s) * ufl.inv(F).T, ufl.grad(dv)) * dx_solid

        return residual

    def A_P(u, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    residual = A_T(u, u_old, v, v_old)
    residual += A_I(u, v, z)
    residual += A_P(u, p)
    residual += theta * A_E(u, v)
    residual += (1.0 - theta) * A_E(u_old, v_old)

    # Add in u_s with zero constants for correct sparsity patterns
    residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(u, du) * dx_solid
    residual += dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(v, du) * dx_solid

    # --------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])

    residual_blocked = ufl.extract_blocks(residual)
    # Cap the quadrature degree before the Jacobian is derived from the residual, so that it keeps the cap.
    residual_blocked = [cap_quadrature_degree(form, max_quadrature_degree) for form in residual_blocked]

    max_iter = 20
    atol = 1.0e-7
    rtol = 1.0e-12

    if mesh_motion_scaling is None:
        pc_options = {
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 80,
            "mat_mumps_cntl_1": 1e-4,
        }
    else:
        pc_options = {"pc_type": "none"}  # Replaced by the ScaledLU shell preconditioner below.

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocked,
        [u, v, p, z],
        bcs=bcs,
        petsc_options_prefix="fsi2_biharmonic_fulljacobian_",
        entity_maps=entity_maps,
        petsc_options={
            "ksp_type": "preonly",
            **pc_options,
            "snes_linesearch_type": "none",
            "snes_max_it": max_iter,
            "snes_atol": atol,
            "snes_rtol": rtol,
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
            # "snes_monitor": "ascii:output/logs/fsi2_biharmonic_fulljacobian_snes_log.txt",
            "snes_converged_reason": None,
        },
    )

    # (t, step) are the time and number of completed steps of the current state
    t, step, checkpointer, (output_path, output_path_p) = load_start_state(
        mesh,
        [u, v, p, z],
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

    # Add extra callbacks to change how residual and jacobian is assembled,
    # to account for test function restriction on the interface.

    problem: dolfinx.fem.petsc.NonlinearProblem

    solver = problem.solver
    b_vec, (fem_residual, res_args, res_kargs) = solver.getFunction()
    J_mat, P_mat, (fem_jacobian, jac_args, jac_kargs) = solver.getJacobian()

    # Prevent possibly overwriting the sparsity pattern on zeroRows.
    problem.A.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)

    # The same cap as for the residual above, before the derivatives are taken.
    res_miss_ufl = cap_quadrature_degree(
        rho_s * ufl.inner((u - u_old) / dt - theta * v - (1 - theta) * v_old, du) * dx_solid, max_quadrature_degree
    )
    res_miss = dolfinx.fem.form(res_miss_ufl)
    jac_uu = dolfinx.fem.form(ufl.derivative(res_miss_ufl, u, ufl.TrialFunction(U)))
    jac_uv = dolfinx.fem.form(ufl.derivative(res_miss_ufl, v, ufl.TrialFunction(V)))

    # Global rows of the interface u-dofs owned by this process. The owned rows
    # of the block matrix are numbered [u, v, p, z] from the start of the
    # ownership range, and every interface row is owned by a process that
    # located it, so zeroing owned rows covers all of them.
    dofs_d, num_owned_d = bc_deactivate.dof_indices()
    off_own, _ = problem.b.getAttr("_blocks")
    rows_d = (J_mat.getOwnershipRange()[0] + off_own[0] + dofs_d[:num_owned_d]).astype(PETSc.IntType)

    def zero_block(test_space, trial_space, **kwargs):
        return dolfinx.fem.form(
            ufl.ZeroBaseForm((ufl.TestFunction(test_space), ufl.TrialFunction(trial_space))),
            **kwargs,
        )

    res_post = [
        res_miss,  # rho_s[(u-u_old)/dt - θv - (1-θ)v_old]·du_u dx_solid
        dolfinx.fem.form(ufl.ZeroBaseForm((dv,))),
        dolfinx.fem.form(ufl.ZeroBaseForm((dp,))),
        dolfinx.fem.form(ufl.ZeroBaseForm((dz,))),
    ]

    jac_post = [
        [jac_uu, jac_uv, None, None],
        [None, zero_block(V, V), None, None],
        [None, None, zero_block(P, P), None],
        [None, None, None, zero_block(Z, Z)],
    ]

    bcs_post = [u_s_bc, *bcs]
    bcs_rows = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(res_post), bcs_post)
    bcs_cols = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(jac_post, 1), bcs_post)

    def pre_jacobian(x: PETSc.Vec, J: PETSc.Mat) -> None:
        pass

    def post_jacobian(x: PETSc.Vec, J: PETSc.Mat) -> None:
        J.zeroRows(rows_d, diag=0.0)

        dolfinx.fem.petsc.assemble_matrix(J, jac_post, bcs=[u_s_bc, *bcs])
        J.assemble()

    def wrapped_jacobian(snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat) -> None:
        pre_jacobian(x, J)
        fem_jacobian(snes, x, J, P, *jac_args, **jac_kargs)
        post_jacobian(x, J)
        if mesh_motion_scaling is not None:
            J.copy(P, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            P.diagonalScale(L=d_vec)

    def pre_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:
        pass

    def post_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:

        dolfinx.fem.petsc.set_bc(b, [[bc_deactivate], [], [], []], alpha=0.0)  # zero restricted rows

        with b.localForm() as bl:
            # Remove ghost-entries.
            bl.array[b.getLocalSize() :] = 0.0

        dolfinx.fem.petsc.assemble_vector(
            b,
            res_post,
        )
        dolfinx.fem.petsc.apply_lifting(b, jac_post, bcs=bcs_cols, x0=x, alpha=-1.0)

        b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)

        dolfinx.fem.petsc.set_bc(b, bcs_rows, x0=x, alpha=-1.0)

        b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        pass

    def wrapped_residual(snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        pre_residual(x, b)  # x is not yet assigned to u here
        fem_residual(snes, x, b, *res_args, **res_kargs)
        post_residual(x, b)  # u is now updated and b assembled (BCs applied)

    if mesh_motion_scaling is None:
        solver.setJacobian(wrapped_jacobian, J_mat, P_mat)
    else:
        # Row scaling for the mesh-motion equations, used only in the factorization.
        n_u_own = U.dofmap.index_map.size_local * U.dofmap.index_map_bs

        def unroll(dofs, bs=U.dofmap.index_map_bs):
            return (bs * dofs[:, None] + np.arange(bs)).ravel()

        u_fluid_rows = unroll(
            dolfinx.fem.locate_dofs_topological(U, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"]))
        )
        u_fluid_rows = np.setdiff1d(u_fluid_rows, dofs_d)  # interface rows hold the solid equation
        for bc in (u_f_bc, u_s_bc):  # Dirichlet rows hold x - g
            u_fluid_rows = np.setdiff1d(u_fluid_rows, bc.dof_indices()[0])
        u_fluid_rows = u_fluid_rows[u_fluid_rows < n_u_own]  # owned rows only

        d_vec = problem.b.duplicate()
        d_vec.set(1.0)
        d_vec.array_w[off_own[0] + u_fluid_rows] = mesh_motion_scaling
        d_vec.array_w[off_own[3] : off_own[4]] = mesh_motion_scaling

        class ScaledLU:
            """Shell preconditioner applying (D*J)^-1 D = J^-1 with LU of D*J."""

            def __init__(self, options_prefix: str):
                self.lu = PETSc.KSP().create(comm)
                self.lu.setOptionsPrefix(options_prefix)
                opts = PETSc.Options(options_prefix)
                opts["ksp_type"] = "preonly"
                opts["pc_type"] = "lu"
                opts["pc_factor_mat_solver_type"] = "mumps"
                opts["mat_mumps_icntl_14"] = 80
                self.lu.setFromOptions()
                self.tmp = d_vec.duplicate()

            def setUp(self, pc: PETSc.PC) -> None:
                _, P = pc.getOperators()
                self.lu.setOperators(P)
                self.lu.setUp()

            def apply(self, pc: PETSc.PC, r: PETSc.Vec, y: PETSc.Vec) -> None:
                self.tmp.pointwiseMult(r, d_vec)
                self.lu.solve(self.tmp, y)
                # Report a failed factorization or solve to the outer KSP, which
                # otherwise takes y as a valid preconditioned residual
                if self.lu.getConvergedReason() < 0:
                    reason = self.lu.getPC().getFailedReason()
                    pc.setFailedReason(reason if reason != 0 else PETSc.PC.FailedReason.SUBPC_ERROR)

        pc = solver.getKSP().getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        scaled_lu = ScaledLU(f"{solver.getOptionsPrefix()}scaled_lu_")
        pc.setPythonContext(scaled_lu)

        # Preconditioner matrix D*J, factorized by MUMPS instead of J. J is
        # assembled once first, so that P gets J's final parallel layout and can
        # be refilled with a same-nonzero-pattern copy in every Newton iteration.
        dolfinx.fem.petsc.assign([u, v, p, z], problem.x)
        fem_jacobian(solver, problem.x, J_mat, P_mat, *jac_args, **jac_kargs)
        post_jacobian(problem.x, J_mat)
        P_scaled = J_mat.duplicate(copy=True)

        solver.setJacobian(wrapped_jacobian, J_mat, P_scaled)
    solver.setFunction(wrapped_residual, b_vec)

    init_qoi_file(qoi_path, comm, restart, t, dt_val)
    if newton_log_path is not None:
        blocks = field_blocks(problem, ["u", "v", "p", "z"])
        newton_log = NewtonLog(newton_log_path, blocks, comm, restart, t, dt_val)
        solver.setMonitor(newton_log)

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

        if newton_log_path is not None:
            newton_log.start_step(t)
        problem.solve()
        check_converged(problem, f"t = {t:.4f}", writers=writers)

        if comm.rank == 0:
            sys.stdout.flush()

        step += 1
        write_output(t, step)

        if checkpoint_every is not None and step % checkpoint_every == 0:
            checkpointer.write([u, v, p, z], t, step, dt_val, t0)

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")
        print(f"\nSolver setup:\n  Quadrature degree: {quadrature_description(max_quadrature_degree)}")

    for writer in writers:
        writer.close()

    if plot_matrices:
        from xfsi_solver.tools.matrix_blocks import matrix_blocks_dir, plot_matrix_blocks

        dolfinx.fem.petsc.assign([u, v, p, z], problem.x)
        fem_jacobian(solver, problem.x, J_mat, P_mat, *jac_args, **jac_kargs)
        post_jacobian(problem.x, J_mat)

        # Global rows of the owned dofs of each block in [u, v, p, z], and the owned u dofs
        # supported on solid cells, which are u_S in eq. (1); the others are u_I.
        first_row = J_mat.getOwnershipRange()[0]
        rows_u, rows_v, rows_p, rows_z = (first_row + np.arange(off_own[k], off_own[k + 1]) for k in range(4))
        bs = U.dofmap.index_map_bs
        solid_dofs = dolfinx.fem.locate_dofs_topological(
            U, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"])
        )
        u_solid = np.isin(np.arange(len(rows_u)), (bs * solid_dofs[:, None] + np.arange(bs)).ravel())
        partitions = {
            "storage": ([rows_u, rows_v, rows_p, rows_z], ["$u$", "$v$", "$p$", "$z$"]),
            "eq1": (
                [rows_u[u_solid], rows_v, rows_p, rows_z, rows_u[~u_solid]],
                ["$u_S$", "$v$", "$p$", "$z$", "$u_I$"],
            ),
            "split": (
                [rows_v, rows_p, rows_u[u_solid], rows_z, rows_u[~u_solid]],
                ["$v$", "$p$", "$u_S$", "$z$", "$u_I$"],
            ),
        }
        plot_matrix_blocks(J_mat, partitions, "Full Jacobian", matrix_blocks_dir(__file__), t)

    # Destroying the MUMPS factorization is collective, so destroy it on all ranks here instead of leaving it
    # to garbage collection, which can run at different points on different ranks, such as after rank 0 has
    # plotted the matrices.
    solver.destroy()
    if mesh_motion_scaling is not None:
        scaled_lu.lu.destroy()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument("--initial-state-dir", help="start from the latest checkpoint of another run in this directory")
    args = parser.parse_args()

    mesh_path = "data/meshes/fsi2/mesh_sec.xdmf"
    T = 15.5
    dt_val = 0.0025
    output_path = "output/pv/fsi2_biharmonic_fulljacobian.bp"
    output_path_p = "output/pv/fsi2_biharmonic_fulljacobian_p.bp"
    qoi_path = "output/qoi/fsi2_biharmonic_fulljacobian.txt"
    newton_log_path = "output/qoi/fsi2_biharmonic_fulljacobian_newton.txt"
    checkpoint_dir = "output/checkpoints/fsi2_biharmonic_fulljacobian"
    checkpoint_every = 100
    vtx_save_every = 4
    mesh_motion_scaling = None
    # mesh_motion_scaling = 1.0e-9
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
        newton_log_path=newton_log_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        vtx_save_every=vtx_save_every,
        mesh_motion_scaling=mesh_motion_scaling,
        max_quadrature_degree=max_quadrature_degree,
        restart=restart,
        initial_state_dir=initial_state_dir,
    )

    if MPI.COMM_WORLD.rank == 0:
        plot_run_qois(qoi_path, T)


if __name__ == "__main__":
    main()
