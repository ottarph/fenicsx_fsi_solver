# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""fsi2_biharmonic_fulljacobian with a prediction of the displacements as the initial guess of each
Newton solve, set in an initial-guess hook of the SNES with a mesh motion solve, as fsi2_simplified does.

The Jacobian can also be copied, after each assembly, into a MatNest over the groups of dofs
(v, p, u_s, z_int, u_int, z_f, u_f), for experiments on condensation (see ``jacobian_callback``)."""

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

from xfsi_solver.mesh_motion import biharmonic_mixed_mesh_motion
from xfsi_solver.scripts.plot_qois import plot_run_qois
from xfsi_solver.tools.checkpoint import check_start_options, load_start_state, start_time
from xfsi_solver.tools.convergence import check_converged
from xfsi_solver.tools.matrix_nest import NestView
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
from xfsi_solver.tools.submesh import all_cells

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

# The groups of dofs of the nest Jacobian, in the order of its blocks.
NEST_GROUPS = ("v", "p", "u_s", "z_int", "u_int", "z_f", "u_f")


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
    plot_matrices_dir=None,
    restart=False,
    initial_state_dir=None,
    max_quadrature_degree=None,
    newton_log_path=None,
    predict_displacements=True,
    snes_converged_reason_on=True,
    snes_atol=1.0e-7,
    snes_rtol=1.0e-12,
    snes_max_it=20,
    jacobian_callback=None,
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
    final state are saved in ``plot_matrices_dir``, by default
    ``output/figures/matrix_blocks/fsi2_biharmonic_fulljacobian_initialguess/``, with the dofs ordered by
    the storage blocks [u, v, p, z], by the blocks of eq. (1) of
    docs/restricted-iterative-solver.md, in the order (v, p, u_S, z, u_I) in which the
    fieldsplit of fsi2_biharmonic_mixed.py solves them, and by the groups of the nest Jacobian
    (see ``jacobian_callback``). The matrix is gathered on rank 0.

    If ``max_quadrature_degree`` is given, the quadrature degree of the residual, and so of its
    Jacobian, and of the drag and lift is capped at it: integrals for which UFL estimates a higher
    degree are integrated inexactly at that degree, and the others keep their estimate. With the
    default None, every integral keeps the degree UFL estimates.

    With ``newton_log_path``, every Newton iterate is recorded in that file by
    ``xfsi_solver.tools.newton_log.NewtonLog``: the norms of the residual and of the Newton update,
    in total and on each block u, v, p and z, whether the Jacobian of the step was assembled anew,
    and the wall time of each iteration. A restart drops the rows after the checkpoint time, as it
    does from the QoI file.

    With ``predict_displacements=True``, the default, the initial guess of each Newton solve is not
    the state of the previous step but a prediction of the displacements: u = u_old + dt * v_old at
    the dofs of the solid cells, the interface included, and on the fluid cells u and z from the
    biharmonic mesh motion in mixed form with these interface values, as fsi2_simplified predicts
    them. v and p stay those of the previous step. The interface rows of the residual, the solid
    kinematic relation, then hold exactly, and the mesh velocity (u - u_old) / dt in the fluid is no
    longer zero. The mesh motion is solved on the fluid submesh, by MUMPS LU on the x component,
    factorized once for the whole run, and its z is the z of the solver. It is set in an
    initial-guess hook of the SNES, which runs once per Newton solve, before the residual of the
    initial guess is evaluated. With ``False``, the solver is fsi2_biharmonic_fulljacobian.

    ``snes_atol``, ``snes_rtol`` and ``snes_max_it`` are the stopping tolerances of the SNES, for
    example a tight relative tolerance to follow the rate of the iterations over more of them.

    With ``jacobian_callback``, the Jacobian is copied after each assembly, in every Newton
    iteration, into a MatNest over the groups of dofs in ``NEST_GROUPS``, (v, p, u_s, z_int, u_int,
    z_f, u_f): v and p all their dofs, u_s the dofs of u on the solid cells off the interface, u_int
    and z_int the dofs of u and z on the solid-fluid interface, and u_f and z_f the other dofs of u
    and z, those of the fluid cells off the interface, the outer fluid boundary included. Dofs with
    Dirichlet conditions stay in their group, with their identity rows. The Jacobian is the one
    Newton solves with: the interface rows of u hold the solid kinematic relation. Then
    ``jacobian_callback(snes, nest_view)`` is called, with the SNES, whose ``getSolution()`` is the
    iterate and ``getFunction()[0]`` its residual, and the
    ``xfsi_solver.tools.matrix_nest.NestView``: its ``nest`` is the MatNest, ``block(row, col)``
    gives a block by the names of its groups, and ``index_sets`` the rows of each group in the
    global numbering of the Jacobian, which the nest keeps, so that it acts on the vectors of the
    SNES. The nest is only a copy: Newton still solves with the assembled Jacobian, by MUMPS LU, so
    changes to it do not change the run. The callback must be collective, as the blocks are
    distributed.

    With ``snes_converged_reason_on=False``, the SNES does not print why each Newton solve stopped.
    Printing it makes PETSc keep a stdout viewer for the communicator of the mesh, in a table of
    fixed size, until PetscFinalize, so a process that creates a few hundred meshes, such as the
    test suite, which turns it off in tests/conftest.py, overflows the table and crashes.
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

    pc_options = {
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": 80,
        "mat_mumps_cntl_1": 1e-4,
    }

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocked,
        [u, v, p, z],
        bcs=bcs,
        petsc_options_prefix="fsi2_biharmonic_fulljacobian_initialguess_",
        entity_maps=entity_maps,
        petsc_options={
            "ksp_type": "preonly",
            **pc_options,
            "snes_linesearch_type": "none",
            "snes_max_it": snes_max_it,
            "snes_atol": snes_atol,
            "snes_rtol": snes_rtol,
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
            # "snes_monitor": "ascii:output/logs/fsi2_biharmonic_fulljacobian_snes_log.txt",
            **({"snes_converged_reason": None} if snes_converged_reason_on else {}),
        },
    )

    # (t, step) are the time and number of completed steps of the current state
    t, step, checkpointer, (output_path, output_path_p) = load_start_state(
        mesh,
        [u, v, p],
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

    # The groups of NEST_GROUPS, as the global rows of the owned dofs in each. locate_dofs_topological
    # also finds the owned dofs on entities that only other ranks have marked, so each group has all
    # its owned rows.
    fluid_mesh.topology.create_connectivity(fluid_mesh.topology.dim - 1, fluid_mesh.topology.dim)
    fluid_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
        facet_tags, fluid_mesh, fluid_vertex_map, fluid_cell_map
    )
    first_row = J_mat.getOwnershipRange()[0]

    def owned_rows(block: int, space: dolfinx.fem.FunctionSpace, dofs: np.ndarray | None = None) -> np.ndarray:
        """The global rows of the owned dofs among the (blocked) local ``dofs`` of ``space``, all
        of them if None, which is block ``block`` of [u, v, p, z]."""
        num_owned = space.dofmap.index_map.size_local
        dofs = np.arange(num_owned) if dofs is None else dofs[dofs < num_owned]
        bs = space.dofmap.index_map_bs
        return (first_row + off_own[block] + bs * dofs[:, None] + np.arange(bs)).ravel().astype(PETSc.IntType)

    solid_dofs_u = dolfinx.fem.locate_dofs_topological(U, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"]))
    interface_dofs_z = dolfinx.fem.locate_dofs_topological(
        Z, fluid_mesh.topology.dim - 1, fluid_facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])
    )
    all_dofs_u = np.arange(U.dofmap.index_map.size_local)
    all_dofs_z = np.arange(Z.dofmap.index_map.size_local)
    u_s_dofs = np.setdiff1d(solid_dofs_u, dofs_interface)
    groups = {
        "v": owned_rows(1, V),
        "p": owned_rows(2, P),
        "u_s": owned_rows(0, U, u_s_dofs),
        "z_int": owned_rows(3, Z, interface_dofs_z),
        "u_int": owned_rows(0, U, dofs_interface),
        "z_f": owned_rows(3, Z, np.setdiff1d(all_dofs_z, interface_dofs_z)),
        "u_f": owned_rows(0, U, np.setdiff1d(all_dofs_u, np.union1d(solid_dofs_u, dofs_interface))),
    }
    # With the same element for u and v, their dofs are numbered alike, and nest_data["v_rows"] holds the
    # rows of v at the dofs of the rows of u_s and u_int, in the same order, for the condensation of the solid.
    nest_data = {}
    if elements["U"] == elements["V"]:
        nest_data["v_rows"] = {"u_s": owned_rows(1, V, u_s_dofs), "u_int": owned_rows(1, V, dofs_interface)}
    nest_view = None  # made at the first assembly of the Jacobian, since blocks are only taken of an assembled matrix

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
        if jacobian_callback is not None:
            nonlocal nest_view
            if nest_view is None:
                nest_view = NestView(J, [groups[name] for name in NEST_GROUPS], list(NEST_GROUPS), nest_data)
            else:
                nest_view.update()
            jacobian_callback(snes, nest_view)

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

    solver.setJacobian(wrapped_jacobian, J_mat, P_mat)
    solver.setFunction(wrapped_residual, b_vec)

    if predict_displacements:
        if elements["U"] != elements["V"]:
            raise ValueError(
                f"The prediction needs the same element for U and V, not {elements['U']} and {elements['V']}"
            )
        # U and V have the same element on the same mesh, so their dofs are numbered alike, and the
        # prediction u = u_old + dt * v_old holds dof by dof on the solid cells, the interface
        # included.
        block_size = U.dofmap.index_map_bs

        def unrolled(dofs: np.ndarray) -> np.ndarray:
            return (block_size * dofs[:, None] + np.arange(block_size)).ravel()

        solid_u = unrolled(
            dolfinx.fem.locate_dofs_topological(U, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"]))
        )

        # The mesh motion lives on the fluid submesh: u_f with the element of u, and z itself, with
        # zero Dirichlet values on the outer fluid boundary and the solid displacement on the
        # interface. The dofmaps of the fluid cells give the dof of u at each local dof of u_f. Every
        # dof of u_f on a rank lies in one of its fluid cells, so this covers the owned and ghost dofs.
        U_f = dolfinx.fem.functionspace(fluid_mesh, elements["U"])
        u_f = dolfinx.fem.Function(U_f, name="u_f")
        u_f_mesh_bc_func = dolfinx.fem.Function(U_f)
        u_f_mesh_bc_dofs = dolfinx.fem.locate_dofs_topological(
            U_f, fluid_mesh.topology.dim - 1, dolfinx.mesh.exterior_facet_indices(fluid_mesh.topology)
        )
        u_f_mesh_bc = dolfinx.fem.dirichletbc(u_f_mesh_bc_func, u_f_mesh_bc_dofs)
        mesh_motion = biharmonic_mixed_mesh_motion(
            u_f,
            [u_f_mesh_bc],
            z=z,
            petsc_options_prefix="fsi2_biharmonic_fulljacobian_initialguess_mesh_motion_",
            max_quadrature_degree=max_quadrature_degree,
        )

        fluid_cells = all_cells(fluid_mesh)
        fluid_parent_cells = fluid_cell_map.sub_topology_to_topology(fluid_cells, inverse=False)
        num_u_f_dofs = U_f.dofmap.index_map.size_local + U_f.dofmap.index_map.num_ghosts
        u_dof_of_u_f = np.empty(num_u_f_dofs, dtype=np.int32)
        u_dof_of_u_f[U_f.dofmap.list[fluid_cells]] = U.dofmap.list[fluid_parent_cells]
        u_of_u_f = unrolled(u_dof_of_u_f)  # the dof of u at each unrolled dof of u_f
        fluid_interface_dofs = dolfinx.fem.locate_dofs_topological(
            U_f,
            fluid_mesh.topology.dim - 1,
            fluid_facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"]),
            remote=True,
        )
        interface_u_f = unrolled(fluid_interface_dofs)

        def predict(snes: PETSc.SNES, x: PETSc.Vec) -> None:
            """Set the initial guess x of the Newton solve to the predicted displacements, with the v,
            p of the previous step. Called by SNES before the residual of the initial guess, when the
            functions still hold the previous step."""
            u.x.array[solid_u] = u_old.x.array[solid_u] + dt_val * v_old.x.array[solid_u]
            # From u_old and v_old, not u: a rank may hold an interface dof only through a fluid cell
            # (remote=True above), where the line above has not set u.
            interface_u = u_of_u_f[interface_u_f]
            u_f_mesh_bc_func.x.array[interface_u_f] = u_old.x.array[interface_u] + dt_val * v_old.x.array[interface_u]
            mesh_motion.solve()
            u.x.array[u_of_u_f] = u_f.x.array
            u.x.scatter_forward()
            dolfinx.fem.petsc.assign([u, v, p, z], x)

        solver.setInitialGuess(predict)

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
        initial_guess = (
            "predicted displacements, u = u_old + dt * v_old on the solid and the mesh motion on the fluid"
            if predict_displacements
            else "the previous step"
        )
        print(f"  Initial guess: {initial_guess}")

    for writer in writers:
        writer.close()

    if plot_matrices:
        from xfsi_solver.tools.matrix_blocks import matrix_blocks_dir, plot_matrix_blocks

        dolfinx.fem.petsc.assign([u, v, p, z], problem.x)
        fem_jacobian(solver, problem.x, J_mat, P_mat, *jac_args, **jac_kargs)
        post_jacobian(problem.x, J_mat)

        # The owned u dofs supported on solid cells, the interface included, are u_S in eq. (1); the
        # others are u_I.
        rows_u = np.union1d(np.union1d(groups["u_s"], groups["u_int"]), groups["u_f"])
        rows_z = np.union1d(groups["z_int"], groups["z_f"])
        rows_u_solid = np.union1d(groups["u_s"], groups["u_int"])
        rows_v, rows_p, rows_u_fluid = groups["v"], groups["p"], groups["u_f"]
        partitions = {
            "storage": ([rows_u, rows_v, rows_p, rows_z], ["$u$", "$v$", "$p$", "$z$"]),
            "eq1": (
                [rows_u_solid, rows_v, rows_p, rows_z, rows_u_fluid],
                ["$u_S$", "$v$", "$p$", "$z$", "$u_I$"],
            ),
            "split": (
                [rows_v, rows_p, rows_u_solid, rows_z, rows_u_fluid],
                ["$v$", "$p$", "$u_S$", "$z$", "$u_I$"],
            ),
            "nest": (
                [groups[name] for name in NEST_GROUPS],
                ["$v$", "$p$", "$u_s$", "$z_{int}$", "$u_{int}$", "$z_f$", "$u_f$"],
            ),
        }
        if plot_matrices_dir is None:
            plot_matrices_dir = matrix_blocks_dir(__file__)
        plot_matrix_blocks(J_mat, partitions, "Full Jacobian", plot_matrices_dir, t)

    # Destroying the MUMPS factorization is collective, so destroy it on all ranks here instead of leaving it
    # to garbage collection, which can run at different points on different ranks, such as after rank 0 has
    # plotted the matrices.
    solver.destroy()
    if predict_displacements:
        mesh_motion.destroy()
    if nest_view is not None:
        nest_view.destroy()

    return


# Define callbacks here


def human_count(n: float) -> str:
    """``n`` with k for thousands and M for millions, such as 291.5k or 3.43M."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


def _nonzeros(blocks) -> float:
    """The number of entries in the patterns of ``blocks``, a list of lists of Mats or None. Collective."""
    return sum(
        block.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)["nz_used"] for row in blocks for block in row if block is not None
    )


def _nonzero_columns(mat: PETSc.Mat) -> PETSc.IS:
    """The columns of ``mat`` with an entry in its pattern on any rank, each in the IS on the rank
    that owns it in the column layout of ``mat``. Collective."""
    comm = mat.getComm()
    _, cols, _ = mat.getValuesCSR()
    cols = np.unique(np.concatenate(comm.tompi4py().allgather(np.unique(cols))))
    start, end = mat.getOwnershipRangeColumn()
    return PETSc.IS().createGeneral(cols[(cols >= start) & (cols < end)].astype(PETSc.IntType), comm=comm)


def _lu(A: PETSc.Mat) -> PETSc.KSP:
    """A KSP that solves with ``A`` by MUMPS LU."""
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()
    return ksp


def _schur_blocks(nest_view: NestView, rest: list[str], updates: dict) -> tuple[list, list[PETSc.Mat]]:
    """The blocks of S = D - updates over the groups ``rest``: the blocks of the nest Jacobian, less
    ``updates[(a, b)]`` where given, which are used up. Returns the blocks and those made here."""
    blocks, made = [], []
    for a in rest:
        row = []
        for b in rest:
            block = nest_view.block(a, b)
            if (a, b) in updates:
                update = updates.pop((a, b))
                if block is None:
                    update.scale(-1.0)
                    block = update
                else:
                    block = block.duplicate(copy=True)
                    block.axpy(-1.0, update, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                    update.destroy()
                made.append(block)
            row.append(block)
        blocks.append(row)
    return blocks, made


def condense_u_s(nest_view: NestView, drop_tol: float = 1e-14) -> tuple[PETSc.Mat, list, list[PETSc.Mat]]:
    """Condense u_s out of the nest Jacobian: the Schur complement S = D - C A^-1 B of
    A = J[u_s, u_s], with B = J[u_s, rest], C = J[rest, u_s] and D = J[rest, rest], as a MatNest over
    the other groups of ``NEST_GROUPS``, in their order and with the contiguous layout of a nest.

    Only the blocks of S that u_s couples to change: (v, v), (v, u_int), (u_int, v) and (u_int,
    u_int). The others are the blocks of the nest Jacobian themselves. A^-1 B is computed by MUMPS LU
    of A on the columns of B that have entries, as a dense matrix, whose entries below ``drop_tol``
    times its largest are dropped before it is multiplied by C, so that round-off does not fill S.
    Collective. Returns S, its blocks (None where zero) and the matrices made here, which the caller
    destroys with S.
    """
    rest = [name for name in NEST_GROUPS if name != "u_s"]
    comm = nest_view.mat.getComm()
    ksp = _lu(nest_view.block("u_s", "u_s"))
    updates = {}  # (a, b): C_a A^-1 B_b, for the blocks of S that change
    for b in rest:
        B = nest_view.block("u_s", b)
        if B is None:
            continue
        rows = PETSc.IS().createStride(B.getLocalSize()[0], B.getOwnershipRange()[0], 1, comm=comm)
        cols = _nonzero_columns(B)
        B_dense = B.createSubMatrix(rows, cols).convert("dense")  # converted in place
        X = B_dense.duplicate()
        ksp.matSolve(B_dense, X)
        values = X.getDenseArray(readonly=False)
        largest = comm.tompi4py().allreduce(np.abs(values).max(initial=0.0), op=MPI.MAX)
        values[np.abs(values) < drop_tol * largest] = 0.0
        X.assemble()
        X_sparse = X.convert("aij", out=PETSc.Mat())
        # Q puts the columns of X back at the columns of B they were taken from.
        num_cols = cols.getLocalSize()
        Q = PETSc.Mat().createAIJ(
            ((num_cols, None), (B.getLocalSize()[1], None)),
            csr=(np.arange(num_cols + 1, dtype=PETSc.IntType), cols.getIndices(), np.ones(num_cols)),
            comm=comm,
        )
        for a in rest:
            C = nest_view.block(a, "u_s")
            if C is not None:
                CX = C.matMult(X_sparse)
                updates[(a, b)] = CX.matMult(Q)
                CX.destroy()
        for obj in (rows, cols, B_dense, X, X_sparse, Q):
            obj.destroy()
    ksp.destroy()
    blocks, made = _schur_blocks(nest_view, rest, updates)
    return PETSc.Mat().createNest(blocks, comm=comm), blocks, made


SOLID_GROUPS = ("u_s", "u_int")


def condense_solid_exact(nest_view: NestView) -> tuple[PETSc.Mat, list, list[PETSc.Mat], float]:
    """Condense the solid displacement, u_s and u_int, out of the nest Jacobian without inverting
    its mass matrix, by the kinematic relation of the solid.

    The rows of u_s and u_int are rho_s M (du - theta dt dv) on the solid, and those of the
    interface the same since the fluid is restricted out of them. u and v share their shape
    functions, so M_vu = M_u and du = theta dt dv + M^-1 r on the solid dofs: A^-1 B = K, where K
    takes v at the dof of each solid row, times -theta dt, and B = J[solid, v] is the only nonzero block of
    B (the interface rows keep zero entries in z and u_f in their pattern). The factor is read from the Jacobian, as B[i, v(i)] / A[i, i], which is 0 on the Dirichlet
    rows. Then S = D - C K changes only in the columns of v, and C K is C with its columns moved from
    u to v at the same dofs, so the fill is cell-local: z_int, z_f and u_f get blocks in v.

    Needs ``nest_view.data["v_rows"]``, as set by ``solve`` when u and v have the same element.
    Collective. Returns S, as a MatNest over (v, p, z_int, z_f, u_f), its blocks, the matrices made
    here, which the caller destroys with S, and |A K - B| / |B|, which is round-off when the
    relation holds.
    """
    rest = [name for name in NEST_GROUPS if name not in SOLID_GROUPS]
    for g in SOLID_GROUPS:
        for b in rest:
            # The restriction zeroes the fluid entries of the interface rows but keeps their pattern.
            if b != "v" and nest_view.block(g, b) is not None and nest_view.block(g, b).norm() != 0.0:
                raise ValueError(f"The rows of {g} reach {b}, so they are not the kinematic relation alone")
    comm = nest_view.mat.getComm()
    v_is = nest_view.index_sets[nest_view.names.index("v")]
    v_rows, v_start = v_is.getIndices(), nest_view.block("v", "v").getOwnershipRangeColumn()[0]

    K = {}
    for g in SOLID_GROUPS:
        A, B = nest_view.block(g, g), nest_view.block(g, "v")
        # The column of v in the v block, at the dof of each row of g.
        positions = np.searchsorted(v_rows, nest_view.data["v_rows"][g])
        assert np.array_equal(v_rows[positions], nest_view.data["v_rows"][g])
        v_cols = (v_start + positions).astype(PETSc.IntType)
        indptr, cols, values = B.getValuesCSR()
        factor = np.zeros(len(v_cols))
        for i, col in enumerate(v_cols):
            row_cols = cols[indptr[i] : indptr[i + 1]]
            k = np.searchsorted(row_cols, col)
            if k < len(row_cols) and row_cols[k] == col:
                factor[i] = values[indptr[i] + k]
        factor /= A.getDiagonal().array
        K[g] = PETSc.Mat().createAIJ(
            ((len(v_cols), None), (B.getLocalSize()[1], None)),
            csr=(np.arange(len(v_cols) + 1, dtype=PETSc.IntType), v_cols, factor),
            comm=comm,
        )

    # A K - B, with A = J[solid, solid]
    residual, size = 0.0, 0.0
    for g in SOLID_GROUPS:
        AK = nest_view.block(g, "v").duplicate(copy=True)
        AK.scale(-1.0)
        size += AK.norm() ** 2
        for h in SOLID_GROUPS:
            if nest_view.block(g, h) is not None:
                product = nest_view.block(g, h).matMult(K[h])
                AK.axpy(1.0, product, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                product.destroy()
        residual += AK.norm() ** 2
        AK.destroy()

    updates = {}  # (a, v): C_a K
    for a in rest:
        for g in SOLID_GROUPS:
            C = nest_view.block(a, g)
            if C is None:
                continue
            CK = C.matMult(K[g])
            if (a, "v") in updates:
                updates[(a, "v")].axpy(1.0, CK, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                CK.destroy()
            else:
                updates[(a, "v")] = CK
    for k in K.values():
        k.destroy()
    blocks, made = _schur_blocks(nest_view, rest, updates)
    return PETSc.Mat().createNest(blocks, comm=comm), blocks, made, np.sqrt(residual / size)


def check_condensation(nest_view: NestView, eliminated: list[str], S: PETSc.Mat) -> float:
    """The relative error of a condensation S of the nest Jacobian J that eliminates the groups
    ``eliminated``: for a random x and y = J x, S must map the other groups of x to
    y_rest - J[rest, E] J[E, E]^-1 y_E, where J[E, E] is solved with by MUMPS LU. Collective."""
    J = nest_view.mat
    comm = J.getComm()
    index_sets = dict(zip(nest_view.names, nest_view.index_sets, strict=True))
    rest = [name for name in nest_view.names if name not in eliminated]
    E = PETSc.IS().createGeneral(
        np.sort(np.concatenate([index_sets[name].getIndices() for name in eliminated])).astype(PETSc.IntType),
        comm=comm,
    )
    A = J.createSubMatrix(E, E)
    ksp = _lu(A)
    x, y = J.createVecs()
    x.setRandom()
    J.mult(x, y)
    y_E = y.getSubVector(E)
    w = y_E.duplicate()
    ksp.solve(y_E, w)
    y.restoreSubVector(E, y_E)

    # The vectors of S are plain vectors, with the groups at the index sets of the nest.
    x_rest, Sx = S.createVecs()
    expected = Sx.duplicate()
    S_index_sets, _ = S.getNestISs()
    for name, S_is in zip(rest, S_index_sets, strict=True):
        for full, condensed in ((x, x_rest), (y, expected)):
            sub, condensed_sub = full.getSubVector(index_sets[name]), condensed.getSubVector(S_is)
            sub.copy(condensed_sub)
            if condensed is expected:
                C = J.createSubMatrix(index_sets[name], E)
                Cw = condensed_sub.duplicate()
                C.mult(w, Cw)
                condensed_sub.axpy(-1.0, Cw)
                for obj in (C, Cw):
                    obj.destroy()
            full.restoreSubVector(index_sets[name], sub)
            condensed.restoreSubVector(S_is, condensed_sub)
    S.mult(x_rest, Sx)
    Sx.axpy(-1.0, expected)
    error = Sx.norm() / expected.norm()
    for obj in (E, A, ksp, x, y, w, x_rest, Sx, expected):
        obj.destroy()
    return error


def _report(snes, nest_view, label, eliminated, S, S_blocks, elapsed, extra=""):
    """Check the condensation S and print the error and the nonzeros of J, D and S on rank 0."""
    error = check_condensation(nest_view, eliminated, S)
    rest = [name for name in NEST_GROUPS if name not in eliminated]
    nnz_J = _nonzeros(nest_view.blocks)
    nnz_D = _nonzeros([[nest_view.block(a, b) for b in rest] for a in rest])
    nnz_S = _nonzeros(S_blocks)
    num_dofs = sum(nest_view.index_sets[nest_view.names.index(name)].getSize() for name in eliminated)
    if nest_view.mat.getComm().getRank() == 0:
        print(
            f"  {label} ({num_dofs} dofs) in {elapsed:.3f} s at it {snes.getIterationNumber()}:"
            f" error {error:.1e}{extra}, nonzeros J {human_count(nnz_J)}, D {human_count(nnz_D)},"
            f" S {human_count(nnz_S)}",
            flush=True,
        )


def J_0_jacobian_hook(snes: PETSc.SNES, nest_view: NestView) -> None:
    """Condense u_s out of the Jacobian with ``condense_u_s``, check it with ``check_condensation``
    and print the error and the nonzeros before and after, on rank 0."""
    start = timer()
    S, S_blocks, made = condense_u_s(nest_view)
    elapsed = timer() - start
    _report(snes, nest_view, "Schur condensed u_s", ["u_s"], S, S_blocks, elapsed)
    for obj in (S, *made):
        obj.destroy()


def exact_solid_condensation_hook(snes: PETSc.SNES, nest_view: NestView) -> None:
    """Condense u_s and u_int out of the Jacobian with ``condense_solid_exact``, check it with
    ``check_condensation`` and print the error, |A K - B| / |B| and the nonzeros, on rank 0."""
    start = timer()
    S, S_blocks, made, relation_error = condense_solid_exact(nest_view)
    elapsed = timer() - start
    _report(
        snes,
        nest_view,
        "exactly condensed u_s, u_int",
        list(SOLID_GROUPS),
        S,
        S_blocks,
        elapsed,
        extra=f", |A K - B| / |B| {relation_error:.1e}",
    )
    for obj in (S, *made):
        obj.destroy()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument("--initial-state-dir", help="start from the latest checkpoint of another run in this directory")
    args = parser.parse_args()

    mesh_path = "data/meshes/fsi2/mesh_sec.xdmf"
    dt_val = 0.0025
    T = 16.0 + 2 * dt_val
    output_path = "output/pv/fsi2_biharmonic_fulljacobian_initialguess.bp"
    output_path_p = "output/pv/fsi2_biharmonic_fulljacobian_initialguess_p.bp"
    qoi_path = "output/qoi/fsi2_biharmonic_fulljacobian_initialguess.txt"
    newton_log_path = "output/qoi/fsi2_biharmonic_fulljacobian_initialguess_newton.txt"
    checkpoint_dir = "output/checkpoints/fsi2_biharmonic_fulljacobian_initialguess"
    checkpoint_every = None
    vtx_save_every = None
    max_quadrature_degree = None  # Default None, no cap.
    restart = args.restart
    initial_state_dir = args.initial_state_dir
    initial_state_dir = "output/snapshots/fsi2_simplified/t16.0000"
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
        max_quadrature_degree=max_quadrature_degree,
        restart=restart,
        initial_state_dir=initial_state_dir,
        plot_matrices=False,
        plot_matrices_dir="./",
        jacobian_callback=J_0_jacobian_hook,
        # jacobian_callback=exact_solid_condensation_hook,
    )

    if MPI.COMM_WORLD.rank == 0:
        plot_run_qois(qoi_path, T)


if __name__ == "__main__":
    main()
