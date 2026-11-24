# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""fsi2_harmonic_fulljacobian with the solid displacement predicted from the velocity as the initial
guess of each Newton solve, an option to condense the solid displacement out of the solid stress in
UFL, and the Jacobian copied into a MatNest over the groups of dofs (v, p, u_s, u_int, u_f) for
experiments on condensation (see ``jacobian_callback``)."""

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

from xfsi_solver.mesh_motion import harmonic_mesh_motion
from xfsi_solver.scripts.plot_qois import plot_run_qois
from xfsi_solver.tools.checkpoint import check_start_options, load_start_state, start_time
from xfsi_solver.tools.condensation import J_0_jacobian_hook, exact_solid_condensation_hook  # noqa: F401
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
NEST_GROUPS = ("v", "p", "u_s", "u_int", "u_f")


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
    move_reference_domain=False,
    max_quadrature_degree=None,
    newton_log_path=None,
    predict_displacements=True,
    condense_solid=False,
    snes_converged_reason_on=True,
    snes_atol=1.0e-7,
    snes_rtol=1.0e-12,
    snes_max_it=20,
    jacobian_callback=None,
):
    """Solve the FSI2 benchmark with harmonic mesh motion and restricted displacement test
    functions, with Newton's method on the full Jacobian and MUMPS LU on the whole system.

    This is fsi2_harmonic_fulljacobian.py, without its mesh motion scaling, with a prediction of the
    solid displacement as the initial guess, an option to condense the solid displacement out of the
    solid stress, and a nest copy of the Jacobian for experiments. That solver is
    fsi2_biharmonic_fulljacobian.py with harmonic mesh motion instead of biharmonic:
    the unknowns are (u, v, p), the mesh equation is (grad u, grad psi)_f on the reference
    fluid domain, without a scaling or an interface flux term, and the interface rows of the
    displacement test functions hold the solid kinematic equation, as in the biharmonic
    solver. The mesh rows of the Jacobian are A_IS and A_II of the harmonic case of
    docs/restricted-iterative-solver.md.

    The benchmark is solved up to time ``T``, in ``round((T - t0) / dt_val)`` time
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

    With ``move_reference_domain=True``, the mesh motion acts on the displacement since the start
    state instead of on the whole displacement: the mesh equation is that of u - u_ref, with u_ref
    the u of the start state, fixed through the run. The start state then satisfies the mesh
    equation, so a run from the state of a solver with another mesh motion, such as fsi2_simplified,
    starts without a jump in the mesh velocity (u - u_old) / dt in the fluid. The other equations
    still use the whole displacement u. u_ref is stored in the checkpoints and read back on a
    restart, which must also pass ``move_reference_domain=True``; a restart without it would measure
    the mesh motion against the original domain again. On a run from rest, u_ref = 0, and the run is
    the same as without it.

    The function spaces are given by ``elements``, a dict from the name of each space ("U", "V", "P") to
    the element passed to ``dolfinx.fem.functionspace``. The default is quadratic vector Lagrange for
    u, v and linear Lagrange for p.

    The physical constants are ``rho_f``, ``nu_f`` (fluid density and kinematic viscosity), ``rho_s``,
    ``mu_s`` (solid density and shear modulus), ``U_bar`` (mean inflow speed scale) and ``H`` (channel
    height), with defaults 1e3, 1e-3, 1e4, 5e5, 1.0 and 0.41. The first Lamé parameter of the solid is
    ``lambda_s``, or computed from the Poisson ratio ``nu_s``; giving both raises ``ValueError``, and
    with neither, ``nu_s = 0.4``. Theta of the time stepping is 0.5 + ``absolute_theta_shift``, or
    0.5 + ``relative_theta_shift`` * ``dt_val``; giving both raises ``ValueError``, and with neither,
    ``relative_theta_shift = 1.0``.

    With ``plot_matrices=True``, a spy plot and a block norm plot of the Jacobian at the
    final state are saved in ``plot_matrices_dir``, by default
    ``output/figures/matrix_blocks/fsi2_harmonic_fulljacobian_initialguess/``, with the dofs ordered
    by the storage blocks [u, v, p], by the blocks (u_S, v, p, u_I) of the harmonic version
    of eq. (1) of docs/restricted-iterative-solver.md, in the order (v, p, u_S, u_I) in which
    the fieldsplit of fsi2_harmonic_condensed.py solves them, and by the groups of the nest
    Jacobian (see ``jacobian_callback``). The matrix is gathered on rank 0.

    If ``max_quadrature_degree`` is given, the quadrature degree of the residual, and so of its
    Jacobian, and of the drag and lift is capped at it: integrals for which UFL estimates a higher
    degree are integrated inexactly at that degree, and the others keep their estimate. With the
    default None, every integral keeps the degree UFL estimates.

    With ``newton_log_path``, every Newton iterate is recorded in that file by
    ``xfsi_solver.tools.newton_log.NewtonLog``: the norms of the residual and of the Newton update,
    in total and on each block u, v and p, whether the Jacobian of the step was assembled anew, and
    the wall time of each iteration. A restart drops the rows after the checkpoint time, as it
    does from the QoI file.

    With ``predict_displacements=True``, the default, the initial guess of each Newton solve is not
    the state of the previous step but u = u_old + dt * v_old at the dofs of the solid cells, the
    interface included, and on the fluid u_old plus the harmonic extension of the increment
    dt * v_old on the interface, zero on the outer boundary, with v and p of the previous step. With
    v = v_old, this solves the solid kinematic relation, which the rows of u on the solid hold, and,
    when u_old does, the mesh equation, exactly. The fluid has to move with the solid: predicting the
    solid alone, with u_old on the fluid, deforms the fluid cells at the interface by the jump of the
    increment there, and from the fsi2_simplified snapshots at t = 10 and 16 on mesh_sec Newton then
    diverges in the first step. The extension is solved on the fluid submesh, by MUMPS LU, factorized
    once for the whole run. The prediction is set in an initial-guess hook of the SNES, which runs
    once per Newton solve, before the residual of the initial guess is evaluated. It needs the same
    element for u and v.

    With ``condense_solid=True``, the solid stress of the momentum equation at the new time is taken
    at u_old + dt * (theta * v + (1 - theta) * v_old) instead of at u, as the solid kinematic
    relation gives it: the solid displacement is condensed out of it in UFL, so its derivative is in
    the (v, v) block of the Jacobian instead of the (v, u) block on the solid. u stays an unknown on
    the solid, held by the kinematic relation, since the fluid uses it on the interface. As the
    relation is linear, Newton satisfies it from its first step on, so the discrete solution is the
    same as without the option, up to the Newton tolerance.

    With ``jacobian_callback``, the Jacobian is copied after each assembly, in every Newton
    iteration, into a MatNest over the groups of dofs in ``NEST_GROUPS``, (v, p, u_s, u_int, u_f):
    v and p all their dofs, u_s the dofs of u on the solid cells off the interface, u_int those on
    the solid-fluid interface, and u_f the other dofs of u, those of the fluid cells off the
    interface, the outer fluid boundary included. Dofs with Dirichlet conditions stay in their
    group, with their identity rows. The Jacobian is the one Newton solves with: the interface rows
    of u hold the solid kinematic relation. Then ``jacobian_callback(snes, nest_view)`` is called,
    with the SNES, whose ``getSolution()`` is the iterate and ``getFunction()[0]`` its residual, and
    the ``xfsi_solver.tools.matrix_nest.NestView``: its ``nest`` is the MatNest, ``block(row, col)``
    gives a block by the names of its groups, and ``index_sets`` the rows of each group in the
    global numbering of the Jacobian, which the nest keeps, so that it acts on the vectors of the
    SNES. With the same element for u and v, ``nest_view.data["v_rows"]`` holds the rows of v at
    the dofs of the rows of u_s and u_int, for ``xfsi_solver.tools.condensation``. The nest is only
    a copy: Newton still solves with the assembled Jacobian, by MUMPS LU. The callback must be
    collective, as the blocks are distributed.

    ``snes_atol``, ``snes_rtol`` and ``snes_max_it`` are the stopping tolerances of the SNES.

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
    W = ufl.MixedFunctionSpace(U, V, P)

    # create functions

    u, v, p = (
        dolfinx.fem.Function(U, name="u"),
        dolfinx.fem.Function(V, name="v"),
        dolfinx.fem.Function(P, name="p"),
    )
    u_old, v_old = dolfinx.fem.Function(U), dolfinx.fem.Function(V)
    # The u of the start state, which the mesh motion is measured against with
    # move_reference_domain=True, set once the start state is read
    u_ref = dolfinx.fem.Function(U, name="u_ref")

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

    def A_I(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        # Harmonic mesh motion on the reference fluid domain, of the displacement since the start
        # state with move_reference_domain. The interface rows of du are restricted, so no
        # interface flux term is needed.
        u_mesh = u - u_ref if move_reference_domain else u
        residual = ufl.inner(ufl.grad(u_mesh), ufl.grad(du)) * dx_fluid

        residual += ufl.div(J * ufl.inv(F) * v) * dp * dx_fluid

        return residual

    def A_E(u, v, u_solid):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx_fluid

        residual += ufl.inner(J * Fluid.NS_velocity(u, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        # This one stays because the test function is dv, not du. The solid displacement is u_solid,
        # which is u, or what the kinematic relation gives with condense_solid.
        F_s = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_solid)
        J_s = ufl.det(F_s)
        residual += ufl.inner(J_s * Solid.STVK(u_solid, lambda_s, mu_s) * ufl.inv(F_s).T, ufl.grad(dv)) * dx_solid

        return residual

    def A_P(u, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual = J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    residual = A_T(u, u_old, v, v_old)
    residual += A_I(u, v)
    residual += A_P(u, p)
    # With condense_solid, the solid displacement of the solid stress is condensed out by the solid
    # kinematic relation, (u - u_old) / dt = theta * v + (1 - theta) * v_old.
    u_solid = u_old + dt * (theta * v + (1.0 - theta) * v_old) if condense_solid else u
    residual += theta * A_E(u, v, u_solid)
    residual += (1.0 - theta) * A_E(u_old, v_old, u_old)

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
        [u, v, p],
        bcs=bcs,
        petsc_options_prefix="fsi2_harmonic_fulljacobian_initialguess_",
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
            # "snes_monitor": "ascii:output/logs/fsi2_harmonic_fulljacobian_initialguess_snes_log.txt",
            **({"snes_converged_reason": None} if snes_converged_reason_on else {}),
        },
    )

    # The checkpoints of a run with move_reference_domain also hold u_ref, which a restart reads
    # back; a run from another run's state takes u_ref from u instead.
    checkpointed = [u, v, p, u_ref] if move_reference_domain else [u, v, p]

    # (t, step) are the time and number of completed steps of the current state
    t, step, checkpointer, (output_path, output_path_p) = load_start_state(
        mesh,
        checkpointed if restart else [u, v, p],
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
    if move_reference_domain and not restart:
        u_ref.x.array[:] = u.x.array

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
    # of the block matrix are numbered [u, v, p] from the start of the
    # ownership range, and every interface row is owned by a process that
    # located it, so zeroing owned rows covers all of them.
    dofs_d, num_owned_d = bc_deactivate.dof_indices()
    off_own, _ = problem.b.getAttr("_blocks")
    rows_d = (J_mat.getOwnershipRange()[0] + off_own[0] + dofs_d[:num_owned_d]).astype(PETSc.IntType)

    # The groups of NEST_GROUPS, as the global rows of the owned dofs in each. locate_dofs_topological
    # also finds the owned dofs on entities that only other ranks have marked, so each group has all
    # its owned rows.
    first_row = J_mat.getOwnershipRange()[0]

    def owned_rows(block: int, space: dolfinx.fem.FunctionSpace, dofs: np.ndarray | None = None) -> np.ndarray:
        """The global rows of the owned dofs among the (blocked) local ``dofs`` of ``space``, all
        of them if None, which is block ``block`` of [u, v, p]."""
        num_owned = space.dofmap.index_map.size_local
        dofs = np.arange(num_owned) if dofs is None else dofs[dofs < num_owned]
        bs = space.dofmap.index_map_bs
        return (first_row + off_own[block] + bs * dofs[:, None] + np.arange(bs)).ravel().astype(PETSc.IntType)

    solid_dofs_u = dolfinx.fem.locate_dofs_topological(U, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"]))
    all_dofs_u = np.arange(U.dofmap.index_map.size_local)
    u_s_dofs = np.setdiff1d(solid_dofs_u, dofs_interface)
    groups = {
        "v": owned_rows(1, V),
        "p": owned_rows(2, P),
        "u_s": owned_rows(0, U, u_s_dofs),
        "u_int": owned_rows(0, U, dofs_interface),
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
    ]

    jac_post = [
        [jac_uu, jac_uv, None],
        [None, zero_block(V, V), None],
        [None, None, zero_block(P, P)],
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

        dolfinx.fem.petsc.set_bc(b, [[bc_deactivate], [], []], alpha=0.0)  # zero restricted rows

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
        # included. The solid dofs hold the owned ones, also where the owner has no solid cell.
        block_size = U.dofmap.index_map_bs

        def unrolled(dofs: np.ndarray) -> np.ndarray:
            return (block_size * dofs[:, None] + np.arange(block_size)).ravel()

        solid_u = unrolled(solid_dofs_u)

        # The increment w of the fluid displacement lives on the fluid submesh, with the element of
        # u, zero on the outer fluid boundary and dt * v_old on the interface. The dofmaps of the
        # fluid cells give the dof of u at each local dof of w. Every dof of w on a rank lies in one
        # of its fluid cells, so this covers the owned and ghost dofs.
        U_f = dolfinx.fem.functionspace(fluid_mesh, elements["U"])
        w = dolfinx.fem.Function(U_f, name="w")
        w_bc_func = dolfinx.fem.Function(U_f)
        fluid_mesh.topology.create_connectivity(fluid_mesh.topology.dim - 1, fluid_mesh.topology.dim)
        w_bc = dolfinx.fem.dirichletbc(
            w_bc_func,
            dolfinx.fem.locate_dofs_topological(
                U_f, fluid_mesh.topology.dim - 1, dolfinx.mesh.exterior_facet_indices(fluid_mesh.topology)
            ),
        )
        extension = harmonic_mesh_motion(
            w,
            [w_bc],
            petsc_options_prefix="fsi2_harmonic_fulljacobian_initialguess_extension_",
            max_quadrature_degree=max_quadrature_degree,
        )
        fluid_cells = all_cells(fluid_mesh)
        fluid_parent_cells = fluid_cell_map.sub_topology_to_topology(fluid_cells, inverse=False)
        u_dof_of_w = np.empty(U_f.dofmap.index_map.size_local + U_f.dofmap.index_map.num_ghosts, dtype=np.int32)
        u_dof_of_w[U_f.dofmap.list[fluid_cells]] = U.dofmap.list[fluid_parent_cells]
        u_of_w = unrolled(u_dof_of_w)  # the dof of u at each unrolled dof of w
        fluid_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
            facet_tags, fluid_mesh, fluid_vertex_map, fluid_cell_map
        )
        interface_w = unrolled(
            dolfinx.fem.locate_dofs_topological(
                U_f,
                fluid_mesh.topology.dim - 1,
                fluid_facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"]),
                remote=True,
            )
        )

        def predict(snes: PETSc.SNES, x: PETSc.Vec) -> None:
            """Set the initial guess x of the Newton solve to the predicted displacement, with v and p
            of the previous step. Called by SNES before the residual of the initial guess, when the
            functions still hold the previous step."""
            # From v_old, not u: a rank may hold an interface dof only through a fluid cell.
            w_bc_func.x.array[interface_w] = dt_val * v_old.x.array[u_of_w[interface_w]]
            extension.solve()
            u.x.array[u_of_w] = u_old.x.array[u_of_w] + w.x.array
            u.x.array[solid_u] = u_old.x.array[solid_u] + dt_val * v_old.x.array[solid_u]
            u.x.scatter_forward()
            dolfinx.fem.petsc.assign([u, v, p], x)

        solver.setInitialGuess(predict)

    init_qoi_file(qoi_path, comm, restart, t, dt_val)
    if newton_log_path is not None:
        newton_log = NewtonLog(newton_log_path, field_blocks(problem, ["u", "v", "p"]), comm, restart, t, dt_val)
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
            checkpointer.write(checkpointed, t, step, dt_val, t0)

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")
        print(f"\nSolver setup:\n  Quadrature degree: {quadrature_description(max_quadrature_degree)}")
        initial_guess = "the previous step"
        if predict_displacements:
            initial_guess = "u = u_old + dt * v_old on the solid, and its harmonic extension on the fluid"
        print(f"  Initial guess: {initial_guess}")
        print(f"  Solid displacement condensed out of the solid stress: {condense_solid}")

    for writer in writers:
        writer.close()

    if plot_matrices:
        from xfsi_solver.tools.matrix_blocks import matrix_blocks_dir, plot_matrix_blocks

        dolfinx.fem.petsc.assign([u, v, p], problem.x)
        fem_jacobian(solver, problem.x, J_mat, P_mat, *jac_args, **jac_kargs)
        post_jacobian(problem.x, J_mat)

        # The owned u dofs supported on solid cells, the interface included, are u_S in eq. (1); the
        # others are u_I.
        rows_u = np.union1d(np.union1d(groups["u_s"], groups["u_int"]), groups["u_f"])
        rows_u_solid = np.union1d(groups["u_s"], groups["u_int"])
        rows_v, rows_p, rows_u_fluid = groups["v"], groups["p"], groups["u_f"]
        partitions = {
            "storage": ([rows_u, rows_v, rows_p], ["$u$", "$v$", "$p$"]),
            "eq1": ([rows_u_solid, rows_v, rows_p, rows_u_fluid], ["$u_S$", "$v$", "$p$", "$u_I$"]),
            "split": ([rows_v, rows_p, rows_u_solid, rows_u_fluid], ["$v$", "$p$", "$u_S$", "$u_I$"]),
            "nest": ([groups[name] for name in NEST_GROUPS], ["$v$", "$p$", "$u_s$", "$u_{int}$", "$u_f$"]),
        }
        if plot_matrices_dir is None:
            plot_matrices_dir = matrix_blocks_dir(__file__)
        plot_matrix_blocks(J_mat, partitions, "Full Jacobian", plot_matrices_dir, t)

    # Destroying the MUMPS factorization is collective, so destroy it on all ranks here instead of leaving it
    # to garbage collection, which can run at different points on different ranks, such as after rank 0 has
    # plotted the matrices.
    solver.destroy()
    if predict_displacements:
        extension.destroy()
    if nest_view is not None:
        nest_view.destroy()

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument("--initial-state-dir", help="start from the latest checkpoint of another run in this directory")
    args = parser.parse_args()

    mesh_path = "data/meshes/fsi2/mesh_sec.xdmf"
    dt_val = 0.0025
    T = 16.0 + 2 * dt_val
    output_path = "output/pv/fsi2_harmonic_fulljacobian_initialguess.bp"
    output_path_p = "output/pv/fsi2_harmonic_fulljacobian_initialguess_p.bp"
    qoi_path = "output/qoi/fsi2_harmonic_fulljacobian_initialguess.txt"
    newton_log_path = "output/qoi/fsi2_harmonic_fulljacobian_initialguess_newton.txt"
    checkpoint_dir = "output/checkpoints/fsi2_harmonic_fulljacobian_initialguess"
    checkpoint_every = None
    vtx_save_every = None
    max_quadrature_degree = None  # Default None, no cap.
    condense_solid = False
    # condense_solid = True
    restart = args.restart
    # A snapshot of the biharmonic mesh motion of fsi2_simplified, so the mesh motion is measured
    # against its domain, with move_reference_domain.
    initial_state_dir = args.initial_state_dir or "output/snapshots/fsi2_simplified/t16.0000"
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
        move_reference_domain=True,
        condense_solid=condense_solid,
        jacobian_callback=J_0_jacobian_hook,
        # jacobian_callback=exact_solid_condensation_hook,
    )

    if MPI.COMM_WORLD.rank == 0:
        plot_run_qois(qoi_path, T)


if __name__ == "__main__":
    main()
