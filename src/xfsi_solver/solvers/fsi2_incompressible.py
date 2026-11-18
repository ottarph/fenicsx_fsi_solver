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

from xfsi_solver.fsi.materials import Fluid, Solid
from xfsi_solver.mesh_motion import biharmonic_mixed_mesh_motion
from xfsi_solver.petsc_options.fsi2d import (
    DEFAULT_DIRECT_SOLVER_OPTIONS,
    DEFAULT_ITERATIVE_SOLVER_OPTIONS,
    MUMPS_ON_SOLID,
)
from xfsi_solver.scripts.plot_qois import INCOMPRESSIBLE_QOI_LABELS, plot_run_qois
from xfsi_solver.tools import skip_dirichlet_lifting as dirichlet_lifting
from xfsi_solver.tools.checkpoint import SnapshotWriter, check_start_options, load_start_state, start_time
from xfsi_solver.tools.convergence import check_converged
from xfsi_solver.tools.newton_log import NewtonLog, field_blocks
from xfsi_solver.tools.parameters import theta_value
from xfsi_solver.tools.qoi import (
    INCOMPRESSIBLE_QOI_HEADER,
    append_qoi_row,
    assemble_force,
    drag_lift_forms,
    find_point_dof,
    incompressibility_violation,
    init_qoi_file,
    point_value,
)
from xfsi_solver.tools.quadrature import cap_quadrature_degree, quadrature_description
from xfsi_solver.tools.rigid_body_modes import rigid_body_modes
from xfsi_solver.tools.submesh import SubmeshComposite, all_cells
from xfsi_solver.tools.vtkhdf import VTKHDFWriter

PHYSICAL_MARKERS = {
    "solid": 1,
    "ALE_fluid": 2,
    #
    "solid_fluid_interface": 11,
    #
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
    qoi_path,
    elements={  # noqa: B006
        "U": ("P", 2, (2,)),
        "V": ("P", 2, (2,)),
        "P": ("P", 1),
        "P_S": ("P", 1),
    },
    solver_options=None,
    mesh_motion_type=None,
    rho_f=1.0e3,
    nu_f=1.0e-3,
    rho_s=1.0e4,
    c1=0.0,
    c2=2.5e5,
    U_bar=1.0,
    H=0.41,
    absolute_theta_shift=None,
    relative_theta_shift=None,
    checkpoint_dir=None,
    checkpoint_every=None,
    output_every=None,
    restart=False,
    initial_state_dir=None,
    snapshot_dir=None,
    snapshot_times=(),
    gamma_reassemble=0.05,
    max_quadrature_degree=None,
    mesh_motion_options=None,
    requires_v_sub_IS=False,
    newton_log_path=None,
    predict_displacements=True,
    skip_dirichlet_lifting=True,
    snes_converged_reason_on=True,
):
    """Solve the FSI2i benchmark, FSI2 with an incompressible solid, with Newton's method on the
    velocities and pressures only.

    The solid is an incompressible Mooney-Rivlin material with c1 = 0 and c2 = 2.5e5 (Wichrowski et
    al., Int. J. Numer. Methods Eng. 124 (2023), Sec. 2.3.2 and Table 4; their mu1 = 0, mu2 = 5e5 Pa
    are c1 = mu1 / 2, c2 = mu2 / 2). The constraint det(F) = 1 is imposed by a solid pressure p_S, on
    the solid submesh, in addition to the fluid pressure p, on the fluid submesh. The elastic stress,
    shifted by its value 2 (c1 - c2) I in the reference configuration, so that the solid is stress
    free at rest with p_S = 0, is theta-averaged in time, while the pressure stress -p_S I and the
    constraint are fully implicit. The time stepping is shifted Crank-Nicolson, with theta = 0.5 +
    ``absolute_theta_shift``, or theta = 0.5 + ``relative_theta_shift`` * ``dt_val``; with neither
    given, ``relative_theta_shift=2.0``, twice the shift of fsi2_simplified, and giving both raises
    ``ValueError``. The shift damps the stiff modes of the solid velocity by the factor
    (1 - theta) / theta per step, among them a mode that flips sign every step and that the
    displacement does not see.

    The solver is fsi2_simplified with this solid. The solid stress, the solid pressure stress and the
    constraint are written in terms of v through the kinematic relation u_s = u_s_old + dt * (theta * v
    + (1 - theta) * v_old) (Failer & Richter, J. Sci. Comput. 82:28, 2020, eq. (5)), and the Jacobian
    is that of the momentum, continuity and constraint equations with respect to (v, p, p_S), solved as
    ``solver_options`` sets: by default ``DEFAULT_DIRECT_SOLVER_OPTIONS``, MUMPS LU, or by FGMRES,
    preconditioned by the block upper triangular Schur complement factor over v and the combined
    pressure (p, p_S), with one BoomerAMG V-cycle on the selfp approximation of the Schur complement
    and, for the velocity block,

    - with ``DEFAULT_ITERATIVE_SOLVER_OPTIONS``, one GAMG V-cycle (rigid body modes as near-nullspace);
    - with ``MUMPS_ON_SOLID`` and ``requires_v_sub_IS=True``, a multiplicative fieldsplit into the
      velocity dofs on solid cells (with the interface), solved by MUMPS LU, and the fluid dofs, by one
      GAMG V-cycle (translations as near-nullspace) with two under-relaxed SOR steps per level.

    ``requires_v_sub_IS=True`` builds the index sets of the inner split, and leaves the rigid body
    modes off the velocity block; it is needed with, and only with, a fieldsplit of the velocity block.
    The fieldsplit options name the splits ``v_0`` and ``p_1``, as for fsi2_simplified, where ``p_1``
    here holds p and p_S. The Jacobian is one AIJ matrix, not a nest matrix as in fsi2_simplified,
    since the selfp approximation is built by products of the off-diagonal blocks, which PETSc does
    not form for the nest submatrix over the two pressures.

    The displacements are not unknowns of the Newton system: after each linear solve, a post-solve hook
    of the KSP updates v to the next iterate and sets the displacements from the kinematic relation,
    node by node: the solid displacement u_s, and the same at the interface dofs of the fluid
    displacement u_f, which the mesh motion ``mesh_motion_type``, one of the functions of
    ``xfsi_solver.mesh_motion``, by default ``biharmonic_mixed_mesh_motion``, extends into the fluid,
    before SNES evaluates the residual there. By default, the mesh motion does not couple the x and y
    components, which have the same matrix, so only the x component is factorized, by MUMPS once for
    the whole run, and both components are solved with that factorization at once, by one MatMatSolve.
    ``mesh_motion_options``, the PETSc options of its KSP, replace that default.

    With ``predict_displacements=True``, the default, the displacements are also set before each Newton
    solve, by an initial-guess hook of the SNES, as after a linear solve but from the velocity of the
    previous step: u_s = u_s_old + dt * v_old, the same on the interface for u_f, and its mesh motion.
    With ``predict_displacements=False``, the initial guess is the state of the previous step as it is.

    The momentum equation is multiplied by dt, as in fsi2_simplified; the continuity equation and the
    constraint are not. ``snes_atol`` therefore applies to dt times the momentum residual.

    The output, the restart checkpoints and the snapshots are VTKHDF files of the same layout
    (``xfsi_solver.tools.vtkhdf``), each of which opens in ParaView: u, v, p and p_S at the nodes of the
    mesh, with the cell tags and the owning rank of each cell. p, on the fluid submesh, is 0 on the
    solid cells, and p_S, on the solid submesh, is 0 on the fluid cells, so in ParaView their values
    blend to 0 across the cells next to the interface; a Threshold on ``cell_tags`` shows one part
    alone. With ``output_every``, the state is written to ``output_path`` every ``output_every`` steps;
    a restart writes a new file, ``output_path`` renamed by
    ``xfsi_solver.tools.checkpoint.restart_output_path``, so the output before it is kept.

    Checkpoints store the displacement as one function u on the parent mesh, with v, p and p_S. They
    hold no field of the mesh motion, which each solve computes again, so a run can start from the
    io4dolfinx ``.bp`` checkpoints of fsi2_biharmonic_mixed_incompressible, the solver this one
    replaces, whose extra field z is not read, as well as from those of this solver.

    With ``snapshot_dir``, the state at the time steps closest to the times ``snapshot_times`` is
    also kept, each in a checkpoint directory ``snapshot_dir/t<t:.4f>`` of its own, from which a run
    starts with ``initial_state_dir``, by ``xfsi_solver.tools.checkpoint.SnapshotWriter``: a time
    after ``T`` raises ``ValueError``, and a new run removes the snapshots of an earlier run in
    ``snapshot_dir``, while a restart keeps those up to the step it resumes from.

    The QoI file has, after the drag, lift and tip displacement, three measures of the violation of
    det(F) = 1 in the solid, by ``xfsi_solver.tools.qoi.incompressibility_violation``.

    If ``max_quadrature_degree`` is given, the quadrature degree of the residual, and so of its
    Jacobian, and of the drag and lift is capped at it: integrals for which UFL estimates a higher
    degree are integrated inexactly at that degree, and the others keep their estimate. With the
    default None, every integral keeps the degree UFL estimates. The measures of the violation of
    det(F) = 1 are not capped.

    With ``skip_dirichlet_lifting=True``, the residual is assembled without lifting the Dirichlet
    conditions of v, and every Newton iterate holds their values exactly, by
    ``xfsi_solver.tools.skip_dirichlet_lifting.skip_dirichlet_lifting``: the values are set in the
    initial guess, before the displacement prediction, and the Newton update is held at zero on the
    Dirichlet dofs, in the post-solve hook before v is updated with it. The converged solution is the
    same as with False, where dolfinx lifts the conditions in every residual evaluation.

    ``elements`` gives the finite element of each function space ``U``, ``V``, ``P`` and ``P_S``, as
    the tuple passed to ``dolfinx.fem.functionspace``; U and V must be the same. The physical constants
    are the fluid density ``rho_f`` and kinematic viscosity ``nu_f``, the solid density ``rho_s`` and
    Mooney-Rivlin coefficients ``c1`` and ``c2``, the mean inflow velocity ``U_bar``, and the channel
    height ``H``. ``newton_log_path`` has the blocks v, p and p_S. ``gamma_reassemble`` sets the
    Jacobian reuse, as in fsi2_simplified.

    Returns the statistics of the time steps of this run, the same on every rank: the wall time of the
    time loop, ``elapsed_time``, ``num_steps``, ``num_newton_iterations``, ``num_assemblies`` of the
    Jacobian, and the Newton iterations and Jacobian assemblies of each step, ``iterations_per_step``
    and ``assemblies_per_step``.

    With ``snes_converged_reason_on=False``, the SNES does not print why each Newton solve stopped.
    Printing it makes PETSc keep a stdout viewer for the communicator of the mesh, in a table of
    fixed size, until PetscFinalize, so a process that creates a few hundred meshes, such as the
    test suite, which turns it off in tests/conftest.py, overflows the table and crashes.
    """
    check_start_options(restart, checkpoint_dir, checkpoint_every, initial_state_dir, snapshot_dir, snapshot_times)
    theta_val = theta_value(dt_val, absolute_theta_shift, relative_theta_shift, 2.0)

    if solver_options is None:
        solver_options = DEFAULT_DIRECT_SOLVER_OPTIONS
    if mesh_motion_type is None:
        mesh_motion_type = biharmonic_mixed_mesh_motion

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

    # create submesh for solid

    solid_mesh, solid_cell_map, solid_vertex_map, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"])
    )

    # create entity maps for mixed mesh integration

    entity_maps = [fluid_cell_map, solid_cell_map]

    # Need facet tags on the solid mesh, for its Dirichlet condition
    solid_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
        facet_tags, solid_mesh, solid_vertex_map, solid_cell_map
    )

    fluid_mesh.topology.create_connectivity(fluid_mesh.topology.dim - 1, fluid_mesh.topology.dim)
    solid_mesh.topology.create_connectivity(solid_mesh.topology.dim - 1, solid_mesh.topology.dim)

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
    c1 = dolfinx.fem.Constant(mesh, c1)
    c2 = dolfinx.fem.Constant(mesh, c2)

    t0 = start_time(mesh, restart, checkpoint_dir, initial_state_dir)
    dt = dolfinx.fem.Constant(mesh, dt_val)

    theta = dolfinx.fem.Constant(mesh, theta_val)

    num_steps = round((T - t0) / dt_val)
    if output_every is not None and num_steps < output_every:
        warnings.warn(
            f"output_every ({output_every}) is larger than the total number of time "
            f"steps ({num_steps}); at most one time will be written to "
            f"{output_path!r}, which is not a usable time series in ParaView.",
            stacklevel=2,
        )

    # create function spaces
    V = dolfinx.fem.functionspace(mesh, elements["V"])
    P = dolfinx.fem.functionspace(fluid_mesh, elements["P"])
    P_S = dolfinx.fem.functionspace(solid_mesh, elements["P_S"])
    U_s = dolfinx.fem.functionspace(solid_mesh, elements["U"])
    U_f = dolfinx.fem.functionspace(fluid_mesh, elements["U"])
    W = ufl.MixedFunctionSpace(V, P, P_S)

    # create functions

    v, p, p_S, u_s, u_f = (
        dolfinx.fem.Function(V, name="v"),
        dolfinx.fem.Function(P, name="p"),
        dolfinx.fem.Function(P_S, name="p_S"),
        dolfinx.fem.Function(U_s, name="u_s"),
        dolfinx.fem.Function(U_f, name="u_f"),
    )
    v_old, u_s_old, u_f_old = dolfinx.fem.Function(V), dolfinx.fem.Function(U_s), dolfinx.fem.Function(U_f)

    dv, dp, dp_S = ufl.TestFunctions(W)

    # The displacement on the parent mesh, u_s on the solid cells and u_f on the fluid cells, which
    # the checkpoints and the output hold in place of u_s and u_f. The checkpoints hold no other
    # field of the mesh motion, such as z of the mixed form: every mesh motion solve computes those
    # again from u_s, so the state is (u, v, p, p_S) whichever mesh motion is used.
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, elements["U"]), name="u")
    checkpoint_fields = (u, v, p, p_S)

    displacement = SubmeshComposite(u, [(u_s, solid_cell_map), (u_f, fluid_cell_map)])

    # Set up output, checkpointing, and qoi tracking.

    # (t, step) are the time and number of completed steps of the current state
    submeshes = [(fluid_mesh, fluid_cell_map), (solid_mesh, solid_cell_map)]
    t, step, checkpointer, (output_path,) = load_start_state(
        mesh,
        checkpoint_fields,
        t0=t0,
        T=T,
        dt=dt_val,
        num_steps=num_steps,
        restart=restart,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        initial_state_dir=initial_state_dir,
        submeshes=submeshes,
        output_paths=[output_path],
        write_vtx=output_every is not None,
        file_format="vtkhdf",
        cell_tags=cell_tags,
    )
    displacement.scatter()  # u_s, u_f <- u, which is 0 on a new run from rest
    snapshots = SnapshotWriter(
        snapshot_dir,
        mesh,
        submeshes,
        snapshot_times,
        t0=t0,
        dt=dt_val,
        num_steps=num_steps,
        step=step,
        restart=restart,
        file_format="vtkhdf",
        cell_tags=cell_tags,
    )

    writers = []
    if output_every is not None:
        writers = [VTKHDFWriter(output_path, mesh, checkpoint_fields, submeshes, cell_tags)]

    # Quantities of interest: the displacement at the tip of the structure, the drag and lift on the
    # obstacle and on the fluid side of the solid-fluid interface, and the violation of det(F) = 1 in
    # the solid. At convergence, u_s is the displacement that the constraint is imposed on.
    spot_dof = find_point_dof(U_s, np.array([0.6, 0.2, 0.0], dtype=np.float64))
    drag_forms, lift_forms = drag_lift_forms(
        mesh,
        u_f,
        v,
        p,
        nu_f,
        rho_f,
        [ds(PHYSICAL_MARKERS["obstacle"]), ds_interface_fluid],
        entity_maps,
        max_quadrature_degree=max_quadrature_degree,
    )
    measure_incompressibility = incompressibility_violation(u_s, ufl.dx(domain=solid_mesh), all_cells(solid_mesh))

    def write_output(t, step):
        """Write the QoI row, and every output_every steps the output, of the state at time t."""
        if output_every is not None and step % output_every == 0:
            displacement.gather()  # u is written with v
            for writer in writers:
                writer.write(t)

        u_spot = point_value(u_s, spot_dof)
        drag = assemble_force(drag_forms, comm)
        lift = assemble_force(lift_forms, comm)
        append_qoi_row(qoi_path, comm, t, drag, lift, u_spot, *measure_incompressibility())

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

    u_f_bc_func = dolfinx.fem.Function(U_f)
    u_f_bc_func.x.array[:] = 0.0
    u_f_bc_facets = dolfinx.mesh.exterior_facet_indices(fluid_mesh.topology)
    u_f_bc_dofs = dolfinx.fem.locate_dofs_topological(U_f, fluid_mesh.geometry.dim - 1, u_f_bc_facets)
    u_f_bc = dolfinx.fem.dirichletbc(u_f_bc_func, u_f_bc_dofs)

    # Create solid Dirichlet boundary condition

    u_s_bc_func = dolfinx.fem.Function(U_s)
    u_s_bc_func.x.array[:] = 0.0
    u_s_bc_facets = reduce(
        np.union1d,
        [
            solid_facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
        ],
    )
    u_s_bc_dofs = dolfinx.fem.locate_dofs_topological(U_s, solid_mesh.geometry.dim - 1, u_s_bc_facets)
    u_s_bc = dolfinx.fem.dirichletbc(u_s_bc_func, u_s_bc_dofs)

    # Create Dirichlet boundary condition for restricting test functions.

    total_interface_facets_found = mesh.comm.allreduce(interface_facets.size, op=MPI.SUM)
    assert total_interface_facets_found > 0, "Interface not found."

    # Collect Dirichlet boundary conditions to pass to NonlinearProblem

    bcs = [inflow_bc, noslip_bc]

    # DESCRIBE FSI PROBLEM
    # FLUID: Parabolic inflow on left side, no-slip on top, bottom, and obstacle, do-nothing on right side
    # SOLID: Homogeneous Dirichlet on left side

    # create residual form with u condensed out.

    def A_T_mesh_transport(v, v_old, u_f, u_f_old):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = -rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * ((u_f - u_f_old) / dt), dv) * dx_fluid

        return residual

    def A_T_mass(v, v_old, u_f, u_f_old):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)
        F_old = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f_old)
        J_old = ufl.det(F_old)
        J_mid = 0.5 * (J + J_old)

        residual = rho_f * J_mid * ufl.inner((v - v_old) / dt, dv) * dx_fluid

        residual += rho_s * ufl.inner((v - v_old) / dt, dv) * dx_solid

        return residual

    def A_T(v, v_old, u_f, u_f_old):
        return A_T_mass(v, v_old, u_f, u_f_old) + A_T_mesh_transport(v, v_old, u_f, u_f_old)

    def A_I_fluid(v, u_f):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = ufl.div(J * ufl.inv(F) * v) * dp * dx_fluid

        return residual

    def A_I_solid(u_s):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_s)
        J = ufl.det(F)

        residual = (J - 1) * dp_S * dx_solid

        return residual

    def A_E_fluid_viscous(v, u_f):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = ufl.inner(J * Fluid.NS_velocity(u_f, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    def A_E_fluid_transport(v, u_f):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx_fluid

        return residual

    def A_E_fluid(v, u_f):

        return A_E_fluid_viscous(v, u_f) + A_E_fluid_transport(v, u_f)

    def A_E_solid(u_s):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_s)
        J = ufl.det(F)

        # The elastic stress less its value in the reference configuration, so that the solid at rest is
        # stress free with p_S = 0.
        sigma_ref = 2.0 * (c1 - c2) * ufl.Identity(mesh.geometry.dim)
        residual = ufl.inner(J * (Solid.IMR_elastic(u_s, c1, c2) - sigma_ref) * ufl.inv(F).T, ufl.grad(dv)) * dx_solid

        return residual

    def A_P_fluid(p, u_f):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_f)
        J = ufl.det(F)

        residual = J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    def A_P_solid(p_S, u_s):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u_s)
        J = ufl.det(F)

        residual = J * ufl.inner(Solid.IMR_pressure(p_S) * ufl.inv(F).T, ufl.grad(dv)) * dx_solid

        return residual

    # The momentum equation is multiplied by dt; the continuity equation and the constraint are not.
    residual = dt * A_T(v, v_old, u_f, u_f_old)
    residual += A_I_fluid(v, u_f)
    residual += dt * A_P_fluid(p, u_f)
    residual += dt * theta * A_E_fluid(v, u_f)
    residual += dt * (1.0 - theta) * A_E_fluid(v_old, u_f_old)

    # The elastic stress is theta-averaged; the solid pressure and the constraint are fully implicit.
    u_s_condensed = u_s_old + dt * ((1.0 - theta) * v_old + theta * v)
    residual += dt * theta * A_E_solid(u_s_condensed)
    residual += dt * (1.0 - theta) * A_E_solid(u_s_old)
    residual += dt * A_P_solid(p_S, u_s_condensed)
    residual += A_I_solid(u_s_condensed)

    residual_blocks = ufl.extract_blocks(residual)

    # Cap the quadrature degree before taking derivatives, so that the Jacobian blocks keep it.
    residual_blocks = [cap_quadrature_degree(form, max_quadrature_degree) for form in residual_blocks]

    jacobian = dolfinx.fem.forms.derivative_block(residual_blocks, (v, p, p_S))

    # --------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])

    # One AIJ matrix, also for the fieldsplit, whose splits are set by index sets below: the pressure
    # split holds p and p_S, and the selfp approximation of its Schur complement, a product of the
    # off-diagonal blocks, is not formed from the submatrices of a nest matrix over two pressures.
    use_fieldsplit = solver_options.get("pc_type") == "fieldsplit"
    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocks,
        (v, p, p_S),
        J=jacobian,
        bcs=bcs,
        petsc_options_prefix="solver_",
        entity_maps=entity_maps,
        petsc_options=solver_options
        if snes_converged_reason_on
        else {key: value for key, value in solver_options.items() if key != "snes_converged_reason"},
    )

    opts = PETSc.Options()
    opts["options_left"] = None

    if use_fieldsplit:
        # Each rank holds its owned rows as [v | p | p_S], from its first global row on, and
        # offsets_owned[k] is where block k starts within them.
        offsets_owned, _ = problem.b.getAttr("_blocks")
        first_row = problem.A.getOwnershipRange()[0]
        rows_v = first_row + np.arange(offsets_owned[0], offsets_owned[1])
        rows_pressure = first_row + np.arange(offsets_owned[1], offsets_owned[3])
        is_v = PETSc.IS().createGeneral(rows_v.astype(PETSc.IntType), comm=comm)
        is_pressure = PETSc.IS().createGeneral(rows_pressure.astype(PETSc.IntType), comm=comm)

        if not requires_v_sub_IS:
            is_v.setBlockSize(V.dofmap.index_map_bs)
            is_v.compose("nearnullspace", rigid_body_modes(V))

        problem.solver.getKSP().getPC().setFieldSplitIS(("v_0", is_v), ("p_1", is_pressure))

        # The fieldsplit calls setFromOptions on its sub-KSPs in PCSetUp, in the first solve, after
        # dolfinx means to have deleted the options in petsc_options. Its deletion, under a pushed
        # prefix, has no effect with petsc4py 3.25, so they stay; set those of the sub-KSPs again, so
        # that this does not depend on it.
        opts.prefixPush(problem.solver.getOptionsPrefix())
        for key, value in solver_options.items():
            if key.startswith("fieldsplit_"):
                opts[key] = value
        opts.prefixPop()

        # The solid (with the interface) and fluid dofs of v, in the numbering of the v block, which the
        # outer fieldsplit numbers by the position in is_v: V's own global numbering, not the Jacobian's.
        bs = V.dofmap.index_map_bs
        num_owned_nodes = V.dofmap.index_map.size_local
        solid_nodes = dolfinx.fem.locate_dofs_topological(
            V, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"])
        )
        solid_nodes = solid_nodes[solid_nodes < num_owned_nodes]
        in_solid = np.zeros(bs * num_owned_nodes, dtype=bool)
        in_solid[(bs * solid_nodes[:, None] + np.arange(bs)).ravel()] = True
        offset = bs * V.dofmap.index_map.local_range[0]
        is_v_s = PETSc.IS().createGeneral((offset + np.flatnonzero(in_solid)).astype(PETSc.IntType), comm=comm)
        is_v_f = PETSc.IS().createGeneral((offset + np.flatnonzero(~in_solid)).astype(PETSc.IntType), comm=comm)
        is_v_f.setBlockSize(bs)  # GAMG aggregates per node, with the translations as near-nullspace

    # The linear solve for the end-of-run summary, built here so that options without the keys it
    # reads fail before the time loop, not after it
    options = solver_options
    if options in (DEFAULT_ITERATIVE_SOLVER_OPTIONS, MUMPS_ON_SOLID):

        def smoothing_description(prefix):
            """The GAMG smoothing of the sub-KSP with the options prefix ``prefix``."""
            description = (
                f"{options[prefix + 'mg_levels_ksp_type']} with {options[prefix + 'mg_levels_pc_type']} smoothing"
            )
            details = []
            if prefix + "mg_levels_ksp_max_it" in options:
                details.append(f"{options[prefix + 'mg_levels_ksp_max_it']} steps")
            if prefix + "mg_levels_pc_sor_omega" in options:
                details.append(f"omega = {options[prefix + 'mg_levels_pc_sor_omega']}")
            return description + (f" ({', '.join(details)})" if details else "")

        if options == DEFAULT_ITERATIVE_SOLVER_OPTIONS:
            v_pc_description = (
                "one V-cycle of GAMG built from A (rigid body modes as near-nullspace),\n"
                f"      {smoothing_description('fieldsplit_v_0_')}"
            )
        else:
            s_prefix, f_prefix = "fieldsplit_v_0_fieldsplit_s_", "fieldsplit_v_0_fieldsplit_f_"
            v_pc_description = (
                f"a {options['fieldsplit_v_0_pc_fieldsplit_type']} fieldsplit of A into the velocity dofs on\n"
                f"      solid cells (with the interface), by {options[s_prefix + 'pc_factor_mat_solver_type'].upper()} "
                f"{options[s_prefix + 'pc_type'].upper()}, and the fluid dofs, by one V-cycle of GAMG\n"
                f"      (translations as near-nullspace), {smoothing_description(f_prefix)}"
            )
        p_pc_description = options["fieldsplit_p_1_pc_type"]
        if p_pc_description == "hypre":
            p_pc_description += f" {options.get('fieldsplit_p_1_pc_hypre_type', 'boomeramg')}"  # the default type
        linear_solve_description = (
            f"{options['ksp_type'].upper()} (rtol {options['ksp_rtol']:g}, atol {options['ksp_atol']:g}) , "
            f"preconditioned by the\n    {options['pc_fieldsplit_schur_fact_type']} Schur factor over v and (p, p_S), "
            f"{v_pc_description}, for A^{{-1}},\n"
            f"    one V-cycle of {p_pc_description} on -B diag(A)^{{-1}} G "
            f"({options['pc_fieldsplit_schur_precondition']}), for S^{{-1}}\n"
        )
    elif options == DEFAULT_DIRECT_SOLVER_OPTIONS:
        linear_solve_description = f"{options['pc_factor_mat_solver_type'].upper()} {options['pc_type'].upper()}\n"
    else:
        linear_solve_description = "custom PETSc options\n" + "".join(f"    {k}: {v}\n" for k, v in options.items())

    # Add extra callbacks to change how residual and jacobian is assembled,
    # to account for test function restriction on the interface.

    solver = problem.solver
    J_mat, P_mat, (fem_jacobian, jac_args, jac_kargs) = solver.getJacobian()

    # State of the Jacobian reuse in wrapped_jacobian_approximate. previous_norm is the
    # residual norm at the previous Newton iterate of the current time step.
    jacobian_state = {
        "assembled": False,
        "previous_norm": None,
        "num_assemblies": 0,
        "inner_split_set": False,
    }

    def wrapped_jacobian(snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat) -> None:
        norm = snes.getFunctionNorm()  # residual norm at the current iterate x
        previous_norm = jacobian_state["previous_norm"] if snes.getIterationNumber() > 0 else None
        jacobian_state["previous_norm"] = norm
        # Reuse J if the last Newton iteration converged fast enough, or at the first iteration
        # of a time step, where there is no rate yet. Leaving J unchanged makes PETSc skip
        # PCSetUp, so the fieldsplit submatrices and the factorization of K_c are reused too.
        # With gamma_reassemble=None, J is never reused.
        reuse = gamma_reassemble is not None and jacobian_state["assembled"]
        if reuse and (previous_norm is None or norm <= gamma_reassemble * previous_norm):
            return
        fem_jacobian(snes, x, J, P, *jac_args, **jac_kargs)
        jacobian_state["assembled"] = True
        jacobian_state["num_assemblies"] += 1

        if use_fieldsplit and requires_v_sub_IS and not jacobian_state["inner_split_set"]:
            # The sub-KSPs of the outer split exist only after its PCSetUp, which needs the assembled J.
            ksp = snes.getKSP()
            ksp.setOperators(J, P)
            ksp.getPC().setUp()
            ksp_v, _ = ksp.getPC().getFieldSplitSubKSP()
            ksp_v.getPC().setFieldSplitIS(("s", is_v_s), ("f", is_v_f))
            jacobian_state["inner_split_set"] = True

    solver.setJacobian(wrapped_jacobian, J_mat, P_mat)

    # The solid displacement is updated node by node by the kinematic relation. U_s and V have the same
    # element, so the dofs of a solid cell are in the same order in both, and the dofmaps of the solid
    # cells give the dof of v at each local dof of u_s.
    if elements["U"] != elements["V"]:
        raise ValueError(
            f"The nodal update needs the same element for U and V, not {elements['U']} and {elements['V']}"
        )
    solid_cells, fluid_cells = all_cells(solid_mesh), all_cells(fluid_mesh)
    solid_parent_cells = solid_cell_map.sub_topology_to_topology(solid_cells, inverse=False)
    num_u_s_dofs = U_s.dofmap.index_map.size_local + U_s.dofmap.index_map.num_ghosts
    v_dof_of_u_s = np.empty(num_u_s_dofs, dtype=np.int32)
    v_dof_of_u_s[U_s.dofmap.list[solid_cells]] = V.dofmap.list[solid_parent_cells]
    block_size_u = U_s.dofmap.index_map_bs
    v_dof_of_u_s = (block_size_u * v_dof_of_u_s[:, None] + np.arange(block_size_u)).ravel()  # unrolled

    # The mesh motion is linear, with a matrix that is the same for the whole run, so it is assembled
    # once, and by default factorized once, in the first solve.
    mesh_motion = mesh_motion_type(u_f, [u_f_bc], solver_options=mesh_motion_options)
    mesh_motion_name = mesh_motion_type.__name__.removesuffix("_mesh_motion")
    if mesh_motion_options is None:
        mesh_motion_description = (
            f"{mesh_motion_name}, MUMPS LU on the x component, once for the whole run, for both\n"
            "    components at once by MatMatSolve"
        )
    else:
        mesh_motion_description = f"{mesh_motion_name}, with the KSP options {mesh_motion_options}"

    num_v = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    # The Dirichlet values of u_f are 0 on the fluid boundary and, on the interface, the displacement of
    # the solid, which follows from the same kinematic relation as u_s: u_f = u_f_old + dt * (theta * v +
    # (1 - theta) * v_old) at the interface dofs. U_f and V have the same element, so the dofmaps of the
    # fluid cells give the dof of v at each local dof of u_f. Every dof of u_f on a rank lies in one of
    # its fluid cells, whose dofs of v the rank holds too, and v is up to date on its ghosts, so the
    # values are set on the owned and ghost dofs alike, without communication. locate_dofs_topological
    # finds the interface dofs also on ranks that hold them only through fluid cells that touch the
    # interface at a vertex (remote=True), which no interface facet of the rank would reach.
    fluid_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
        facet_tags, fluid_mesh, fluid_vertex_map, fluid_cell_map
    )

    fluid_movable_dofs = dolfinx.fem.locate_dofs_topological(
        U_f,
        fluid_mesh.topology.dim - 1,
        fluid_facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"]),
        remote=True,
    )
    fluid_parent_cells = fluid_cell_map.sub_topology_to_topology(fluid_cells, inverse=False)
    num_u_f_dofs = U_f.dofmap.index_map.size_local + U_f.dofmap.index_map.num_ghosts
    v_dof_of_u_f = np.empty(num_u_f_dofs, dtype=np.int32)
    v_dof_of_u_f[U_f.dofmap.list[fluid_cells]] = V.dofmap.list[fluid_parent_cells]
    interface_u_f = (block_size_u * fluid_movable_dofs[:, None] + np.arange(block_size_u)).ravel()  # unrolled
    interface_v = (block_size_u * v_dof_of_u_f[fluid_movable_dofs][:, None] + np.arange(block_size_u)).ravel()

    def update_displacements() -> None:
        """Set u_s and the interface values of u_f from v by the kinematic relation, node by node, and u_f
        by the mesh motion."""
        # u_s = u_s_old + dt * (theta * v + (1 - theta) * v_old), at the owned and ghost dofs. The
        # Dirichlet values hold already, since v and v_old vanish there, and are set again exactly.
        u_s.x.array[:] = u_s_old.x.array + dt_val * (
            theta_val * v.x.array[v_dof_of_u_s] + (1.0 - theta_val) * v_old.x.array[v_dof_of_u_s]
        )
        u_s_bc.set(u_s.x.array)
        u_f_bc_func.x.array[interface_u_f] = u_f_old.x.array[interface_u_f] + dt_val * (
            theta_val * v.x.array[interface_v] + (1.0 - theta_val) * v_old.x.array[interface_v]
        )
        mesh_motion.solve()

    def post_velocity_solve_update(ksp: PETSc.KSP, b: PETSc.Vec, x: PETSc.Vec) -> None:
        """Update the displacements to the next Newton iterate. Called at the end of the linear solve,
        where x is the Newton update Y of (v, p, p_S) and v still holds the current iterate, before SNES
        sets the next iterate to v - Y and evaluates the residual there."""
        if lifting_hooks is not None:
            # Zero the update on the Dirichlet dofs before v is updated with it: the line-search pre-check
            # that does so comes after this hook.
            lifting_hooks["set_dirichlet_update"](solver.getSolution(), x)
        # The owned v rows come first in x.
        v.x.array[:num_v] -= x.getArray(readonly=True)[:num_v]
        v.x.scatter_forward()
        update_displacements()

    # A KSP has one post-solve hook, so KSPConvCheck, which sets its own, cannot be attached to this KSP.
    # check_converged reports a failure of the linear solve through the SNES.
    solver.getKSP().setPostSolve(post_velocity_solve_update)

    lifting_hooks = dirichlet_lifting.skip_dirichlet_lifting(problem) if skip_dirichlet_lifting else None
    if lifting_hooks is not None:
        solver.setFunction(lifting_hooks["residual"], problem.b)
        solver.setLineSearchPreCheck(lifting_hooks["set_dirichlet_update"])

    def initial_guess(snes: PETSc.SNES, x: PETSc.Vec) -> None:
        """SNES calls this once per solve, before it evaluates the residual of the initial guess x, when v
        is still v_old."""
        if lifting_hooks is not None:
            # The Dirichlet values of v of the new step, in x.
            lifting_hooks["set_dirichlet_values"](snes, x)
        if predict_displacements:
            # The displacements become u_s_old + dt * v_old, and the same on the interface.
            update_displacements()

    if lifting_hooks is not None or predict_displacements:
        solver.setInitialGuess(initial_guess)

    # Open qoi file on fresh runs, discard entries after the restart time if a restarted run.
    init_qoi_file(qoi_path, comm, restart, t, dt_val, header=INCOMPRESSIBLE_QOI_HEADER)
    if newton_log_path is not None:
        newton_log = NewtonLog(newton_log_path, field_blocks(problem, ["v", "p", "p_S"]), comm, restart, t, dt_val)
        solver.setMonitor(newton_log)

    if not restart:
        write_output(t, step)
        if snapshots.due(step):
            snapshots.write(checkpoint_fields, t, step, dt_val, t0)  # u holds the start state

    PETSc.Sys.Print()
    mesh.comm.barrier()

    first_step = step
    num_newton_iterations = 0
    # Newton iterations and Jacobian assemblies of each time step of this run
    iterations_per_step, assemblies_per_step = [], []
    start = timer()
    while step < num_steps:
        t = t0 + (step + 1) * dt_val
        inflow_bc_func.interpolate(InflowFunc(t))
        inflow_bc_func.x.scatter_forward()

        v_old.x.array[:] = v.x.array[:]
        u_s_old.x.array[:] = u_s.x.array[:]
        u_f_old.x.array[:] = u_f.x.array[:]

        PETSc.Sys.Print(f"\n\n{t = :.3f}")

        if newton_log_path is not None:
            newton_log.start_step(t)
        num_assemblies_before = jacobian_state["num_assemblies"]
        problem.solve()
        num_newton_iterations += solver.getIterationNumber()
        iterations_per_step.append(solver.getIterationNumber())
        assemblies_per_step.append(jacobian_state["num_assemblies"] - num_assemblies_before)
        check_converged(problem, f"t = {t:.4f}", writers=writers, ksp_check=None)

        step += 1
        write_output(t, step)

        restart_checkpoint_due = checkpoint_every is not None and step % checkpoint_every == 0
        if snapshots.due(step) or restart_checkpoint_due:
            displacement.gather()
        # The snapshot goes first, so that every snapshot up to the step of a restart checkpoint is complete
        if snapshots.due(step):
            snapshots.write(checkpoint_fields, t, step, dt_val, t0)
        if restart_checkpoint_due:
            checkpointer.write(checkpoint_fields, t, step, dt_val, t0)

    end = timer()
    if gamma_reassemble is None:
        reuse_description = "none, reassembled in every Newton iteration"
    else:
        reuse_description = f"reassembled when ||R_l|| > gamma_reassemble * ||R_(l-1)||, with {gamma_reassemble = }"
    if predict_displacements:
        initial_guess_description = "the previous step, with the displacements predicted from dt * v_old"
    else:
        initial_guess_description = "the previous step"
    if skip_dirichlet_lifting:
        initial_guess_description += ", with the Dirichlet values of the step set in it"
        dirichlet_description = "skipped, since every Newton iterate holds the values exactly"
    else:
        dirichlet_description = "in every residual evaluation (dolfinx's assemble_residual)"
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")
        print(
            "\nSolver setup:\n"
            f"  Quadrature degree: {quadrature_description(max_quadrature_degree)}\n"
            f"  Time stepping: theta = 0.5 + {theta_val - 0.5:g} = {theta_val:g}\n"
            f"  Solid: incompressible Mooney-Rivlin, c1 = {c1.value:g}, c2 = {c2.value:g}, with solid pressure p_S\n"
            "    for det(F) = 1, elastic stress theta-averaged, pressure and constraint fully implicit\n"
            "  Solid stress: in terms of v, via u_s = u_s_old + dt * (theta * v + (1 - theta) * v_old)\n"
            "  Newton system: (v, p, p_S) only; u_s and the interface values of u_f set from v node by node,\n"
            "    and u_f by the mesh motion, after each linear solve\n"
            f"  Linear solve: {linear_solve_description}"
            f"  Mesh motion: {mesh_motion_description}\n"
            f"  Initial guess: {initial_guess_description}\n"
            f"  Lifting of the Dirichlet conditions of v: {dirichlet_description}\n"
            f"  Jacobian reuse: {reuse_description}\n"
            f"  Jacobian assemblies: {jacobian_state['num_assemblies']} in {num_newton_iterations} Newton iterations "
            f"over {step - first_step} time steps\n"
        )

    for writer in writers:
        writer.close()

    # Destroying the MUMPS factorization and the AMG hierarchies is collective, so destroy them on
    # all ranks here instead of leaving it to garbage collection, which can run at different points
    # on different ranks.
    solver.destroy()
    mesh_motion.destroy()

    return {
        "elapsed_time": end - start,
        "num_steps": step - first_step,
        "num_newton_iterations": num_newton_iterations,
        "num_assemblies": jacobian_state["num_assemblies"],
        "iterations_per_step": iterations_per_step,
        "assemblies_per_step": assemblies_per_step,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument("--initial-state-dir", help="start from the latest checkpoint of another run in this directory")
    parser.add_argument(
        "--snapshot-times", type=float, nargs="*", default=[], help="keep the state at the steps closest to these times"
    )
    args = parser.parse_args()

    # mesh_sec with the element size halved in the solid, from create_mesh_FSI2.py with solid_refinement=2
    mesh_path = "data/meshes/fsi2/mesh_sec_solid2x.xdmf"
    run_name = "fsi2_incompressible"
    dt_val = 0.0025
    T = 15.0
    output_path = f"output/pv/{run_name}.vtkhdf"
    qoi_path = f"output/qoi/{run_name}.txt"
    newton_log_path = f"output/qoi/{run_name}_newton.txt"
    checkpoint_dir = f"output/checkpoints/{run_name}"
    # Only with --snapshot-times, since a new run clears the snapshots of an earlier one; a restart
    # needs the same times again to write the snapshots after its checkpoint
    snapshot_times = args.snapshot_times
    snapshot_dir = f"output/snapshots/{run_name}" if snapshot_times else None
    checkpoint_every = 400
    output_every = None
    gamma_reassemble = 0.05  # Default 0.05.
    max_quadrature_degree = None  # Default None, no cap.
    solver_options = DEFAULT_DIRECT_SOLVER_OPTIONS
    # solver_options = DEFAULT_ITERATIVE_SOLVER_OPTIONS
    requires_v_sub_IS = False
    # solver_options = MUMPS_ON_SOLID
    # requires_v_sub_IS = True
    initial_state_dir = args.initial_state_dir
    restart = args.restart
    # restart = True

    # opts = PETSc.Options()
    # opts["solver_snes_monitor"] = None
    # opts["solver_ksp_converged_reason"] = None
    # opts["solver_ksp_monitor"] = None
    solve(
        mesh_path=mesh_path,
        T=T,
        dt_val=dt_val,
        output_path=output_path,
        qoi_path=qoi_path,
        newton_log_path=newton_log_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        snapshot_dir=snapshot_dir,
        snapshot_times=snapshot_times,
        output_every=output_every,
        gamma_reassemble=gamma_reassemble,  # Default 0.05.
        max_quadrature_degree=max_quadrature_degree,  # Default None, no cap.
        solver_options=solver_options,
        requires_v_sub_IS=requires_v_sub_IS,
        restart=restart,
        initial_state_dir=initial_state_dir,
    )
    # opts.view()

    if MPI.COMM_WORLD.rank == 0:
        # FSI2i has no FeatFlow reference, and the last 2 s is the window of the Wichrowski et al. reference.
        plot_run_qois(
            qoi_path,
            T,
            qoi_labels=INCOMPRESSIBLE_QOI_LABELS,
            reference_file=None,
            closeup_interval=(max(T - 2.0, 0.0), T),
        )


if __name__ == "__main__":
    main()
