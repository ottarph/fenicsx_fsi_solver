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

from xfsi_solver.fsi.forms import FSIForms
from xfsi_solver.mesh_motion import biharmonic_c0ip_mesh_motion, default_mesh_motion_solver_options
from xfsi_solver.petsc_options.fsi3d import (
    DEFAULT_DIRECT_SOLVER_OPTIONS,
    DEFAULT_ITERATIVE_SOLVER_OPTIONS,
    MUMPS_ON_SOLID,
)
from xfsi_solver.scripts.plot_qois import QOI_LABELS_3D, plot_run_qois
from xfsi_solver.tools import skip_dirichlet_lifting as dirichlet_lifting
from xfsi_solver.tools.checkpoint import SnapshotWriter, check_start_options, load_start_state, start_time
from xfsi_solver.tools.convergence import check_converged
from xfsi_solver.tools.newton_log import NewtonLog, field_blocks
from xfsi_solver.tools.parameters import lame_lambda, solid_material_coefficients, theta_value
from xfsi_solver.tools.qoi import (
    DET_F_QOI_HEADER_3D,
    append_qoi_row,
    assemble_force,
    deformation_jacobian_range,
    drag_lift_forms,
    find_point_dof,
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
        "U": ("P", 2, (3,)),
        "V": ("P", 2, (3,)),
        "P": ("P", 1),
    },
    solver_options=None,
    mesh_motion_type=None,
    rho_f=1.0e3,
    nu_f=1.0e-3,
    rho_s=1.0e3,
    mu_s=2.0e6,
    nu_s=None,
    lambda_s=None,
    solid_material="stvk",
    mooney_rivlin_3=None,
    U_bar=1.75,
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
    gamma_reassemble=0.1,
    max_quadrature_degree=8,
    mesh_motion_options=None,
    mesh_motion_max_quadrature_degree=8,
    det_F_lattice_degree=4,
    requires_v_sub_IS=False,
    predict_displacements=True,
    skip_dirichlet_lifting=True,
    newton_log_path=None,
    snes_converged_reason_on=True,
):
    """Solve the 3d FSI benchmark of Failer & Richter (J. Sci. Comput. 82:28, 2020, Sect. 5.1.2), an
    elastic beam attached to a cylinder in a channel of square cross section, with Newton's method on
    the velocity and pressure only.

    This is the 3d version of fsi2_simplified.py, with the benchmark set-up of threeDfsi_c0ip.py: the
    physical constants, by default those of the 3d configuration in Table 1 of Failer & Richter (with
    neither ``nu_s`` nor ``lambda_s`` given, ``lambda_s = 8.0e6``), the bi-parabolic inflow with mean
    ``U_bar``, ramped up over 0 <= t < 1, and the QoIs: the drag and the lift on the cylinder and the
    beam, and the displacement of the point B = (0.9, 0.2, 0.3) on the back face of the beam.

    The solid stress is written in terms of v through the kinematic relation, and the Jacobian is that
    of the momentum and continuity equations with respect to (v, p), solved as ``solver_options`` sets:
    with ``DEFAULT_DIRECT_SOLVER_OPTIONS``, by MUMPS LU, or by FGMRES on a nest Jacobian, preconditioned
    by the block upper triangular Schur complement factor, with one BoomerAMG V-cycle (strong threshold
    0.5, as hypre recommends in 3d) on the selfp approximation of the Schur complement and, for the
    velocity block,

    - by default, with ``DEFAULT_ITERATIVE_SOLVER_OPTIONS``, as in the K_c options of threeDfsi_c0ip, one
      GAMG V-cycle (rigid body modes as near-nullspace, three under-relaxed SOR steps per level);
    - with ``MUMPS_ON_SOLID`` and ``requires_v_sub_IS=True``, a multiplicative fieldsplit into the
      velocity dofs on solid cells (with the interface), solved by MUMPS LU, and the fluid dofs, by
      block Jacobi with ILU(0) on each rank. The solid rows are dominated by the solid stiffness, where
      GAMG's SOR smoothing, Jacobi between ranks, depends on the partition; the exact solve does not.
      The fluid rows are dominated by the mass, which needs no coarse grid.

    ``requires_v_sub_IS=True`` builds the index sets of the inner split, and leaves the rigid body modes
    off the velocity block; it is needed with, and only with, a fieldsplit of the velocity block.

    The displacements are not unknowns of the Newton system: after each linear solve, a post-solve hook
    of the KSP updates v to the next iterate and sets the displacements from the kinematic relation,
    node by node: the solid displacement u_s = u_s_old + dt * (theta * v + (1 - theta) * v_old), and the
    same at the interface dofs of the fluid displacement u_f, which the mesh motion
    ``mesh_motion_type``, one of the functions of ``xfsi_solver.mesh_motion``, by default
    ``biharmonic_c0ip_mesh_motion``, the mesh motion of threeDfsi_c0ip, extends into the fluid, before
    SNES evaluates the residual there. The interface values come from v on the fluid cells, so no values
    pass between the meshes. The mesh motion does not couple the x, y and z components, which have the
    same matrix, so only the x component is factorized, by MUMPS once for the whole run, and the three
    components are solved with that factorization at once, by one MatMatSolve: by Cholesky for the
    symmetric positive definite C0IP matrix, as in threeDfsi_c0ip, and by LU for the other mesh motions.
    ``mesh_motion_options``, the PETSc options of its KSP, replace that default. The quadrature degree
    of the mesh motion forms is capped at ``mesh_motion_max_quadrature_degree``, by default 8, as
    threeDfsi_c0ip caps its C0IP forms: UFL estimates degree 12 on the facets for P2 on the P2
    tetrahedral mesh, and the mesh motion residual is assembled in every Newton iteration. This cap is
    required too, and None raises ``ValueError``.

    As in fsi2_simplified, with ``predict_displacements=True``, the default, the displacements are also
    set before each Newton solve, by an initial-guess hook of the SNES, as after a linear solve but from
    the velocity of the previous step: u_s = u_s_old + dt * v_old, the same on the interface for u_f,
    and its mesh motion. Without it, u_f in the initial guess is still the fluid displacement of the
    previous step, so the mesh velocity (u_f - u_f_old) / dt in its residual vanishes, and the first
    Newton step, whose Jacobian has no derivative through the mesh motion, is followed by a mesh update
    by the displacement of a whole time step. In fsi2_simplified on mesh_sec, the prediction cut the
    first-iteration contraction from 0.16 to 3e-3, and the time per step by 11% with the default
    Jacobian reuse and by 27% without (32bd021:notes/newton-convergence/report.md).

    The Jacobian is reassembled when the residual norm decreased by less than a factor
    ``gamma_reassemble`` in the last Newton iteration, and otherwise reused, also across time steps;
    with ``gamma_reassemble=None``, it is reassembled in every Newton iteration.

    With ``newton_log_path``, every Newton iterate is recorded in that file by
    ``xfsi_solver.tools.newton_log.NewtonLog``, with the norms on the blocks v and p, as in
    fsi2_simplified.

    The output, the restart checkpoints and the snapshots are VTKHDF files of the same layout
    (``xfsi_solver.tools.vtkhdf``), each of which opens in ParaView: u, v and p at the nodes of the
    mesh, with the cell tags and the owning rank of each cell. p, on the fluid submesh, is 0 on the
    solid cells, so in ParaView its values there blend from the interface value to 0 across the
    solid cells at the interface; a Threshold on ``cell_tags`` shows the fluid alone. With
    ``output_every``, the state is written to ``output_path`` every ``output_every`` steps; a restart
    writes a new file, ``output_path`` renamed by ``xfsi_solver.tools.checkpoint.restart_output_path``,
    so the output before it is kept.

    Checkpoints store the displacement as one function u on the parent mesh, with v and p, as the 3d
    solvers do, so runs can restart from or start from each other's states: the checkpoints of the
    other 3d solvers, and those of this solver before it wrote VTKHDF, are io4dolfinx ``.bp`` files,
    which are read as well. They hold no field of the mesh motion, which each solve computes again, so
    a run can also start from the state of a run with another mesh motion.

    With ``snapshot_dir``, the state at the time steps closest to the times ``snapshot_times`` is
    also kept, each in a checkpoint directory ``snapshot_dir/t<t:.4f>`` of its own, from which a run
    starts with ``initial_state_dir``, by ``xfsi_solver.tools.checkpoint.SnapshotWriter``: a time
    after ``T`` raises ``ValueError``, and a new run removes the snapshots of an earlier run in
    ``snapshot_dir``, while a restart keeps those up to the step it resumes from.

    The quadrature degree of the residual, and so of its Jacobian, and of the drag and lift is capped
    at ``max_quadrature_degree``, by default 8, as in threeDfsi_c0ip: integrals for which UFL estimates
    a higher degree are integrated inexactly at that degree, and the others keep their estimate. The
    cap is required, and ``max_quadrature_degree=None`` raises ``ValueError``. The other arguments are
    those of ``threeDfsi_c0ip.solve``.

    The solid is the St. Venant-Kirchhoff material with the Lame parameters ``mu_s`` and ``lambda_s`` (or ``nu_s``)
    by default, ``solid_material="stvk"``. With ``solid_material="mooney_rivlin_3"`` it is the compressible
    three-term Mooney-Rivlin material of ``xfsi_solver.fsi.materials.Solid.MR3_piola``, with the coefficients
    ``mooney_rivlin_3``, a dict of ``"C10"``, ``"C01"`` and ``"C11"`` in Pa and optionally ``"D1"`` in 1/Pa. The Cij
    set the shear response, with initial shear modulus 2 (C10 + C01); without ``"D1"``, the bulk modulus K = 2 / D1
    is that of the SVK parameters, lambda_s + 2 mu_s / 3
    (``xfsi_solver.tools.parameters.mooney_rivlin_3_coefficients``). Its stress is condensed in v as that of the SVK
    solid, and its integrals are capped at ``max_quadrature_degree`` as the others are. The smallest and largest J =
    det(F) of the solid and of the fluid mesh are measured at every QoI row, at the points of a lattice of degree
    ``det_F_lattice_degree`` on each cell, and written as the four columns after the displacement of B
    (``DET_F_QOI_HEADER_3D``). A J <= 0 in the solid or on the fluid mesh, with either material, raises
    ``RuntimeError`` after its row is written. The drag and lift are those at t^{n+theta} = t - (1 - theta) dt, half
    a step before the time of their row for theta near 1/2: the pressure of a step is that of t^{n+theta}, and they
    are computed with the theta averages of v and u_f (see ``xfsi_solver.tools.qoi.drag_lift_forms``), which makes
    them second order in time; from v and u_f of t^{n+1} they would be first order. In the row of a start state they
    are those of the state.

    Returns the statistics of the time steps of this run: the wall time of the time loop,
    ``elapsed_time``, ``num_steps``, ``num_newton_iterations``, ``num_linear_iterations``,
    ``num_assemblies`` of the Jacobian, and per step, ``iterations_per_step``, ``assemblies_per_step``,
    ``linear_iterations_per_step`` and the wall time ``step_times``, and the (minimum, maximum) of J of each
    step in the solid, ``J_solid_per_step``, and on the fluid mesh, ``J_fluid_per_step``. The counts and J
    are the same on every rank; the wall times are those of each rank.

    With ``skip_dirichlet_lifting=True``, the residual is assembled without lifting the Dirichlet
    conditions of v, and every Newton iterate holds their values exactly, by
    ``xfsi_solver.tools.skip_dirichlet_lifting.skip_dirichlet_lifting``: the values are set in the
    initial guess, before the displacement prediction, and the Newton update is held at zero on the
    Dirichlet dofs, in the post-solve hook before v is updated with it. dolfinx's residual lifts every
    condition, J (g - x), which is zero where x = g but runs the Jacobian kernels on every cell with a
    Dirichlet dof, in every residual evaluation. The converged solution is the same; the iterates differ
    from those of the default, False, in the first Newton step of a time step, which starts here from the
    new inflow values, and agree to the Newton tolerance.

    With ``snes_converged_reason_on=False``, the SNES does not print why each Newton solve stopped.
    Printing it makes PETSc keep a stdout viewer for the communicator of the mesh, in a table of
    fixed size, until PetscFinalize, so a process that creates a few hundred meshes, such as the
    test suite, which turns it off in tests/conftest.py, overflows the table and crashes.
    """
    check_start_options(restart, checkpoint_dir, checkpoint_every, initial_state_dir, snapshot_dir, snapshot_times)
    if max_quadrature_degree is None:
        raise ValueError(
            "the 3d solver needs a quadrature cap, so max_quadrature_degree cannot be None: the degrees that "
            "UFL estimates (19 in the residual, 25 in the Jacobian) are far too expensive on tetrahedra"
        )
    if mesh_motion_max_quadrature_degree is None:
        raise ValueError(
            "the 3d solver needs a quadrature cap on the mesh motion, so mesh_motion_max_quadrature_degree cannot "
            "be None: its residual is assembled in every Newton iteration"
        )
    theta_val = theta_value(dt_val, absolute_theta_shift, relative_theta_shift, 1.0)
    if nu_s is None and lambda_s is None:
        lambda_s = 8.0e6
    lambda_s_val = lame_lambda(mu_s, nu_s, lambda_s)
    mooney_rivlin_3 = solid_material_coefficients(solid_material, mooney_rivlin_3, mu_s, lambda_s_val)

    if solver_options is None:
        solver_options = DEFAULT_ITERATIVE_SOLVER_OPTIONS
    if mesh_motion_type is None:
        mesh_motion_type = biharmonic_c0ip_mesh_motion

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

    # create problem parameters (Table 1 of Failer & Richter, 3d configuration)

    gdim = mesh.geometry.dim

    rho_f = dolfinx.fem.Constant(mesh, rho_f)
    nu_f = dolfinx.fem.Constant(mesh, nu_f)

    rho_s = dolfinx.fem.Constant(mesh, rho_s)
    mu_s = dolfinx.fem.Constant(mesh, mu_s)
    lambda_s = dolfinx.fem.Constant(mesh, lambda_s_val)
    if mooney_rivlin_3 is not None:
        mooney_rivlin_3_constants = {name: dolfinx.fem.Constant(mesh, value) for name, value in mooney_rivlin_3.items()}
    else:
        mooney_rivlin_3_constants = None

    # U_bar is the mean inflow velocity, H the channel height (y) and width (z)

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
    U_s = dolfinx.fem.functionspace(solid_mesh, elements["U"])
    U_f = dolfinx.fem.functionspace(fluid_mesh, elements["U"])
    W = ufl.MixedFunctionSpace(V, P)

    # create functions

    v, p, u_s, u_f = (
        dolfinx.fem.Function(V, name="v"),
        dolfinx.fem.Function(P, name="p"),
        dolfinx.fem.Function(U_s, name="u_s"),
        dolfinx.fem.Function(U_f, name="u_f"),
    )
    v_old, u_s_old, u_f_old = dolfinx.fem.Function(V), dolfinx.fem.Function(U_s), dolfinx.fem.Function(U_f)

    dv, dp = ufl.TestFunctions(W)

    # The displacement on the parent mesh, u_s on the solid cells and u_f on the fluid cells, which
    # the checkpoints and the output hold in place of u_s and u_f. The checkpoints hold no other
    # field of the mesh motion: every mesh motion solve computes those again from u_s, so the state
    # is (u, v, p) whichever mesh motion is used, as in the other 3d solvers.
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, elements["U"]), name="u")
    checkpoint_fields = (u, v, p)

    displacement = SubmeshComposite(u, [(u_s, solid_cell_map), (u_f, fluid_cell_map)])

    # Set up output, checkpointing, and qoi tracking.

    # (t, step) are the time and number of completed steps of the current state
    submeshes = [(fluid_mesh, fluid_cell_map)]
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

    # Quantities of interest: the displacement at the point B on the back face of the beam, and
    # the drag and lift on the obstacle and on the fluid side of the solid-fluid interface.
    spot_dof = find_point_dof(U_s, np.array([0.9, 0.2, 0.3], dtype=np.float64))
    # The drag and lift are those at t^{n+theta}: the pressure of a step approximates that at t^{n+theta}, and
    # with v and u_f of t^{n+1} the force would be only first order in time, so they are taken from the theta
    # averages of v and u_f. v_old and u_f_old hold the previous step when the row of a step is written.
    drag_forms, lift_forms = drag_lift_forms(
        mesh,
        theta * u_f + (1.0 - theta) * u_f_old,
        theta * v + (1.0 - theta) * v_old,
        p,
        nu_f,
        rho_f,
        [ds(PHYSICAL_MARKERS["obstacle"]), ds_interface_fluid],
        entity_maps,
        max_quadrature_degree=max_quadrature_degree,
    )

    # The range of det(F) on the solid and on the fluid mesh, sampled at the points of a lattice on each cell
    solid_J_range = deformation_jacobian_range(u_s, all_cells(solid_mesh), det_F_lattice_degree)
    fluid_J_range = deformation_jacobian_range(u_f, all_cells(fluid_mesh), det_F_lattice_degree)

    def write_output(t, step):
        """Write the QoI row, and every output_every steps the output, of the state at time t. Return the
        (minimum, maximum) of det(F) on the solid and on the fluid mesh, which the row holds."""
        if output_every is not None and step % output_every == 0:
            displacement.gather()  # u is written with v
            for writer in writers:
                writer.write(t)

        u_spot = point_value(u_s, spot_dof)
        drag = assemble_force(drag_forms, comm)
        lift = assemble_force(lift_forms, comm)
        J_solid, J_fluid = solid_J_range(), fluid_J_range()
        append_qoi_row(qoi_path, comm, t, drag, lift, u_spot, *J_solid, *J_fluid)
        return J_solid, J_fluid

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
    # FLUID: Bi-parabolic inflow at x = 0, no-slip on the channel walls and the cylinder, do-nothing at x = 2.8
    # SOLID: Homogeneous Dirichlet where the beam is attached to the cylinder

    # create residual form with u condensed out.
    forms = FSIForms(
        dv=dv,
        dp=dp,
        dx_fluid=dx_fluid,
        dx_solid=dx_solid,
        rho_f=rho_f,
        nu_f=nu_f,
        rho_s=rho_s,
        mu_s=mu_s,
        lambda_s=lambda_s,
        dt=dt,
        theta=theta,
        solid_material=solid_material,
        mooney_rivlin_3=mooney_rivlin_3_constants,
    )
    residual = forms.build_residual(v, v_old, p, u_f, u_f_old, u_s_old)

    residual_blocks = ufl.extract_blocks(residual)
    # Cap the quadrature degree before taking derivatives, so that the Jacobian blocks keep it.
    residual_blocks = [cap_quadrature_degree(form, max_quadrature_degree) for form in residual_blocks]

    jacobian = dolfinx.fem.forms.derivative_block(residual_blocks, (v, p))

    # --------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])

    # The fieldsplit preconditioner takes the (v, v) and (p, p) blocks of the Jacobian, which a nest
    # matrix holds as separate matrices. MUMPS also works on nest matrices.
    use_fieldsplit = solver_options.get("pc_type") == "fieldsplit"
    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocks,
        (v, p),
        J=jacobian,
        bcs=bcs,
        petsc_options_prefix="solver_",
        entity_maps=entity_maps,
        kind="nest" if use_fieldsplit else None,
        petsc_options=solver_options
        if snes_converged_reason_on
        else {key: value for key, value in solver_options.items() if key != "snes_converged_reason"},
    )

    opts = PETSc.Options()
    opts["options_left"] = None

    if use_fieldsplit:
        is_v, is_p = problem.A.getNestISs()[0]

        if not requires_v_sub_IS:
            is_v.setBlockSize(V.dofmap.index_map_bs)
            is_v.compose("nearnullspace", rigid_body_modes(V))

        problem.solver.getKSP().getPC().setFieldSplitIS(("v_0", is_v), ("p_1", is_p))

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
        # outer fieldsplit takes from the nest as is: V's own global numbering, not the nest's.
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
        # With GAMG on v_f (DEFAULT_V_OPTIONS), GAMG aggregates per node, with the translations as
        # near-nullspace.
        is_v_f.setBlockSize(bs)

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
            if options[f_prefix + "pc_type"] == "gamg":
                f_description = (
                    f"one V-cycle of GAMG\n      (translations as near-nullspace), {smoothing_description(f_prefix)}"
                )
            else:
                f_pc_name = {"bjacobi": "block Jacobi"}.get(
                    options[f_prefix + "pc_type"], options[f_prefix + "pc_type"]
                )
                f_description = f"{f_pc_name} with {options[f_prefix + 'sub_pc_type'].upper()} on each rank"
            v_pc_description = (
                f"a {options['fieldsplit_v_0_pc_fieldsplit_type']} fieldsplit of A into the velocity dofs on\n"
                f"      solid cells (with the interface), by {options[s_prefix + 'pc_factor_mat_solver_type'].upper()} "
                f"{options[s_prefix + 'pc_type'].upper()}, and the fluid dofs, by {f_description}"
            )
        p_pc_description = options["fieldsplit_p_1_pc_type"]
        if p_pc_description == "hypre":
            p_pc_description += f" {options.get('fieldsplit_p_1_pc_hypre_type', 'boomeramg')}"  # the default type
            if "fieldsplit_p_1_pc_hypre_boomeramg_strong_threshold" in options:
                p_pc_description += (
                    f" (strong threshold {options['fieldsplit_p_1_pc_hypre_boomeramg_strong_threshold']})"
                )
        linear_solve_description = (
            f"{options['ksp_type'].upper()} (rtol {options['ksp_rtol']:g}, atol {options['ksp_atol']:g}) on the nest "
            f"Jacobian, preconditioned by the\n    {options['pc_fieldsplit_schur_fact_type']} Schur factor, "
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
            # PCSetUp, not KSPSetUp, which here leaves FGMRES with nest work vectors that are not set up.
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
    # once, and by default factorized once, in the first solve. The C0IP matrix is symmetric positive
    # definite, so it is factorized by Cholesky, as in threeDfsi_c0ip.
    mesh_motion_name = mesh_motion_type.__name__.removesuffix("_mesh_motion")
    if mesh_motion_options is None:
        factorization = "LU"
        if mesh_motion_type is biharmonic_c0ip_mesh_motion:
            mesh_motion_options = default_mesh_motion_solver_options(U_f.dofmap.index_map_bs)
            mesh_motion_options["block_pc_type"] = "cholesky"
            factorization = "Cholesky"
        mesh_motion_description = (
            f"{mesh_motion_name}, MUMPS {factorization} on the x component, once for the whole run, for all\n"
            "    three components at once by MatMatSolve"
        )
    else:
        mesh_motion_description = f"{mesh_motion_name}, with the KSP options {mesh_motion_options}"
    mesh_motion = mesh_motion_type(
        u_f, [u_f_bc], solver_options=mesh_motion_options, max_quadrature_degree=mesh_motion_max_quadrature_degree
    )

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
        where x is the Newton update Y of (v, p) and v still holds the current iterate, before SNES
        sets the next iterate to v - Y and evaluates the residual there."""
        if lifting_hooks is not None:
            # Zero the update on the Dirichlet dofs before v is updated with it: the line-search pre-check
            # that does so comes after this hook.
            lifting_hooks["set_dirichlet_update"](solver.getSolution(), x)
        # The owned v rows come first in x, or are its first block if x is a nest vector.
        x_v = x.getNestSubVecs()[0] if x.getType() == PETSc.Vec.Type.NEST else x
        v.x.array[:num_v] -= x_v.getArray(readonly=True)[:num_v]
        v.x.scatter_forward()
        update_displacements()

    # A KSP has one post-solve hook, so KSPConvCheck, which sets its own, cannot be attached to this KSP.
    # A failed linear solve of (v, p) makes the SNES stop with DIVERGED_LINEAR_SOLVE, which
    # check_converged reports.
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
    init_qoi_file(qoi_path, comm, restart, t, dt_val, header=DET_F_QOI_HEADER_3D)
    if newton_log_path is not None:
        newton_log = NewtonLog(newton_log_path, field_blocks(problem, ["v", "p"]), comm, restart, t, dt_val)
        solver.setMonitor(newton_log)

    if not restart:
        # The start state has no previous step: its drag and lift are those of the start state
        v_old.x.array[:] = v.x.array
        u_f_old.x.array[:] = u_f.x.array
        write_output(t, step)
        if snapshots.due(step):
            snapshots.write(checkpoint_fields, t, step, dt_val, t0)  # u holds the start state

    PETSc.Sys.Print()
    mesh.comm.barrier()

    first_step = step
    num_newton_iterations = 0
    num_linear_iterations = 0
    # Newton iterations, Jacobian assemblies, linear iterations and wall time of each time step of this run
    iterations_per_step, assemblies_per_step, linear_iterations_per_step, step_times = [], [], [], []
    J_solid_per_step, J_fluid_per_step = [], []
    start = timer()
    while step < num_steps:
        step_start = timer()
        t = t0 + (step + 1) * dt_val
        inflow_bc_func.interpolate(InflowFunc(t))
        inflow_bc_func.x.scatter_forward()

        v_old.x.array[:] = v.x.array[:]
        u_s_old.x.array[:] = u_s.x.array[:]
        u_f_old.x.array[:] = u_f.x.array[:]

        PETSc.Sys.Print(f"\n{t = :.3f}")

        if newton_log_path is not None:
            newton_log.start_step(t)
        num_assemblies_before = jacobian_state["num_assemblies"]
        problem.solve()
        step_newton_iterations = solver.getIterationNumber()
        step_linear_iterations = solver.getLinearSolveIterations()
        num_newton_iterations += step_newton_iterations
        num_linear_iterations += step_linear_iterations
        iterations_per_step.append(step_newton_iterations)
        assemblies_per_step.append(jacobian_state["num_assemblies"] - num_assemblies_before)
        linear_iterations_per_step.append(step_linear_iterations)
        check_converged(problem, f"t = {t:.4f}", writers=writers, ksp_check=None)

        step_times.append(timer() - step_start)
        PETSc.Sys.Print(
            f"  step time {step_times[-1]:.2f} s, linear iterations {step_linear_iterations} in "
            f"{step_newton_iterations} solves ({step_linear_iterations / max(step_newton_iterations, 1):.1f} "
            "per solve)"
        )

        if comm.rank == 0:
            sys.stdout.flush()

        step += 1
        J_solid, J_fluid = write_output(t, step)
        J_solid_per_step.append(J_solid)
        J_fluid_per_step.append(J_fluid)
        if J_solid[0] <= 0.0 or J_fluid[0] <= 0.0:
            # A solid with det(F) <= 0 has passed through itself, and a fluid mesh with det(F) <= 0 has inverted
            # cells, on which the ALE terms change sign without failing. Close the output so it stays readable.
            for writer in writers:
                writer.close()
            raise RuntimeError(
                f"At t = {t:.4f}, the smallest det(F) is not positive: {J_solid[0]:g} in the solid, "
                f"{J_fluid[0]:g} on the fluid mesh"
            )

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
    if solid_material == "mooney_rivlin_3":
        solid_description = "compressible three-term Mooney-Rivlin, P = dW/dF by UFL, " + ", ".join(
            f"{name} = {value:g}" for name, value in mooney_rivlin_3.items()
        )
    else:
        solid_description = f"St. Venant-Kirchhoff, mu_s = {float(mu_s.value):g}, lambda_s = {lambda_s_val:g}"
    if J_solid_per_step:
        J_description = (
            f"solid [{min(r[0] for r in J_solid_per_step):.6f}, {max(r[1] for r in J_solid_per_step):.6f}], "
            f"fluid mesh [{min(r[0] for r in J_fluid_per_step):.6f}, {max(r[1] for r in J_fluid_per_step):.6f}]"
        )
    else:
        J_description = "no time steps"
    if skip_dirichlet_lifting:
        initial_guess_description += ", with the Dirichlet values of the step set in it"
        dirichlet_description = "skipped, since every Newton iterate holds the values exactly"
    else:
        dirichlet_description = "in every residual evaluation (dolfinx's assemble_residual)"
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / max(step - first_step, 1):.3f} s")
        if len(step_times) > 1:
            # The first step also sets up the mesh motion factorization and the AMG hierarchies.
            print(f"Time per step after the first: {sum(step_times[1:]) / (len(step_times) - 1):.3f} s")
        if use_fieldsplit and num_newton_iterations > 0:
            print(f"Linear iterations per solve: {num_linear_iterations / num_newton_iterations:.1f}")
        print(
            "\nSolver setup:\n"
            f"  Quadrature degree: {quadrature_description(max_quadrature_degree)}\n"
            f"  Solid material: {solid_description}\n"
            f"  Range of J = det(F) over the time steps: {J_description}\n"
            "  Solid stress: in terms of v, via u_s = u_s_old + dt * (theta * v + (1 - theta) * v_old)\n"
            "  Newton system: (v, p) only; u_s and the interface values of u_f set from v node by node,\n"
            "    and u_f by the mesh motion, after each linear solve\n"
            f"  Linear solve: {linear_solve_description}"
            f"  Mesh motion: {mesh_motion_description}\n"
            f"  Mesh motion quadrature degree: {quadrature_description(mesh_motion_max_quadrature_degree)}\n"
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
        "num_linear_iterations": num_linear_iterations,
        "num_assemblies": jacobian_state["num_assemblies"],
        "iterations_per_step": iterations_per_step,
        "assemblies_per_step": assemblies_per_step,
        "linear_iterations_per_step": linear_iterations_per_step,
        "step_times": step_times,
        "J_solid_per_step": J_solid_per_step,
        "J_fluid_per_step": J_fluid_per_step,
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

    mesh_path = "data/meshes/fsi3d/mesh_sec_medium.xdmf"
    run_name = "threeDfsi_simplified_medium"
    dt_val = 0.004
    T = 10.0 + 8 * dt_val
    output_path = f"output/pv/{run_name}.vtkhdf"
    qoi_path = f"output/qoi/{run_name}.txt"
    newton_log_path = f"output/qoi/{run_name}_newton.txt"
    checkpoint_dir = f"output/checkpoints/{run_name}"
    # Only with --snapshot-times, since a new run clears the snapshots of an earlier one; a restart
    # needs the same times again to write the snapshots after its checkpoint
    snapshot_times = args.snapshot_times
    snapshot_dir = f"output/snapshots/{run_name}" if snapshot_times else None
    checkpoint_every = 250
    output_every = None
    gamma_reassemble = 0.1  # Default 0.1.
    max_quadrature_degree = 6  # Default 8.
    mesh_motion_max_quadrature_degree = 6  # Default 8.
    # solver_options = DEFAULT_ITERATIVE_SOLVER_OPTIONS
    # requires_v_sub_IS = False
    solver_options = MUMPS_ON_SOLID
    requires_v_sub_IS = True
    restart = args.restart
    initial_state_dir = args.initial_state_dir
    if not initial_state_dir:
        initial_state_dir = "output/checkpoints/threeDfsi_mixed_medium"
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
        gamma_reassemble=gamma_reassemble,
        max_quadrature_degree=max_quadrature_degree,
        mesh_motion_max_quadrature_degree=mesh_motion_max_quadrature_degree,
        solver_options=solver_options,
        requires_v_sub_IS=requires_v_sub_IS,
        restart=restart,
        initial_state_dir=initial_state_dir,
    )

    if MPI.COMM_WORLD.rank == 0:
        plot_run_qois(qoi_path, T, qoi_labels=QOI_LABELS_3D, reference_file=None, closeup_interval=(9.0, 10.0))


if __name__ == "__main__":
    main()
