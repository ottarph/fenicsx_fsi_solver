# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import sys
import warnings
from pathlib import Path
from timeit import default_timer as timer

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from petsc4py import PETSc

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

    max_iter = 20
    atol = 1.0e-7
    rtol = 1.0e-12

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual_blocked,
        [u, v, p, z],
        bcs=bcs,
        petsc_options_prefix="fsi2_biharmonic_diffmesh_restr_split_",
        entity_maps=entity_maps,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 80,
            "mat_mumps_cntl_1": 1e-4,
            "snes_linesearch_type": "none",
            "snes_max_it": max_iter,
            "snes_atol": atol,
            "snes_rtol": rtol,
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
            # "snes_monitor": "ascii:output/logs/fsi2_biharm_restr_split_snes_log.txt",
            "snes_monitor": None,
        },
    )

    from xfsi_solver.tools.checkpoint import Checkpointer, restart_output_path, truncate_qoi_file

    # (t, step) are the time and number of completed steps of the current state
    t = t0
    step = 0
    checkpointer = None
    if checkpoint_dir is not None:
        checkpointer = Checkpointer(checkpoint_dir, mesh, submeshes=[(fluid_mesh, fluid_cell_map)])
    if restart:
        t, step = checkpointer.read([u, v, p, z], dt_val)
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

    dm_loc_size = U.dofmap.index_map.size_local
    spot = np.array([0.6, 0.2, 0.0], dtype=np.float64)
    spot_dof_cand = np.flatnonzero(
        np.all(np.isclose(U.tabulate_dof_coordinates()[:dm_loc_size, :], spot, atol=1e-6), axis=1)
    )
    spot_dof = spot_dof_cand[0] if len(spot_dof_cand) > 0 else None
    assert comm.allreduce(len(spot_dof_cand), op=MPI.SUM) == 1, "None or multiple dofs found for measurement point"

    loc_u_spot = np.zeros(2, dtype=np.float64)

    normal = ufl.FacetNormal(mesh)
    e_x = dolfinx.fem.Constant(mesh, (-1.0, 0.0))
    e_y = dolfinx.fem.Constant(mesh, (0.0, 1.0))
    F = ufl.Identity(2) + ufl.grad(u)
    transformed_normal = ufl.dot(ufl.inv(F.T), normal)

    drag_form_obstacle = (
        ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_x)
        * ufl.det(F)
        * ds(PHYSICAL_MARKERS["obstacle"])
    )
    lift_form_obstacle = (
        ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_y)
        * ufl.det(F)
        * ds(PHYSICAL_MARKERS["obstacle"])
    )

    drag_form_interface = (
        ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_x) * ufl.det(F) * ds_interface_fluid
    )
    lift_form_interface = (
        ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_y) * ufl.det(F) * ds_interface_fluid
    )

    drag_form_obstacle = dolfinx.fem.form(drag_form_obstacle, entity_maps=entity_maps)
    drag_form_interface = dolfinx.fem.form(drag_form_interface, entity_maps=entity_maps)
    lift_form_obstacle = dolfinx.fem.form(lift_form_obstacle, entity_maps=entity_maps)
    lift_form_interface = dolfinx.fem.form(lift_form_interface, entity_maps=entity_maps)

    if comm.rank == 0:
        Path(qoi_path).parent.mkdir(parents=True, exist_ok=True)
        if restart and Path(qoi_path).exists():
            truncate_qoi_file(qoi_path, t, dt_val)
        else:
            with open(qoi_path, "wb") as f:
                np.savetxt(f, [], fmt="%.6e", delimiter="\t", header="t\tdrag\tlift\tA_x\tA_y")

    # Add extra callbacks to change how residual and jacobian is assembled,
    # to account for test function restriction on the interface.

    problem: dolfinx.fem.petsc.NonlinearProblem

    solver = problem.solver
    b_vec, (fem_residual, res_args, res_kargs) = solver.getFunction()
    J_mat, P_mat, (fem_jacobian, jac_args, jac_kargs) = solver.getJacobian()

    # Prevent possibly overwriting the sparsity pattern on zeroRows.
    problem.A.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)

    res_miss_ufl = rho_s * ufl.inner((u - u_old) / dt - theta * v - (1 - theta) * v_old, du) * dx_solid
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

    def write_output(t, step):
        """Write the QoI row, and every save_every steps the VTX snapshot, of the state at time t."""
        if step % save_every == 0:
            writer.write(t)
            writer_p.write(t)

        loc_u_spot[:] = u.x.array[2 * spot_dof : 2 * (spot_dof + 1)] if spot_dof is not None else 0.0
        u_spot = comm.reduce(loc_u_spot, op=MPI.SUM, root=0)
        drag = comm.reduce(
            dolfinx.fem.assemble_scalar(drag_form_obstacle) + dolfinx.fem.assemble_scalar(drag_form_interface)
        )
        lift = comm.reduce(
            dolfinx.fem.assemble_scalar(lift_form_obstacle) + dolfinx.fem.assemble_scalar(lift_form_interface)
        )
        if comm.rank == 0:
            with open(qoi_path, "ab") as f:
                np.savetxt(f, [[t, drag, lift, *u_spot]], fmt="%.6e", delimiter="\t")

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

        problem.solve()
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

    writer.close()
    writer_p.close()

    import matplotlib.pyplot as plt

    # Spy plots and block norms of the Jacobian at the current state, with the
    # dofs reordered block by block (the global numbering is [u, v, p, z] per rank).
    dolfinx.fem.petsc.assign([u, v, p, z], problem.x)
    wrapped_jacobian(solver, problem.x, J_mat, P_mat)

    row_start = J_mat.getOwnershipRange()[0]
    indptr, cols, vals = J_mat.getValuesCSR()
    rows = row_start + np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    nonzero = vals != 0.0
    rows, cols, vals = rows[nonzero], cols[nonzero], vals[nonzero]
    rows = comm.gather(rows, root=0)
    cols = comm.gather(cols, root=0)
    vals = comm.gather(vals, root=0)

    def solid_supported(space, num_dofs):
        """Mask of the owned unrolled dofs of space that are supported on solid cells."""
        bs = space.dofmap.index_map_bs
        dofs = dolfinx.fem.locate_dofs_topological(space, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["solid"]))
        return np.isin(np.arange(num_dofs), (bs * dofs[:, None] + np.arange(bs)).ravel())

    def gather_dofs(dofs):
        """Global dofs of all ranks on rank 0, in rank order."""
        dofs = comm.gather(dofs, root=0)
        return np.concatenate(dofs) if comm.rank == 0 else None

    # Owned global dofs of the storage blocks [u, v, p, z]. As in eq. (1) of
    # docs/restricted-iterative-solver.md, u is split into the solid-supported dofs
    # u_S (including the interface) and the remaining fluid-interior dofs u_I.
    u_dofs, v_dofs, p_dofs, z_dofs = (
        row_start + np.arange(off_own[k], off_own[k + 1]) for k in range(len(off_own) - 1)
    )
    u_solid = solid_supported(U, len(u_dofs))
    v_solid = solid_supported(V, len(v_dofs))
    dofs_u_S = gather_dofs(u_dofs[u_solid])
    dofs_u_I = gather_dofs(u_dofs[~u_solid])
    dofs_u = gather_dofs(u_dofs)
    dofs_v = gather_dofs(v_dofs)
    dofs_v_fluid = gather_dofs(v_dofs[~v_solid])
    dofs_p = gather_dofs(p_dofs)
    dofs_z = gather_dofs(z_dofs)

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

        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        n = J_mat.getSize()[0]
        J_full = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))

        names_eq1 = ["$u_S$", "$v$", "$p$", "$z$", "$u_I$"]
        blocks_eq1 = [dofs_u_S, dofs_v, dofs_p, dofs_z, dofs_u_I]
        plot_blocks(J_full, [dofs_u, dofs_v, dofs_p, dofs_z], ["u", "v", "p", "z"], "Jacobian", "storage")
        plot_blocks(J_full, blocks_eq1, names_eq1, "Jacobian", "eq1")

        def indicator(dofs):
            mask = np.zeros(n, dtype=bool)
            mask[dofs] = True
            return mask

        in_u_S, in_u_I, in_v, in_v_fluid, in_p = map(indicator, (dofs_u_S, dofs_u_I, dofs_v, dofs_v_fluid, dofs_p))
        drop = (
            (in_v[rows] & in_u_I[cols])
            | (in_p[rows] & (in_u_S[cols] | in_u_I[cols]))
            | (in_v_fluid[rows] & in_u_S[cols])
        )

        # J_0 of eq. (3) in docs/restricted-iterative-solver.md (following Failer & Richter):
        # drop the fluid ALE derivatives F_I (v, u_I), D_S (p, u_S) and D_I (p, u_I), and
        # F_S (v, u_S) in the velocity rows not supported on the solid. In the interface
        # velocity rows, F_S is summed with E_S in the same entries and is kept, since
        # separating them needs E_S assembled on its own. J_0 is still block triangular
        # as in (8) and condenses as in (6), but with H_c = H + θΔt (E_S + F_S^Γ) R_S,
        # where F_S^Γ is the interface part of F_S.
        J_0 = scipy.sparse.coo_matrix((vals[~drop], (rows[~drop], cols[~drop])), shape=(n, n))
        plot_blocks(J_0, blocks_eq1, names_eq1, "$J_0$", "eq3")

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
        T=15.5 + 2 * 0.0025,
        dt_val=0.0025,
        output_path="output/pv/fsi2_biharm_dm_restr_split.bp",
        output_path_p="output/pv/fsi2_biharm_p_dm_restr_split.bp",
        qoi_path="output/qoi/fsi2_biharm_qoi_restr_split.txt",
        checkpoint_dir="output/checkpoints/fsi2_biharm_dm_restr",
        checkpoint_every=None,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
