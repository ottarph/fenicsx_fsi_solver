# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import numpy as np
import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import ufl
from petsc4py import PETSc
import sys
import warnings
from pathlib import Path
from timeit import default_timer as timer

from mpi4py.MPI import COMM_WORLD as comm
from mpi4py import MPI

PHYSICAL_MARKERS = {
    "solid": 1,
    "ALE_fluid": 2,

    "solid_fluid_interface": 11,

    "obstacle": 21,                 # no-slip for fluid
    "inflow": 22,                   # parabolic inflow for fluid
    "outflow": 23,                  # do-nothing for fluid
    "channel_side": 24,             # no-slip for fluid
    "solid_obstacle_interface": 25, # homogeneous Dirichlet BC for solid
}

def solve(mesh_path, T, dt_val, output_path, output_path_p, qoi_path):


    # load mesh and meshtags

    with dfx.io.XDMFFile(comm, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name= "Cell tags")
        mesh.topology.create_connectivity(1, 2)
        facet_tags = infile.read_meshtags(mesh, name= "Facet tags")

    assert len(np.setdiff1d(np.union1d(cell_tags.values, facet_tags.values), [PHYSICAL_MARKERS[i] for i in PHYSICAL_MARKERS])) == 0, "Physical markers and cell tags do not match"
    

    # create submesh for fluid

    fluid_mesh, fluid_cell_map, fluid_vertex_map, _ = dfx.mesh.create_submesh(mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"]))


    # create entity maps for mixed mesh integration

    entity_maps = [fluid_cell_map]

    # Create measure with  meshtags

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    dx_fluid = dx(PHYSICAL_MARKERS["ALE_fluid"])
    dx_solid = dx(PHYSICAL_MARKERS["solid"])

    fluid_volume = comm.reduce(dfx.fem.assemble_scalar(dfx.fem.form(ufl.as_ufl(1.0) * dx_fluid)))
    solid_volume = comm.reduce(dfx.fem.assemble_scalar(dfx.fem.form(ufl.as_ufl(1.0) * dx_solid)))
    if comm.rank == 0:
        print(f"{fluid_volume = }")
        print(f"{solid_volume = }")


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

    rho_f = dfx.fem.Constant(mesh, 1.0e3)
    nu_f = dfx.fem.Constant(mesh, 1.0e-3)

    rho_s = dfx.fem.Constant(mesh, 1.0e4)
    mu_s = dfx.fem.Constant(mesh, 5.0e5)
    nu_s = dfx.fem.Constant(mesh, 0.4)
    lambda_s = dfx.fem.Constant(mesh, -mu_s.value / (1 - 0.5 / nu_s.value))
    assert np.isclose(lambda_s.value, 2e6), "Lambda value is not as expected"
    # lambda_s = dfx.fem.Constant(mesh, 2e6)

    U_bar = 1.0
    H = 0.41

    t0 = 0.0
    dt = dfx.fem.Constant(mesh, dt_val)

    theta = dfx.fem.Constant(mesh, 0.5 + dt.value)

    save_every = 4

    total_steps = int(np.ceil((T - t0) / dt_val))
    if total_steps <= save_every:
        warnings.warn(
            f"save_every ({save_every}) is larger than the total number of time "
            f"steps ({total_steps}); at most one VTX snapshot will be written to "
            f"{output_path!r} or {output_path_p!r}, which is not a usable time "
            f"series in ParaView."
        )

    
    # create function spaces

    U = dfx.fem.functionspace(mesh, ("CG", 2, (2, )))
    V = dfx.fem.functionspace(mesh, ("CG", 2, (2, )))
    P = dfx.fem.functionspace(fluid_mesh, ("CG", 1))
    Z = dfx.fem.functionspace(fluid_mesh, ("CG", 2, (2, )))
    W = ufl.MixedFunctionSpace(U, V, P, Z)


    # create functions

    u, v, p, z = dfx.fem.Function(U, name="u"), dfx.fem.Function(V, name="v"), dfx.fem.Function(P, name="p"), dfx.fem.Function(Z, name="z")
    u_old, v_old = dfx.fem.Function(U), dfx.fem.Function(V)

    
    du, dv, dp, dz = ufl.TestFunctions(W)


    # create Dirichlet boundary condition

    from functools import reduce

    inflow_bc_func = dfx.fem.Function(V)
    inflow_bc_func.x.array[:] = 0.0
    inflow_bc_facets = reduce(np.union1d, [
        facet_tags.find(PHYSICAL_MARKERS["inflow"]),
        ])
    inflow_bc_dofs = dfx.fem.locate_dofs_topological(V, mesh.geometry.dim - 1, inflow_bc_facets)

    inflow_bc = dfx.fem.dirichletbc(inflow_bc_func, inflow_bc_dofs)

    class InflowFunc:
        def __init__(self, t: float = 0.0):
            self.t = t
        def __call__(self, x: np.ndarray) -> np.ndarray:
            values = np.zeros((2, x.shape[1]), dtype=x.dtype)
            values[0] = 1.5 * U_bar * 4 * x[1] * (H - x[1]) / H**2
            if self.t < 2.0:
                values[0] *= 0.5 * (1.0 - np.cos(0.5*np.pi * self.t))
            return values

    noslip_bc_func = dfx.fem.Function(V)
    noslip_bc_func.x.array[:] = 0.0
    noslip_bc_facets = reduce(np.union1d, [
        facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
        facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
        facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
        ])
    noslip_bc_dofs = dfx.fem.locate_dofs_topological(V, mesh.geometry.dim - 1, noslip_bc_facets)
    
    noslip_bc = dfx.fem.dirichletbc(noslip_bc_func, noslip_bc_dofs)

    
    # Create fluid ALE Dirichlet boundary condition
    
    u_f_bc_func = dfx.fem.Function(U)
    u_f_bc_func.x.array[:] = 0.0
    u_f_bc_facets = reduce(np.union1d, [
        facet_tags.find(PHYSICAL_MARKERS["inflow"]),
        facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
        facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
        facet_tags.find(PHYSICAL_MARKERS["outflow"]),
    ])
    u_f_bc_dofs = dfx.fem.locate_dofs_topological(U, mesh.geometry.dim - 1, u_f_bc_facets)
    u_f_bc = dfx.fem.dirichletbc(u_f_bc_func, u_f_bc_dofs)


    # Create solid Dirichlet boundary condition

    u_s_bc_func = dfx.fem.Function(U)
    u_s_bc_func.x.array[:] = 0.0
    u_s_bc_facets = reduce(np.union1d, [
        facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
    ])
    u_s_bc_dofs = dfx.fem.locate_dofs_topological(U, mesh.geometry.dim - 1, u_s_bc_facets)
    u_s_bc = dfx.fem.dirichletbc(u_s_bc_func, u_s_bc_dofs)    


    # Create Dirichlet boundary condition for restricting test functions.

    total_interface_facets_found = mesh.comm.allreduce(interface_facets.size, op=MPI.SUM)
    assert total_interface_facets_found > 0, "Interface not found."
    dofs_interface = dfx.fem.locate_dofs_topological(
        U, mesh.topology.dim - 1, interface_facets
    )
    bc_deactivate = dfx.fem.dirichletbc(
        dfx.fem.Constant(mesh, (0.0, 0.0)), dofs_interface, U
    )


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

        residual  = rho_f * J_mid * ufl.inner((v - v_old) / dt, dv) * dx_fluid

        residual -= rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * ((u - u_old) / dt), dv) * dx_fluid

        residual += rho_s * ufl.inner((v - v_old) / dt, dv) * dx_solid

        return residual
    
    def A_I(u, v, z):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual  = ufl.inner(z, dz) * dx_fluid
        residual -= ufl.inner(ufl.grad(u), ufl.grad(dz)) * dx_fluid

        residual += ufl.inner(ufl.grad(z), ufl.grad(du)) * dx_fluid
        residual += dfx.fem.Constant(mesh, 0.0) * ufl.inner(u, du) * dx_fluid

        residual += ufl.div(J * ufl.inv(F) * v) * dp * dx_fluid

        return residual

    def A_E(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual  = rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx_fluid

        residual += ufl.inner(J * Fluid.NS_velocity(u, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        # This one stays because the test function is dv, not du.
        residual += ufl.inner(J * Solid.STVK(u, lambda_s, mu_s) * ufl.inv(F).T, ufl.grad(dv)) * dx_solid

        return residual
    
    def A_P(u, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual  = J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    residual  = A_T(u, u_old, v, v_old)
    residual += A_I(u, v, z)
    residual += A_P(u, p)
    residual += theta * A_E(u, v)
    residual += (1.0 - theta) * A_E(u_old, v_old)

    # Add in u_s with zero constants for correct sparsity patterns
    residual += dfx.fem.Constant(mesh, 0.0) * ufl.inner(u, du) * dx_solid
    residual += dfx.fem.Constant(mesh, 0.0) * ufl.inner(v, du) * dx_solid


    #--------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])


    residual_blocked = ufl.extract_blocks(residual)

    max_iter = 20
    atol = 1.0e-7
    rtol = 1.0e-12

    problem = dfx.fem.petsc.NonlinearProblem(
        residual_blocked, [u, v, p, z], bcs=bcs,
        petsc_options_prefix="fsi2_biharmonic_diffmesh_restr_scaled_",
        entity_maps=entity_maps,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "none", # Replaced by the ScaledLU shell preconditioner below.
            "snes_linesearch_type": "none",
            "snes_max_it": max_iter,
            "snes_atol": atol,
            "snes_rtol": rtol,
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
            # "snes_monitor": "ascii:output/logs/fsi2_biharm_restr_snes_log.txt",
            "snes_monitor": None,
        },
    )


    writer = dfx.io.VTXWriter(comm, output_path, [u,v])
    writer_p = dfx.io.VTXWriter(comm, output_path_p, [p])

    dm_loc_size = U.dofmap.index_map.size_local
    spot = np.array([0.6, 0.2, 0.0], dtype=np.float64)
    spot_dof_cand = np.flatnonzero(np.all(np.isclose(U.tabulate_dof_coordinates()[:dm_loc_size,:], spot, atol=1e-6), axis=1))
    spot_dof = spot_dof_cand[0] if len(spot_dof_cand) > 0 else None
    assert comm.allreduce(len(spot_dof_cand), op=MPI.SUM) == 1, "None or multiple dofs found for measurement point"

    loc_u_spot = np.zeros(2, dtype=np.float64)

    normal = ufl.FacetNormal(mesh)
    e_x = dfx.fem.Constant(mesh, (-1.0, 0.0))
    e_y = dfx.fem.Constant(mesh, (0.0, 1.0))
    F = ufl.Identity(2) + ufl.grad(u)
    transformed_normal = ufl.dot(ufl.inv(F.T), normal)

    drag_form_obstacle = ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_x) * ufl.det(F) * ds(PHYSICAL_MARKERS["obstacle"])
    lift_form_obstacle = ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_y) * ufl.det(F) * ds(PHYSICAL_MARKERS["obstacle"])

    drag_form_interface = ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_x) * ufl.det(F) * ds_interface_fluid
    lift_form_interface = ufl.dot(ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal), e_y) * ufl.det(F) * ds_interface_fluid

    drag_form_obstacle = dfx.fem.form(drag_form_obstacle, entity_maps=entity_maps)
    drag_form_interface = dfx.fem.form(drag_form_interface, entity_maps=entity_maps)
    lift_form_obstacle = dfx.fem.form(lift_form_obstacle, entity_maps=entity_maps)
    lift_form_interface = dfx.fem.form(lift_form_interface, entity_maps=entity_maps)


    if comm.rank == 0:
        Path(qoi_path).parent.mkdir(parents=True, exist_ok=True)
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



    res_miss_ufl = rho_s * ufl.inner(
        (u - u_old) / dt - theta * v - (1 - theta) * v_old, du
        ) * dx_solid
    res_miss = dfx.fem.form(res_miss_ufl)
    jac_uu = dfx.fem.form(ufl.derivative(res_miss_ufl, u, ufl.TrialFunction(U)))
    jac_uv = dfx.fem.form(ufl.derivative(res_miss_ufl, v, ufl.TrialFunction(V)))


    # Global rows of the interface u-dofs owned by this process. The owned rows
    # of the block matrix are numbered [u, v, p, z] from the start of the
    # ownership range, and every interface row is owned by a process that
    # located it, so zeroing owned rows covers all of them.
    dofs_d, num_owned_d = bc_deactivate.dof_indices()
    off_own, _ = problem.b.getAttr("_blocks")
    rows_d = (J_mat.getOwnershipRange()[0] + off_own[0] + dofs_d[:num_owned_d]).astype(PETSc.IntType)

    def zero_block(test_space, trial_space, **kwargs):
        return dfx.fem.form(
            ufl.ZeroBaseForm((ufl.TestFunction(test_space), ufl.TrialFunction(trial_space))),
            **kwargs,
        )

    res_post = [
        res_miss,    # rho_s[(u-u_old)/dt - θv - (1-θ)v_old]·du_u dx_solid
        dfx.fem.form(ufl.ZeroBaseForm((dv,))),
        dfx.fem.form(ufl.ZeroBaseForm((dp,))),
        dfx.fem.form(ufl.ZeroBaseForm((dz,))),
    ]

    jac_post = [
        [jac_uu,           jac_uv,             None,             None],
        [  None, zero_block(V, V),             None,             None],
        [  None,             None, zero_block(P, P),             None],
        [  None,             None,             None, zero_block(Z, Z)],
    ]

    bcs_post = [u_s_bc, *bcs]
    bcs_rows = dfx.fem.bcs_by_block(dfx.fem.extract_function_spaces(res_post), bcs_post)
    bcs_cols = dfx.fem.bcs_by_block(dfx.fem.extract_function_spaces(jac_post, 1), bcs_post)

    # Row scaling for the mesh-motion equations, used only in the factorization.
    # With restricted test functions the mesh-motion rows (the z block and the
    # fluid-only u rows) are decoupled from the interface, so scaling them by
    # alpha does not change the Newton step. It does help MUMPS pivoting, see
    # restricted-test-functions-solver-time.md. The residual and Jacobian seen by
    # SNES stay unscaled, so the convergence tests see the alpha = 1 residual.

    mesh_motion_scaling = 1.0e-9

    n_u_own = U.dofmap.index_map.size_local * U.dofmap.index_map_bs

    def unroll(dofs, bs=U.dofmap.index_map_bs):
        return (bs * dofs[:, None] + np.arange(bs)).ravel()

    u_fluid_rows = unroll(dfx.fem.locate_dofs_topological(
        U, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"])
    ))
    u_fluid_rows = np.setdiff1d(u_fluid_rows, dofs_d)    # interface rows hold the solid equation
    for bc in (u_f_bc, u_s_bc):                           # Dirichlet rows hold x - g
        u_fluid_rows = np.setdiff1d(u_fluid_rows, bc.dof_indices()[0])
    u_fluid_rows = u_fluid_rows[u_fluid_rows < n_u_own]   # owned rows only

    d_vec = problem.b.duplicate()
    d_vec.set(1.0)
    d_vec.array_w[off_own[0] + u_fluid_rows] = mesh_motion_scaling
    d_vec.array_w[off_own[3]:off_own[4]] = mesh_motion_scaling

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

    pc = solver.getKSP().getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(ScaledLU(f"{solver.getOptionsPrefix()}scaled_lu_"))

    def pre_jacobian(x: PETSc.Vec, J: PETSc.Mat) -> None:
        pass

    def post_jacobian(x: PETSc.Vec, J: PETSc.Mat) -> None:
        J.zeroRows(rows_d, diag=0.0)

        dolfinx.fem.petsc.assemble_matrix(
            J, jac_post, bcs=[u_s_bc, *bcs]
        )
        J.assemble()

    def wrapped_jacobian(
            snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat
        ) -> None:
        pre_jacobian(x, J)
        fem_jacobian(snes, x, J, P, *jac_args, **jac_kargs)
        post_jacobian(x, J)
        J.copy(P, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        P.diagonalScale(L=d_vec)


    def pre_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:
        pass

    def post_residual(x: PETSc.Vec, b: PETSc.Vec) -> None:

        dolfinx.fem.petsc.set_bc(b, [[bc_deactivate], [], [], []], alpha=0.0)   # zero restricted rows

        with b.localForm() as bl:
            # Remove ghost-entries.
            bl.array[b.getLocalSize():] = 0.0

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
        pre_residual(x, b) # x is not yet assigned to u here
        fem_residual(snes, x, b, *res_args, **res_kargs)
        post_residual(x, b) # u is now updated and b assembled (BCs applied)
        
    
    # Preconditioner matrix D*J, factorized by MUMPS instead of J. J is
    # assembled once first, so that P gets J's final parallel layout and can
    # be refilled with a same-nonzero-pattern copy in every Newton iteration.
    dfx.fem.petsc.assign([u, v, p, z], problem.x)
    fem_jacobian(solver, problem.x, J_mat, P_mat, *jac_args, **jac_kargs)
    post_jacobian(problem.x, J_mat)
    P_scaled = J_mat.duplicate(copy=True)

    solver.setJacobian(wrapped_jacobian, J_mat, P_scaled)
    solver.setFunction(wrapped_residual, b_vec)

    t = t0
    step = 0
    max_steps = np.inf
    start = timer()
    while step < max_steps and t < T:

        inflow_bc_func.interpolate(InflowFunc(t))
        inflow_bc_func.x.scatter_forward()

        u_old.x.array[:] = u.x.array[:]
        v_old.x.array[:] = v.x.array[:]

        if comm.rank == 0:
            print(f"\n{t = :.3f}")

        problem.solve()
        converged = problem.solver.getConvergedReason()

        if converged < 0:
            writer.close()
            writer_p.close()
            break


        if comm.rank == 0:
            sys.stdout.flush()
        
        if step % save_every == 0:
            writer.write(t)
            writer_p.write(t)

        loc_u_spot[:] = u.x.array[2*spot_dof:2*(spot_dof+1)] if spot_dof is not None else 0.0
        u_spot = comm.reduce(loc_u_spot, op=MPI.SUM, root=0)
        drag = comm.reduce(dfx.fem.assemble_scalar(drag_form_obstacle) + dfx.fem.assemble_scalar(drag_form_interface))
        lift = comm.reduce(dfx.fem.assemble_scalar(lift_form_obstacle) + dfx.fem.assemble_scalar(lift_form_interface))
        if comm.rank == 0:
            with open(qoi_path, "ab") as f:
                np.savetxt(f, [[t, drag, lift, *u_spot]], fmt="%.6e", delimiter="\t")


        step += 1
        t += dt.value
        

    end = timer()
    if comm.rank == 0:
        print(f"\n{comm.size = }")
        print(f"Elapsed time: {end - start:.3f} s")
        print(f"Time per step: {(end - start) / (step+1):.3f} s")


    if problem.solver.getConvergedReason() > 0:
        writer.close()
        writer_p.close()


    return


def main():
    solve(
        mesh_path="data/meshes/fsi2/mesh_sec.xdmf",
        T=15.0,
        dt_val=0.0025,
        output_path="output/pv/fsi2_biharm_dm_restr_scaled.bp",
        output_path_p="output/pv/fsi2_biharm_p_dm_restr_scaled.bp",
        qoi_path="output/qoi/fsi2_biharm_qoi_restr_scaled.txt",
    )


if __name__ == "__main__":
    main()
