# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""C0IP biharmonic mesh motion alone, on the fluid domain of the FSI2 or the 3d FSI benchmark.

This is the mesh-motion block K_m of ``solvers/fsi2_biharmonic_c0ip.py``, solved for one
deformation of the fluid boundary, to study the linear solvers for it without running the FSI
simulation. The deformation is the displacement u of a checkpoint of an FSI run: it is zero on
the channel walls and the cylinder, and the solid's displacement on the fluid-solid interface.
The problem is posed on the fluid submesh with u as Dirichlet condition on its whole boundary.
The unknowns are then those of K_m; the interface dofs, which the FSI solver keeps in the solid
block u_S, become identity rows here. There is no C0IP FSI solver in 3d yet; there, the penalty
parameters are those of triangles carried over to tetrahedra (get_dg_penalty_parameters_tetrahedron).

The deformation can be interpolated to another mesh of the same geometry, to solve for the same
deformation on meshes of different resolution. Run with MPI, for example

    mpirun -n 4 python src/xfsi_solver/component_solvers/c0ip_mesh_motion.py --case fsi2 \\
        --mesh data/meshes/fsi2/mesh_fine_sec.xdmf --solver cg_gamg --compare
"""

import argparse
import resource
from timeit import default_timer as timer

import basix
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from petsc4py import PETSc

from xfsi_solver.tools.checkpoint import Checkpointer
from xfsi_solver.tools.dg_optpar import (
    compute_cell_volumes,
    compute_facet_diameters_and_areas,
    get_dg_penalty_parameters_tetrahedron,
    get_dg_penalty_parameters_triangle,
)
from xfsi_solver.tools.petsc_reuse import ReusePC
from xfsi_solver.tools.qoi import find_point_dof, point_value

PHYSICAL_MARKERS = {"solid": 1, "ALE_fluid": 2}

# The point of the solid on the interface whose displacement the benchmarks report: A in FSI2, B
# in the 3d benchmark. Printed for the checkpoint and for the boundary condition, to check the
# transfer of the deformation between meshes.
TIP_POINTS = {2: np.array([0.6, 0.2, 0.0]), 3: np.array([0.9, 0.2, 0.3])}

# "direct": MUMPS LU on the whole vector system, as a reference.
# "direct_component": MUMPS Cholesky on the x component, reused for the others, as in the FSI solver.
# "cg_gamg": CG with GAMG on each component, with the GAMG of the x component reused for the others.
# "cg_laplacian_squared": CG on each component, preconditioned by LaplacianSquaredPC, set up on the
#   x component and reused for the others.
SOLVERS = ("direct", "direct_component", "cg_gamg", "cg_laplacian_squared")

CASES = {
    "fsi2": {
        "mesh_path": "data/meshes/fsi2/mesh_sec.xdmf",
        "checkpoint_dir": "output/checkpoints/fsi2_biharm_c0ip",
        "checkpoint_dt": 0.0025,
    },
    "fsi3d": {
        "mesh_path": "data/meshes/fsi3d/mesh_sec_medium.xdmf",
        "checkpoint_dir": "output/checkpoints/fsi3d_iterative_medium",
        "checkpoint_dt": 0.004,
    },
}


def read_mesh(mesh_path):
    """The mesh and cell tags of an FSI benchmark mesh file."""
    with dolfinx.io.XDMFFile(comm, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name="Cell tags")
    return mesh, cell_tags


def read_displacement(mesh, checkpoint_dir, checkpoint_dt):
    """The displacement u on the whole mesh in the latest checkpoint of an FSI run on mesh, and
    the time of the checkpoint."""
    U = dolfinx.fem.functionspace(mesh, ("CG", 2, (mesh.geometry.dim,)))
    u = dolfinx.fem.Function(U, name="u")
    t, _ = Checkpointer(checkpoint_dir, mesh).read([u], checkpoint_dt)
    return u, t


def boundary_displacement(u_checkpoint, U_fluid, fluid_cell_map, same_mesh):
    """u_checkpoint on the fluid submesh space U_fluid.

    On the same mesh, it is copied cell by cell through the submesh's entity map. On another
    mesh, it is interpolated at the dof coordinates of U_fluid.
    """
    u_bc = dolfinx.fem.Function(U_fluid, name="u_bc")
    fluid_mesh = U_fluid.mesh
    index_map = fluid_mesh.topology.index_map(fluid_mesh.topology.dim)
    sub_cells = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    if same_mesh:
        parent_cells = fluid_cell_map.sub_topology_to_topology(sub_cells, inverse=False)
        u_bc.interpolate(u_checkpoint, cells0=parent_cells, cells1=sub_cells)
    else:
        # A small padding, so that dof coordinates on the boundary of the fluid domain find a cell
        # of the checkpoint mesh although its curved boundary is approximated differently.
        interpolation_data = dolfinx.fem.create_interpolation_data(
            U_fluid, u_checkpoint.function_space, sub_cells, padding=1e-8
        )
        u_bc.interpolate_nonmatching(u_checkpoint, sub_cells, interpolation_data)
    u_bc.x.scatter_forward()
    return u_bc


def c0ip_forms(U, a_val):
    """Bilinear and linear forms of the C0IP biharmonic mesh motion on the fluid submesh space U,
    as in fsi2_biharmonic_c0ip.py, with a zero normal derivative weakly imposed on the whole
    boundary, and the entity map of the facet sizes in the penalty."""
    mesh = U.mesh
    gdim = mesh.geometry.dim
    order = U.element.basix_element.degree
    vol = compute_cell_volumes(mesh)
    h_facet, area_facet, facet_entity_map = compute_facet_diameters_and_areas(mesh)
    if mesh.basix_cell() == basix.CellType.triangle:
        beta, beta_boundary = get_dg_penalty_parameters_triangle(mesh, a_val=a_val, p=order, vol=vol, h=h_facet)
    elif mesh.basix_cell() == basix.CellType.tetrahedron:
        beta, beta_boundary = get_dg_penalty_parameters_tetrahedron(
            mesh, a_val=a_val, p=order, vol=vol, h=h_facet, area=area_facet
        )
    else:
        raise NotImplementedError(f"C0IP penalty parameters are not implemented for {mesh.basix_cell()} cells")

    u, du = ufl.TrialFunction(U), ufl.TestFunction(U)
    n = ufl.FacetNormal(mesh)
    # The facet sizes live on a submesh of the facets, so the measures need the domain.
    dx = ufl.Measure("dx", domain=mesh)
    dS = ufl.Measure("dS", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    def lap(w):
        return ufl.div(ufl.grad(w))

    a = ufl.inner(lap(u), lap(du)) * dx
    a -= ufl.inner(ufl.avg(lap(u)), ufl.jump(ufl.grad(du), n)) * dS
    a -= ufl.inner(ufl.avg(lap(du)), ufl.jump(ufl.grad(u), n)) * dS
    a += beta / h_facet("+") * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(du), n)) * dS
    a -= ufl.inner(lap(u), ufl.grad(du) * n) * ds
    a -= ufl.inner(lap(du), ufl.grad(u) * n) * ds
    a += beta_boundary / h_facet * ufl.inner(ufl.grad(u) * n, ufl.grad(du) * n) * ds

    L = ufl.inner(dolfinx.fem.Constant(mesh, np.zeros(gdim)), du) * dx

    return a, L, facet_entity_map


class LaplacianSquaredPC:
    """PCPYTHON context applying L^{-1} M L^{-1}, with L the stiffness matrix of the Laplacian and
    M the mass matrix, and L^{-1} one V-cycle of algebraic multigrid (multigrid "hypre" for
    BoomerAMG or "gamg"), or a MUMPS LU solve (multigrid "lu"), to check the V-cycles.

    The biharmonic operator is the square of the Laplacian, and its discretization is close to
    L M^{-1} L. With the same symmetric V-cycle for both L^{-1}, the preconditioner is symmetric,
    so it can be used in CG. For the clamped boundary conditions of the mesh motion, the condition
    number of the preconditioned operator grows like h^{-1} (Braess and Peisker, IMA J. Numer.
    Anal. 6, 1986), instead of h^{-4} without a preconditioner. L and M must have the identity rows
    of the Dirichlet dofs of the biharmonic system.

    Call destroy_solvers when done: a method named destroy would be called by petsc4py when the
    PC is destroyed.
    """

    def __init__(self, L: PETSc.Mat, M: PETSc.Mat, multigrid: str):
        self.M = M
        self.ksp = PETSc.KSP().create(L.getComm())
        self.ksp.setOperators(L)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType(multigrid)
        if multigrid == "hypre":
            self.ksp.getPC().setHYPREType("boomeramg")
        elif multigrid == "lu":
            self.ksp.getPC().setFactorSolverType("mumps")
        self.Linv_x, self.M_Linv_x = M.createVecs()

    def setUp(self, pc: PETSc.PC) -> None:
        self.ksp.setUp()

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        self.ksp.solve(x, self.Linv_x)
        self.M.mult(self.Linv_x, self.M_Linv_x)
        self.ksp.solve(self.M_Linv_x, y)

    def destroy_solvers(self) -> None:
        self.ksp.destroy()
        self.Linv_x.destroy()
        self.M_Linv_x.destroy()


def laplacian_and_mass(U, bc, rows):
    """The stiffness matrix of the Laplacian and the mass matrix on the vector space U, with the
    identity rows of bc, restricted to the rows and columns in the index set rows."""
    u, du = ufl.TrialFunction(U), ufl.TestFunction(U)
    matrices = []
    for form in [ufl.inner(ufl.grad(u), ufl.grad(du)) * ufl.dx, ufl.inner(u, du) * ufl.dx]:
        A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(form), bcs=[bc])
        A.assemble()
        matrices.append(A.createSubMatrix(rows, rows))
        A.destroy()
    return matrices


def split_components(ksp, component_names, component_rows, split_options, x_options):
    """Set up the additive fieldsplit PC of ksp over the components, with the rows of each in
    component_rows, split_options for every component, x_options for the x component, and the
    preconditioner of the x component reused for the others. Returns the components' KSPs."""
    pc = ksp.getPC()
    pc.setFieldSplitIS(*zip(component_names, component_rows, strict=True))
    # Set through the options database, since the fieldsplit calls setFromOptions on its
    # sub-KSPs in PCSetUp, which would override types set directly on them.
    opts = PETSc.Options()
    component_ksps = pc.getFieldSplitSubKSP()
    for name, component_ksp in zip(component_names, component_ksps, strict=True):
        opts.prefixPush(component_ksp.getOptionsPrefix())
        for key, value in (split_options | (x_options if name == "x" else {"pc_type": "python"})).items():
            opts[key] = value
        opts.prefixPop()
        component_ksp.setFromOptions()
    pc_reuse = ReusePC(component_ksps[0].getPC())
    for component_ksp in component_ksps[1:]:
        component_ksp.getPC().setType(PETSc.PC.Type.PYTHON)
        component_ksp.getPC().setPythonContext(pc_reuse)
    return component_ksps


def solver_options(solver, rtol, atol, max_it):
    """Options of the KSP, and of the splits of its components, for one of SOLVERS."""
    if solver == "direct":
        options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_cntl_1": 1e-4,
        }
        return options, {}

    options = {"ksp_type": "preonly", "pc_type": "fieldsplit", "pc_fieldsplit_type": "additive"}
    if solver == "direct_component":
        split_options = {"ksp_type": "preonly"}
        x_options = {"pc_type": "cholesky", "pc_factor_mat_solver_type": "mumps", "mat_mumps_cntl_1": 1e-4}
    else:
        split_options = {
            "ksp_type": "cg",
            "ksp_norm_type": "unpreconditioned",
            "ksp_rtol": rtol,
            "ksp_atol": atol,
            "ksp_max_it": max_it,
        }
        if solver == "cg_gamg":
            x_options = {"pc_type": "gamg", "pc_gamg_type": "agg", "pc_gamg_threshold": 0.2}
        else:
            x_options = {"pc_type": "python"}
    return options, {"split": split_options, "x": x_options}


def solve(
    mesh_path,
    checkpoint_dir,
    checkpoint_dt,
    checkpoint_mesh_path=None,
    solver="cg_gamg",
    rtol=1e-5,
    atol=1e-9,
    max_it=1000,
    a_val=2.0,
    multigrid="hypre",
    compare_with_direct=False,
    output_path=None,
):
    """Solve the C0IP mesh motion on the fluid domain of mesh_path, for the deformation of the
    fluid boundary in the latest checkpoint in checkpoint_dir, written with time step
    checkpoint_dt by a run on checkpoint_mesh_path (by default mesh_path), and print the size,
    timings, iterations and memory of the solve.

    The solve is done twice: the first includes the setup of the solver (factorization or GAMG
    hierarchy), the second applies the solver that is set up, as in all but the first Newton
    iterations of the FSI solver, where K_m stays the same. rtol, atol and max_it are those of CG
    on each component, a_val scales the C0IP penalty. multigrid is the PETSc PC type of the
    solves with the Laplacian in cg_laplacian_squared: "hypre" (a BoomerAMG V-cycle), "gamg" (a
    GAMG V-cycle) or "lu" (MUMPS). With compare_with_direct, the system is also solved with
    MUMPS Cholesky on each component, to print the error of the solution relative to it, which for this
    fourth-order operator can be much larger than the relative residual. With output_path, the
    displacement is written to a VTX file.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, not {solver!r}")
    checkpoint_mesh_path = mesh_path if checkpoint_mesh_path is None else checkpoint_mesh_path

    mesh, cell_tags = read_mesh(mesh_path)
    same_mesh = checkpoint_mesh_path == mesh_path
    checkpoint_mesh = mesh if same_mesh else read_mesh(checkpoint_mesh_path)[0]
    u_checkpoint, t_checkpoint = read_displacement(checkpoint_mesh, checkpoint_dir, checkpoint_dt)

    tdim = mesh.topology.dim
    fluid_mesh, fluid_cell_map, _, _ = dolfinx.mesh.create_submesh(
        mesh, tdim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"])
    )
    fluid_mesh.topology.create_connectivity(tdim - 1, tdim)

    U = dolfinx.fem.functionspace(fluid_mesh, ("CG", 2, (tdim,)))
    u_bc = boundary_displacement(u_checkpoint, U, fluid_cell_map, same_mesh)

    tip_checkpoint = point_value(u_checkpoint, find_point_dof(u_checkpoint.function_space, TIP_POINTS[tdim]))
    tip_bc = point_value(u_bc, find_point_dof(U, TIP_POINTS[tdim]))

    boundary_facets = dolfinx.mesh.exterior_facet_indices(fluid_mesh.topology)
    bc = dolfinx.fem.dirichletbc(u_bc, dolfinx.fem.locate_dofs_topological(U, tdim - 1, boundary_facets))

    a, L, facet_entity_map = c0ip_forms(U, a_val)
    uh = dolfinx.fem.Function(U, name="u")
    options, component_options = solver_options(solver, rtol, atol, max_it)
    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        u=uh,
        petsc_options_prefix="mesh_motion_",
        petsc_options=options,
        entity_maps=[facet_entity_map],
    )
    ksp = problem.solver

    # The owned dofs are stored interleaved per node, (x, y[, z]), so the rows of a component are
    # every tdim-th owned row. The components are not coupled and have the same matrix.
    component_names = ["x", "y", "z"][:tdim]
    start, end = problem.A.getOwnershipRange()
    component_rows = [
        PETSc.IS().createGeneral(np.arange(start + c, end, tdim, dtype=np.int32), comm=comm) for c in range(tdim)
    ]
    laplacian_squared = None
    if solver != "direct":
        component_ksps = split_components(
            ksp, component_names, component_rows, component_options["split"], component_options["x"]
        )
        if solver == "cg_laplacian_squared":
            L_x, M_x = laplacian_and_mass(U, bc, component_rows[0])
            laplacian_squared = LaplacianSquaredPC(L_x, M_x, multigrid)
            component_ksps[0].getPC().setType(PETSc.PC.Type.PYTHON)
            component_ksps[0].getPC().setPythonContext(laplacian_squared)

    # First solve: assembly, setup of the solver, and the solve. The setup time is the time in
    # PCSetUp and PCSetUpOnBlocks, the slowest rank's: the fieldsplit sets up the preconditioners
    # of the components in PCSetUpOnBlocks.
    PETSc.Log.begin()
    start_time = timer()
    problem.solve()
    first_solve_time = timer() - start_time
    setup_time = comm.allreduce(
        sum(PETSc.Log.Event(name).getPerfInfo()["time"] for name in ["PCSetUp", "PCSetUpOnBlocks"]), op=MPI.MAX
    )
    iterations = [k.getIterationNumber() for k in component_ksps] if solver != "direct" else []
    reason_names = {value: name for name, value in vars(PETSc.KSP.ConvergedReason).items() if isinstance(value, int)}
    reasons = [reason_names[k.getConvergedReason()] for k in component_ksps] if solver != "direct" else []

    # Second solve: the solver set up in the first one, applied to the same right-hand side.
    x = problem.x.duplicate()
    start_time = timer()
    ksp.solve(problem.b, x)
    second_solve_time = timer() - start_time

    residual = problem.b.duplicate()
    problem.A.mult(problem.x, residual)
    residual.axpy(-1.0, problem.b)
    relative_residual = residual.norm() / problem.b.norm()

    if compare_with_direct:
        # The same assembled system, solved exactly by MUMPS Cholesky on the x component, reused for
        # the others, which needs much less memory than a factorization of the whole system.
        ksp_direct = PETSc.KSP().create(comm)
        ksp_direct.setOptionsPrefix("reference_")
        ksp_direct.setOperators(problem.A)
        ksp_direct.setType("preonly")
        ksp_direct.getPC().setType("fieldsplit")
        ksp_direct.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
        _, direct_options = solver_options("direct_component", rtol, atol, max_it)
        split_components(ksp_direct, component_names, component_rows, direct_options["split"], direct_options["x"])
        x_direct = problem.x.duplicate()
        ksp_direct.solve(problem.b, x_direct)
        x_direct.axpy(-1.0, problem.x)
        relative_error = x_direct.norm() / problem.x.norm()
        x_direct.destroy()
        ksp_direct.destroy()

    # Mesh quality of the deformed fluid domain, from det(I + grad u) at the cell midpoints.
    DG0 = dolfinx.fem.functionspace(fluid_mesh, ("DG", 0))
    det_F = dolfinx.fem.Function(DG0)
    det_F.interpolate(
        dolfinx.fem.Expression(ufl.det(ufl.Identity(tdim) + ufl.grad(uh)), DG0.element.interpolation_points)
    )
    num_owned_cells = fluid_mesh.topology.index_map(tdim).size_local
    min_det_F = comm.allreduce(det_F.x.array[:num_owned_cells].min(), op=MPI.MIN)

    factors = []
    if solver == "direct":
        factors = [ksp.getPC().getFactorMatrix()]
    elif solver == "direct_component":
        factors = [component_ksps[0].getPC().getFactorMatrix()]
    mumps_memory = sum(F.getMumpsInfog(22) for F in factors)  # MB, summed over the ranks
    peak_memory = comm.allreduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, op=MPI.SUM)  # kB

    u_norm = np.sqrt(
        comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(uh, uh) * ufl.dx)), op=MPI.SUM)
    )

    if comm.rank == 0:
        num_dofs = U.dofmap.index_map.size_global * tdim
        print(f"Mesh: {mesh_path}, fluid cells: {fluid_mesh.topology.index_map(tdim).size_global}")
        print(f"Deformation: checkpoint at t = {t_checkpoint:.4f} in {checkpoint_dir}, on {checkpoint_mesh_path}")
        print(f"  tip displacement in the checkpoint: {tip_checkpoint}, in the boundary condition: {tip_bc}")
        print(f"Unknowns: {num_dofs} ({num_dofs // tdim} per component), ranks: {comm.size}")
        print(
            f"Solver: {solver}"
            + (f" (rtol {rtol:g}, atol {atol:g}, max_it {max_it})" if solver.startswith("cg") else "")
            + (f", V-cycles of {multigrid}" if solver == "cg_laplacian_squared" else "")
        )
        if solver != "direct":
            print(f"  iterations per component: {iterations}, converged reasons: {reasons}")
        print(f"  first solve (assembly, setup and solve): {first_solve_time:.3f} s, of which setup {setup_time:.3f} s")
        print(f"  second solve (solve only): {second_solve_time:.3f} s")
        print(f"  relative residual: {relative_residual:.3e}")
        if factors:
            print(f"  MUMPS memory: {mumps_memory} MB")
        print(f"  peak memory, summed over the ranks: {peak_memory / 1024**2:.2f} GB")
        if compare_with_direct:
            print(f"  error relative to MUMPS Cholesky on each component (l2 norm of the dofs): {relative_error:.3e}")
        print(f"L2 norm of the displacement: {u_norm:.10e}")
        print(f"Smallest det(I + grad u) at the cell midpoints: {min_det_F:.4f}")

    if output_path is not None:
        with dolfinx.io.VTXWriter(comm, output_path, [uh]) as writer:
            writer.write(t_checkpoint)

    x.destroy()
    residual.destroy()
    # Destroying a MUMPS factorization is collective, so destroy the solver on all ranks here
    # instead of leaving it to garbage collection.
    ksp.destroy()
    if laplacian_squared is not None:
        laplacian_squared.destroy_solvers()
        L_x.destroy()
        M_x.destroy()

    return uh


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=CASES, default="fsi2", help="geometry and checkpoint of the deformation")
    parser.add_argument("--mesh", help="mesh to solve on, if not the checkpoint's; the deformation is interpolated")
    parser.add_argument("--solver", choices=SOLVERS, default="cg_gamg")
    parser.add_argument("--rtol", type=float, default=1e-5, help="relative tolerance of CG")
    parser.add_argument(
        "--multigrid",
        choices=["hypre", "gamg", "lu"],
        default="hypre",
        help="solves with the Laplacian in cg_laplacian_squared",
    )
    parser.add_argument("--compare", action="store_true", help="also solve with MUMPS Cholesky and print the error")
    parser.add_argument("--output", help="VTX file for the displacement")
    args = parser.parse_args()

    case = CASES[args.case]
    solve(
        mesh_path=case["mesh_path"] if args.mesh is None else args.mesh,
        checkpoint_dir=case["checkpoint_dir"],
        checkpoint_dt=case["checkpoint_dt"],
        checkpoint_mesh_path=case["mesh_path"],
        solver=args.solver,
        rtol=args.rtol,
        multigrid=args.multigrid,
        compare_with_direct=args.compare,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
