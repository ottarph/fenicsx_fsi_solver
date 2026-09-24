# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import ufl
from petsc4py import PETSc

import sys
import warnings
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from timeit import default_timer as timer

from mpi4py.MPI import COMM_WORLD as comm
from mpi4py import MPI

from xfsi_solver.fsi.forms import nonzero, restrict_to_cells
from xfsi_solver.fsi.materials import Fluid, Solid
from xfsi_solver.linalg.fieldsplit import field_dof_rows, field_norms
from xfsi_solver.solvers.fsi2_harmonic_diffmesh_fieldsplit import (
    FieldSplitConfig,
    FieldSplitSolver,
    InstrumentedSolver,
)

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

U_BAR = 1.0
CHANNEL_HEIGHT = 0.41

JACOBIAN_MODES = ("full", "no_ale")
LINEAR_SOLVERS = ("direct", "fieldsplit")


class InflowFunc:
    def __init__(self, t: float = 0.0):
        self.t = t
    def __call__(self, x: np.ndarray) -> np.ndarray:
        H = CHANNEL_HEIGHT
        values = np.zeros((2, x.shape[1]), dtype=x.dtype)
        values[0] = 1.5 * U_BAR * 4 * x[1] * (H - x[1]) / H**2
        if self.t < 2.0:
            values[0] *= 0.5 * (1.0 - np.cos(0.5*np.pi * self.t))
        return values


@dataclass
class SolverConfig:
    """Nonlinear/linear solver configuration for :func:`solve`.

    ``jacobian_mode`` selects the operator used by Newton's method:
    ``"full"`` is the exact derivative of the residual, ``"no_ale"`` omits
    the derivatives of the fluid momentum and incompressibility terms with
    respect to the fluid displacement (see :func:`jacobian_forms`). The
    residual itself is identical in both modes.

    ``preconditioner_mode`` selects the matrix handed to the linear solver as
    preconditioning operator; ``None`` means the Jacobian itself.

    ``linear_solver="direct"`` is MUMPS LU on the Jacobian.
    ``linear_solver="fieldsplit"`` is FGMRES with a Schur field split
    configured by ``fieldsplit`` (a :class:`FieldSplitConfig`, or its
    ``variant`` as a string), see
    :mod:`xfsi_solver.solvers.fsi2_harmonic_diffmesh_fieldsplit`. The
    ``ksp_*`` options apply to FGMRES only.
    """
    jacobian_mode: str = "full"
    preconditioner_mode: str | None = None
    linear_solver: str = "direct"
    snes_max_it: int = 20
    snes_atol: float = 1.0e-7
    snes_rtol: float = 1.0e-12
    snes_monitor: bool = True
    fieldsplit: FieldSplitConfig = field(default_factory=FieldSplitConfig)
    ksp_rtol: float = 1.0e-6
    ksp_atol: float = 1.0e-50
    ksp_max_it: int = 500
    ksp_restart: int = 100
    ksp_monitor: bool = False

    def __post_init__(self):
        if isinstance(self.fieldsplit, str):
            self.fieldsplit = FieldSplitConfig(variant=self.fieldsplit)
        if self.jacobian_mode not in JACOBIAN_MODES:
            raise ValueError(f"Unknown jacobian_mode {self.jacobian_mode!r}, expected one of {JACOBIAN_MODES}")
        if self.preconditioner_mode not in (None, *JACOBIAN_MODES):
            raise ValueError(f"Unknown preconditioner_mode {self.preconditioner_mode!r}")
        if self.linear_solver not in LINEAR_SOLVERS:
            raise ValueError(f"Unknown linear_solver {self.linear_solver!r}, expected one of {LINEAR_SOLVERS}")
        if self.linear_solver == "direct" and self.preconditioner_mode not in (None, self.jacobian_mode):
            raise ValueError("A separate preconditioning matrix requires an iterative linear solver")


@dataclass
class FSIProblem:
    """Discrete FSI problem on the shared fluid-solid displacement/velocity spaces."""
    mesh: dfx.mesh.Mesh
    fluid_mesh: dfx.mesh.Mesh
    fluid_vertex_map: dfx.mesh.EntityMap
    cell_tags: dfx.mesh.MeshTags
    facet_tags: dfx.mesh.MeshTags
    entity_maps: list
    ds: ufl.Measure
    dx_fluid: ufl.Measure
    dx_solid: ufl.Measure
    ds_interface_fluid: ufl.Measure
    constants: dict
    U: dfx.fem.FunctionSpace
    V: dfx.fem.FunctionSpace
    P: dfx.fem.FunctionSpace
    u: dfx.fem.Function
    v: dfx.fem.Function
    p: dfx.fem.Function
    u_old: dfx.fem.Function
    v_old: dfx.fem.Function
    bcs_u: list
    bcs_v: list
    inflow_bc_func: dfx.fem.Function
    residual: list

    @property
    def solution(self):
        return [self.u, self.v, self.p]

    @property
    def bcs(self):
        return [*self.bcs_u, *self.bcs_v]

    def solid_displacement_dofs(self) -> np.ndarray:
        """Mask of the owned (block-expanded) displacement DOFs of solid cells, interface included."""
        U = self.U
        bs = U.dofmap.index_map_bs
        n_owned = U.dofmap.index_map.size_local
        solid_cells = self.cell_tags.find(PHYSICAL_MARKERS["solid"])
        nodes = dfx.fem.locate_dofs_topological(U, self.mesh.topology.dim, solid_cells)
        nodes = nodes[nodes < n_owned]
        mask = np.zeros(n_owned * bs, dtype=bool)
        for k in range(bs):
            mask[bs * nodes + k] = True
        return mask

    def set_inflow(self, t: float):
        self.inflow_bc_func.interpolate(InflowFunc(t))
        self.inflow_bc_func.x.scatter_forward()


def build_problem(mesh_path, dt_val) -> FSIProblem:

    # load mesh and meshtags

    # mesh_path alternatives:
    #   "data/meshes/fsi2/mesh_quad.xdmf"
    #   "data/meshes/fsi2/mesh_quad_fine_sec.xdmf"
    #   "data/meshes/fsi2/mesh_sec.xdmf"

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
    # lambda_s = dfx.fem.Constant(mesh, 2e6)

    dt = dfx.fem.Constant(mesh, dt_val)

    theta = dfx.fem.Constant(mesh, 0.5 + dt.value)

    alpha_u = dfx.fem.Constant(mesh, 1e-9)


    # create function spaces

    U = dfx.fem.functionspace(mesh, ("CG", 2, (2, )))
    V = dfx.fem.functionspace(mesh, ("CG", 2, (2, )))
    P = dfx.fem.functionspace(fluid_mesh, ("CG", 1))
    W = ufl.MixedFunctionSpace(U, V, P)


    # create functions

    u, v, p = dfx.fem.Function(U, name="u"), dfx.fem.Function(V, name="v"), dfx.fem.Function(P, name="p")
    u_old, v_old = dfx.fem.Function(U), dfx.fem.Function(V)


    du, dv, dp = ufl.TestFunctions(W)


    # create Dirichlet boundary condition

    inflow_bc_func = dfx.fem.Function(V)
    inflow_bc_func.x.array[:] = 0.0
    inflow_bc_facets = reduce(np.union1d, [
        facet_tags.find(PHYSICAL_MARKERS["inflow"]),
        ])
    inflow_bc_dofs = dfx.fem.locate_dofs_topological(V, mesh.geometry.dim - 1, inflow_bc_facets)

    inflow_bc = dfx.fem.dirichletbc(inflow_bc_func, inflow_bc_dofs)

    noslip_bc_func = dfx.fem.Function(V)
    noslip_bc_func.x.array[:] = 0.0
    noslip_bc_facets = reduce(np.union1d, [
        facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
        facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
        facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
        ])
    noslip_bc_dofs = dfx.fem.locate_dofs_topological(V, mesh.geometry.dim - 1, noslip_bc_facets)

    noslip_bc = dfx.fem.dirichletbc(noslip_bc_func, noslip_bc_dofs)


    # Create ALE Dirichlet boundary condition

    u_bc_func = dfx.fem.Function(U)
    u_bc_func.x.array[:] = 0.0
    u_bc_facets = reduce(np.union1d, [
        facet_tags.find(PHYSICAL_MARKERS["inflow"]),
        facet_tags.find(PHYSICAL_MARKERS["obstacle"]),
        facet_tags.find(PHYSICAL_MARKERS["solid_obstacle_interface"]),
        facet_tags.find(PHYSICAL_MARKERS["channel_side"]),
        facet_tags.find(PHYSICAL_MARKERS["outflow"]),
    ])
    u_bc_dofs = dfx.fem.locate_dofs_topological(U, mesh.geometry.dim - 1, u_bc_facets)
    u_bc = dfx.fem.dirichletbc(u_bc_func, u_bc_dofs)



    # DESCRIBE FSI PROBLEM
    # FLUID: Parabolic inflow on left side, no-slip on top, bottom, and obstacle, do-nothing on right side
    # SOLID: Homogeneous Dirichlet on left side

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

        residual += rho_s * ufl.inner((u - u_old) / dt, du) * dx_solid

        return residual

    def A_I(u, v, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)
        normal = ufl.FacetNormal(mesh)

        residual  = ufl.inner(alpha_u * ufl.grad(u), ufl.grad(du)) * dx_fluid
        residual -= ufl.inner(alpha_u * ufl.grad(u) * normal, du) * ds_interface_fluid

        residual += ufl.div(J * ufl.inv(F) * v) * dp * dx_fluid

        return residual

    def A_E(u, v):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual  = rho_f * J * ufl.inner(ufl.grad(v) * ufl.inv(F) * v, dv) * dx_fluid

        residual += ufl.inner(J * Fluid.NS_velocity(u, v, nu_f, rho_f) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        residual += ufl.inner(J * Solid.STVK(u, lambda_s, mu_s) * ufl.inv(F).T, ufl.grad(dv)) * dx_solid

        residual -= rho_s * ufl.inner(v, du) * dx_solid

        return residual

    def A_P(u, p):
        F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
        J = ufl.det(F)

        residual  = J * ufl.inner(Fluid.NS_pressure(p) * ufl.inv(F).T, ufl.grad(dv)) * dx_fluid

        return residual

    residual  = A_T(u, u_old, v, v_old)
    residual += A_I(u, v, p)
    residual += A_P(u, p)
    residual += theta * A_E(u, v)
    residual += (1.0 - theta) * A_E(u_old, v_old)


    #--------------------------------------------

    # Do-nothing condition
    # residual -= rho_f * nu_f * ufl.inner(ufl.grad(v).T * n, dv) * ds(PHYSICAL_MARKERS["outflow"])


    residual_blocked = ufl.extract_blocks(residual)

    constants = dict(
        rho_f=rho_f, nu_f=nu_f, rho_s=rho_s, mu_s=mu_s, nu_s=nu_s, lambda_s=lambda_s,
        dt=dt, theta=theta, alpha_u=alpha_u,
    )

    return FSIProblem(
        mesh=mesh, fluid_mesh=fluid_mesh, fluid_vertex_map=fluid_vertex_map, cell_tags=cell_tags, facet_tags=facet_tags,
        entity_maps=entity_maps, ds=ds, dx_fluid=dx_fluid, dx_solid=dx_solid,
        ds_interface_fluid=ds_interface_fluid, constants=constants,
        U=U, V=V, P=P, u=u, v=v, p=p, u_old=u_old, v_old=v_old,
        bcs_u=[u_bc], bcs_v=[inflow_bc, noslip_bc], inflow_bc_func=inflow_bc_func, residual=residual_blocked,
    )


def jacobian_forms(problem: FSIProblem, mode: str = "full"):
    """Block Jacobian forms, rows and columns ordered as ``(u, v, p)``.

    ``mode="full"`` differentiates the complete residual. ``mode="no_ale"``
    omits the derivatives of the fluid momentum and incompressibility
    residuals with respect to the displacement: block ``(v, u)`` is replaced
    by the derivative of the solid momentum contribution alone, and block
    ``(p, u)`` is dropped. The shared interface rows of block ``(v, u)``
    contain both fluid and solid contributions, so the selection is made by
    integration domain, not by rows. All velocity and pressure derivatives,
    the mesh equation and the kinematic equation are retained.
    """
    if mode not in JACOBIAN_MODES:
        raise ValueError(f"Unknown Jacobian mode {mode!r}, expected one of {JACOBIAN_MODES}")

    F = problem.residual
    w = problem.solution
    dw = [ufl.TrialFunction(w_j.function_space) for w_j in w]

    def derivative(form, j):
        return nonzero(ufl.derivative(form, w[j], dw[j]))

    J = [[None if (mode == "no_ale" and i > 0 and j == 0) else derivative(F[i], j)
          for j in range(len(w))] for i in range(len(F))]

    if mode == "no_ale":
        J[1][0] = derivative(restrict_to_cells(F[1], PHYSICAL_MARKERS["solid"]), 0)

    return J


@dataclass
class StepInfo:
    t: float
    snes_iterations: int
    linear_iterations: int
    converged_reason: int
    residual_history: np.ndarray
    time: float
    field_residuals: np.ndarray
    linear_solves: list
    timings: dict
    preconditioner_statistics: dict
    drag: float
    lift: float
    tip_displacement: np.ndarray


@dataclass
class SolveResult:
    problem: FSIProblem
    config: SolverConfig
    steps: list = field(default_factory=list)
    elapsed: float = 0.0


DIAGNOSTIC_FIELDS = ("u_solid", "u_fluid", "v", "p")


def diagnostic_index_sets(A, problem: FSIProblem) -> list:
    """Index sets of the rows of ``DIAGNOSTIC_FIELDS`` in the block matrix ``A``.

    The displacement is split into the DOFs of solid cells (kinematic rows)
    and the fluid interior (mesh rows, scaled by the tiny mesh-extension
    coefficient), so that residuals of the latter are not hidden by the former.
    """
    u_rows, v_rows, p_rows = field_dof_rows(A, [problem.U, problem.V, problem.P])
    solid = problem.solid_displacement_dofs()
    return [PETSc.IS().createGeneral(np.sort(rows), comm=A.comm)
            for rows in (u_rows[solid], u_rows[~solid], v_rows, p_rows)]


class FieldResidualMonitor:
    """Records the nonlinear residual norm of each of ``DIAGNOSTIC_FIELDS`` at every Newton iterate.

    The mesh equation is scaled by the tiny mesh-extension coefficient, so the
    total residual norm alone can hide a poorly converged displacement.
    """

    def __init__(self, nonlinear_problem, problem: FSIProblem):
        self.field_is = diagnostic_index_sets(nonlinear_problem.A, problem)
        self.history = []
        nonlinear_problem.solver.setMonitor(self)

    def __call__(self, snes, its, rnorm):
        if its == 0:
            self.history = []
        self.history.append(field_norms(snes.getFunction()[0], self.field_is))

    def destroy(self):
        for index_set in self.field_is:
            index_set.destroy()


def create_nonlinear_problem(problem: FSIProblem, config: SolverConfig):
    """The ``NonlinearProblem`` and the ``InstrumentedSolver`` (a ``FieldSplitSolver`` for fieldsplit) of its KSP."""
    J = jacobian_forms(problem, config.jacobian_mode)
    P = None
    if config.preconditioner_mode not in (None, config.jacobian_mode):
        P = jacobian_forms(problem, config.preconditioner_mode)

    petsc_options = {
        "snes_linesearch_type": "none",
        "snes_max_it": config.snes_max_it,
        "snes_atol": config.snes_atol,
        "snes_rtol": config.snes_rtol,
        "snes_error_if_not_converged": True,
        "ksp_error_if_not_converged": True,
        # "snes_monitor": "ascii:output/logs/fsi2_harm_dm_snes_log.txt",
    }
    if config.linear_solver == "direct":
        petsc_options |= {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 80,
        }
    if config.snes_monitor:
        petsc_options["snes_monitor"] = None
    if config.ksp_monitor:
        petsc_options["ksp_monitor_true_residual"] = None

    prefix = "fsi2_harmonic_diffmesh_"
    nonlinear_problem = dfx.fem.petsc.NonlinearProblem(
        problem.residual, problem.solution, bcs=problem.bcs, J=J, P=P,
        petsc_options_prefix=prefix,
        entity_maps=problem.entity_maps,
        petsc_options=petsc_options,
    )
    # NonlinearProblem removes its options with prefixPush + ClearValue, which
    # does not apply the pushed prefix, so they stay in the options database and
    # would configure any later solver with the same prefix (DOLFINx 0.11.0,
    # PETSc 3.25.5).
    opts = PETSc.Options()
    for key in petsc_options:
        if opts.hasName(f"{prefix}{key}"):
            del opts[f"{prefix}{key}"]
    nonlinear_problem.solver.setConvergenceHistory(reset=True)

    if config.linear_solver == "fieldsplit":
        linear_solver = FieldSplitSolver(nonlinear_problem, problem, config)
    else:
        linear_solver = InstrumentedSolver(nonlinear_problem, problem)
    return nonlinear_problem, linear_solver


def state_path(path) -> Path:
    """The file holding this rank's part of the state ``path`` (suffixed by rank on several ranks)."""
    path = Path(path)
    if comm.size > 1:
        path = path.with_name(f"{path.stem}_rank{comm.rank}of{comm.size}{path.suffix}")
    return path


def save_state(problem: FSIProblem, t_next: float, path) -> None:
    """Save the current ``(u, v, p)`` for a restart at time ``t_next``.

    The arrays are stored per rank in DOLFINx's local ordering, so a state
    can only be loaded with the same mesh file and number of ranks.
    """
    path = state_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, t_next=t_next, comm_size=comm.size, **{f.name: f.x.array for f in problem.solution})


def load_state(problem: FSIProblem, path) -> float:
    """Load a state saved by :func:`save_state`; returns the time of the next step."""
    path = state_path(path)
    data = np.load(path)
    if int(data["comm_size"]) != comm.size:
        raise ValueError(f"State {path} was saved on {int(data['comm_size'])} ranks, not {comm.size}")
    for f in problem.solution:
        if data[f.name].shape != f.x.array.shape:
            raise ValueError(f"State {path} does not match the mesh ({f.name})")
        f.x.array[:] = data[f.name]
    return float(data["t_next"])


def solve(mesh_path, T, dt_val, output_path, output_path_p, qoi_path,
          config: SolverConfig | None = None, *, initial_state=None, checkpoint_dir=None,
          checkpoint_every: float | None = None) -> SolveResult:
    """Time step the FSI problem up to time ``T``.

    ``initial_state`` is a file written by :func:`save_state` to restart from;
    with ``checkpoint_dir`` and ``checkpoint_every``, states are saved to
    ``checkpoint_dir/state_t<time>.npz`` about every ``checkpoint_every``
    time units.
    """

    config = SolverConfig() if config is None else config

    problem = build_problem(mesh_path, dt_val)
    mesh = problem.mesh
    U = problem.U
    u, v, p = problem.solution
    rho_f, nu_f = problem.constants["rho_f"], problem.constants["nu_f"]
    ds, ds_interface_fluid = problem.ds, problem.ds_interface_fluid
    entity_maps = problem.entity_maps

    t0 = 0.0 if initial_state is None else load_state(problem, initial_state)
    dt = problem.constants["dt"]
    next_checkpoint = None if checkpoint_every is None else t0 + checkpoint_every

    save_every = 4

    total_steps = int(np.ceil((T - t0) / dt_val))
    if total_steps <= save_every:
        warnings.warn(
            f"save_every ({save_every}) is larger than the total number of time "
            f"steps ({total_steps}); at most one VTX snapshot will be written to "
            f"{output_path!r} or {output_path_p!r}, which is not a usable time "
            f"series in ParaView."
        )

    nonlinear_problem, linear_solver = create_nonlinear_problem(problem, config)
    field_monitor = FieldResidualMonitor(nonlinear_problem, problem)
    snes = nonlinear_problem.solver


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

    result = SolveResult(problem=problem, config=config)

    writer = dfx.io.VTXWriter(comm, output_path, [u,v])
    writer_p = dfx.io.VTXWriter(comm, output_path_p, [p])

    try:
        t = t0
        step = -1
        max_steps = np.inf
        start = timer()
        while step < max_steps and t < T:

            problem.set_inflow(t)

            problem.u_old.x.array[:] = u.x.array[:]
            problem.v_old.x.array[:] = v.x.array[:]

            if comm.rank == 0:
                print(f"\n{t = :.3f}")

            linear_solver.reset_statistics()
            step_start = timer()
            nonlinear_problem.solve()
            step_time = timer() - step_start

            if comm.rank == 0:
                sys.stdout.flush()

            if step % save_every == 0:
                writer.write(t)
                writer_p.write(t)

            loc_u_spot[:] = u.x.array[2*spot_dof:2*(spot_dof+1)] if spot_dof is not None else 0.0
            u_spot = comm.allreduce(loc_u_spot, op=MPI.SUM)
            drag = comm.allreduce(dfx.fem.assemble_scalar(drag_form_obstacle) + dfx.fem.assemble_scalar(drag_form_interface))
            lift = comm.allreduce(dfx.fem.assemble_scalar(lift_form_obstacle) + dfx.fem.assemble_scalar(lift_form_interface))
            if comm.rank == 0:
                with open(qoi_path, "ab") as f:
                    np.savetxt(f, [[t, drag, lift, *u_spot]], fmt="%.6e", delimiter="\t")

            history, _ = snes.getConvergenceHistory()
            result.steps.append(StepInfo(
                t=t,
                snes_iterations=snes.getIterationNumber(),
                linear_iterations=snes.getLinearSolveIterations(),
                converged_reason=snes.getConvergedReason(),
                residual_history=np.array(history),
                time=step_time,
                field_residuals=np.array(field_monitor.history),
                linear_solves=linear_solver.linear_solves,
                timings=dict(linear_solver.timings),
                preconditioner_statistics=linear_solver.context_statistics(),
                drag=drag,
                lift=lift,
                tip_displacement=u_spot.copy(),
            ))

            step += 1
            t += dt.value

            if next_checkpoint is not None and t >= next_checkpoint - 1e-9 * dt.value:
                save_state(problem, t, Path(checkpoint_dir) / f"state_t{t:.4f}.npz")
                next_checkpoint += checkpoint_every

        end = timer()
        result.elapsed = end - start
        if comm.rank == 0:
            print(f"\n{comm.size = }")
            print(f"Elapsed time: {end - start:.3f} s")
            print(f"Time per step: {(end - start) / (step+1):.3f} s")
    finally:
        writer.close()
        writer_p.close()
        field_monitor.destroy()
        linear_solver.destroy()

    return result


def main():
    solve(
        mesh_path="data/meshes/fsi2/mesh_sec.xdmf",
        T=15.0, # Will fail at around t=7s
        dt_val=0.0025,
        output_path="output/pv/fsi2_harm_dm.bp",
        output_path_p="output/pv/fsi2_harm_p_dm.bp",
        qoi_path="output/qoi/fsi2_harm_dm_qoi.txt",
    )


if __name__ == "__main__":
    main()
