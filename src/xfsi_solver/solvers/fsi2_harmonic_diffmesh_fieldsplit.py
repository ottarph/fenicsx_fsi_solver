# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""FGMRES / Schur field-split linear solver for the shared-space FSI solver.

The Newton system is ordered ``(u, v, p)`` and split into displacement ``u``
and momentum-pressure ``vp = (v, p)``. The outer preconditioner is a full
Schur factorization of the preconditioning matrix (the no-ALE Jacobian in the
production configuration)::

    FGMRES on J
    +-- Schur fieldsplit: u | (v,p)
        +-- u: solid/interface mass solve + fluid-interior mesh extension
        +-- (v,p): Schur fieldsplit on an assembled auxiliary operator P_vp
            +-- v: AMG on the coupled fluid-solid effective velocity operator
            +-- p: pressure Schur complement approximation

Eliminating the displacement from the no-ALE Jacobian
``[[A, C, 0], [E_s, H, G], [0, B, 0]]`` gives the momentum-pressure Schur
complement ``[[H - E_s A^{-1} C, G], [B, 0]]``. With the solid kinematic
equation ``A ~ rho_s/dt M_s`` and ``C = -theta rho_s M_s`` on the solid DOFs,
``-E_s A^{-1} C ~ theta dt E_s = theta^2 dt K_s``, which motivates the
assembled auxiliary operator ``P_vp = [[H_hat, G], [B, 0]]`` with
``H_hat = H + theta dt E_s`` (``convection=True``, the default) or
``H_hat = rho_f J_mid/dt M_f + theta A_visc + rho_s/dt M_s + theta dt E_s``
without fluid convection. In this discretization ``A^{-1} C = -theta dt`` on
the solid DOFs up to the ``alpha``-scaled interface terms, so with exact
subsolves ``P_vp`` is nearly the exact Schur complement (1-3 FGMRES
iterations per Newton step). The reduced right-hand side and the
displacement recovery of the outer factorization use the actual blocks of
the preconditioning matrix.

``FieldSplitConfig.variant``:

``"exact"``
    Diagnostic. LU for the displacement block and dense LU on the explicitly
    computed Schur complement (``schur_precondition=full``). Every outer
    iteration is then an exact solve with the preconditioning matrix. Forming
    the Schur complement costs one displacement solve per ``(v, p)`` unknown
    and dense storage, so this is for coarse meshes only.

``"auxiliary"``
    ``P_vp`` is the user Schur preconditioning matrix. The subsolvers are
    selected by the remaining ``FieldSplitConfig`` fields; any of them can be
    replaced by LU to isolate the quality of a single approximation.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from timeit import default_timer as timer

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.fsi.forms import nonzero, restrict_to_cells
from xfsi_solver.fsi.materials import Fluid
from xfsi_solver.linalg.fieldsplit import (
    field_dof_rows,
    field_index_sets,
    field_norms,
    nested_index_sets,
    union_index_set,
)

FIELDSPLIT_VARIANTS = ("exact", "auxiliary")


@dataclass
class FieldSplitConfig:
    """Preconditioner hierarchy of the ``fieldsplit`` linear solver.

    Attributes:
        variant: ``"exact"`` or ``"auxiliary"``, see the module docstring.
        displacement: ``"lu"`` on the displacement block of the
            preconditioning matrix, or ``"block_triangular"``: a solid mass
            solve on the solid/interface DOFs followed by a solve with the
            fluid-interior mesh extension (harmonic or elastic, extracted from
            the preconditioning matrix), retaining the coupling ``A_fI``.
        displacement_fluid: solver for the fluid-interior block ``A_ff`` of
            ``"block_triangular"``: ``"cholesky"`` (MUMPS, factored once: the
            block is the linear mesh equation on the reference configuration
            and does not change), ``"amg"`` (BoomerAMG-preconditioned CG to
            a relative tolerance of 1e-8, unknown-based coarsening of the two
            displacement components) or ``"gamg"`` (GAMG-preconditioned CG
            to 1e-8 with the rigid body modes of the fluid-interior DOFs as
            near-nullspace, for vector elasticity with variable coefficients:
            the modes are not in the kernel of the Dirichlet-restricted block).
            For the harmonic extension on the default mesh a Cholesky solve
            takes 3 ms after a one-time 70 ms factorization, the BoomerAMG
            solve 29 ms (7 iterations); AMG is the option that scales to 3D.
        momentum: ``"lu"`` on ``P_vp``, or ``"schur"``: an inner full Schur
            factorization ``v | p`` of ``P_vp``.
        velocity: ``"lu"``, ``"hypre"`` (one BoomerAMG V-cycle) or ``"gamg"``
            (one GAMG V-cycle with the rigid body modes as near-nullspace) on
            ``H_hat``; used with ``momentum="schur"``. ``H_hat`` is symmetric
            positive definite without convection. On the default mesh,
            BoomerAMG-preconditioned CG reduces the residual by 1e-8 in 10
            iterations, GAMG with rigid body modes in 25 and BoomerAMG with
            nodal coarsening in 18: the solid is a small part of the domain and
            the mass term dominates at the time steps used.
        pressure: ``"cahouet_chabard"`` applies
            ``rho_f/dt K_p^{-1} + theta mu_f M_p^{-1}`` once;
            ``"selfp"`` applies one BoomerAMG V-cycle to the assembled
            ``S_p = B diag(H_hat)^{-1} G`` (PETSc ``schur_precondition=selfp``),
            which inherits the boundary conditions and the solid mass on the
            interface velocity DOFs algebraically; ``"accurate"`` solves the inner pressure Schur complement of
            ``P_vp`` with GMRES preconditioned by it (diagnostic, requires
            ``velocity="lu"``).
        convection: take ``H_hat`` from the velocity block of the
            preconditioning matrix (including convection by
            ``theta v - (u - u_old)/dt``), so that ``P_vp`` is built
            algebraically; otherwise assemble ``H_hat`` without convection
            from forms. With exact subsolves convection reduces the FGMRES
            iterations at developed coarse-mesh states from 5.5 to 3; with
            BoomerAMG and ``selfp`` it makes no difference, but the algebraic
            construction is 3x cheaper to assemble.
    """
    variant: str = "auxiliary"
    displacement: str = "block_triangular"
    displacement_fluid: str = "cholesky"
    momentum: str = "schur"
    velocity: str = "hypre"
    pressure: str = "selfp"
    convection: bool = True

    def __post_init__(self):
        choices = {
            "variant": FIELDSPLIT_VARIANTS,
            "displacement": ("lu", "block_triangular"),
            "displacement_fluid": ("cholesky", "amg", "gamg"),
            "momentum": ("lu", "schur"),
            "velocity": ("lu", "hypre", "gamg"),
            "pressure": ("cahouet_chabard", "selfp", "accurate"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"Unknown {name} {getattr(self, name)!r}, expected one of {allowed}")
        if self.pressure == "accurate" and self.velocity != "lu":
            raise ValueError("pressure='accurate' requires velocity='lu'")


@contextmanager
def temporary_options(prefix: str, options: dict):
    """Insert ``options`` under ``prefix`` into the PETSc options database for the duration of the block.

    Full option names are used rather than ``prefixPush``, because clearing an
    option does not apply the pushed prefix.
    """
    opts = PETSc.Options()
    try:
        for key, value in options.items():
            opts[f"{prefix}{key}"] = value
        yield
    finally:
        for key in options:
            del opts[f"{prefix}{key}"]


def _mumps_lu(prefix: str) -> dict:
    return {
        f"{prefix}ksp_type": "preonly",
        f"{prefix}pc_type": "lu",
        f"{prefix}pc_factor_mat_solver_type": "mumps",
        f"{prefix}mat_mumps_icntl_14": 80,
    }


def _cholesky(prefix: str) -> dict:
    return {
        f"{prefix}ksp_type": "preonly",
        f"{prefix}pc_type": "cholesky",
        f"{prefix}pc_factor_mat_solver_type": "mumps",
    }


def _dense_lu(prefix: str) -> dict:
    # The explicitly computed Schur complement is a dense matrix. MUMPS does not
    # factor it (INFOG(1)=-3 in the solve phase) and ScaLAPACK does not accept
    # MPIDENSE, so it is gathered onto every rank and factored there.
    return {
        f"{prefix}ksp_type": "preonly",
        f"{prefix}pc_type": "redundant",
        f"{prefix}redundant_pc_type": "lu",
        f"{prefix}redundant_pc_factor_mat_solver_type": "petsc",
    }


def _preonly_python(prefix: str) -> dict:
    # The Python context is attached after the outer setup
    return {f"{prefix}ksp_type": "preonly", f"{prefix}pc_type": "none"}


def _hypre(prefix: str) -> dict:
    return {
        f"{prefix}ksp_type": "preonly",
        f"{prefix}pc_type": "hypre",
        f"{prefix}pc_hypre_type": "boomeramg",
    }


def _gamg(prefix: str) -> dict:
    return {
        f"{prefix}ksp_type": "preonly",
        f"{prefix}pc_type": "gamg",
        f"{prefix}pc_gamg_threshold": 0.01,
        f"{prefix}mg_levels_ksp_type": "chebyshev",
        f"{prefix}mg_levels_pc_type": "jacobi",
    }


def _hypre_cg(prefix: str, rtol: float) -> dict:
    return _hypre(prefix) | {
        f"{prefix}ksp_type": "cg",
        f"{prefix}ksp_rtol": rtol,
        f"{prefix}ksp_max_it": 200,
        f"{prefix}ksp_norm_type": "unpreconditioned",
    }


def _gamg_cg(prefix: str, rtol: float) -> dict:
    return _gamg(prefix) | {
        f"{prefix}ksp_type": "cg",
        f"{prefix}ksp_rtol": rtol,
        f"{prefix}ksp_max_it": 200,
        f"{prefix}ksp_norm_type": "unpreconditioned",
    }


def rigid_body_modes(space: dfx.fem.FunctionSpace, template: PETSc.Vec,
                     rows: np.ndarray | None = None) -> PETSc.NullSpace:
    """Orthonormalized 2D rigid body modes of the owned DOFs of the vector space ``space``.

    Entry ``j`` of ``template`` holds the block-expanded owned DOF ``rows[j]``
    (default: all owned DOFs in local order).
    """
    bs = space.dofmap.index_map_bs
    if bs != 2:
        raise NotImplementedError("Rigid body modes are implemented for 2D vector spaces")
    n_owned = space.dofmap.index_map.size_local
    x = space.tabulate_dof_coordinates()[:n_owned]
    expanded = np.repeat(x[:, :2], bs, axis=0)
    component = np.tile(np.arange(bs), n_owned)
    modes = [(component == 0).astype(float), (component == 1).astype(float),
             np.where(component == 0, -expanded[:, 1], expanded[:, 0])]
    rows = np.arange(n_owned * bs) if rows is None else rows
    vectors = []
    for mode in modes:
        vec = template.duplicate()
        vec.array[:] = mode[rows]
        vectors.append(vec)
    for i, vec in enumerate(vectors):
        for prev in vectors[:i]:
            vec.axpy(-vec.dot(prev), prev)
        vec.normalize()
    return PETSc.NullSpace().create(vectors=vectors, comm=template.comm)


def _mass_solver(prefix: str, rtol: float = 1e-6) -> dict:
    # Diagonally scaled P2/P1 mass matrices are well conditioned
    return {
        f"{prefix}ksp_type": "cg",
        f"{prefix}pc_type": "jacobi",
        f"{prefix}ksp_rtol": rtol,
        f"{prefix}ksp_max_it": 100,
        f"{prefix}ksp_norm_type": "unpreconditioned",
    }


def configure_schur_fieldsplit(ksp: PETSc.KSP, is_u: PETSc.IS, is_vp: PETSc.IS, config: FieldSplitConfig,
                               P_vp: PETSc.Mat | None = None) -> dict:
    """Configure ``ksp``'s PC as the outer ``u | (v,p)`` full Schur factorization.

    Returns the sub-solver options (relative to the KSP's prefix) that must be
    in the options database when the PC is first set up, see
    :func:`setup_with_options`. Sub-solvers implemented as Python PCs are
    created with ``pc_type=none`` and must be attached after that setup.
    """
    if isinstance(config, str):
        config = FieldSplitConfig(variant=config)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.FIELDSPLIT)
    pc.setFieldSplitIS(("u", is_u), ("vp", is_vp))
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.FieldSplitSchurFactType.FULL)
    if config.variant == "exact":
        pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.FULL)
    else:
        if P_vp is None:
            raise ValueError("The auxiliary variant requires the assembled P_vp")
        pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, P_vp)

    # Every block of the factorization, including the off-diagonal coupling,
    # comes from the preconditioning matrix, never from the Newton operator.
    options = {
        "pc_fieldsplit_diag_use_amat": False,
        "pc_fieldsplit_off_diag_use_amat": False,
    }
    if config.variant == "exact":
        options |= _mumps_lu("fieldsplit_u_") | _dense_lu("fieldsplit_vp_")
        return options

    if config.displacement == "lu":
        options |= _mumps_lu("fieldsplit_u_")
    else:
        options |= _preonly_python("fieldsplit_u_")
        # The solid mass block is constant and factored once. The fluid-interior
        # rows are scaled by alpha and invisible in the outer residual norm, so
        # the harmonic extension is solved exactly or to a tolerance.
        options |= _cholesky("fieldsplit_u_solid_")
        if config.displacement_fluid == "cholesky":
            options |= _cholesky("fieldsplit_u_fluid_")
        elif config.displacement_fluid == "amg":
            options |= _hypre_cg("fieldsplit_u_fluid_", rtol=1e-8)
        else:
            options |= _gamg_cg("fieldsplit_u_fluid_", rtol=1e-8)

    if config.momentum == "lu":
        options |= _mumps_lu("fieldsplit_vp_")
    else:
        options |= _preonly_python("fieldsplit_vp_")
        # not "inner_": PETSc reserves fieldsplit_<split>_inner_ for the A00 solve
        # inside the Schur complement
        inner = "fieldsplit_vp_aux_"
        options |= {"lu": _mumps_lu, "hypre": _hypre, "gamg": _gamg}[config.velocity](f"{inner}fieldsplit_v_")
        if config.pressure == "selfp":
            options |= _hypre(f"{inner}fieldsplit_p_")
        elif config.pressure == "accurate":
            options |= {
                f"{inner}fieldsplit_p_ksp_type": "gmres",
                f"{inner}fieldsplit_p_ksp_rtol": 1e-10,
                f"{inner}fieldsplit_p_ksp_max_it": 500,
                f"{inner}fieldsplit_p_pc_type": "none",
            }
        else:
            options |= _preonly_python(f"{inner}fieldsplit_p_")
        options |= _mass_solver(f"{inner}pressure_mass_") | _hypre(f"{inner}pressure_laplace_")
    return options


def setup_with_options(ksp: PETSc.KSP, options: dict) -> None:
    """First setup of ``ksp`` (operators already set) with the sub-solver ``options``."""
    with temporary_options(ksp.getOptionsPrefix() or "", options):
        ksp.getPC().setFromOptions()
        ksp.setUp()


def _sub_ksp(prefix: str, comm) -> PETSc.KSP:
    ksp = PETSc.KSP().create(comm)
    ksp.setOptionsPrefix(prefix)
    ksp.setFromOptions()
    return ksp


def _local_positions(rows: np.ndarray, parent_rows: np.ndarray) -> np.ndarray:
    """Local position of each global row in ``rows`` within the sorted owned ``parent_rows``."""
    pos = np.searchsorted(parent_rows, rows)
    if np.any(pos >= parent_rows.size) or np.any(parent_rows[np.minimum(pos, parent_rows.size - 1)] != rows):
        raise ValueError("Rows are not owned rows of the parent")
    return pos


class AuxiliaryOperators:
    """Assembled auxiliary operators for the approximate subsolvers.

    ``P_vp`` and the pressure operators ``K_p``, ``M_p`` depend on the
    current state and are reassembled with every Jacobian; the solid mass
    matrix is constant and assembled once.

    With ``convection``, ``H_hat`` is the velocity block ``H`` of the
    preconditioning matrix plus ``theta dt E_s``, so ``P_vp`` is obtained
    algebraically: the ``(v,p)`` block of the preconditioning matrix is copied
    into ``P_vp`` and ``theta dt E_s``, an integral over the solid cells only,
    is assembled on top. Without convection, ``P_vp`` is assembled from forms.
    """

    def __init__(self, problem, config: FieldSplitConfig, jacobian_forms, P_mat: PETSc.Mat | None = None,
                 is_vp: PETSc.IS | None = None, prefix: str = "fsi2_harmonic_diffmesh_"):
        self.problem = problem
        self.config = config
        c = problem.constants
        u, v, _ = problem.solution
        entity_maps = problem.entity_maps
        dim = problem.mesh.geometry.dim
        F = ufl.Identity(dim) + ufl.grad(u)
        J = ufl.det(F)
        J_old = ufl.det(ufl.Identity(dim) + ufl.grad(problem.u_old))
        J_mid = 0.5 * (J + J_old)
        dx_fluid, dx_solid = problem.dx_fluid, problem.dx_solid

        # arguments shared with the Jacobian blocks (v,p) and (p,v)
        G_form, B_form = jacobian_forms[1][2], jacobian_forms[2][1]
        dv = G_form.arguments()[0]
        w = B_form.arguments()[1]

        # theta dt E_s: the solid momentum residual already carries one theta
        solid_momentum = restrict_to_cells(problem.residual[1], problem.dx_solid.subdomain_id())
        T_s = c["theta"] * c["dt"] * nonzero(ufl.derivative(solid_momentum, u, w))
        self.vp_bcs = problem.bcs_v
        P = problem.P
        p_trial, q = ufl.TrialFunction(P), ufl.TestFunction(P)

        if config.convection:
            if P_mat is None or is_vp is None:
                raise ValueError("The algebraic P_vp requires the preconditioning matrix and the (v,p) index set")
            self._P_mat, self._is_vp = P_mat, is_vp
            self._P_mat_vp = None
            # A structurally present, zero (p,p) block keeps the p rows of the
            # added form nonempty; the pattern of the (v,p) block of P_mat is a
            # subset of the resulting pattern.
            zero = dfx.fem.Constant(problem.mesh, 0.0)
            Z_pp = zero * p_trial * q * dx_fluid
            self.T_forms = dfx.fem.form([[T_s, None], [None, Z_pp]], entity_maps=entity_maps)
            # Assembled once to fix the sparsity pattern (zeroEntries keeps it)
            pattern_forms = dfx.fem.form([[jacobian_forms[1][1], G_form], [B_form, Z_pp]], entity_maps=entity_maps)
            self.P_vp = dfx.fem.petsc.assemble_matrix(pattern_forms, bcs=self.vp_bcs, diag=1.0)
            self.P_vp.assemble()
        else:
            H_hat = c["rho_f"] * J_mid / c["dt"] * ufl.inner(w, dv) * dx_fluid
            H_hat += c["theta"] * ufl.inner(J * Fluid.NS_velocity(u, w, c["nu_f"], c["rho_f"]) * ufl.inv(F).T,
                                            ufl.grad(dv)) * dx_fluid
            H_hat += c["rho_s"] / c["dt"] * ufl.inner(w, dv) * dx_solid
            H_hat += T_s
            self.P_vp_forms = dfx.fem.form([[H_hat, G_form], [B_form, None]], entity_maps=entity_maps)
            self.P_vp = dfx.fem.petsc.create_matrix(self.P_vp_forms)
        self.P_vp.setOptionsPrefix(f"{prefix}P_vp_")

        # Unsteady Stokes pressure Schur complement approximation on the current
        # (ALE) configuration. B M^{-1} B^T with mass density J_mid and the
        # divergence carrying J gives the Laplacian weight J^2 / J_mid.
        Finv = ufl.inv(F)
        K_p = J**2 / J_mid * ufl.inner(ufl.dot(Finv.T, ufl.grad(p_trial)), ufl.dot(Finv.T, ufl.grad(q))) * dx_fluid
        M_p = J * p_trial * q * dx_fluid
        self.K_p_form = dfx.fem.form(K_p, entity_maps=entity_maps)
        self.M_p_form = dfx.fem.form(M_p, entity_maps=entity_maps)
        self.K_p = dfx.fem.petsc.create_matrix(self.K_p_form)
        self.M_p = dfx.fem.petsc.create_matrix(self.M_p_form)
        self.pressure_bcs = [self._outflow_pressure_bc()]
        self.pressure_coefficients = (
            float(c["rho_f"].value / c["dt"].value),
            float(c["theta"].value * c["rho_f"].value * c["nu_f"].value),
        )

        # constant solid mass for the solid/interface displacement block
        U = problem.U
        u_trial, du = ufl.TrialFunction(U), ufl.TestFunction(U)
        mass = dfx.fem.form(c["rho_s"] / c["dt"] * ufl.inner(u_trial, du) * dx_solid)
        self.u_bcs = problem.bcs_u
        self.M_u = dfx.fem.petsc.assemble_matrix(mass, bcs=self.u_bcs, diag=1.0)
        self.M_u.assemble()

        self.assemblies = 0

    def _outflow_pressure_bc(self):
        """Homogeneous Dirichlet condition for ``K_p`` at the do-nothing outflow."""
        from xfsi_solver.solvers.fsi2_harmonic_diffmesh import PHYSICAL_MARKERS

        problem = self.problem
        mesh = problem.mesh
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        f2v = mesh.topology.connectivity(tdim - 1, 0)
        facets = problem.facet_tags.find(PHYSICAL_MARKERS["outflow"])
        parent_vertices = np.unique(np.concatenate([f2v.links(f) for f in facets])) if facets.size else \
            np.empty(0, dtype=np.int32)
        sub_vertices = np.asarray(problem.fluid_vertex_map.sub_topology_to_topology(parent_vertices, True),
                                  dtype=np.int32)
        sub_vertices = sub_vertices[sub_vertices >= 0]
        problem.fluid_mesh.topology.create_connectivity(0, tdim)
        dofs = dfx.fem.locate_dofs_topological(problem.P, 0, sub_vertices)
        if problem.mesh.comm.allreduce(len(dofs), op=MPI.SUM) == 0:
            raise RuntimeError("No pressure DOFs found on the outflow boundary")
        return dfx.fem.dirichletbc(0.0, dofs, problem.P)

    def assemble(self):
        if self.config.convection:
            self._P_mat_vp = self._P_mat.createSubMatrix(self._is_vp, self._is_vp, submat=self._P_mat_vp)
            self.P_vp.zeroEntries()
            self.P_vp.axpy(1.0, self._P_mat_vp, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
            # Adds theta dt E_s. DOLFINx inserts (does not add) the diagonal of
            # Dirichlet rows, so it must be the unit diagonal of P_mat.
            dfx.fem.petsc.assemble_matrix(self.P_vp, self.T_forms, bcs=self.vp_bcs, diag=1.0)
            self.P_vp.assemble()
        else:
            self.P_vp.zeroEntries()
            dfx.fem.petsc.assemble_matrix(self.P_vp, self.P_vp_forms, bcs=self.vp_bcs, diag=1.0)
            self.P_vp.assemble()
        # M_p: user Schur preconditioning matrix of the inner split and part of
        # Cahouet-Chabard; K_p: Cahouet-Chabard only
        pressure = []
        if self.config.momentum == "schur" and self.config.pressure != "selfp":
            pressure.append((self.M_p, self.M_p_form, []))
            pressure.append((self.K_p, self.K_p_form, self.pressure_bcs))
        for A, form, bcs in pressure:
            A.zeroEntries()
            dfx.fem.petsc.assemble_matrix(A, form, bcs=bcs, diag=1.0)
            A.assemble()
        self.assemblies += 1

    def destroy(self):
        for A in (self.P_vp, self.K_p, self.M_p, self.M_u, getattr(self, "_P_mat_vp", None)):
            if A is not None:
                A.destroy()


class Statistics(dict):
    """Counters of a preconditioner context, reset per time step by :class:`FieldSplitSolver`."""

    def __init__(self, *names):
        super().__init__({name: 0 for name in names})

    def reset(self):
        for name in self:
            self[name] = 0


class DisplacementPC:
    """Block lower-triangular approximate inverse of the displacement block ``A``.

    The displacement DOFs are split into ``I``, all DOFs of solid cells
    (including the shared interface, one copy each), and ``f``, the fluid
    interior. The inverse solves ``M_II y_I = r_I`` with the constant solid
    mass matrix (factored once), then ``A_ff y_f = r_f - A_fI y_I``. ``A_If`` and the
    fluid-mesh contributions to ``A_II`` are dropped here only; they scale
    with the mesh coefficient ``alpha`` and remain in the Jacobian. The
    fluid block is used as assembled: a factorization is unaffected by the
    ``alpha`` scaling, and BoomerAMG's coarsening, interpolation and
    smoothers are invariant under the row scaling that distinguishes the
    ``alpha``-scaled mesh rows from the unit Dirichlet rows, so no
    normalization is applied. ``A_ff`` and ``A_fI`` are whatever mesh
    extension the preconditioning matrix contains (harmonic, or the
    stiffened elastic extension with its weights); ``A_ff`` is symmetric
    (the interface flux only enters interface rows, which belong to ``I``).
    """

    def __init__(self, problem, aux: AuxiliaryOperators, u_rows: np.ndarray, is_u: PETSc.IS, prefix: str):
        U = problem.U
        bs = U.dofmap.index_map_bs
        is_solid = problem.solid_displacement_dofs()

        # u-split numbering, ordered by position in the u split
        u_parent = is_u.getIndices()
        positions = _local_positions(u_rows, u_parent)
        offset = is_u.comm.tompi4py().exscan(u_parent.size, op=MPI.SUM) or 0
        order_I = np.flatnonzero(is_solid)[np.argsort(positions[is_solid])]
        order_f = np.flatnonzero(~is_solid)[np.argsort(positions[~is_solid])]
        comm = is_u.comm
        self.is_I = PETSc.IS().createGeneral((offset + positions[order_I]).astype(PETSc.IntType), comm=comm)
        self.is_f = PETSc.IS().createGeneral((offset + positions[order_f]).astype(PETSc.IntType), comm=comm)
        self.is_f.setBlockSize(bs)
        # owned block-expanded displacement DOF of every entry of is_f, in order
        self.fluid_dofs = order_f
        self.space = U
        self.near_nullspace = aux.config.displacement_fluid == "gamg"

        # the same DOFs, in the same order, in the numbering of the solid mass matrix
        mass_rows = field_dof_rows(aux.M_u, [U])[0]
        is_mass = PETSc.IS().createGeneral(mass_rows[order_I].astype(PETSc.IntType), comm=comm)
        self.M_II = aux.M_u.createSubMatrix(is_mass, is_mass)
        is_mass.destroy()

        self.ksp_solid = _sub_ksp(f"{prefix}solid_", comm)
        self.ksp_solid.setOperators(self.M_II)
        self.ksp_fluid = _sub_ksp(f"{prefix}fluid_", comm)
        self.A_ff = None
        self.A_fI = None
        self.statistics = Statistics("applications", "solid_iterations", "fluid_iterations", "fluid_setups",
                                     "solid_time", "fluid_time", "time")

    def setUp(self, pc):
        _, P = pc.getOperators()
        # The fluid-interior rows are the linear mesh equation on the reference
        # configuration, so A_ff does not change between Jacobians unless the
        # formulation does; its AMG hierarchy is rebuilt only when it changes.
        A_ff_new = P.createSubMatrix(self.is_f, self.is_f)
        rebuild = self.A_ff is None
        if not rebuild:
            A_ff_new.axpy(-1.0, self.A_ff)
            rebuild = A_ff_new.norm() > 1e-14 * self.A_ff.norm()
            A_ff_new.destroy()
        if rebuild:
            if self.A_ff is not None:
                self.A_ff.destroy()
            self.A_ff = P.createSubMatrix(self.is_f, self.is_f)
            if self.near_nullspace:
                self.A_ff.setNearNullSpace(rigid_body_modes(self.space, self.A_ff.createVecLeft(),
                                                            rows=self.fluid_dofs))
            self.ksp_fluid.setOperators(self.A_ff)
            if self.A_ff.getSize()[0] > 0:
                self.ksp_fluid.setUp()
            self.statistics["fluid_setups"] += 1
        self.A_fI = P.createSubMatrix(self.is_f, self.is_I, submat=self.A_fI)
        self.ksp_solid.setUp()

    def apply(self, pc, x, y):
        start = timer()
        x_I, x_f = x.getSubVector(self.is_I), x.getSubVector(self.is_f)
        y_I, y_f = y.getSubVector(self.is_I), y.getSubVector(self.is_f)
        t0 = timer()
        self.ksp_solid.solve(x_I, y_I)
        t1 = timer()
        r_f = x_f.copy()
        self.A_fI.mult(y_I, y_f)
        r_f.axpy(-1.0, y_f)
        self.ksp_fluid.solve(r_f, y_f)
        t2 = timer()
        r_f.destroy()
        x.restoreSubVector(self.is_I, x_I)
        x.restoreSubVector(self.is_f, x_f)
        y.restoreSubVector(self.is_I, y_I)
        y.restoreSubVector(self.is_f, y_f)
        stats = self.statistics
        stats["applications"] += 1
        stats["solid_iterations"] += self.ksp_solid.getIterationNumber()
        stats["fluid_iterations"] += self.ksp_fluid.getIterationNumber()
        stats["solid_time"] += t1 - t0
        stats["fluid_time"] += t2 - t1
        stats["time"] += timer() - start

    def view(self, pc, viewer):
        viewer.printfASCII("Block triangular displacement PC: solid/interface mass, then fluid interior\n")
        self.ksp_solid.view(viewer)
        self.ksp_fluid.view(viewer)

    def destroy(self, pc=None):
        for obj in (self.ksp_solid, self.ksp_fluid, self.M_II, self.A_ff, self.A_fI, self.is_I, self.is_f):
            if obj is not None:
                obj.destroy()


class PressureSchurPC:
    """``S_p^{-1} ~ rho_f/dt K_p^{-1} + theta mu_f M_p^{-1}`` (a sum of inverse actions).

    ``K_p`` is the ALE-metric pressure Laplacian with a homogeneous Dirichlet
    condition at the do-nothing outflow and natural conditions elsewhere,
    including the moving interface (which omits the structural impedance).
    ``M_p`` is the pressure mass matrix on the current configuration. Both
    are assembled on the pressure space, whose numbering is related to the
    pressure split by ``order`` (split entry ``j`` is pressure DOF ``order[j]``).
    """

    def __init__(self, aux: AuxiliaryOperators, order: np.ndarray, prefix: str, comm):
        self.aux = aux
        self.order = order
        self.ksp_K = _sub_ksp(f"{prefix}pressure_laplace_", comm)
        self.ksp_M = _sub_ksp(f"{prefix}pressure_mass_", comm)
        self.ksp_K.setOperators(aux.K_p)
        self.ksp_M.setOperators(aux.M_p)
        self.z, self.a = aux.K_p.createVecs()
        self.b = self.a.duplicate()

    def setUp(self, pc):
        self.ksp_K.setUp()
        self.ksp_M.setUp()

    def apply(self, pc, x, y):
        c_K, c_M = self.aux.pressure_coefficients
        self.z.array[self.order] = x.array_r
        self.ksp_K.solve(self.z, self.a)
        self.ksp_M.solve(self.z, self.b)
        self.a.scale(c_K)
        self.a.axpy(c_M, self.b)
        y.array[:] = self.a.array_r[self.order]

    def view(self, pc, viewer):
        viewer.printfASCII("Cahouet-Chabard pressure Schur PC: rho/dt K_p^{-1} + theta mu M_p^{-1}\n")
        self.ksp_K.view(viewer)
        self.ksp_M.view(viewer)

    def destroy(self, pc=None):
        for obj in (self.ksp_K, self.ksp_M, self.z, self.a, self.b):
            obj.destroy()


class MomentumPressurePC:
    """Inner full Schur factorization ``v | p`` of the assembled ``P_vp``.

    The outer Schur KSP passes vectors in the numbering of the ``(v,p)``
    split of the Jacobian, which :class:`FieldSplitSolver` checks to coincide
    with the numbering of ``P_vp``.
    """

    def __init__(self, problem, aux: AuxiliaryOperators, config: FieldSplitConfig, prefix: str):
        self.aux = aux
        self.config = config
        self.prefix = f"{prefix}aux_"
        P_vp = aux.P_vp
        self.is_v, self.is_p = field_index_sets(P_vp, [problem.V, problem.P])
        v_rows, p_rows = field_dof_rows(P_vp, [problem.V, problem.P])
        self.v_order = np.argsort(v_rows)
        self.p_order = np.argsort(p_rows)
        self.problem = problem
        self.inner = None
        self.pressure_pc = None
        self.statistics = Statistics("applications", "time")

    def _rigid_body_modes(self, template: PETSc.Vec) -> PETSc.NullSpace:
        V = self.problem.V
        n_owned = V.dofmap.index_map.size_local
        x = V.tabulate_dof_coordinates()[:n_owned]
        modes = []
        for values in (np.array([1.0, 0.0]) + 0 * x[:, :2], np.array([0.0, 1.0]) + 0 * x[:, :2],
                       np.column_stack((-x[:, 1], x[:, 0]))):
            vec = template.duplicate()
            vec.array[:] = values.reshape(-1)[self.v_order]
            modes.append(vec)
        # orthonormalize for GAMG
        for i, mode in enumerate(modes):
            for prev in modes[:i]:
                mode.axpy(-mode.dot(prev), prev)
            mode.normalize()
        return PETSc.NullSpace().create(vectors=modes, comm=template.comm)

    def setUp(self, pc):
        _, P = pc.getOperators()
        if self.inner is None:
            inner = PETSc.PC().create(pc.comm)
            inner.setOptionsPrefix(self.prefix)
            inner.setOperators(P, P)
            inner.setType(PETSc.PC.Type.FIELDSPLIT)
            inner.setFieldSplitIS(("v", self.is_v), ("p", self.is_p))
            inner.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
            inner.setFieldSplitSchurFactType(PETSc.PC.FieldSplitSchurFactType.FULL)
            if self.config.pressure == "selfp":
                inner.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.SELFP)
            else:
                inner.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, self.aux.M_p)
            inner.setFromOptions()
            inner.setUp()
            ksp_v, ksp_p = inner.getFieldSplitSubKSP()
            if self.config.velocity == "gamg":
                H = ksp_v.getOperators()[1]
                H.setNearNullSpace(self._rigid_body_modes(H.createVecLeft()))
            if self.config.pressure != "selfp":
                self.pressure_pc = PressureSchurPC(self.aux, self.p_order, self.prefix, pc.comm)
                p_pc = ksp_p.getPC()
                p_pc.setType(PETSc.PC.Type.PYTHON)
                p_pc.setPythonContext(self.pressure_pc)
            self.inner = inner
        else:
            self.inner.setOperators(P, P)
            self.inner.setUp()

    def apply(self, pc, x, y):
        start = timer()
        self.inner.apply(x, y)
        self.statistics["applications"] += 1
        self.statistics["time"] += timer() - start

    def view(self, pc, viewer):
        viewer.printfASCII("Inner v|p Schur fieldsplit on the assembled P_vp\n")
        if self.inner is not None:
            self.inner.view(viewer)

    def destroy(self, pc=None):
        for obj in (self.inner, self.is_v, self.is_p):
            if obj is not None:
                obj.destroy()
        if self.pressure_pc is not None:
            self.pressure_pc.destroy()


class InstrumentedSolver:
    """Instruments the linear solver of a ``NonlinearProblem``.

    Replaces the SNES Jacobian callback by one that calls
    ``dolfinx.fem.petsc.assemble_jacobian`` and then sets up the KSP, so that
    Jacobian assembly, auxiliary assembly, preconditioner setup (e.g. the LU
    factorization) and the Krylov solve are timed separately; SNES's own
    ``KSPSetOperators`` with the same, unchanged matrices does not repeat the
    setup. Every linear solve records its iterations and the true residual
    ``b - A x`` with the Newton operator ``A``, in total and per diagnostic
    field. The matrices, vectors and SNES remain owned by the
    ``NonlinearProblem``.
    """

    def __init__(self, nonlinear_problem, problem):
        from xfsi_solver.solvers.fsi2_harmonic_diffmesh import diagnostic_index_sets

        self.nonlinear_problem = nonlinear_problem
        self.problem = problem
        A = nonlinear_problem.A
        self.diagnostic_is = diagnostic_index_sets(A, problem)
        self.snes = nonlinear_problem.solver
        self.ksp = self.snes.getKSP()
        self.prefix = self.ksp.getOptionsPrefix()
        self.contexts = []

        self._jacobian_ctx = {
            "u": problem.solution,
            "jacobian": nonlinear_problem.J,
            "preconditioner": nonlinear_problem.preconditioner,
            "bcs": problem.bcs,
        }
        P_mat = nonlinear_problem.P_mat if nonlinear_problem.P_mat is not None else A
        self.snes.setJacobian(self._assemble_jacobian, A, P_mat)
        self.ksp.setPreSolve(self._pre_solve)
        self.ksp.setPostSolve(self._post_solve)
        self.reset_statistics()

    def reset_statistics(self):
        self.linear_solves = []
        self.timings = {"jacobian": 0.0, "auxiliary": 0.0, "setup": 0.0, "linear_solve": 0.0}
        for ctx in self.contexts:
            ctx.statistics.reset()

    def context_statistics(self) -> dict:
        """Counters of the Python PC contexts, keyed by context class."""
        return {type(ctx).__name__: dict(ctx.statistics) for ctx in self.contexts}

    def _assemble_auxiliary(self):
        pass

    def _setup(self):
        self.ksp.setUp()

    def _assemble_jacobian(self, snes, x, J, P):
        start = timer()
        dfx.fem.petsc.assemble_jacobian(snes, x, J, P, **self._jacobian_ctx)
        self.timings["jacobian"] += timer() - start

        start = timer()
        self._assemble_auxiliary()
        self.timings["auxiliary"] += timer() - start

        start = timer()
        self.ksp.setOperators(J, P)
        self._setup()
        self.timings["setup"] += timer() - start

    def _pre_solve(self, ksp, b, x):
        self._solve_start = timer()

    def _post_solve(self, ksp, b, x):
        self.timings["linear_solve"] += timer() - self._solve_start
        A, _ = ksp.getOperators()
        r = b.duplicate()
        A.mult(x, r)
        r.aypx(-1.0, b)
        b_norm = b.norm()
        self.linear_solves.append({
            "iterations": ksp.getIterationNumber(),
            "reason": ksp.getConvergedReason(),
            "true_relative_residual": r.norm() / b_norm if b_norm > 0 else r.norm(),
            "field_true_residuals": field_norms(r, self.diagnostic_is),
            "field_rhs": field_norms(b, self.diagnostic_is),
        })
        r.destroy()

    def view(self, viewer=None):
        self.ksp.view(viewer)

    def destroy(self):
        for index_set in self.diagnostic_is:
            index_set.destroy()


class FieldSplitSolver(InstrumentedSolver):
    """Configures the linear solver of a ``NonlinearProblem`` as FGMRES + Schur fieldsplit.

    Owns the index sets, the auxiliary operators and the Python PC contexts
    it creates.
    """

    def __init__(self, nonlinear_problem, problem, config, preconditioner_forms):
        """``preconditioner_forms``: the UFL block forms of the preconditioning matrix (of the Jacobian if none)."""
        super().__init__(nonlinear_problem, problem)
        self.config = config
        fs_config = config.fieldsplit

        A = nonlinear_problem.A
        spaces = [problem.U, problem.V, problem.P]
        self.field_is = field_index_sets(A, spaces)
        self.is_u = self.field_is[0]
        self.is_vp = union_index_set(self.field_is[1:])
        self.vp_field_is = nested_index_sets(self.is_vp, self.field_is[1:])

        self.ksp.setType(PETSc.KSP.Type.FGMRES)
        self.ksp.setTolerances(rtol=config.ksp_rtol, atol=config.ksp_atol, max_it=config.ksp_max_it)
        self.ksp.setGMRESRestart(config.ksp_restart)
        self.ksp.setErrorIfNotConverged(True)

        self.aux = None
        if fs_config.variant == "auxiliary":
            P_mat = nonlinear_problem.P_mat if nonlinear_problem.P_mat is not None else A
            self.aux = AuxiliaryOperators(problem, fs_config, preconditioner_forms, P_mat, self.is_vp,
                                          prefix=self.prefix)
            self._check_vp_layout(A, spaces)
        self._options = configure_schur_fieldsplit(self.ksp, self.is_u, self.is_vp, fs_config,
                                                   None if self.aux is None else self.aux.P_vp)
        self._u_rows = field_dof_rows(A, spaces)[0]

        # Nested sub-solvers are set up lazily (e.g. the GAMG levels at the first
        # application) and read their options then, so the options stay in the
        # database until destroy(). They are set explicitly here, overriding any
        # left by an earlier solver with the same prefix.
        self._full_options = {f"{self.prefix}{k}": v for k, v in self._options.items()}
        opts = PETSc.Options()
        for key, value in self._full_options.items():
            opts[key] = value
        self._configured = False

    def _check_vp_layout(self, A, spaces):
        """The ``(v,p)`` split of ``A`` and ``P_vp`` must number the local DOFs identically."""
        J_rows = field_dof_rows(A, spaces)[1:]
        vp_parent = self.is_vp.getIndices()
        P_rows = field_dof_rows(self.aux.P_vp, spaces[1:])
        P_start = self.aux.P_vp.getOwnershipRange()[0]
        for J_field, P_field in zip(J_rows, P_rows, strict=True):
            if not np.array_equal(_local_positions(J_field, vp_parent), P_field - P_start):
                raise NotImplementedError("P_vp and the (v,p) split of the Jacobian number the DOFs differently")

    def _attach_python_contexts(self):
        fs_config = self.config.fieldsplit
        if fs_config.variant != "auxiliary":
            return
        ksp_u, ksp_vp = self.ksp.getPC().getFieldSplitSubKSP()
        if fs_config.displacement == "block_triangular":
            ctx = DisplacementPC(self.problem, self.aux, self._u_rows, self.is_u, f"{self.prefix}fieldsplit_u_")
            self._set_python_pc(ksp_u, ctx)
        if fs_config.momentum == "schur":
            ctx = MomentumPressurePC(self.problem, self.aux, fs_config, f"{self.prefix}fieldsplit_vp_")
            self._set_python_pc(ksp_vp, ctx)

    def _check_hierarchy(self):
        """Fail if options from the database replaced the configured outer solver."""
        pc = self.ksp.getPC()
        if self.ksp.getType() != PETSc.KSP.Type.FGMRES or pc.getType() != PETSc.PC.Type.FIELDSPLIT:
            raise RuntimeError(f"Expected FGMRES/fieldsplit, found {self.ksp.getType()}/{pc.getType()}")
        prefixes = [ksp.getOptionsPrefix() for ksp in pc.getFieldSplitSubKSP()]
        if prefixes != [f"{self.prefix}fieldsplit_u_", f"{self.prefix}fieldsplit_vp_"]:
            raise RuntimeError(f"Unexpected field split prefixes {prefixes}")

    def _set_python_pc(self, ksp, ctx):
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(ctx)
        self.contexts.append(ctx)

    def _assemble_auxiliary(self):
        if self.aux is not None:
            self.aux.assemble()

    def _setup(self):
        if not self._configured:
            self.ksp.getPC().setFromOptions()
            self.ksp.setUp()
            self._check_hierarchy()
            self._attach_python_contexts()
            self._configured = True
        self.ksp.setUp()

    def destroy(self):
        opts = PETSc.Options()
        for key in self._full_options:
            if opts.hasName(key):
                del opts[key]
        for ctx in self.contexts:
            ctx.destroy()
        if self.aux is not None:
            self.aux.destroy()
        for index_set in (*self.field_is, self.is_vp, *self.vp_field_is):
            index_set.destroy()
        super().destroy()
