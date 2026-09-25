# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Mesh-extension operators of the shared-space ALE FSI solvers.

The displacement ``u`` is one field on the whole mesh. On solid cells it is
the structural displacement; on fluid cells it is the ALE mesh displacement,
an extension of the interface displacement into the fluid domain. The mesh
equation contributes to the displacement residual

    R_m(u; phi) = alpha [ int_{Omega_f^0} sigma_m(u) : grad(phi) dX
                          - int_{Gamma_fs^0} (sigma_m(u) n_f) . phi dS ],

with the harmonic (``sigma_m = grad u``) or the stiffened linear-elastic
mesh stress. The interface term uses the fluid-side trace of the stress and
the outward normal of the fluid cell, and cancels the fluid-cell volume
contribution to the interface rows up to the strong-form residual, so that
the mesh equation does not act as a load on the solid kinematic equation.
All derivatives are with respect to the reference coordinates ``X``, and
the mesh geometry is never updated (non-incremental extension).

Stiffened elasticity (Shamanskiy and Simeon, Comput. Mech. 67, 2021,
sections 3.3 and 3.5) uses

    sigma_m(u) = w (2 mu_0 eps(u) + lambda_0 tr(eps(u)) I),
    w = (j_star / j_0)^chi,    j_0 = |det D_xi G_0|,

where ``G_0`` maps the parent cell to the initial physical cell, ``mu_0``,
``lambda_0`` are the Lame parameters of modulus ``E_0`` and Poisson ratio
``nu_m``, and ``j_star`` is a global normalization: by default the mean
initial fluid-cell volume divided by the parent-cell volume, so that
``w = 1`` in an affine cell of mean size. ``j_0`` is evaluated pointwise, as
the UFL ``JacobianDeterminant`` of the unchanged mesh, also on the curved
cells of quadratic geometry. The weight does not depend on ``u``, so the
mesh equation stays linear in ``u``.
"""

import math
from dataclasses import asdict, dataclass

import dolfinx as dfx
import numpy as np
import ufl
from mpi4py import MPI

# Volume of the DOLFINx/Basix parent cells
PARENT_CELL_VOLUME = {"interval": 1.0, "triangle": 0.5, "quadrilateral": 1.0, "tetrahedron": 1.0 / 6.0,
                      "hexahedron": 1.0}

# Default number of subintervals per parent-cell edge of the sampling lattice
SAMPLE_RESOLUTION = 6

# Default quadrature degree of the stiffened elastic mesh forms. On the FSI2
# meshes, doubling it changes the assembled volume residual of a smooth
# displacement by 8e-10 (quadratic triangles, mesh_sec_coarse) and 9e-8
# (strongly non-affine quadratic quadrilaterals, mesh_quad_ssq_sec).
DEFAULT_QUADRATURE_DEGREE = {"triangle": 6, "quadrilateral": 10}


def sample_points(cell_name: str, n: int = SAMPLE_RESOLUTION) -> np.ndarray:
    """Lattice of parent-cell points with ``n`` subintervals per edge, vertices and edges included."""
    s = np.arange(n + 1) / n
    if cell_name == "triangle":
        return np.array([(a, b) for i, a in enumerate(s) for b in s[:n + 1 - i]])
    if cell_name == "quadrilateral":
        return np.array([(a, b) for a in s for b in s])
    raise NotImplementedError(f"Sampling points for {cell_name} cells")


def evaluate(expr, mesh: dfx.mesh.Mesh, cells: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Values of ``expr`` at ``points`` of each of ``cells``, shape ``(cells, points, *value_shape)``."""
    shape = expr.ufl_shape
    cells = np.asarray(cells, dtype=np.int32)
    expression = dfx.fem.Expression(expr, points, comm=mesh.comm)
    values = expression.eval(mesh, cells) if cells.size else np.empty((0, points.shape[0] * math.prod(shape)))
    return values.reshape(cells.size, points.shape[0], *shape)


def global_range(values: np.ndarray, comm) -> tuple[float, float, int]:
    """Global minimum, maximum and number of non-finite entries of the local ``values``."""
    values = np.asarray(values).ravel()
    finite = values[np.isfinite(values)]
    lo = comm.allreduce(float(finite.min()) if finite.size else np.inf, op=MPI.MIN)
    hi = comm.allreduce(float(finite.max()) if finite.size else -np.inf, op=MPI.MAX)
    bad = comm.allreduce(int(values.size - finite.size), op=MPI.SUM)
    return lo, hi, bad


class FluidReferenceGeometry:
    """The initial (reference) fluid domain on which the mesh equation is posed.

    Args:
        mesh: The whole fluid-solid mesh, never moved.
        cell_tags: Cell markers of ``mesh``.
        fluid_marker: Marker of the fluid cells.
        dx_fluid: Cell measure of the fluid cells.
        ds_interface_fluid: One-sided measure of the fluid-solid interface,
            integrating over each interface facet from its fluid cell.
    """

    def __init__(self, mesh, cell_tags, fluid_marker: int, dx_fluid: ufl.Measure,
                 ds_interface_fluid: ufl.Measure | None):
        self.mesh = mesh
        self.dx_fluid = dx_fluid
        self.ds_interface_fluid = ds_interface_fluid
        n_owned = mesh.topology.index_map(mesh.topology.dim).size_local
        fluid = cell_tags.find(fluid_marker)
        self.owned_fluid_cells = fluid[fluid < n_owned]
        self.cell_name = mesh.topology.cell_name()
        self.parent_volume = PARENT_CELL_VOLUME[self.cell_name]
        # positive volume factor of the parent-to-initial-cell map
        self.signed_jacobian_determinant = ufl.JacobianDeterminant(mesh)
        self.jacobian_determinant = abs(self.signed_jacobian_determinant)

    @property
    def comm(self):
        return self.mesh.comm

    def fluid_volume(self) -> float:
        local = dfx.fem.assemble_scalar(dfx.fem.form(ufl.as_ufl(1.0) * self.dx_fluid))
        return self.comm.allreduce(local, op=MPI.SUM)

    def num_fluid_cells(self) -> int:
        return self.comm.allreduce(int(self.owned_fluid_cells.size), op=MPI.SUM)

    def mean_parent_jacobian(self) -> float:
        """Mean initial fluid-cell volume divided by the parent-cell volume (owned cells only)."""
        n = self.num_fluid_cells()
        if n == 0:
            raise ValueError("The mesh has no fluid cells")
        return self.fluid_volume() / n / self.parent_volume

    def sample(self, expr, n: int = SAMPLE_RESOLUTION) -> np.ndarray:
        """``expr`` at the sampling lattice of every owned fluid cell."""
        return evaluate(expr, self.mesh, self.owned_fluid_cells, sample_points(self.cell_name, n))

    def check_reference_validity(self, n: int = SAMPLE_RESOLUTION) -> dict:
        """Sampled range of the parent-to-initial-cell Jacobian; raises for degenerate geometry.

        Orientation signs are allowed to differ between cells (they are a
        property of the cell numbering, not of the geometry), but not within
        a cell, and the volume factor must be bounded away from zero.
        """
        values = self.sample(self.signed_jacobian_determinant, n)
        lo, hi, bad = global_range(np.abs(values), self.comm)
        mixed = self.comm.allreduce(int(np.sum((values.min(axis=1) < 0) & (values.max(axis=1) > 0)))
                                    if values.size else 0, op=MPI.SUM)
        negative = self.comm.allreduce(int(np.sum(values.max(axis=1) < 0)) if values.size else 0, op=MPI.SUM)
        info = {"j0_min": lo, "j0_max": hi, "j0_nonfinite": bad, "cells_with_mixed_orientation": mixed,
                "negatively_oriented_cells": negative, "sample_resolution": n}
        if bad or mixed or not lo > 1e-12 * hi:
            raise ValueError(f"Degenerate reference fluid geometry: {info}")
        return info


class HarmonicMeshExtension:
    """Harmonic extension, ``sigma_m(u) = grad(u)``."""

    name = "harmonic"

    def bind(self, geometry: FluidReferenceGeometry) -> "HarmonicMeshOperator":
        return HarmonicMeshOperator(geometry)

    def parameters(self) -> dict:
        return {"mesh_extension": self.name}


class HarmonicMeshOperator:
    def __init__(self, geometry: FluidReferenceGeometry):
        self.geometry = geometry
        self.info = {"mesh_extension": "harmonic"}

    def residual(self, u, du, alpha) -> ufl.Form:
        g = self.geometry
        normal = ufl.FacetNormal(g.mesh)
        residual = ufl.inner(alpha * ufl.grad(u), ufl.grad(du)) * g.dx_fluid
        residual -= ufl.inner(alpha * ufl.grad(u) * normal, du) * g.ds_interface_fluid
        return residual


MESH_WEIGHTINGS = ("pointwise", "cell_volume")


@dataclass
class StiffenedElasticMeshExtension:
    """Reference-configuration linear elasticity with mesh-Jacobian stiffening.

    Attributes:
        stiffening_exponent: ``chi >= 0``; 0 gives unweighted elasticity.
        poisson_ratio: ``nu_m`` in ``[0, 0.49]``.
        modulus: ``E_0 > 0``, a dimensionless normalization, not a material
            parameter. The equation scale ``alpha`` is separate.
        quadrature_degree: Quadrature degree of the mesh-equation volume and
            interface integrals, ``None`` for ``DEFAULT_QUADRATURE_DEGREE`` of
            the cell type. The weight is a fractional power of the Jacobian
            determinant, which is not integrated exactly on curved cells; see
            ``tests/test_mesh_extension.py`` for the quadrature-increase check.
        weighting: ``"pointwise"`` (the default, ``j_0`` at every quadrature
            point) or ``"cell_volume"``: experimental, ``j_0`` replaced by the
            cell volume over the parent-cell volume, which is equivalent on
            affine cells only.
        j_star: Normalization of ``j_0``; ``None`` for the mean initial
            fluid-cell volume over the parent-cell volume.
    """
    stiffening_exponent: float = 2.5
    poisson_ratio: float = 0.3
    modulus: float = 1.0
    quadrature_degree: int | None = None
    weighting: str = "pointwise"
    j_star: float | None = None

    name = "stiffened_elastic"

    def __post_init__(self):
        for key in ("stiffening_exponent", "poisson_ratio", "modulus"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if self.stiffening_exponent < 0.0:
            raise ValueError("stiffening_exponent must be nonnegative")
        if not 0.0 <= self.poisson_ratio <= 0.49:
            raise ValueError("poisson_ratio must be in [0, 0.49]")
        if not self.modulus > 0.0:
            raise ValueError("modulus must be positive")
        if self.quadrature_degree is not None and (int(self.quadrature_degree) != self.quadrature_degree
                                                   or self.quadrature_degree < 1):
            raise ValueError("quadrature_degree must be a positive integer")
        if self.weighting not in MESH_WEIGHTINGS:
            raise ValueError(f"Unknown weighting {self.weighting!r}, expected one of {MESH_WEIGHTINGS}")
        if self.j_star is not None and not (math.isfinite(self.j_star) and self.j_star > 0.0):
            raise ValueError("j_star must be positive and finite")

    def lame_parameters(self) -> tuple[float, float]:
        E, nu = self.modulus, self.poisson_ratio
        return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def bind(self, geometry: FluidReferenceGeometry) -> "ElasticMeshOperator":
        return ElasticMeshOperator(self, geometry)

    def parameters(self) -> dict:
        return {"mesh_extension": self.name, **asdict(self)}


class ElasticMeshOperator:
    """:class:`StiffenedElasticMeshExtension` on a reference geometry."""

    def __init__(self, extension: StiffenedElasticMeshExtension, geometry: FluidReferenceGeometry):
        self.extension = extension
        self.geometry = geometry
        mesh = geometry.mesh
        reference = geometry.check_reference_validity()
        j_star = extension.j_star if extension.j_star is not None else geometry.mean_parent_jacobian()

        self.j_star = dfx.fem.Constant(mesh, j_star)
        self.exponent = dfx.fem.Constant(mesh, float(extension.stiffening_exponent))
        mu, lam = extension.lame_parameters()
        self.mu = dfx.fem.Constant(mesh, mu)
        self.lam = dfx.fem.Constant(mesh, lam)
        if extension.weighting == "pointwise":
            j0 = geometry.jacobian_determinant
        else:
            # experimental: cell volume over parent-cell volume, constant per cell
            Q = dfx.fem.functionspace(mesh, ("DG", 0))
            q = ufl.TestFunction(Q)
            volumes = dfx.fem.assemble_vector(dfx.fem.form(q * ufl.dx(domain=mesh)))
            volumes.scatter_reverse(dfx.la.InsertMode.add)
            j0 = dfx.fem.Function(Q, name="j0")
            j0.x.array[:] = volumes.array / geometry.parent_volume
            j0.x.scatter_forward()
        self.weight = (self.j_star / j0) ** self.exponent

        degree = extension.quadrature_degree
        self.quadrature_degree = int(DEFAULT_QUADRATURE_DEGREE[geometry.cell_name] if degree is None else degree)
        metadata = {"quadrature_degree": self.quadrature_degree}
        self.dx = geometry.dx_fluid(metadata=metadata)
        self.ds = None if geometry.ds_interface_fluid is None else geometry.ds_interface_fluid(metadata=metadata)

        w_min, w_max, w_bad = global_range(geometry.sample(self.weight), geometry.comm)
        if w_bad:
            raise ValueError("Non-finite mesh stiffening weight")
        self.info = extension.parameters() | {
            "quadrature_degree": self.quadrature_degree,
            "j_star": j_star,
            "j_star_definition": ("mean initial fluid-cell volume / parent-cell volume"
                                  if extension.j_star is None else "user"),
            "lame_mu": mu, "lame_lambda": lam,
            "weight_min": w_min, "weight_max": w_max, "reference_geometry": reference,
            "fluid_cells": geometry.num_fluid_cells(), "fluid_volume": geometry.fluid_volume(),
        }

    @staticmethod
    def strain(u):
        return ufl.sym(ufl.grad(u))

    def stress(self, u):
        eps = self.strain(u)
        return self.weight * (2.0 * self.mu * eps + self.lam * ufl.tr(eps) * ufl.Identity(u.ufl_shape[0]))

    def volume_form(self, u, du) -> ufl.Form:
        """``int sigma_m(u) : eps(du)`` over the reference fluid domain, without equation scale."""
        return ufl.inner(self.stress(u), self.strain(du)) * self.dx

    def interface_form(self, u, du) -> ufl.Form:
        """``int (sigma_m(u) n_f) . du`` over the interface, fluid-side trace."""
        normal = ufl.FacetNormal(self.geometry.mesh)
        return ufl.inner(ufl.dot(self.stress(u), normal), du) * self.ds

    def residual(self, u, du, alpha) -> ufl.Form:
        if self.ds is None:
            raise ValueError("The coupled mesh residual requires the interface measure")
        return alpha * self.volume_form(u, du) - alpha * self.interface_form(u, du)
