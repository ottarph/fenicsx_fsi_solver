# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Regenerate the fluid mesh from its current (possibly deformed) geometry.

Implements notes/remeshing/implementation-plan.md §6's recommended
mechanism (per expert input, see notes/remeshing/literature-review.md §5):
feed the current mesh into gmsh as a *discrete* mesh (Dokken's
DOLFINx-mesh-to-gmsh-discrete-entity pattern), then let gmsh's own
``classifySurfaces``/``createGeometry`` recover proper, remeshable CAD
curves from it, rather than hand-building boundary splines.

Empirical findings from prototyping this against the FSI2 fluid domain
(``data/meshes/fsi2/mesh.xdmf`` / ``mesh_sec.xdmf``), which the choices
below encode:

- The angle threshold passed to both ``classifySurfaces`` and its
  ``curveAngle`` matters a lot. 30 degrees (a plausible-looking default)
  makes ``classifySurfaces`` split the boundary into hundreds of spurious
  curves from ordinary mesh-resolution "noise" along a smoothly bent
  boundary, and the pipeline can hang for minutes rather than erroring.
  60 degrees cleanly recovers the true 8-curve topology (4 straight channel
  walls, the obstacle arc, and the flag's 3 straight edges) in both the
  undeformed and moderately-deformed case, in well under a second.
- This only works while the deformed boundary is still a *simple* (non
  -self-intersecting) curve -- i.e. before any cell has actually inverted.
  Feeding it an already-inverted mesh does not fail cleanly; it can hang.
  This is not a corner case to special-case around: it is the reason the
  remesh trigger (quality.py / §4 of the plan) has to fire on early
  degradation, not on outright inversion -- by the time cells have
  inverted, this mechanism can no longer rescue the mesh.
- For the discrete-entity input, only the 3 corner nodes of each cell are
  used, regardless of whether the source mesh is first- or second-order
  (``mesh.geometry.dofmaps[0][:, :3]`` -- basix/DOLFINx always list a
  cell's vertices before any higher-order/edge DOFs). The source mesh's
  curvature does not need to survive this step: the *new* mesh gets its
  own fresh (optionally curved) geometry from gmsh afterwards. This
  sidesteps the DOLFINx<->gmsh node-ordering mismatch for higher-order
  cells entirely, rather than needing per-cell-type ordering permutations.
"""

import math
from dataclasses import dataclass

import dolfinx as dfx
import gmsh
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from xfsi_solver.remeshing import fsi2_geometry as geo
from xfsi_solver.remeshing.fluid_domain import FluidDomain
from xfsi_solver.remeshing.markers import PHYSICAL_MARKERS

#: Angle threshold (both for classifySurfaces' surface-splitting and its
#: curveAngle corner-detection) -- see module docstring for why this is 60
#: and not a tighter-looking default like 30.
CLASSIFY_ANGLE = math.pi / 3

#: Absolute tolerance (metres) for deciding a curve's bounding box lies
#: exactly on a fixed boundary (inflow/outflow/channel walls, which never
#: move -- see deformation.py) vs. being part of the deforming interface.
_FIXED_BOUNDARY_TOL = 1e-6

#: Extra padding (metres) around the obstacle's bounding box used only to
#: decide whether a curve is the (fixed) obstacle boundary -- looser than
#: _FIXED_BOUNDARY_TOL since the obstacle radius itself is only recovered
#: to meshing precision, not exactly.
_OBSTACLE_BBOX_PAD = 1e-3


@dataclass
class SizingField:
    """Graded triangle-size target: ``size_near`` within ``distance`` of the
    obstacle/interface, linearly growing to ``size_far`` beyond that."""

    size_near: float = 0.01
    size_far: float = 0.05
    distance: float = 0.1


def _classify_curve(bbox: tuple[float, float, float, float, float, float]) -> str:
    """Classify a recovered boundary curve into a ``PHYSICAL_MARKERS`` name
    from its bounding box alone.

    Robust to deformation because the four channel-wall/inflow/outflow
    pieces and the obstacle are never moved by construction (see
    ``deformation.py``): whatever curve is left over, by elimination, is
    the (possibly deformed) fluid-solid interface.
    """
    xmin, ymin, _, xmax, ymax, _ = bbox
    if abs(xmin) < _FIXED_BOUNDARY_TOL and abs(xmax) < _FIXED_BOUNDARY_TOL:
        return "inflow"
    if abs(xmin - geo.L) < _FIXED_BOUNDARY_TOL and abs(xmax - geo.L) < _FIXED_BOUNDARY_TOL:
        return "outflow"
    if (abs(ymin) < _FIXED_BOUNDARY_TOL and abs(ymax) < _FIXED_BOUNDARY_TOL) or (
        abs(ymin - geo.H) < _FIXED_BOUNDARY_TOL and abs(ymax - geo.H) < _FIXED_BOUNDARY_TOL
    ):
        return "channel_side"
    if (
        xmin >= geo.C_X - geo.R - _OBSTACLE_BBOX_PAD
        and xmax <= geo.C_X + geo.R + _OBSTACLE_BBOX_PAD
        and ymin >= geo.C_Y - geo.R - _OBSTACLE_BBOX_PAD
        and ymax <= geo.C_Y + geo.R + _OBSTACLE_BBOX_PAD
    ):
        return "obstacle"
    return "solid_fluid_interface"


def regenerate_fluid_mesh(
    mesh: dfx.mesh.Mesh,
    deformed_geometry: np.ndarray,
    sizing: SizingField | None = None,
    geometry_degree: int | None = None,
) -> FluidDomain:
    """Regenerate the fluid mesh from a (possibly deformed) node cloud.

    Args:
        mesh: the current fluid-only mesh (e.g. from
            ``fluid_domain.load_fsi2_fluid_domain``). Only its topology and
            corner-node connectivity are used.
        deformed_geometry: an array shaped like ``mesh.geometry.x``, giving
            the *current* physical position of every geometry node -- e.g.
            ``mesh.geometry.x`` plus a prescribed or solved displacement.
        sizing: graded target triangle size for the new mesh. Defaults to
            ``SizingField()``.
        geometry_degree: geometric degree of the regenerated mesh (1 for
            straight-sided, 2 for curved). Defaults to ``mesh``'s own
            geometry degree.

    Returns:
        A fresh :class:`FluidDomain`, unrelated to ``mesh`` (new dolfinx
        mesh object, new numbering) but covering the same physical region.
    """
    if sizing is None:
        sizing = SizingField()
    if geometry_degree is None:
        geometry_degree = mesh.geometry.cmaps[0].degree

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    corner_dofmap = mesh.geometry.dofmaps[0][:num_cells, :3]

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)

        surface = gmsh.model.addDiscreteEntity(2, 1)
        node_tags = np.arange(deformed_geometry.shape[0], dtype=np.int64) + 1
        gmsh.model.mesh.addNodes(2, surface, node_tags, deformed_geometry.flatten())
        elem_tags = np.arange(num_cells, dtype=np.int64) + 1
        gmsh.model.mesh.addElementsByType(surface, 2, elem_tags, (corner_dofmap + 1).flatten())
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.classifySurfaces(CLASSIFY_ANGLE, curveAngle=CLASSIFY_ANGLE)
        gmsh.model.mesh.createGeometry()

        curve_groups: dict[str, list[int]] = {}
        for _, curve_tag in gmsh.model.getEntities(1):
            label = _classify_curve(gmsh.model.getBoundingBox(1, curve_tag))
            curve_groups.setdefault(label, []).append(curve_tag)
        missing = {"inflow", "outflow", "channel_side", "obstacle", "solid_fluid_interface"} - set(curve_groups)
        if missing:
            raise RuntimeError(f"regenerate_fluid_mesh: failed to recover boundary pieces {missing}")

        for label, tags in curve_groups.items():
            gmsh.model.addPhysicalGroup(1, tags, PHYSICAL_MARKERS[label], name=label)
        surface_tags = [tag for _, tag in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, surface_tags, PHYSICAL_MARKERS["ALE_fluid"], name="ALE_fluid")

        _set_sizing_field(curve_groups, sizing)

        gmsh.model.mesh.generate(2)
        if geometry_degree != 1:
            gmsh.model.mesh.setOrder(geometry_degree)

        out = dfx.io.gmsh.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    finally:
        gmsh.finalize()

    out.mesh.topology.create_connectivity(1, 2)
    return FluidDomain(mesh=out.mesh, facet_tags=out.facet_tags)


def _set_sizing_field(curve_groups: dict[str, list[int]], sizing: SizingField) -> None:
    """Grade the target triangle size: fine near the obstacle/interface,
    growing linearly to ``sizing.size_far`` beyond ``sizing.distance``."""
    near_curves = curve_groups["obstacle"] + curve_groups["solid_fluid_interface"]
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", near_curves)
    gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 100)

    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", sizing.size_near)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", sizing.size_far)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", sizing.distance)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * sizing.distance)

    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
