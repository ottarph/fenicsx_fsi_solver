# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Regenerate the fluid mesh from its current (possibly deformed) geometry.

Implements notes/remeshing/implementation-plan.md §6's recommended
mechanism (per expert input, see notes/remeshing/literature-review.md §5):
feed the current mesh into gmsh as a *discrete* mesh (Dokken's
DOLFINx-mesh-to-gmsh-discrete-entity pattern), then reparametrize it into
proper, remeshable CAD curves -- rather than hand-building boundary
splines.

The boundary curves are built *directly from the mesh's own facet tags*
(``dolfinx.mesh.entities_to_geometry`` maps each tagged facet to its
geometry node indices, per Jørgen Dokken's suggestion), not recovered by
asking gmsh's ``classifySurfaces`` to guess the boundary from the discrete
surface's geometry and then re-classifying the result by bounding box.
This replaced an earlier version of this module that did exactly that;
building curves from the meshtags directly turned out to be both simpler
and more robust -- see the findings below and
notes/remeshing/implementation-log.md.

Empirical findings from prototyping this against the FSI2 fluid domain
(``data/meshes/fsi2/mesh.xdmf`` / ``mesh_sec.xdmf``), which the choices
below encode:

- Tags are now exact by construction (the same ``PHYSICAL_MARKERS`` value
  the input facet already had), not inferred from geometry -- no more
  FSI2-specific bounding-box heuristic, and no risk of a curve being
  misclassified near a boundary intersection.
- This also turned out to be considerably more robust to a poor-quality
  *source* mesh than the ``classifySurfaces``-based approach was: since the
  boundary curves are read directly off the facet tags rather than
  inferred from the discrete surface's own (possibly locally distorted)
  triangulation, the 2D discrete surface's interior quality no longer
  matters for whether curve recovery succeeds. A geometry that made the
  previous ``classifySurfaces``-based version struggle now regenerates
  cleanly. The fundamental limitation remains, though: this can still hang
  rather than fail cleanly once the *boundary itself* is no longer a
  simple (non-self-intersecting) curve -- see below.
- A discrete curve entity needs explicit discrete point entities at its
  two endpoints (``gmsh.model.addDiscreteEntity(1, tag, boundary=[...])``)
  or ``createGeometry`` fails outright ("has no begin or end point");
  building it with no ``boundary`` argument is only valid for a genuinely
  closed loop. Endpoints are found as the degree-1 nodes of each tagged
  facet group's local edge graph (a physical group can be geometrically
  disconnected -- e.g. "channel_side" is the top *and* bottom walls under
  one tag -- so curves are built per connected component, not per group).
  Endpoint point entities are shared (looked up by geometry node index,
  not recreated) across the two curves that meet there, so the topology
  stays consistently connected rather than merely visually coincident.
- The 2D surface entity likewise needs `boundary=<every curve tag just
  built>` at creation time (built *after* the curves, for this reason) --
  without it, ``generate`` silently produces an empty mesh ("only 0 nodes
  on the boundary") rather than raising, because the surface has no
  declared relationship to the curves that are supposed to bound it.
- This mechanism only tolerates a still-simple (non-self-intersecting)
  boundary -- i.e. before the boundary curve itself has crossed itself.
  Feeding it a boundary past that point does not fail cleanly; it can
  hang. This is not a corner case to special-case around: it is the reason
  the remesh trigger (quality.py / §4 of the plan) has to fire on early
  degradation, not on outright inversion.
- For both the 2D surface's cells and the 1D boundary facets, only the
  corner nodes are used, regardless of whether the source mesh is first-
  or second-order (``mesh.geometry.dofmaps[0][:, :3]`` for cells,
  ``entities_to_geometry(...)[:, :2]`` for facets -- basix/DOLFINx always
  list an entity's vertices before any higher-order/edge DOFs, confirmed
  for both cells and facets). The source mesh's curvature does not need to
  survive this step: the *new* mesh gets its own fresh (optionally curved)
  geometry from gmsh afterwards. This sidesteps the DOLFINx<->gmsh
  node-ordering mismatch for higher-order cells entirely, rather than
  needing per-cell-type ordering permutations.
"""

from collections import Counter
from dataclasses import dataclass

import dolfinx as dfx
import gmsh
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from xfsi_solver.remeshing.fluid_domain import FluidDomain
from xfsi_solver.remeshing.markers import PHYSICAL_MARKERS

_MARKER_TO_NAME = {v: k for k, v in PHYSICAL_MARKERS.items()}


@dataclass
class SizingField:
    """Graded triangle-size target: ``size_near`` within ``distance`` of the
    obstacle/interface, linearly growing to ``size_far`` beyond that."""

    size_near: float = 0.01
    size_far: float = 0.05
    distance: float = 0.1


def _connected_components(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Group edges (node-index pairs) into connected components, so a
    geometrically disconnected facet-tag group (e.g. "channel_side" is the
    top *and* bottom walls under one tag) becomes one curve per component
    rather than one curve gmsh can't make sense of.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    components: dict[int, list[tuple[int, int]]] = {}
    for a, b in edges:
        components.setdefault(find(a), []).append((a, b))
    return list(components.values())


def _add_boundary_curves(mesh: dfx.mesh.Mesh, facet_tags: dfx.mesh.MeshTags, deformed_geometry: np.ndarray):
    """Build gmsh discrete point + curve entities directly from
    ``facet_tags``, tagged with the exact same ``PHYSICAL_MARKERS`` value
    the facets already carry.

    Returns ``(curve_groups, all_curve_tags)`` where ``curve_groups`` maps
    each present boundary name to the list of curve tags built for it (fed
    straight into ``addPhysicalGroup`` and into the sizing field), and
    ``all_curve_tags`` is every curve tag across all groups (fed into the
    2D surface entity's own ``boundary=``).
    """
    node_to_point_tag: dict[int, int] = {}
    next_tag = [1]  # mutable cell so the nested closures can bump it

    def point_tag_for(node: int) -> int:
        if node not in node_to_point_tag:
            tag = next_tag[0]
            next_tag[0] += 1
            gmsh.model.addDiscreteEntity(0, tag)
            gmsh.model.mesh.addNodes(0, tag, [node + 1], deformed_geometry[node])
            node_to_point_tag[node] = tag
        return node_to_point_tag[node]

    curve_groups: dict[str, list[int]] = {}
    all_curve_tags: list[int] = []
    for marker in np.unique(facet_tags.values):
        name = _MARKER_TO_NAME[int(marker)]
        facets = facet_tags.find(marker)
        facet_geometry = dfx.mesh.entities_to_geometry(mesh, 1, facets)[:, :2]  # corner nodes only

        curve_tags = []
        for component_edges in _connected_components([tuple(row) for row in facet_geometry]):
            degree = Counter()
            for a, b in component_edges:
                degree[a] += 1
                degree[b] += 1
            endpoints = [node for node, d in degree.items() if d == 1]  # [] if this component is a closed loop

            curve_tag = next_tag[0]
            next_tag[0] += 1
            gmsh.model.addDiscreteEntity(1, curve_tag, boundary=[point_tag_for(n) for n in endpoints])
            component_nodes = sorted({n for edge in component_edges for n in edge})
            gmsh.model.mesh.addNodes(
                1, curve_tag, [n + 1 for n in component_nodes], deformed_geometry[component_nodes].flatten()
            )
            edge_tags = np.arange(len(component_edges), dtype=np.int64) + 1
            connectivity = np.array([[a + 1, b + 1] for a, b in component_edges], dtype=np.int64).flatten()
            gmsh.model.mesh.addElementsByType(curve_tag, 1, edge_tags, connectivity)

            curve_tags.append(curve_tag)
            all_curve_tags.append(curve_tag)

        curve_groups[name] = curve_tags
        gmsh.model.addPhysicalGroup(1, curve_tags, int(marker), name=name)

    return curve_groups, all_curve_tags


def regenerate_fluid_mesh(
    domain: FluidDomain,
    deformed_geometry: np.ndarray,
    sizing: SizingField | None = None,
    geometry_degree: int | None = None,
) -> FluidDomain:
    """Regenerate the fluid mesh from a (possibly deformed) node cloud.

    Args:
        domain: the current fluid-only mesh and its facet tags (e.g. from
            ``fluid_domain.load_fsi2_fluid_domain``). Only its topology,
            corner-node connectivity, and facet tags are used.
        deformed_geometry: an array shaped like ``domain.mesh.geometry.x``,
            giving the *current* physical position of every geometry node
            -- e.g. ``domain.mesh.geometry.x`` plus a prescribed or solved
            displacement.
        sizing: graded target triangle size for the new mesh. Defaults to
            ``SizingField()``.
        geometry_degree: geometric degree of the regenerated mesh (1 for
            straight-sided, 2 for curved). Defaults to ``domain.mesh``'s
            own geometry degree.

    Returns:
        A fresh :class:`FluidDomain`, unrelated to ``domain`` (new dolfinx
        mesh object, new numbering) but covering the same physical region.
    """
    if sizing is None:
        sizing = SizingField()
    mesh = domain.mesh
    if geometry_degree is None:
        geometry_degree = mesh.geometry.cmaps[0].degree

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    corner_dofmap = mesh.geometry.dofmaps[0][:num_cells, :3]

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)

        curve_groups, all_curve_tags = _add_boundary_curves(mesh, domain.facet_tags, deformed_geometry)
        missing = {"inflow", "outflow", "channel_side", "obstacle", "solid_fluid_interface"} - set(curve_groups)
        if missing:
            raise RuntimeError(f"regenerate_fluid_mesh: no facets found for boundary pieces {missing}")

        surface = gmsh.model.addDiscreteEntity(2, -1, boundary=all_curve_tags)
        node_tags = np.arange(deformed_geometry.shape[0], dtype=np.int64) + 1
        gmsh.model.mesh.addNodes(2, surface, node_tags, deformed_geometry.flatten())
        elem_tags = np.arange(num_cells, dtype=np.int64) + 1
        gmsh.model.mesh.addElementsByType(surface, 2, elem_tags, (corner_dofmap + 1).flatten())
        gmsh.model.addPhysicalGroup(2, [surface], PHYSICAL_MARKERS["ALE_fluid"], name="ALE_fluid")

        # Every point/curve/surface entity above reuses the same dolfinx
        # geometry-node index as its gmsh node tag wherever a node is
        # shared (e.g. a curve endpoint is also on the surface boundary),
        # so this reconciles them into one connected topology.
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.createGeometry()

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
