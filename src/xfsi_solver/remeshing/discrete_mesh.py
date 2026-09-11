# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Regenerate the full FSI2 mesh from its current (possibly deformed) geometry.

Implements notes/remeshing/implementation-plan.md §6's recommended
mechanism (per expert input, see notes/remeshing/literature-review.md §5):
feed the current mesh into gmsh as a *discrete* mesh (Dokken's
DOLFINx-mesh-to-gmsh-discrete-entity pattern), then reparametrize it into
proper, remeshable CAD curves -- rather than hand-building boundary
splines.

Both the curves and the subdomains are taken directly from the mesh's own
tags (``dolfinx.mesh.entities_to_geometry`` maps each tagged facet to its
geometry node indices, per Jørgen Dokken's suggestion), not recovered by
asking gmsh's ``classifySurfaces`` to guess the boundary from the discrete
surface's geometry and then re-classifying the result by bounding box.
This replaced an earlier version of this module that did exactly that;
building curves from the meshtags directly turned out to be both simpler
and more robust -- see the findings below and
notes/remeshing/implementation-log.md.

Full mesh, not the fluid alone
------------------------------
An earlier version of this module regenerated only the fluid domain,
extracted as a standalone submesh, and used just the five curves bounding
it. It now regenerates the **whole** FSI2 mesh: one gmsh surface per
``cell_tags`` subdomain (``solid`` and ``ALE_fluid``), bounded by the
curves built from *every* marked facet group -- including the two the
fluid-only version had no use for, ``solid_obstacle_interface`` (the
flag's clamped root on the cylinder) and, now as an internal boundary
rather than an outer one, ``solid_fluid_interface``.

Which curves bound which surface is read off the mesh topology rather
than prescribed: each tagged facet's adjacent cells give the subdomain
markers that facet separates (facet-to-cell connectivity + ``cell_tags``),
so the interface curve is handed to *both* surfaces as a shared boundary
and everything else to the one subdomain it touches. Nothing in this is
FSI2-specific: a geometry with different subdomains and curve groups needs
no changes here.

Because the interface curve entity is shared (one tag, listed in both
surfaces' ``boundary=``), ``createGeometry`` reparametrizes it once and
``generate`` meshes both sides conformally against it -- the regenerated
mesh has matching solid- and fluid-side facets on the interface, which is
what the monolithic ALE formulation needs and what a fluid-only remesh
could never give on its own.

Empirical findings from prototyping this against the FSI2 mesh
(``data/meshes/fsi2/mesh.xdmf`` / ``mesh_sec.xdmf``), which the choices
below encode:

- Tags are exact by construction (the same ``PHYSICAL_MARKERS`` value the
  input facet/cell already had), not inferred from geometry -- no
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
  not recreated) across the curves that meet there, so the topology stays
  consistently connected rather than merely visually coincident.
- Each 2D surface entity likewise needs `boundary=<its own curve tags>` at
  creation time (built *after* the curves, for this reason) -- without it,
  ``generate`` silently produces an empty mesh ("only 0 nodes on the
  boundary") rather than raising, because the surface has no declared
  relationship to the curves that are supposed to bound it.
- This mechanism only tolerates a still-simple (non-self-intersecting)
  boundary -- i.e. before the boundary curve itself has crossed itself.
  Feeding it a boundary past that point does not fail cleanly; it can
  hang. This is not a corner case to special-case around: it is the reason
  the remesh trigger (quality.py / §4 of the plan) has to fire on early
  degradation, not on outright inversion.
- For both the 2D surfaces' cells and the 1D boundary facets, only the
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

from xfsi_solver.remeshing.domain import FsiDomain
from xfsi_solver.remeshing.markers import PHYSICAL_MARKERS

_MARKER_TO_NAME = {v: k for k, v in PHYSICAL_MARKERS.items()}


#: Facet groups the regenerated mesh's cell size is graded away from: the
#: flag's two surfaces and the cylinder. Mirrors which boundaries
#: ``scripts/create_mesh_FSI2.py`` attaches ``resolution_close`` to.
REFINED_BOUNDARIES = ("obstacle", "solid_fluid_interface", "solid_obstacle_interface")


@dataclass
class SizingField:
    """Graded target cell size for a regenerated mesh.

    Two regimes, combined by taking the smaller size at every point --
    mirroring the three resolutions ``scripts/create_mesh_FSI2.py`` builds
    the original mesh from:

    - **Refinement around the flag and cylinder** (``REFINED_BOUNDARIES``):
      ``size_near`` on those surfaces, growing linearly with distance from
      them to ``size_outflow`` at ``growth_distance``. This is
      ``resolution_close``'s role.
    - **A streamwise far-field cap**: ``size_far`` everywhere up to
      ``coarsen_from_x``, then growing linearly to ``size_outflow`` at
      ``coarsen_to_x`` and holding there. This is ``resolution_far`` and
      ``resolution_ultra_far``'s role. It is combined as a *minimum*, not a
      maximum, so that it coarsens the wake and the far channel without
      ever coarsening the flag itself -- the flag sits upstream of
      ``coarsen_from_x``, where this field is merely a constant cap, and
      taking a maximum there would instead wipe out the near-flag
      refinement entirely.

    The defaults reproduce the original FSI2 triangle mesh
    (``data/meshes/fsi2/mesh.xdmf``, i.e. ``scripts/create_mesh_FSI2.py``
    with ``quads=False``): mean cell edge length binned by distance from the
    flag/cylinder surface agrees to within a few percent from the surface
    (~0.005) out to the far field (~0.045), and the regenerated mesh has
    5886 cells against the original's 5851. That mesh's own grading is an
    implicit, linearly-interpolated consequence of those three resolutions
    being set at individual CAD *points*; reproducing it here has to state
    the gradient explicitly, because a regenerated mesh has no CAD points
    to hang sizes off.

    These defaults matter: a regenerated mesh is only a usable replacement
    for the one it supersedes if it resolves the same flow. The fluid-only
    version of this module defaulted to ``size_near=0.01``/``size_far=0.05``
    with no streamwise regime at all -- fine for quick prototyping, but 2x
    too coarse at the flag and much worse than that through the wake -- so
    every remesh event silently dropped the simulation onto a coarser mesh
    than it started from, and repeated events kept doing so.
    """

    size_near: float = 0.005
    size_far: float = 0.025
    size_outflow: float = 0.05125
    growth_distance: float = 0.36
    coarsen_from_x: float = 0.6
    coarsen_to_x: float = 2.25


def _connected_components(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Group edges (node-index pairs) into connected components, returning
    *indices into* ``edges`` so the caller can map each component back to
    the facets it came from.

    Needed because a geometrically disconnected facet-tag group (e.g.
    "channel_side" is the top *and* bottom walls under one tag) has to
    become one curve per component rather than one curve gmsh can't make
    sense of.
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

    components: dict[int, list[int]] = {}
    for i, (a, _) in enumerate(edges):
        components.setdefault(find(a), []).append(i)
    return list(components.values())


def _facet_subdomains(mesh: dfx.mesh.Mesh, cell_tags: dfx.mesh.MeshTags, facets: np.ndarray) -> list[set[int]]:
    """For each facet, the set of subdomain markers of the cells it touches.

    This is what makes the curve-to-surface assignment topological rather
    than prescribed: a facet between two differently-tagged cells (the
    fluid-solid interface) reports both markers and so becomes a shared
    boundary of both gmsh surfaces, while an outer facet reports the one
    subdomain it bounds.
    """
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)

    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    marker_of_cell = np.zeros(num_cells, dtype=np.int32)
    marker_of_cell[cell_tags.indices] = cell_tags.values

    return [{int(marker_of_cell[c]) for c in facet_to_cell.links(f)} for f in facets]


def _add_boundary_curves(
    mesh: dfx.mesh.Mesh,
    cell_tags: dfx.mesh.MeshTags,
    facet_tags: dfx.mesh.MeshTags,
    deformed_geometry: np.ndarray,
    next_tag: list[int],
):
    """Build gmsh discrete point + curve entities directly from
    ``facet_tags``, tagged with the exact same ``PHYSICAL_MARKERS`` value
    the facets already carry.

    Returns ``(curve_groups, curves_by_subdomain)`` where ``curve_groups``
    maps each present boundary name to the list of curve tags built for it
    (used by the sizing field), and ``curves_by_subdomain`` maps each
    subdomain marker to the curve tags bounding it (fed into that
    subdomain's surface entity's own ``boundary=``). The interface curve
    appears under both subdomains.
    """
    node_to_point_tag: dict[int, int] = {}

    def point_tag_for(node: int) -> int:
        if node not in node_to_point_tag:
            tag = next_tag[0]
            next_tag[0] += 1
            gmsh.model.addDiscreteEntity(0, tag)
            gmsh.model.mesh.addNodes(0, tag, [node + 1], deformed_geometry[node])
            node_to_point_tag[node] = tag
        return node_to_point_tag[node]

    curve_groups: dict[str, list[int]] = {}
    curves_by_subdomain: dict[int, list[int]] = {}
    for marker in np.unique(facet_tags.values):
        name = _MARKER_TO_NAME[int(marker)]
        facets = facet_tags.find(marker)
        facet_geometry = dfx.mesh.entities_to_geometry(mesh, 1, facets)[:, :2]  # corner nodes only
        subdomains_of_facet = _facet_subdomains(mesh, cell_tags, facets)
        edges = [(int(a), int(b)) for a, b in facet_geometry]

        curve_tags = []
        for component in _connected_components(edges):
            component_edges = [edges[i] for i in component]
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
            edge_tags = np.arange(len(component_edges), dtype=np.int64) + next_tag[0]
            next_tag[0] += len(component_edges)
            connectivity = np.array([[a + 1, b + 1] for a, b in component_edges], dtype=np.int64).flatten()
            gmsh.model.mesh.addElementsByType(curve_tag, 1, edge_tags, connectivity)

            curve_tags.append(curve_tag)
            for subdomain in set().union(*(subdomains_of_facet[i] for i in component)):
                curves_by_subdomain.setdefault(subdomain, []).append(curve_tag)

        curve_groups[name] = curve_tags
        gmsh.model.addPhysicalGroup(1, curve_tags, int(marker), name=name)

    return curve_groups, curves_by_subdomain


def _add_subdomain_surfaces(
    mesh: dfx.mesh.Mesh,
    cell_tags: dfx.mesh.MeshTags,
    deformed_geometry: np.ndarray,
    curves_by_subdomain: dict[int, list[int]],
    next_tag: list[int],
) -> None:
    """Build one gmsh discrete surface entity per ``cell_tags`` subdomain,
    each bounded by the curves found to touch it, and give each the
    physical group its cells were already tagged with."""
    for marker in np.unique(cell_tags.values):
        marker = int(marker)
        cells = cell_tags.find(marker)
        corner_dofmap = mesh.geometry.dofmaps[0][cells][:, :3]
        nodes = np.unique(corner_dofmap)

        surface = gmsh.model.addDiscreteEntity(2, -1, boundary=curves_by_subdomain[marker])
        gmsh.model.mesh.addNodes(2, surface, nodes + 1, deformed_geometry[nodes].flatten())
        elem_tags = np.arange(len(cells), dtype=np.int64) + next_tag[0]
        next_tag[0] += len(cells)
        gmsh.model.mesh.addElementsByType(surface, 2, elem_tags, (corner_dofmap + 1).flatten())
        gmsh.model.addPhysicalGroup(2, [surface], marker, name=_MARKER_TO_NAME[marker])


def regenerate_mesh(
    domain: FsiDomain,
    deformed_geometry: np.ndarray,
    sizing: SizingField | None = None,
    geometry_degree: int | None = None,
) -> FsiDomain:
    """Regenerate the full FSI2 mesh from a (possibly deformed) node cloud.

    Args:
        domain: the current mesh with its cell and facet tags (e.g. from
            ``domain.load_fsi2_domain``). Only its topology, corner-node
            connectivity, and the two tag sets are used.
        deformed_geometry: an array shaped like ``domain.mesh.geometry.x``,
            giving the *current* physical position of every geometry node
            -- e.g. ``domain.mesh.geometry.x`` plus a prescribed or solved
            displacement.
        sizing: graded target cell size for the new mesh. Defaults to
            ``SizingField()``, which reproduces the original FSI2 mesh's
            own resolution.
        geometry_degree: geometric degree of the regenerated mesh (1 for
            straight-sided, 2 for curved). Defaults to ``domain.mesh``'s
            own geometry degree.

    Returns:
        A fresh :class:`FsiDomain`, unrelated to ``domain`` (new dolfinx
        mesh object, new numbering) but covering the same physical region,
        with both subdomains and all curve groups tagged as before.
    """
    if sizing is None:
        sizing = SizingField()
    mesh = domain.mesh
    if geometry_degree is None:
        geometry_degree = mesh.geometry.cmaps[0].degree

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)

        # gmsh entity tags and element tags both have to be unique across
        # the whole model, so every entity and element built below draws
        # from this one running counter.
        next_tag = [1]

        curve_groups, curves_by_subdomain = _add_boundary_curves(
            mesh, domain.cell_tags, domain.facet_tags, deformed_geometry, next_tag
        )
        unbounded = {int(m) for m in np.unique(domain.cell_tags.values)} - set(curves_by_subdomain)
        if unbounded:
            raise RuntimeError(
                f"regenerate_mesh: subdomain(s) {[_MARKER_TO_NAME[m] for m in unbounded]} have no tagged facets "
                "bounding them; every subdomain's boundary must be covered by facet_tags for a surface to be built"
            )

        _add_subdomain_surfaces(mesh, domain.cell_tags, deformed_geometry, curves_by_subdomain, next_tag)

        # Every point/curve/surface entity above reuses the same dolfinx
        # geometry-node index as its gmsh node tag wherever a node is
        # shared (a curve endpoint is also on the surfaces' boundary; an
        # interface node is on both surfaces), so this reconciles them into
        # one connected topology.
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
    return FsiDomain(mesh=out.mesh, cell_tags=out.cell_tags, facet_tags=out.facet_tags)


def _set_sizing_field(curve_groups: dict[str, list[int]], sizing: SizingField) -> None:
    """Build the two-regime background field described by :class:`SizingField`."""
    near_curves = [tag for name in REFINED_BOUNDARIES for tag in curve_groups[name]]
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", near_curves)
    gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 200)

    near_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(near_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(near_field, "SizeMin", sizing.size_near)
    gmsh.model.mesh.field.setNumber(near_field, "SizeMax", sizing.size_outflow)
    gmsh.model.mesh.field.setNumber(near_field, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(near_field, "DistMax", sizing.growth_distance)

    streamwise = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(streamwise, "F", "x")
    far_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(far_field, "InField", streamwise)
    gmsh.model.mesh.field.setNumber(far_field, "SizeMin", sizing.size_far)
    gmsh.model.mesh.field.setNumber(far_field, "SizeMax", sizing.size_outflow)
    gmsh.model.mesh.field.setNumber(far_field, "DistMin", sizing.coarsen_from_x)
    gmsh.model.mesh.field.setNumber(far_field, "DistMax", sizing.coarsen_to_x)

    background = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(background, "FieldsList", [near_field, far_field])

    gmsh.model.mesh.field.setAsBackgroundMesh(background)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
