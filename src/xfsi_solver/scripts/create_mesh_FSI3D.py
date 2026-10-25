# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import sys

import gmsh

gmsh.initialize(sys.argv)
import numpy as np

PHYSICAL_MARKERS = {
    "solid": 1,
    "ALE_fluid": 2,
    "solid_fluid_interface": 11,
    "obstacle": 21,
    "inflow": 22,
    "outflow": 23,
    "channel_side": 24,
    "solid_obstacle_interface": 25,
}


def create_mesh(
    resolution_close: float = 0.02,
    resolution_far: float = 0.08,
    second_order: bool = True,
    suffix: str = "_coarse",
):
    """Generate the tetrahedral mesh of the 3d FSI benchmark of Failer & Richter (J. Sci. Comput.
    82:28, 2020, Sect. 5.1.2, Fig. 1) and write it to ``data/meshes/fsi3d/mesh<suffix>.xdmf``.

    The channel 0 < x < 2.8, 0 < y < 0.41, 0 < z < 0.41 has the cylinder (x - 0.5)^2 + (y - 0.2)^2
    < 0.05^2 cut out of it over its full height. The elastic beam 0.5 < x < 0.9, 0.19 < y < 0.21,
    0.1 < z < 0.3, without the cylinder, is attached to the cylinder. The cell and facet tags are
    those of the 2d FSI2 mesh, with the walls y = 0, y = 0.41, z = 0 and z = 0.41 as the channel
    side.

    Parameters
    ----------
    resolution_close:
        Cell size on the beam and the cylinder.
    resolution_far:
        Cell size away from the beam and the cylinder.
    second_order:
        Use a second order (curved) mesh.
    suffix:
        Appended to the output file name.
    """

    L, H, W = 2.8, 0.41, 0.41  # channel length (x), height (y) and width (z)
    C_x, C_y, r = 0.5, 0.2, 0.05  # cylinder center and radius
    beam_x, beam_y, beam_z = 0.5, 0.19, 0.1  # beam corner, before cutting out the cylinder
    beam_l, beam_h, beam_w = 0.4, 0.02, 0.2  # beam length (x), thickness (y) and width (z)
    B = (0.9, 0.2, 0.3)  # measurement point on the back face of the beam
    gdim = 3

    channel = gmsh.model.occ.addBox(0.0, 0.0, 0.0, L, H, W)
    cylinder = gmsh.model.occ.addCylinder(C_x, C_y, 0.0, 0.0, 0.0, W, r)
    beam = gmsh.model.occ.addBox(beam_x, beam_y, beam_z, beam_l, beam_h, beam_w)

    # Cut the cylinder out of both the channel and the beam, then fragment the channel with the
    # beam, so that the fluid and the solid volume share the interface faces and get a conforming
    # mesh. The point B is fragmented in too, so that the mesh has a vertex there: it lies in the
    # middle of the edge x = 0.9, z = 0.3 of the beam, which is otherwise not a mesh node.
    channel_cut, _ = gmsh.model.occ.cut([(3, channel)], [(3, cylinder)], removeTool=False)
    beam_cut, _ = gmsh.model.occ.cut([(3, beam)], [(3, cylinder)])
    point_B = gmsh.model.occ.addPoint(*B)
    gmsh.model.occ.fragment(channel_cut, beam_cut + [(0, point_B)])
    gmsh.model.occ.synchronize()

    beam_volume = beam_l * beam_h * beam_w
    solid_tags = []
    fluid_tags = []
    for _, tag in gmsh.model.getEntities(dim=3):
        if gmsh.model.occ.getMass(3, tag) < beam_volume:
            solid_tags.append(tag)
        else:
            fluid_tags.append(tag)
    assert len(solid_tags) == 1 and len(fluid_tags) == 1, "Expected one solid and one fluid volume"

    gmsh.model.addPhysicalGroup(3, solid_tags, PHYSICAL_MARKERS["solid"], name="solid")
    gmsh.model.addPhysicalGroup(3, fluid_tags, PHYSICAL_MARKERS["ALE_fluid"], name="ALE_fluid")

    # Classify the surfaces: a surface between the two volumes is the interface. Of the boundary
    # surfaces, the planar ones on the channel walls are found by their bounding box, and the
    # remaining ones lie on the cylinder, where they belong to either the beam or the fluid.
    def lies_in_plane(surface, axis, value):
        # OCC pads the bounding boxes by 1e-7.
        bounding_box = gmsh.model.getBoundingBox(2, surface)
        return np.isclose(bounding_box[axis], value, atol=1e-6) and np.isclose(bounding_box[axis + 3], value, atol=1e-6)

    surface_tags = {name: [] for name in PHYSICAL_MARKERS if name not in ["solid", "ALE_fluid"]}
    for _, surface in gmsh.model.getEntities(dim=2):
        volumes, _ = gmsh.model.getAdjacencies(2, surface)
        if len(volumes) == 2:
            surface_tags["solid_fluid_interface"].append(surface)
        elif lies_in_plane(surface, 0, 0.0):
            surface_tags["inflow"].append(surface)
        elif lies_in_plane(surface, 0, L):
            surface_tags["outflow"].append(surface)
        elif any(lies_in_plane(surface, 1, y) for y in [0.0, H]) or any(lies_in_plane(surface, 2, z) for z in [0.0, W]):
            surface_tags["channel_side"].append(surface)
        elif volumes[0] in solid_tags:
            surface_tags["solid_obstacle_interface"].append(surface)
        else:
            surface_tags["obstacle"].append(surface)

    for name, tags in surface_tags.items():
        assert len(tags) > 0, f"No surfaces found for {name}"
        gmsh.model.addPhysicalGroup(2, tags, PHYSICAL_MARKERS[name], name=name)

    # Refine towards the beam and the cylinder: the cell size grows linearly from resolution_close
    # on their surfaces to resolution_far at a distance of 0.2.
    close_surfaces = surface_tags["solid_fluid_interface"] + surface_tags["obstacle"]
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", close_surfaces)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", resolution_close)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", resolution_far)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.2)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

    # Let the size field alone set the cell size.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.optimize("Netgen")

    if second_order:
        # Use second order mesh, and untangle the cells whose curved faces on the cylinder
        # would otherwise invert them.
        gmsh.model.mesh.setOrder(2)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
        gmsh.model.mesh.optimize("HighOrderElastic")

    import dolfinx.io
    from mpi4py import MPI

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    out = dolfinx.io.gmsh.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
    domain = out.mesh
    cell_markers = out.cell_tags
    facet_markers = out.facet_tags
    cell_markers.name = "Cell tags"
    facet_markers.name = "Facet tags"

    domain.topology.create_connectivity(gdim - 1, gdim)

    num_cells = domain.topology.index_map(gdim).size_global
    num_vertices = domain.topology.index_map(0).size_global
    if mesh_comm.rank == gmsh_model_rank:
        print(f"{num_cells = }, {num_vertices = }")

    from pathlib import Path

    mesh_path = Path(f"data/meshes/fsi3d/mesh{'_sec' if second_order else ''}{suffix}.xdmf")

    from dolfinx.io import XDMFFile

    with XDMFFile(MPI.COMM_WORLD, mesh_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(cell_markers, domain.geometry)
        xdmf.write_meshtags(facet_markers, domain.geometry)

    # gmsh.fltk.run()


def create_hex_mesh(
    resolution_close: float = 0.02,
    resolution_far: float = 0.08,
    second_order: bool = True,
    suffix: str = "_coarse",
):
    """Generate a hexahedral mesh of the 3d FSI benchmark, with the geometry and the tags of
    ``create_mesh``, and write it to ``data/meshes/fsi3d/mesh_hex<suffix>.xdmf``.

    The cross-section z = 0 is meshed with quadrilaterals by gmsh, with the size field of
    ``create_mesh`` restricted to that plane, and extruded in z: the layers along the beam have the
    size on it, and those below and above it grow with the distance to its ends as the size field
    does. gmsh cannot extrude it itself: recombination leaves triangles (and so prisms) in its
    extruded meshes, and its all-quad subdivision cannot split extruded edges. The tags are found
    from the geometry.

    Parameters
    ----------
    resolution_close:
        Cell size on the beam and the cylinder.
    resolution_far:
        Cell size away from the beam and the cylinder, in the cross-section.
    second_order:
        Use a second order (curved) mesh.
    suffix:
        Appended to the output file name.
    """

    L, H, W = 2.8, 0.41, 0.41  # channel length (x), height (y) and width (z)
    C_x, C_y, r = 0.5, 0.2, 0.05  # cylinder center and radius
    beam_x, beam_y, beam_z = 0.5, 0.19, 0.1  # beam corner, before cutting out the cylinder
    beam_l, beam_h, beam_w = 0.4, 0.02, 0.2  # beam length (x), thickness (y) and width (z)
    B = (0.9, 0.2, 0.3)  # measurement point on the back face of the beam
    gdim = 3
    order = 2 if second_order else 1

    gmsh.model.add("fsi3d_hex")

    # The cross-section, with the cylinder cut out of the channel and the beam, and the beam
    # fragmented in. The seam of the cylinder is turned upstream, so that the arc of the beam on it
    # is one curve.
    channel = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, H)
    cylinder = gmsh.model.occ.addDisk(C_x, C_y, 0.0, r, r)
    gmsh.model.occ.rotate([(2, cylinder)], C_x, C_y, 0.0, 0.0, 0.0, 1.0, np.pi)
    beam = gmsh.model.occ.addRectangle(beam_x, beam_y, 0.0, beam_l, beam_h)
    channel_cut, _ = gmsh.model.occ.cut([(2, channel)], [(2, cylinder)], removeTool=False)
    beam_cut, _ = gmsh.model.occ.cut([(2, beam)], [(2, cylinder)])
    gmsh.model.occ.fragment(channel_cut, beam_cut)
    gmsh.model.occ.synchronize()

    sections = gmsh.model.getEntities(dim=2)
    assert len(sections) == 2, "Expected the cross-sections of the beam and of the fluid"
    beam_section, fluid_section = sorted(sections, key=lambda section: gmsh.model.occ.getMass(*section))

    # gmsh recombines a triangulation of twice the cell size and splits every cell into four
    # quadrilaterals (a triangle into three), which leaves no triangles. The beam is a structured
    # strip, since its unstructured cells are poor. After the split, it is an even number of cells
    # thick, at least two, and the middle of its end x = 0.9 is a vertex, which B is at z = 0.3.
    for _, curve in gmsh.model.getBoundary([beam_section], oriented=False):
        x_min, y_min, _, x_max, y_max, _ = gmsh.model.getBoundingBox(1, curve)
        if y_max - y_min > beam_h / 2:  # the arc on the cylinder and the end x = 0.9
            gmsh.model.mesh.setTransfiniteCurve(curve, max(1, round(beam_h / (2 * resolution_close))) + 1)
        else:
            gmsh.model.mesh.setTransfiniteCurve(curve, max(1, round((x_max - x_min) / (2 * resolution_close))) + 1)
    gmsh.model.mesh.setTransfiniteSurface(beam_section[1])

    # The size field of create_mesh, by the distance to the curves of the beam and the cylinder.
    def on_channel_wall(curve):
        # OCC pads the bounding boxes by 1e-7.
        x_min, y_min, _, x_max, y_max, _ = gmsh.model.getBoundingBox(1, curve)
        return any(np.isclose([x_min, x_max], x, atol=1e-6).all() for x in [0.0, L]) or any(
            np.isclose([y_min, y_max], y, atol=1e-6).all() for y in [0.0, H]
        )

    close_curves = [curve for _, curve in gmsh.model.getEntities(dim=1) if not on_channel_wall(curve)]
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", close_curves)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", 2 * resolution_close)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", 2 * resolution_far)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.2)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Frontal-Delaunay gives better quadrilaterals here than Frontal-Delaunay for quads.
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    if second_order:
        gmsh.model.mesh.setOrder(2)

    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    node_index = np.zeros(node_tags.max() + 1, dtype=np.int64)
    node_index[node_tags] = np.arange(len(node_tags))
    x_section = node_coordinates.reshape(-1, 3)[:, :2]

    quad_type = gmsh.model.mesh.getElementType("Quadrangle", order)
    hex_type = gmsh.model.mesh.getElementType("Hexahedron", order)
    num_quad_nodes = gmsh.model.mesh.getElementProperties(quad_type)[3]
    quads = []
    for section in [beam_section, fluid_section]:
        element_types, _, _ = gmsh.model.mesh.getElements(*section)
        assert list(element_types) == [quad_type], "Expected quadrilaterals only"
        _, nodes = gmsh.model.mesh.getElementsByType(quad_type, section[1])
        quads.append(node_index[nodes.reshape(-1, num_quad_nodes)])
    quads = np.vstack(quads)

    # Orient the quadrilaterals counterclockwise, so that the extruded hexahedra have positive
    # Jacobians.
    corners = x_section[quads[:, :4]]
    area = np.sum(
        corners[:, :, 0] * np.roll(corners[:, :, 1], -1, axis=1)
        - np.roll(corners[:, :, 0], -1, axis=1) * corners[:, :, 1],
        axis=1,
    )
    reverse = [0, 3, 2, 1, 7, 6, 5, 4, 8][:num_quad_nodes]
    quads[area < 0] = quads[area < 0][:, reverse]

    # The z of the vertex layers. Below and above the beam, the layers have the size the threshold
    # field gives at their distance d from the beam end, close + slope * d (d < 0.2 here), so that
    # there are log(1 + slope * d / close) / slope of them over a distance d.
    slope = (resolution_far - resolution_close) / 0.2

    def graded_distances(length):
        assert length <= 0.2
        num_layers = np.log1p(slope * length / resolution_close) / slope
        layers = np.arange(1, max(1, round(num_layers)) + 1) * num_layers / max(1, round(num_layers))
        distances = resolution_close * np.expm1(slope * layers) / slope
        distances[-1] = length
        return distances

    num_along = max(1, round(beam_w / resolution_close))
    z_vertices = np.concatenate(
        [
            beam_z - graded_distances(beam_z)[::-1],
            beam_z + beam_w * np.arange(num_along + 1) / num_along,
            beam_z + beam_w + graded_distances(W - beam_z - beam_w),
        ]
    )
    on_B = np.isclose(x_section[np.unique(quads[:, :4])], B[:2]).all(axis=1)
    assert on_B.any() and np.isclose(z_vertices, B[2]).any(), "B is not a vertex"

    z_nodes = np.empty(order * (len(z_vertices) - 1) + 1)
    z_nodes[::order] = z_vertices
    if second_order:
        z_nodes[1::2] = (z_vertices[:-1] + z_vertices[1:]) / 2

    # The nodes are the cross-section nodes at every z, and a hexahedron of the layer k has the
    # nodes of its quadrilateral at the z of the node layers order * k, ..., order * (k + 1), as
    # found from the reference coordinates of their nodes.
    x = np.column_stack([np.tile(x_section, (len(z_nodes), 1)), np.repeat(z_nodes, len(x_section))])
    quad_reference = gmsh.model.mesh.getElementProperties(quad_type)[4].reshape(num_quad_nodes, 2)
    hex_reference = gmsh.model.mesh.getElementProperties(hex_type)[4].reshape(-1, 3)
    hex_quad_node = [np.flatnonzero((quad_reference == node[:2]).all(axis=1))[0] for node in hex_reference]
    hex_node_layer = np.rint((hex_reference[:, 2] + 1) / 2 * order).astype(np.int64)
    cells = np.vstack(
        [quads[:, hex_quad_node] + (order * k + hex_node_layer) * len(x_section) for k in range(len(z_vertices) - 1)]
    )

    import dolfinx.io
    import dolfinx.mesh
    from mpi4py import MPI

    mesh_comm = MPI.COMM_WORLD
    if mesh_comm.rank != 0:
        cells, x = np.empty((0, cells.shape[1]), dtype=np.int64), np.empty((0, gdim))
    cells = cells[:, dolfinx.io.gmsh.cell_perm_array(dolfinx.mesh.CellType.hexahedron, cells.shape[1])]
    domain = dolfinx.mesh.create_mesh(mesh_comm, cells, dolfinx.io.gmsh.ufl_mesh(hex_type, gdim, np.float64), x)
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)

    # Tags, from the vertices of the cells and the facets. The beam has the cells with all
    # vertices in its box (from x = C_x, inside the cylinder), and the interface the facets with all
    # vertices on one of its faces but the cylinder.
    tol = 1e-8

    def in_beam(x):
        return (
            (x[0] > C_x)
            & (x[0] < beam_x + beam_l + tol)
            & (np.abs(x[1] - C_y) < beam_h / 2 + tol)
            & (x[2] > beam_z - tol)
            & (x[2] < beam_z + beam_w + tol)
        )

    def on_beam_face(x):
        on_face = np.isclose(x[0], beam_x + beam_l, atol=tol) | np.isclose(np.abs(x[1] - C_y), beam_h / 2, atol=tol)
        on_face |= np.isclose(x[2], beam_z, atol=tol) | np.isclose(x[2], beam_z + beam_w, atol=tol)
        return in_beam(x) & on_face

    def on_cylinder(x):
        return np.isclose(np.hypot(x[0] - C_x, x[1] - C_y), r, atol=tol)

    num_cells = domain.topology.index_map(tdim).size_local
    cell_values = np.full(num_cells, PHYSICAL_MARKERS["ALE_fluid"], dtype=np.int32)
    cell_values[dolfinx.mesh.locate_entities(domain, tdim, in_beam)] = PHYSICAL_MARKERS["solid"]
    cell_markers = dolfinx.mesh.meshtags(domain, tdim, np.arange(num_cells, dtype=np.int32), cell_values)

    facet_markers = {
        "solid_fluid_interface": on_beam_face,
        "obstacle": on_cylinder,
        "inflow": lambda x: np.isclose(x[0], 0.0, atol=tol),
        "outflow": lambda x: np.isclose(x[0], L, atol=tol),
        "channel_side": lambda x: (
            np.isclose(x[1], 0.0, atol=tol)
            | np.isclose(x[1], H, atol=tol)
            | np.isclose(x[2], 0.0, atol=tol)
            | np.isclose(x[2], W, atol=tol)
        ),
        "solid_obstacle_interface": lambda x: on_cylinder(x) & in_beam(x),
    }
    facet_values = {}
    for name, marker in facet_markers.items():
        if name == "solid_fluid_interface":
            facets = dolfinx.mesh.locate_entities(domain, tdim - 1, marker)
        else:
            facets = dolfinx.mesh.locate_entities_boundary(domain, tdim - 1, marker)
        # The facets of the beam on the cylinder are also on the cylinder: the later tag wins.
        facet_values.update(dict.fromkeys(facets.tolist(), PHYSICAL_MARKERS[name]))
    facets = np.array(sorted(facet_values), dtype=np.int32)
    facet_markers = dolfinx.mesh.meshtags(
        domain, tdim - 1, facets, np.array([facet_values[facet] for facet in facets], dtype=np.int32)
    )
    cell_markers.name = "Cell tags"
    facet_markers.name = "Facet tags"

    num_cells = domain.topology.index_map(tdim).size_global
    num_vertices = domain.topology.index_map(0).size_global
    num_nodes = domain.geometry.index_map().size_global
    if mesh_comm.rank == 0:
        print(f"{num_cells = }, {num_vertices = }, {num_nodes = }, {len(z_vertices) - 1} layers in z")

    from pathlib import Path

    mesh_path = Path(f"data/meshes/fsi3d/mesh_hex{'_sec' if second_order else ''}{suffix}.xdmf")

    from dolfinx.io import XDMFFile

    with XDMFFile(MPI.COMM_WORLD, mesh_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(cell_markers, domain.geometry)
        xdmf.write_meshtags(facet_markers, domain.geometry)


def main():
    create_mesh(
        resolution_close=0.02,
        resolution_far=0.08,
        second_order=True,
        suffix="_coarse",
    )


if __name__ == "__main__":
    main()
