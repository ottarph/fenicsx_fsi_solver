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


def main():
    create_mesh(
        resolution_close=0.02,
        resolution_far=0.08,
        second_order=True,
        suffix="_coarse",
    )


if __name__ == "__main__":
    main()
