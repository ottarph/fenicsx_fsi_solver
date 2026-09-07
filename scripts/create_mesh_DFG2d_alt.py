# Copyright (C) 2023-2025 Jørgen S. Dokken, Ottar Hellan
#
# SPDX-License-Identifier: CC-BY-4.0

# Code adapted from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html,
# by Jørgen S. Dokken under Creative Commons Attribution 4.0 International License 
# https://creativecommons.org/licenses/by/4.0/.


import gmsh
import numpy as np

from mpi4py import MPI
import dolfinx.io

gmsh.initialize()

QUADS = True
MESH_ORDER = 1

L = 2.2
H = 0.41
c_x = c_y = 0.2
r = 0.05

if QUADS:
    res_min = r / 3
    res_max = H / 4
else:
    res_min = r / 6
    res_max = H / 8

gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)


if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()


fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")


inlet_marker, outlet_marker, wall_marker, obstacle_marker = 22, 23, 24, 21
inflow, outflow, walls, obstacle = [], [], [], []
if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H / 2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")


# Create distance field from obstacle.
# Add threshold of mesh sizes based on the distance field
# LcMax -                  /--------
#                      /
# LcMin -o---------/
#        |         |       |
#       Point    DistMin DistMax
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", res_max)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)


if mesh_comm.rank == model_rank:
    # Mesh.Algorithm:
    # 1: MeshAdapt
    # 2: Automatic
    # 3: Initial mesh only
    # 5: Delaunay
    # 6: Frontal-Delaunay (Default)
    # 7: BAMG
    # 8: Frontal-Delaunay for Quads
    # 9: Packing of Parallelograms
    # 11: Quasi-structured Quad

    gmsh.model.mesh.setOrder(MESH_ORDER)

    if QUADS:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    else:
        gmsh.option.setNumber("Mesh.Algorithm", 6)

    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.optimize("Netgen")

    # gmsh.fltk.finalize()
    # gmsh.fltk.run()


out = dolfinx.io.gmsh.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
mesh = out.mesh
ft = out.facet_tags
ft.name = "Facet tags"

with dolfinx.io.XDMFFile(mesh_comm, f"data/meshes/dfg2d_alt/mesh_{'quad' if QUADS else 'tri'}.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)
