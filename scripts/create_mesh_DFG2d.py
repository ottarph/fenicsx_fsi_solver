import gmsh, sys
gmsh.initialize(sys.argv)
import numpy as np

L, H, r = 2.2, 0.41, 0.05
C_x, C_y = 0.2, 0.2
gdim = 2

E_2_left = 0.9


resolution_far = 0.025
resolution_close = resolution_far / 5
# resolution_ultra_far = H / 8
resolution_ultra_far = resolution_far

QUADS = False
# if QUADS:
#     resolution_far *= 2
#     resolution_close *= 2
#     resolution_ultra_far *= 2


obstacle = gmsh.model.occ.addDisk(C_x, C_y, 0.0, r, r)

gmsh.model.occ.synchronize()

channel = gmsh.model.occ.addRectangle(0, 0, 0, L, H)

gmsh.model.occ.synchronize()

tags = gmsh.model.getEntities(dim=2)

cylinder_tag = next(filter(lambda x: np.allclose(gmsh.model.occ.getCenterOfMass(*x), [C_x, C_y, 0], atol=1e-3), gmsh.model.getEntities(dim=2)))
tags, _ = gmsh.model.occ.cut(tags, [cylinder_tag])


gmsh.model.occ.synchronize()


fluid_tags = []
fluid_tags.append(tags[0][1])
gmsh.model.addPhysicalGroup(2, fluid_tags, 2, name="fluid")


inflow_tags = []
outflow_tags = []
obstacle_tags = []
channel_side_tags = []

line_tags = gmsh.model.getEntities(dim=1)
for line_tag in line_tags:
    com = gmsh.model.occ.getCenterOfMass(1, line_tag[1])
    if np.allclose(com, [C_x, C_y, 0.0]):
        obstacle_tags.append(line_tag[1])
    elif np.isclose(com[0], 0.0):
        inflow_tags.append(line_tag[1])
    elif np.isclose(com[0], L):
        outflow_tags.append(line_tag[1])
    elif np.isclose(com[1], 0.0) or np.isclose(com[1], H):
        channel_side_tags.append(line_tag[1])


gmsh.model.addPhysicalGroup(1, obstacle_tags, 21, name="obstacle")
gmsh.model.addPhysicalGroup(1, inflow_tags, 22, name="inflow")
gmsh.model.addPhysicalGroup(1, outflow_tags, 23, name="outflow")
gmsh.model.addPhysicalGroup(1, channel_side_tags, 24, name="channel_side")

gmsh.model.occ.synchronize()


gmsh.model.occ.addPoint(E_2_left, 0.0, 0.0, meshSize=resolution_far)
gmsh.model.occ.addPoint(E_2_left, H, 0.0, meshSize=resolution_far)


gmsh.model.occ.addPoint(C_x, C_y+r, 0.0, meshSize=resolution_close)
gmsh.model.occ.addPoint(C_x, C_y-r, 0.0, meshSize=resolution_close)
gmsh.model.occ.addPoint(C_x+r, C_y, 0.0, meshSize=resolution_close)
gmsh.model.occ.addPoint(C_x-r, C_y, 0.0, meshSize=resolution_close)


gmsh.model.occ.synchronize()

point_tags = gmsh.model.getEntities(dim=0)

for point_tag in point_tags:
    point_x = gmsh.model.getValue(0, point_tag[1], [])
    if point_x[0] > 0.9 * L:
        gmsh.model.mesh.setSize([(0, point_tag[1])], resolution_ultra_far)
    else:
        if np.isclose(point_x[0], 0.0) or np.isclose(point_x[1], 0.0) or np.isclose(point_x[1], H):
            gmsh.model.mesh.setSize([(0, point_tag[1])], resolution_far)


# 1: MeshAdapt
# 2: Automatic
# 3: Initial mesh only
# 5: Delaunay
# 6: Frontal-Delaunay (Default)
# 7: BAMG
# 8: Frontal-Delaunay for Quads
# 9: Packing of Parallelograms
# 11: Quasi-structured Quad

# gmsh.option.setNumber("Mesh.Algorithm", 5)


if QUADS:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)


gmsh.model.occ.synchronize()

gmsh.model.mesh.generate(gdim)

# Use second order mesh
# gmsh.model.mesh.setOrder(2)


# gmsh.write("test.msh")

from dolfinx.io import gmshio
from mpi4py import MPI
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

domain.topology.create_connectivity(1, 2)

if mesh_comm.rank == gmsh_model_rank:
    print(f"{domain.geometry.x.shape = }")

from pathlib import Path
mesh_path = Path("data/meshes/dfg2d/mesh.xdmf")

from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, mesh_path, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_markers, domain.geometry)
    xdmf.write_meshtags(facet_markers, domain.geometry)

gmsh.fltk.finalize()
gmsh.fltk.run()
