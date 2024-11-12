import gmsh, sys
gmsh.initialize(sys.argv)
import numpy as np

L, H, r = 2.5, 0.41, 0.05
C_x, C_y = 0.2, 0.2
l, h, A_x, A_y = 0.35, 0.02, 0.6, 0.2
gdim = 2

flag_left = C_x + r * np.cos(np.arcsin(h/(2*r)))
flag_right = A_x
flag_bot = C_y - h / 2
flag_top = C_y + h / 2

E_1_right = flag_left
E_2_left = 0.9

theta = np.pi / 3

resolution_far = 0.025
resolution_close = resolution_far / 5
resolution_ultra_far = H / 8

QUADS = False
# if QUADS:
#     resolution_far *= 2
#     resolution_close *= 2
#     resolution_ultra_far *= 2

flag_tl_point = gmsh.model.occ.addPoint(flag_left, flag_top, 0.0)
flag_tr_point = gmsh.model.occ.addPoint(flag_right, flag_top, 0.0)
flag_bl_point = gmsh.model.occ.addPoint(flag_left, flag_bot, 0.0)
flag_br_point = gmsh.model.occ.addPoint(flag_right, flag_bot, 0.0)

obstacle_left_point = gmsh.model.occ.addPoint(C_x - r, C_y, 0.0)
obstacle_right_point = gmsh.model.occ.addPoint(C_x + r, C_y, 0.0)

flag_top_curve = gmsh.model.occ.addLine(flag_tl_point, flag_tr_point)
flag_right_curve = gmsh.model.occ.addLine(flag_tr_point, flag_br_point)
flag_bottom_curve = gmsh.model.occ.addLine(flag_br_point, flag_bl_point)
flag_left_curve = gmsh.model.occ.addCircleArc(flag_bl_point, obstacle_right_point, flag_tl_point, center=False)

obstacle_left_curve = gmsh.model.occ.addCircleArc(flag_tl_point, obstacle_left_point, flag_bl_point, center=False)

flag_loop = gmsh.model.occ.addCurveLoop([flag_top_curve, flag_right_curve, flag_bottom_curve, flag_left_curve])
obstacle_loop = gmsh.model.occ.addCurveLoop([flag_left_curve, obstacle_left_curve])

flag = gmsh.model.occ.addPlaneSurface([flag_loop])
obstacle = gmsh.model.occ.addPlaneSurface([obstacle_loop])


gmsh.model.occ.synchronize()

channel = gmsh.model.occ.addRectangle(0, 0, 0, L, H)

gmsh.model.occ.synchronize()

tags = gmsh.model.getEntities(dim=2)


cylinder_tag = next(filter(lambda x: np.allclose(gmsh.model.occ.getCenterOfMass(*x), [C_x, C_y, 0], atol=1e-3), gmsh.model.getEntities(dim=2)))

# Sliver might not exist anymore after change to obstacle creation
try:
    sliver_tag = next(filter(lambda x: gmsh.model.occ.getMass(*x) < 1e-4, gmsh.model.getEntities(dim=2)))
    tags, _ = gmsh.model.occ.cut(tags, [cylinder_tag, sliver_tag])
except StopIteration:
    tags, _ = gmsh.model.occ.cut(tags, [cylinder_tag])


gmsh.model.occ.synchronize()


fluid_tags = []
solid_tags = []
ALE_tags = []


for tag in tags:
    if gmsh.model.occ.getMass(*tag) < 1.1 * (flag_right - flag_left) * h:
    # if np.isclose(gmsh.model.occ.getCenterOfMass(*tag)[0], 0.5*(flag_right+flag_left), atol=1e-6):

        solid_tags.append(tag[1])
    else:
        ALE_tags.append(tag[1])
        print(ALE_tags)


gmsh.model.addPhysicalGroup(2, solid_tags, 1, name="solid")
gmsh.model.addPhysicalGroup(2, ALE_tags, 2, name="ALE_fluid")

solid_fluid_int_tags = []
solid_obstacle_int_tags = []

inflow_tags = []
outflow_tags = []
obstacle_tags = []
channel_side_tags = []

line_tags = gmsh.model.getEntities(dim=1)
for line_tag in line_tags:
    cell_adj, point_adj = gmsh.model.getAdjacencies(1, line_tag[1])
    a, b = [gmsh.model.getValue(0, p, []) for p in point_adj]
    if len(cell_adj) == 2: # Interface boundary
        solid_fluid_int_tags.append(line_tag[1])
    else: # Domain boundary
        if np.isclose(a[0], 0) and np.isclose(b[0], 0):
            inflow_tags.append(line_tag[1])
        elif np.isclose(a[0], L) and np.isclose(b[0], L):
            outflow_tags.append(line_tag[1])
        elif np.isclose(a[1], 0) and np.isclose(b[1], 0):
            channel_side_tags.append(line_tag[1])
        elif np.isclose(a[1], H) and np.isclose(b[1], H):
            channel_side_tags.append(line_tag[1])
        elif np.isclose(gmsh.model.occ.getCenterOfMass(1, line_tag[1])[0], C_x, atol=1e-2):
            obstacle_tags.append(line_tag[1])
        elif np.isclose(gmsh.model.occ.getCenterOfMass(1, line_tag[1])[0], flag_left, atol=2e-3):
            solid_obstacle_int_tags.append(line_tag[1])
        else:
            ValueError


gmsh.model.addPhysicalGroup(1, solid_fluid_int_tags, 11, name="solid_fluid_interface")

gmsh.model.addPhysicalGroup(1, obstacle_tags, 21, name="obstacle")
gmsh.model.addPhysicalGroup(1, inflow_tags, 22, name="inflow")
gmsh.model.addPhysicalGroup(1, outflow_tags, 23, name="outflow")
gmsh.model.addPhysicalGroup(1, channel_side_tags, 24, name="channel_side")
gmsh.model.addPhysicalGroup(1, solid_obstacle_int_tags, 25, name="solid_obstacle_interface")

gmsh.model.occ.synchronize()



for tag in tags:
    
    if gmsh.model.occ.getMass(*tag) < 1.1 * (flag_right - flag_left) * h:
        # The solid domain
        curve_adj = gmsh.model.getAdjacencies(2, tag[1])[1]
        for curve_tag in curve_adj:
            point_adj = gmsh.model.getAdjacencies(1, curve_tag)[1]
            for point_tag in point_adj:
                # Set mesh size close to the solid to resolution_close for all point
                # entities on the solid boundary.
                gmsh.model.mesh.setSize([(0, point_tag)], resolution_close)
    
    else:
        # The fluid domain
        curve_adj = gmsh.model.getAdjacencies(2, tag[1])[1]
        for curve_tag in curve_adj:
            com = gmsh.model.occ.getCenterOfMass(1, curve_tag)
            if np.isclose(com[1], C_y) and C_x - r < com[0] < C_x + r: # The left side of the obstacle
                point_adj = gmsh.model.getAdjacencies(1, curve_tag)[1]
                for point_tag in point_adj:
                    # Set mesh size close to the obstacle to resolution_close for all point
                    # entities on the left obstacle boundary.
                    gmsh.model.mesh.setSize([(0, point_tag)], resolution_close)


gmsh.model.occ.synchronize()

gmsh.model.occ.addPoint(E_2_left, 0.0, 0.0, meshSize=resolution_far)
gmsh.model.occ.addPoint(E_2_left, H, 0.0, meshSize=resolution_far)


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
mesh_path = Path("data/meshes/fsi2/mesh.xdmf")

from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, mesh_path, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_markers, domain.geometry)
    xdmf.write_meshtags(facet_markers, domain.geometry)

# gmsh.fltk.finalize()
# gmsh.fltk.run()
