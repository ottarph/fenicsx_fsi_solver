import dolfinx
import dolfinx.fem
import io4dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from scifem.mesh import create_geometry_function_space


def reconstruct_mesh(mesh, coordinate_element_degree):
    """Copy of ``mesh`` (shared topology) with a coordinate element of the given degree."""
    cmap = dolfinx.fem.coordinate_element(
        mesh.topology.cell_type,
        coordinate_element_degree,
        dtype=mesh.geometry.x.dtype,
    )
    return dolfinx.fem.interpolate_geometry(mesh, cmap)


def u_ex(x):
    return (
        x[0] ** 2 + x[1] ** 2,
        x[0] * x[1],
        x[0] + x[1],
    )


def p_ex(x):
    return x[0] + x[1]


mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
V = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))
Q = dolfinx.fem.functionspace(mesh, ("P", 1))

u = dolfinx.fem.Function(V)
p = dolfinx.fem.Function(Q)
u.interpolate(u_ex)
p.interpolate(p_ex)
u.x.scatter_forward()
p.x.scatter_forward()

mesh_P2 = reconstruct_mesh(mesh, coordinate_element_degree=2)
cell_map = mesh.topology.index_map(mesh.topology.dim)
num_cells_local = cell_map.size_local + cell_map.num_ghosts
V_P2 = create_geometry_function_space(mesh_P2, N=3)
u_p2 = dolfinx.fem.Function(V_P2, name="u")
u_p2.interpolate(
    u,
    cells0=np.arange(num_cells_local, dtype=np.int32),
    cells1=np.arange(num_cells_local, dtype=np.int32),
)

x_P2 = ufl.SpatialCoordinate(mesh_P2)
print(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(
            ufl.inner(u_p2 - ufl.as_vector(u_ex(x_P2)), u_p2 - ufl.as_vector(u_ex(x_P2))) * ufl.dx(domain=mesh_P2)
        )
    )
)

P_P2 = create_geometry_function_space(mesh_P2, N=1)
p_p2 = dolfinx.fem.Function(P_P2, name="p")
p_p2.interpolate(
    p,
    cells0=np.arange(num_cells_local, dtype=np.int32),
    cells1=np.arange(num_cells_local, dtype=np.int32),
)
print(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(p_p2 - p_ex(x_P2), p_p2 - p_ex(x_P2)) * ufl.dx(domain=mesh_P2))
    )
)

io4dolfinx.write_mesh("solution.vtkhdf", mesh_P2, backend="vtkhdf", backend_args={"compression": "gzip"})
io4dolfinx.write_point_data(
    "solution.vtkhdf",
    u_p2,
    time=1.1,
    mode=io4dolfinx.FileMode.append,
    backend="vtkhdf",
    backend_args={"compression": "gzip"},
)
io4dolfinx.write_point_data(
    "solution.vtkhdf",
    p_p2,
    time=1.1,
    mode=io4dolfinx.FileMode.append,
    backend="vtkhdf",
    backend_args={"compression": "gzip"},
)


mesh_in = io4dolfinx.read_mesh("solution.vtkhdf", MPI.COMM_WORLD, backend="vtkhdf")

u_in = io4dolfinx.read_point_data(
    "solution.vtkhdf",
    name="u",
    mesh=mesh_in,
    time=1.1,
    backend="vtkhdf",
)
u_in.name = "u"
p_in = io4dolfinx.read_point_data(
    "solution.vtkhdf",
    name="p",
    mesh=mesh_in,
    time=1.1,
    backend="vtkhdf",
)
p_in.name = "p"


mesh_in_P1 = reconstruct_mesh(mesh_in, coordinate_element_degree=1)
num_cells_local = (
    mesh_in_P1.topology.index_map(mesh_in_P1.topology.dim).size_local
    + mesh_in_P1.topology.index_map(mesh_in_P1.topology.dim).num_ghosts
)
V_in_P1 = dolfinx.fem.functionspace(mesh_in_P1, ("P", 2, (3,)))
u_in_P1 = dolfinx.fem.Function(V_in_P1, name="u")
u_in_P1.interpolate(
    u_in,
    cells0=np.arange(num_cells_local, dtype=np.int32),
    cells1=np.arange(num_cells_local, dtype=np.int32),
)
x_in_P1 = ufl.SpatialCoordinate(mesh_in_P1)
print(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(
            ufl.inner(
                u_in_P1 - ufl.as_vector(u_ex(x_in_P1)),
                u_in_P1 - ufl.as_vector(u_ex(x_in_P1)),
            )
            * ufl.dx(domain=mesh_in_P1)
        )
    )
)

p_in_P1 = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh_in_P1, ("P", 1)), name="p")
p_in_P1.interpolate(
    p_in,
    cells0=np.arange(num_cells_local, dtype=np.int32),
    cells1=np.arange(num_cells_local, dtype=np.int32),
)
print(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(p_in_P1 - p_ex(x_in_P1), p_in_P1 - p_ex(x_in_P1)) * ufl.dx(domain=mesh_in_P1))
    )
)
