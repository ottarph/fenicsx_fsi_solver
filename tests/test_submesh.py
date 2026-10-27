import dolfinx
import numpy as np
from mpi4py import MPI

from xfsi_solver.tools.submesh import SubmeshComposite

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"
FLUID, SOLID = 2, 1  # cell tags in the FSI mesh
ELEMENT = ("Lagrange", 2, (2,))


def displacement(x):
    """A non-polynomial field, which interpolates to the same dof values on any mesh with the element."""
    return np.vstack([np.sin(3 * x[0]) * x[1], np.cos(2 * x[1]) + x[0]])


def _setup():
    """u on the FSI mesh with u_s and u_f on its solid and fluid submeshes, as in the solvers."""
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, MESH, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name="Cell tags")
    solid_mesh, solid_map, _, _ = dolfinx.mesh.create_submesh(mesh, 2, cell_tags.find(SOLID))
    fluid_mesh, fluid_map, _, _ = dolfinx.mesh.create_submesh(mesh, 2, cell_tags.find(FLUID))
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ELEMENT))
    u_s = dolfinx.fem.Function(dolfinx.fem.functionspace(solid_mesh, ELEMENT))
    u_f = dolfinx.fem.Function(dolfinx.fem.functionspace(fluid_mesh, ELEMENT))
    return u, u_s, u_f, SubmeshComposite(u, [(u_s, solid_map), (u_f, fluid_map)])


def test_gather_pieces_parent_from_parts():
    u, u_s, u_f, composite = _setup()
    u_s.interpolate(displacement)
    u_f.interpolate(displacement)
    composite.gather()
    expected = dolfinx.fem.Function(u.function_space)
    expected.interpolate(displacement)
    # Owned and ghost dofs, as gather updates the ghosts
    np.testing.assert_allclose(u.x.array, expected.x.array, atol=1e-14)


def test_scatter_restores_gathered_parts():
    u, u_s, u_f, composite = _setup()
    u.interpolate(displacement)
    composite.scatter()
    for part in (u_s, u_f):
        expected = dolfinx.fem.Function(part.function_space)
        expected.interpolate(displacement)
        np.testing.assert_allclose(part.x.array, expected.x.array, atol=1e-14)


def test_gather_of_different_parts_takes_each_on_its_cells():
    """With different values in the parts, each cell of u holds the values of the part it belongs to."""
    u, u_s, u_f, composite = _setup()
    u_s.x.array[:] = 1.0
    u_f.x.array[:] = 2.0
    composite.gather()
    u_s.x.array[:] = 0.0
    u_f.x.array[:] = 0.0
    composite.scatter()
    # Interface dofs take the value of the part listed last, the fluid. The copies are exact up to
    # round-off in the interpolation.
    is_solid = np.isclose(u_s.x.array, 1.0, rtol=0, atol=1e-14)
    is_fluid = np.isclose(u_s.x.array, 2.0, rtol=0, atol=1e-14)
    assert np.all(is_solid | is_fluid)
    assert is_solid.any() and is_fluid.any()
    np.testing.assert_allclose(u_f.x.array, 2.0, rtol=0, atol=1e-14)
