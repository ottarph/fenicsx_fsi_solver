import dolfinx as dfx
import pytest
import ufl

from xfsi_solver.remeshing.deformation import prescribed_interface_deformation
from xfsi_solver.remeshing.discrete_mesh import SizingField, regenerate_fluid_mesh
from xfsi_solver.remeshing.fluid_domain import load_fsi2_fluid_domain
from xfsi_solver.remeshing.markers import PHYSICAL_MARKERS
from xfsi_solver.remeshing.quality import MeshQuality

MESH_PATHS = ["data/meshes/fsi2/mesh.xdmf", "data/meshes/fsi2/mesh_sec.xdmf"]

# Coarse-ish on purpose to keep the tests fast; not the production sizing.
TEST_SIZING = SizingField(size_near=0.02, size_far=0.06, distance=0.1)

BOUNDARY_NAMES = ["solid_fluid_interface", "obstacle", "inflow", "outflow", "channel_side"]


def _area(mesh: dfx.mesh.Mesh) -> float:
    return dfx.fem.assemble_scalar(dfx.fem.form(dfx.fem.Constant(mesh, 1.0) * ufl.dx(mesh)))


@pytest.mark.parametrize("mesh_path", MESH_PATHS, ids=["tri", "tri_sec"])
def test_regenerate_undeformed_matches_original(mesh_path):
    fd = load_fsi2_fluid_domain(mesh_path)
    original_area = _area(fd.mesh)
    original_degree = fd.mesh.geometry.cmaps[0].degree

    new_fd = regenerate_fluid_mesh(fd, fd.mesh.geometry.x, sizing=TEST_SIZING)

    assert new_fd.mesh.geometry.cmaps[0].degree == original_degree
    assert _area(new_fd.mesh) == pytest.approx(original_area, rel=5e-3)
    for name in BOUNDARY_NAMES:
        n = len(new_fd.facet_tags.find(PHYSICAL_MARKERS[name]))
        assert n > 0, f"regenerated mesh is missing boundary piece {name!r}"


@pytest.mark.parametrize("mesh_path", MESH_PATHS, ids=["tri", "tri_sec"])
def test_regenerate_deformed_produces_valid_mesh(mesh_path):
    """Regenerating from a moderately (but not yet self-intersectingly)
    deformed interface -- see discrete_mesh.py's module docstring for why
    that qualifier matters -- must succeed and give a good-quality mesh."""
    fd = load_fsi2_fluid_domain(mesh_path)
    X = fd.mesh.geometry.x.copy()
    displacement = prescribed_interface_deformation(amplitude=0.05)(X.T)
    X[:, 0] += displacement[0]
    X[:, 1] += displacement[1]

    new_fd = regenerate_fluid_mesh(fd, X, sizing=TEST_SIZING)

    V = dfx.fem.functionspace(new_fd.mesh, ("CG", 1, (2,)))
    mq = MeshQuality(quality_measure="scaled_jacobian", fspace=V)
    quality = mq(dfx.fem.Function(V))
    assert quality.min() > 0.1
    for name in BOUNDARY_NAMES:
        assert len(new_fd.facet_tags.find(PHYSICAL_MARKERS[name])) > 0


def test_regenerate_chains_across_successive_remesh_events():
    """The Phase 4 loop calls this repeatedly, each time on the *previous*
    call's output -- confirm that actually works, not just a single call on
    the original FSI2 mesh."""
    fd = load_fsi2_fluid_domain("data/meshes/fsi2/mesh.xdmf")

    fd_1 = regenerate_fluid_mesh(fd, fd.mesh.geometry.x, sizing=TEST_SIZING)

    X = fd_1.mesh.geometry.x.copy()
    displacement = prescribed_interface_deformation(amplitude=0.03)(X.T)
    X[:, 0] += displacement[0]
    X[:, 1] += displacement[1]
    fd_2 = regenerate_fluid_mesh(fd_1, X, sizing=TEST_SIZING)

    assert fd_2.mesh.topology.index_map(2).size_local > 0
    assert _area(fd_2.mesh) == pytest.approx(_area(fd.mesh), rel=5e-3)
    for name in BOUNDARY_NAMES:
        assert len(fd_2.facet_tags.find(PHYSICAL_MARKERS[name])) > 0
