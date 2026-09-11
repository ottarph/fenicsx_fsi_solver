import dolfinx as dfx
import numpy as np
import pytest
import ufl

from xfsi_solver.remeshing.deformation import prescribed_interface_deformation
from xfsi_solver.remeshing.discrete_mesh import SizingField, regenerate_mesh
from xfsi_solver.remeshing.domain import FsiDomain, load_fsi2_domain
from xfsi_solver.remeshing.markers import PHYSICAL_MARKERS
from xfsi_solver.remeshing.quality import MeshQuality

MESH_PATHS = ["data/meshes/fsi2/mesh.xdmf", "data/meshes/fsi2/mesh_sec.xdmf"]

# Coarse-ish on purpose to keep the tests fast; not the production sizing.
TEST_SIZING = SizingField(size_near=0.02, size_far=0.06, distance=0.1)

BOUNDARY_NAMES = [
    "solid_fluid_interface",
    "solid_obstacle_interface",
    "obstacle",
    "inflow",
    "outflow",
    "channel_side",
]
SUBDOMAIN_NAMES = ["solid", "ALE_fluid"]


def _area(mesh: dfx.mesh.Mesh) -> float:
    return dfx.fem.assemble_scalar(dfx.fem.form(dfx.fem.Constant(mesh, 1.0) * ufl.dx(mesh)))


def _subdomain_areas(domain: FsiDomain) -> dict[str, float]:
    dx = ufl.Measure("dx", domain=domain.mesh, subdomain_data=domain.cell_tags)
    one = dfx.fem.Constant(domain.mesh, 1.0)
    return {
        name: dfx.fem.assemble_scalar(dfx.fem.form(one * dx(PHYSICAL_MARKERS[name]))) for name in SUBDOMAIN_NAMES
    }


def _assert_all_groups_present(domain: FsiDomain) -> None:
    for name in BOUNDARY_NAMES:
        assert len(domain.facet_tags.find(PHYSICAL_MARKERS[name])) > 0, f"missing boundary piece {name!r}"
    for name in SUBDOMAIN_NAMES:
        assert len(domain.cell_tags.find(PHYSICAL_MARKERS[name])) > 0, f"missing subdomain {name!r}"


@pytest.mark.parametrize("mesh_path", MESH_PATHS, ids=["tri", "tri_sec"])
def test_regenerate_undeformed_matches_original(mesh_path):
    fd = load_fsi2_domain(mesh_path)
    original_areas = _subdomain_areas(fd)
    original_degree = fd.mesh.geometry.cmaps[0].degree

    new_fd = regenerate_mesh(fd, fd.mesh.geometry.x, sizing=TEST_SIZING)

    assert new_fd.mesh.geometry.cmaps[0].degree == original_degree
    _assert_all_groups_present(new_fd)

    new_areas = _subdomain_areas(new_fd)
    assert new_areas["ALE_fluid"] == pytest.approx(original_areas["ALE_fluid"], rel=5e-3)
    # Looser for the flag: at TEST_SIZING its curved root arc is resolved by
    # only a handful of facets, so the polygonal area genuinely differs.
    assert new_areas["solid"] == pytest.approx(original_areas["solid"], rel=3e-2)


@pytest.mark.parametrize("mesh_path", MESH_PATHS, ids=["tri", "tri_sec"])
def test_regenerated_interface_is_conforming(mesh_path):
    """The point of regenerating the whole mesh rather than the fluid alone:
    every ``solid_fluid_interface`` facet must be an *interior* facet with a
    solid cell on one side and a fluid cell on the other, so the monolithic
    formulation's shared-DOF interface coupling still holds afterwards."""
    fd = load_fsi2_domain(mesh_path)
    new_fd = regenerate_mesh(fd, fd.mesh.geometry.x, sizing=TEST_SIZING)

    mesh = new_fd.mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
    index_map = mesh.topology.index_map(tdim)
    marker_of_cell = np.zeros(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    marker_of_cell[new_fd.cell_tags.indices] = new_fd.cell_tags.values

    interface = new_fd.facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])
    assert len(interface) > 0
    expected = {PHYSICAL_MARKERS["solid"], PHYSICAL_MARKERS["ALE_fluid"]}
    for facet in interface:
        cells = facet_to_cell.links(facet)
        assert len(cells) == 2, f"interface facet {facet} is not an interior facet"
        assert {int(marker_of_cell[c]) for c in cells} == expected


@pytest.mark.parametrize("mesh_path", MESH_PATHS, ids=["tri", "tri_sec"])
def test_regenerate_deformed_produces_valid_mesh(mesh_path):
    """Regenerating from a moderately (but not yet self-intersectingly)
    deformed interface -- see discrete_mesh.py's module docstring for why
    that qualifier matters -- must succeed and give a good-quality mesh."""
    fd = load_fsi2_domain(mesh_path)
    X = fd.mesh.geometry.x.copy()
    displacement = prescribed_interface_deformation(amplitude=0.05)(X.T)
    X[:, 0] += displacement[0]
    X[:, 1] += displacement[1]

    new_fd = regenerate_mesh(fd, X, sizing=TEST_SIZING)

    V = dfx.fem.functionspace(new_fd.mesh, ("CG", 1, (2,)))
    mq = MeshQuality(quality_measure="scaled_jacobian", fspace=V)
    quality = mq(dfx.fem.Function(V))
    assert quality.min() > 0.1
    _assert_all_groups_present(new_fd)


def test_regenerate_chains_across_successive_remesh_events():
    """The Phase 4 loop calls this repeatedly, each time on the *previous*
    call's output -- confirm that actually works, not just a single call on
    the original FSI2 mesh."""
    fd = load_fsi2_domain("data/meshes/fsi2/mesh.xdmf")

    fd_1 = regenerate_mesh(fd, fd.mesh.geometry.x, sizing=TEST_SIZING)

    X = fd_1.mesh.geometry.x.copy()
    displacement = prescribed_interface_deformation(amplitude=0.03)(X.T)
    X[:, 0] += displacement[0]
    X[:, 1] += displacement[1]
    fd_2 = regenerate_mesh(fd_1, X, sizing=TEST_SIZING)

    assert fd_2.mesh.topology.index_map(2).size_local > 0
    assert _area(fd_2.mesh) == pytest.approx(_area(fd.mesh), rel=5e-3)
    _assert_all_groups_present(fd_2)
