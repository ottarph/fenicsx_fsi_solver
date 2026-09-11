import dolfinx as dfx
import numpy as np
import pytest
import ufl

from xfsi_solver.remeshing import fsi2_geometry as geo
from xfsi_solver.remeshing.deformation import prescribed_interface_deformation
from xfsi_solver.remeshing.discrete_mesh import SizingField, regenerate_mesh
from xfsi_solver.remeshing.domain import FsiDomain, load_fsi2_domain
from xfsi_solver.remeshing.markers import PHYSICAL_MARKERS
from xfsi_solver.remeshing.quality import MeshQuality

MESH_PATHS = ["data/meshes/fsi2/mesh.xdmf", "data/meshes/fsi2/mesh_sec.xdmf"]

# Coarse-ish on purpose to keep the tests fast; not the production sizing.
TEST_SIZING = SizingField(size_near=0.02, size_far=0.06, size_outflow=0.08)

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


def _cell_sizes(domain: FsiDomain) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell mean edge length, and distance of each cell's centroid from
    the flag/cylinder surfaces."""
    n = domain.mesh.topology.index_map(domain.mesh.topology.dim).size_local
    corners = domain.mesh.geometry.x[domain.mesh.geometry.dofmaps[0][:n, :3]][:, :, :2]
    h = np.mean([np.linalg.norm(corners[:, (i + 1) % 3] - corners[:, i], axis=1) for i in range(3)], axis=0)

    centroid = corners.mean(axis=1)
    nearest_on_flag = np.clip(centroid[:, 0], geo.FLAG_LEFT, geo.FLAG_RIGHT)
    to_flag = np.sqrt((centroid[:, 0] - nearest_on_flag) ** 2 + (centroid[:, 1] - geo.C_Y) ** 2)
    to_flag = np.maximum(to_flag - geo.FLAG_THICKNESS / 2, 0.0)
    to_cylinder = np.abs(np.linalg.norm(centroid - np.array([geo.C_X, geo.C_Y]), axis=1) - geo.R)
    return h, np.minimum(to_flag, to_cylinder)


def test_default_sizing_reproduces_the_original_resolution():
    """The default ``SizingField`` must regenerate a mesh at the resolution
    the simulation started from, not a coarser one.

    This is the whole point of stating the grading explicitly (see
    ``SizingField``'s docstring): a remesh that quietly halves the
    resolution near the flag every time it fires is worse than no remesh.
    Checked both overall and specifically in the refined band next to the
    flag/cylinder, since a mesh can match on average while being far too
    coarse exactly where it matters.
    """
    fd = load_fsi2_domain("data/meshes/fsi2/mesh.xdmf")
    new_fd = regenerate_mesh(fd, fd.mesh.geometry.x)

    n_original = fd.mesh.topology.index_map(2).size_local
    n_new = new_fd.mesh.topology.index_map(2).size_local
    assert n_new == pytest.approx(n_original, rel=0.15)

    h_original, d_original = _cell_sizes(fd)
    h_new, d_new = _cell_sizes(new_fd)
    assert h_new.mean() == pytest.approx(h_original.mean(), rel=0.1)

    near = 0.01
    assert h_new[d_new < near].mean() == pytest.approx(h_original[d_original < near].mean(), rel=0.15)


def _n_boundary_nodes(domain: FsiDomain, name: str) -> int:
    facets = domain.facet_tags.find(PHYSICAL_MARKERS[name])
    return len(np.unique(dfx.mesh.entities_to_geometry(domain.mesh, 1, facets)))


def _interface_node_coords(domain: FsiDomain) -> np.ndarray:
    """Sorted (x, y) coordinates of every node on ``solid_fluid_interface``,
    so two regenerated meshes' interface discretizations can be compared
    regardless of their (unrelated) node numbering."""
    facets = domain.facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])
    nodes = np.unique(dfx.mesh.entities_to_geometry(domain.mesh, 1, facets)[:, :2])
    coords = domain.mesh.geometry.x[nodes][:, :2]
    return coords[np.lexsort((coords[:, 1], coords[:, 0]))]


def test_pinned_interface_survives_a_resizing_remesh():
    """``PINNED_BOUNDARIES`` (``discrete_mesh.py``'s docstring) exempts
    ``solid_fluid_interface`` from ``SizingField`` entirely: regenerating
    again with a much finer field must leave the interface's node count and
    positions exactly as they were, while still changing everything else."""
    fd = load_fsi2_domain("data/meshes/fsi2/mesh.xdmf")
    fine_sizing = SizingField(size_near=0.005, size_far=0.02, size_outflow=0.03)

    fd_1 = regenerate_mesh(fd, fd.mesh.geometry.x, sizing=TEST_SIZING)
    fd_2 = regenerate_mesh(fd_1, fd_1.mesh.geometry.x, sizing=fine_sizing)

    interface_1 = _interface_node_coords(fd_1)
    interface_2 = _interface_node_coords(fd_2)
    assert interface_2.shape == interface_1.shape
    np.testing.assert_allclose(interface_2, interface_1, atol=1e-12)

    # The much finer field must still have visibly changed everything else,
    # confirming the interface's stability isn't just because nothing moved.
    assert _n_boundary_nodes(fd_2, "obstacle") > 2 * _n_boundary_nodes(fd_1, "obstacle")
    assert fd_2.mesh.topology.index_map(2).size_local > 2 * fd_1.mesh.topology.index_map(2).size_local


def test_pinned_interface_still_conforms_after_deformation():
    """The interface being pinned must not compromise the conformity check
    that motivated regenerating the whole mesh in the first place (§10 of
    notes/remeshing/implementation-log.md) -- verify it still holds once the
    interface has actually moved, not just on the undeformed mesh."""
    fd = load_fsi2_domain("data/meshes/fsi2/mesh.xdmf")
    X = fd.mesh.geometry.x.copy()
    displacement = prescribed_interface_deformation(amplitude=0.05)(X.T)
    X[:, 0] += displacement[0]
    X[:, 1] += displacement[1]

    new_fd = regenerate_mesh(fd, X, sizing=TEST_SIZING)

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
        assert len(cells) == 2
        assert {int(marker_of_cell[c]) for c in cells} == expected

    # And the interface itself must have been carried through unresampled:
    # same node count as the undeformed input, now at the deformed positions.
    assert _n_boundary_nodes(new_fd, "solid_fluid_interface") == _n_boundary_nodes(fd, "solid_fluid_interface")


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
