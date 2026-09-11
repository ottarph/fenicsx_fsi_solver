import dolfinx as dfx
import numpy as np
import pytest

from xfsi_solver.remeshing.deformation import prescribed_interface_deformation
from xfsi_solver.remeshing.domain import load_fsi2_domain
from xfsi_solver.remeshing.markers import PHYSICAL_MARKERS

MESH_PATHS = ["data/meshes/fsi2/mesh.xdmf", "data/meshes/fsi2/mesh_sec.xdmf"]


@pytest.mark.parametrize("mesh_path", MESH_PATHS, ids=["tri", "tri_sec"])
def test_deformation_vanishes_on_fixed_boundary(mesh_path):
    """The prescribed deformation must leave every non-interface boundary
    piece exactly fixed, so a remesh never needs to move the channel walls,
    inflow/outflow, or the obstacle -- see
    notes/remeshing/implementation-plan.md §6.
    """
    fd = load_fsi2_domain(mesh_path)
    V = dfx.fem.functionspace(fd.mesh, ("CG", 1, (2,)))
    u = dfx.fem.Function(V)
    u.interpolate(prescribed_interface_deformation(amplitude=0.1))
    values = u.x.array.reshape(-1, 2)

    for name in ["obstacle", "inflow", "outflow", "channel_side"]:
        facets = fd.facet_tags.find(PHYSICAL_MARKERS[name])
        dofs = dfx.fem.locate_dofs_topological(V, 1, facets)
        assert len(dofs) > 0, f"no {name} facets found"
        # Not exactly 0.0 everywhere: the flag's root corner sits on the
        # obstacle arc too, and on mesh_sec.xdmf its x coordinate comes back
        # from the mesh file ~3e-17 to the *right* of geo.FLAG_LEFT, so the
        # clamped-root factor t is ~1e-16 rather than 0 there and the
        # deformation is ~1e-34 rather than identically zero. Everywhere
        # else the compactly-supported envelope still gives exact zeros.
        assert np.allclose(values[dofs], 0.0, atol=1e-14), f"deformation is nonzero on fixed boundary {name!r}"


@pytest.mark.parametrize("mesh_path", MESH_PATHS, ids=["tri", "tri_sec"])
def test_deformation_matches_prescribed_shape_on_interface(mesh_path):
    """On the interface, dy should be 0 at the clamped root and amplitude at
    the tip, per the cantilever-like ``t**2`` profile."""
    fd = load_fsi2_domain(mesh_path)
    V = dfx.fem.functionspace(fd.mesh, ("CG", 1, (2,)))
    amplitude = 0.1
    u = dfx.fem.Function(V)
    u.interpolate(prescribed_interface_deformation(amplitude))
    values = u.x.array.reshape(-1, 2)

    facets = fd.facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])
    dofs = dfx.fem.locate_dofs_topological(V, 1, facets)
    dy = values[dofs, 1]

    assert dy.min() == pytest.approx(0.0, abs=1e-12)
    assert dy.max() == pytest.approx(amplitude, abs=1e-12)
    # dx is identically zero (pure vertical bending) everywhere by construction.
    assert np.all(values[dofs, 0] == 0.0)
