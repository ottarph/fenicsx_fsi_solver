import dolfinx as dfx
import numpy as np
import pytest

from xfsi_solver.remeshing.deformation import prescribed_interface_deformation
from xfsi_solver.remeshing.discrete_mesh import SizingField, regenerate_fluid_mesh
from xfsi_solver.remeshing.fluid_domain import load_fsi2_fluid_domain
from xfsi_solver.remeshing.transfer import DEFAULT_PADDING, transfer_field

TEST_SIZING = SizingField(size_near=0.02, size_far=0.06, distance=0.1)


def _known_scalar(x):
    return np.sin(3 * x[0]) * np.cos(2 * x[1])


def _known_vector(x):
    values = np.zeros((2, x.shape[1]))
    values[0] = np.sin(2 * x[0]) * x[1]
    values[1] = np.cos(x[0]) * x[1] ** 2
    return values


@pytest.fixture
def old_and_new_fluid_domain():
    """A regenerated (Phase 2) fluid mesh from a moderately deformed
    (Phase 1) old one -- exactly the pair of meshes a real remesh event
    would produce."""
    old = load_fsi2_fluid_domain("data/meshes/fsi2/mesh.xdmf")
    X = old.mesh.geometry.x.copy()
    displacement = prescribed_interface_deformation(amplitude=0.05)(X.T)
    X[:, 0] += displacement[0]
    X[:, 1] += displacement[1]
    new = regenerate_fluid_mesh(old.mesh, X, sizing=TEST_SIZING)
    return old, new


@pytest.mark.parametrize(
    "degree,shape,fn",
    [(2, (), _known_scalar), (1, (2,), _known_vector)],
    ids=["scalar_cg2", "vector_cg1"],
)
def test_transfer_matches_direct_interpolation(old_and_new_fluid_domain, degree, shape, fn):
    """Interpolating a known analytic function onto the old mesh, then
    transferring it to the new mesh, should agree with interpolating that
    same function directly onto the new mesh -- the check from
    notes/remeshing/implementation-plan.md §7 Phase 3.
    """
    old, new = old_and_new_fluid_domain
    element = ("CG", degree, shape) if shape else ("CG", degree)

    V_old = dfx.fem.functionspace(old.mesh, element)
    f_old = dfx.fem.Function(V_old)
    f_old.interpolate(fn)

    V_new = dfx.fem.functionspace(new.mesh, element)
    f_transferred = transfer_field(f_old, V_new)

    f_direct = dfx.fem.Function(V_new)
    f_direct.interpolate(fn)

    error = np.abs(f_transferred.x.array - f_direct.x.array)
    # Ordinary P1/P2 interpolation error between two different (but
    # geometrically close) triangulations of the same domain, not a
    # boundary-miss artifact -- see the padding sweep this bound is based
    # on in transfer.py's DEFAULT_PADDING docstring.
    assert error.max() < 1e-3


def test_default_padding_matters(old_and_new_fluid_domain):
    """Guard against the DEFAULT_PADDING regressing to a value too small
    for this mesh's resolution: reproduce the failure mode found while
    calibrating it (points near a curved boundary silently left at 0,
    i.e. NOT a small numerical error but a large, easy-to-miss one) at a
    too-small padding, and confirm DEFAULT_PADDING avoids it.
    """
    old, new = old_and_new_fluid_domain
    V_old = dfx.fem.functionspace(old.mesh, ("CG", 2))
    f_old = dfx.fem.Function(V_old)
    f_old.interpolate(_known_scalar)
    V_new = dfx.fem.functionspace(new.mesh, ("CG", 2))
    f_direct = dfx.fem.Function(V_new)
    f_direct.interpolate(_known_scalar)

    too_small_padding = 1e-6
    assert too_small_padding < DEFAULT_PADDING
    f_bad = transfer_field(f_old, V_new, padding=too_small_padding)
    assert np.abs(f_bad.x.array - f_direct.x.array).max() > 0.1

    f_good = transfer_field(f_old, V_new, padding=DEFAULT_PADDING)
    assert np.abs(f_good.x.array - f_direct.x.array).max() < 1e-3
