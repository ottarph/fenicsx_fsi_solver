import dolfinx as dfx
import pytest

from xfsi_solver.remeshing.deformation import prescribed_interface_deformation
from xfsi_solver.remeshing.fluid_domain import load_fsi2_fluid_domain
from xfsi_solver.remeshing.quality import MeshQuality

MESH_PATHS = ["data/meshes/fsi2/mesh.xdmf", "data/meshes/fsi2/mesh_sec.xdmf"]

# Calibrated against data/meshes/fsi2/mesh.xdmf: no inverted cells up to
# amplitude 0.05, some inverted cells by 0.08 (see
# notes/remeshing/implementation-plan.md for how this was found).
HEALTHY_AMPLITUDE = 0.05
DEGENERATE_AMPLITUDE = 0.1


def _n_inverted_cells(V: dfx.fem.FunctionSpace, u: dfx.fem.Function) -> int:
    """Ground truth for "how many triangles inverted", independent of
    pvmeshquality/Verdict, by directly signed-area-checking the CG1
    triangulation warped by ``u``. Used to check the scaled_jacobian metric
    is actually tracking real degeneracy, not just numerical noise.
    """
    coords = V.tabulate_dof_coordinates()[:, :2] + u.x.array.reshape(-1, 2)
    cells = V.dofmap.list
    p = coords[cells]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    signed_area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return int((signed_area <= 0).sum())


@pytest.mark.parametrize("mesh_path", MESH_PATHS, ids=["tri", "tri_sec"])
def test_quality_decreases_with_amplitude(mesh_path):
    fd = load_fsi2_fluid_domain(mesh_path)
    V = dfx.fem.functionspace(fd.mesh, ("CG", 1, (2,)))
    mq = MeshQuality(quality_measure="scaled_jacobian", fspace=V)

    q_undeformed = mq(dfx.fem.Function(V))
    assert q_undeformed.min() > 0.5, "sanity check: the undeformed fluid mesh should already be good quality"

    u_healthy = dfx.fem.Function(V)
    u_healthy.interpolate(prescribed_interface_deformation(HEALTHY_AMPLITUDE))
    q_healthy = mq(u_healthy)

    u_degenerate = dfx.fem.Function(V)
    u_degenerate.interpolate(prescribed_interface_deformation(DEGENERATE_AMPLITUDE))
    q_degenerate = mq(u_degenerate)

    assert q_undeformed.min() > q_healthy.min() > q_degenerate.min()


@pytest.mark.parametrize("mesh_path", MESH_PATHS, ids=["tri", "tri_sec"])
def test_quality_flags_degenerate_deformation(mesh_path):
    """A large-enough prescribed deformation must (a) actually invert some
    cells (ground truth, independent of the quality library) and (b) be
    clearly flagged as such by the trigger metric.

    Note: for triangles, Verdict's ``scaled_jacobian`` saturates towards
    (but never crosses) 0 for inverted cells -- see the note in
    ``quality.py`` -- so "degenerate" here is "close to zero", not
    "non-positive". A remesh trigger threshold (§4 of the plan) has to be
    calibrated the same way, not by checking for negativity.
    """
    fd = load_fsi2_fluid_domain(mesh_path)
    V = dfx.fem.functionspace(fd.mesh, ("CG", 1, (2,)))
    mq = MeshQuality(quality_measure="scaled_jacobian", fspace=V)

    u_healthy = dfx.fem.Function(V)
    u_healthy.interpolate(prescribed_interface_deformation(HEALTHY_AMPLITUDE))
    assert _n_inverted_cells(V, u_healthy) == 0
    assert mq(u_healthy).min() > 0.1

    u_degenerate = dfx.fem.Function(V)
    u_degenerate.interpolate(prescribed_interface_deformation(DEGENERATE_AMPLITUDE))
    assert _n_inverted_cells(V, u_degenerate) > 0
    assert mq(u_degenerate).min() < 0.01


@pytest.mark.parametrize("mesh_path", MESH_PATHS, ids=["tri", "tri_sec"])
def test_quality_cg1_restriction_needs_projection(mesh_path):
    """pvmeshquality.MeshQuality only accepts CG1 vector spaces for warping
    (see quality.py); passing the mesh's own (possibly higher-order)
    geometry directly is not an option, and neither is a CG2 displacement --
    confirm the CG2-to-CG1 interpolate-down path used elsewhere in this
    module actually raises loudly if skipped, rather than silently doing
    the wrong thing.
    """
    fd = load_fsi2_fluid_domain(mesh_path)
    V2 = dfx.fem.functionspace(fd.mesh, ("CG", 2, (2,)))
    with pytest.raises(AssertionError):
        MeshQuality(quality_measure="scaled_jacobian", fspace=V2)
