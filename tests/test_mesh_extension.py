"""Checks of the reference-weighted elastic mesh-extension operator.

MPI-aware; ``test_mesh_extension_on_two_ranks`` reruns this module on two ranks.
"""

import os
import subprocess
import sys
from pathlib import Path

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from xfsi_solver.fsi.mesh_extension import (
    DEFAULT_QUADRATURE_DEGREE,
    FluidReferenceGeometry,
    StiffenedElasticMeshExtension,
    evaluate,
    sample_points,
)
from xfsi_solver.solvers.fsi2_harmonic_diffmesh import PHYSICAL_MARKERS, build_problem

ROOT = Path(__file__).resolve().parent.parent
CURVED_TRIANGLES = str(ROOT / "data/meshes/fsi2/mesh_sec_coarse.xdmf")
AFFINE_TRIANGLES = str(ROOT / "data/meshes/fsi2/mesh_coarse.xdmf")
CURVED_QUADRILATERALS = str(ROOT / "data/meshes/fsi2/mesh_quad_ssq_sec.xdmf")
DT = 0.0025
comm = MPI.COMM_WORLD


def read_mesh(path):
    with dfx.io.XDMFFile(comm, path, "r") as f:
        mesh = f.read_mesh()
        cell_tags = f.read_meshtags(mesh, name="Cell tags")
    return mesh, cell_tags


def geometry_of(path):
    mesh, cell_tags = read_mesh(path)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    return FluidReferenceGeometry(mesh, cell_tags, PHYSICAL_MARKERS["ALE_fluid"], dx(PHYSICAL_MARKERS["ALE_fluid"]),
                                  None)


@pytest.fixture(scope="module")
def curved():
    return geometry_of(CURVED_TRIANGLES)


@pytest.fixture(scope="module")
def affine():
    return geometry_of(AFFINE_TRIANGLES)


def cell_volumes(mesh):
    """Volume of every local cell (ghosts included)."""
    Q = dfx.fem.functionspace(mesh, ("DG", 0))
    b = dfx.fem.assemble_vector(dfx.fem.form(ufl.TestFunction(Q) * ufl.dx(domain=mesh)))
    b.scatter_reverse(dfx.la.InsertMode.add)
    f = dfx.fem.Function(Q)
    f.x.array[:] = b.array
    f.x.scatter_forward()
    return f.x.array


def test_zero_exponent_gives_unit_weight(curved):
    op = StiffenedElasticMeshExtension(stiffening_exponent=0.0).bind(curved)
    np.testing.assert_array_equal(curved.sample(op.weight), 1.0)
    assert op.info["weight_min"] == op.info["weight_max"] == 1.0


@pytest.mark.parametrize("chi", [1.0, 2.5])
def test_affine_weight_is_inverse_volume_power(affine, chi):
    op = StiffenedElasticMeshExtension(stiffening_exponent=chi).bind(affine)
    mesh = affine.mesh
    volumes = cell_volumes(mesh)
    cells = affine.owned_fluid_cells
    n_fluid = comm.allreduce(cells.size, op=MPI.SUM)
    assert n_fluid == 419
    # the normalization is the global mean fluid-cell volume over the parent-cell volume
    mean_volume = comm.allreduce(volumes[cells].sum(), op=MPI.SUM) / n_fluid
    assert op.j_star.value == pytest.approx(mean_volume / 0.5, rel=1e-12)

    w = affine.sample(op.weight)
    # constant in each affine cell, and (mean |K| / |K|)^chi
    np.testing.assert_allclose(w, w[:, :1] * np.ones_like(w), rtol=1e-12)
    np.testing.assert_allclose(w[:, 0], (mean_volume / volumes[cells]) ** chi, rtol=1e-10)
    # coefficient ratio of two unequal cells
    lo, hi = np.argmin(volumes[cells]), np.argmax(volumes[cells])
    assert w[lo, 0] / w[hi, 0] == pytest.approx((volumes[cells][hi] / volumes[cells][lo]) ** chi, rel=1e-10)


def test_pointwise_weight_varies_in_curved_cells(curved, affine):
    op = StiffenedElasticMeshExtension().bind(curved)
    w = curved.sample(op.weight)
    variation = w.max(axis=1) / w.min(axis=1)
    assert comm.allreduce(variation.max(initial=1.0), op=MPI.MAX) > 1.1
    # the experimental cell-volume weight is constant per cell and differs there
    op_volume = StiffenedElasticMeshExtension(weighting="cell_volume").bind(curved)
    w_volume = curved.sample(op_volume.weight)
    np.testing.assert_allclose(w_volume, w_volume[:, :1] * np.ones_like(w_volume), rtol=1e-12)
    curved_cells = variation > 1 + 1e-8
    assert comm.allreduce(int(curved_cells.sum()), op=MPI.SUM) > 0
    # on straight cells the two coincide
    np.testing.assert_allclose(w[~curved_cells], w_volume[~curved_cells], rtol=1e-8)

    # also on an affine mesh
    w_affine = affine.sample(StiffenedElasticMeshExtension().bind(affine).weight)
    w_affine_volume = affine.sample(StiffenedElasticMeshExtension(weighting="cell_volume").bind(affine).weight)
    np.testing.assert_allclose(w_affine, w_affine_volume, rtol=1e-10)


def test_cell_volume_weight_has_consistent_ghosts(affine):
    """The DG0 volume weight on ghost cells equals the owner's value."""
    op = StiffenedElasticMeshExtension(weighting="cell_volume").bind(affine)
    mesh = affine.mesh
    n_local = mesh.topology.index_map(2).size_local + mesh.topology.index_map(2).num_ghosts
    cells = np.arange(n_local, dtype=np.int32)
    midpoint = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    w_dg0 = evaluate(op.weight, mesh, cells, midpoint)[:, 0]
    w_exact = evaluate(StiffenedElasticMeshExtension().bind(affine).weight, mesh, cells, midpoint)[:, 0]
    np.testing.assert_allclose(w_dg0, w_exact, rtol=1e-10)


def test_rigid_motions_are_stress_free(curved):
    op = StiffenedElasticMeshExtension().bind(curved)
    U = dfx.fem.functionspace(curved.mesh, ("CG", 2, (2,)))
    u = dfx.fem.Function(U)
    for motion in (lambda x: np.vstack((0.3 + 0 * x[0], -0.2 + 0 * x[0])),
                   lambda x: np.vstack((-x[1], x[0]))):  # translation, infinitesimal rotation
        u.interpolate(motion)
        stress = curved.sample(op.stress(u))
        assert np.abs(stress).max(initial=0.0) < 1e-10
        b = dfx.fem.assemble_vector(dfx.fem.form(op.volume_form(u, ufl.TestFunction(U))))
        b.scatter_reverse(dfx.la.InsertMode.add)
        assert np.abs(b.array).max(initial=0.0) < 1e-10


def test_affine_displacement_follows_lame_law(curved):
    ext = StiffenedElasticMeshExtension(poisson_ratio=0.35, stiffening_exponent=2.0)
    op = ext.bind(curved)
    U = dfx.fem.functionspace(curved.mesh, ("CG", 2, (2,)))
    u = dfx.fem.Function(U)
    G = np.array([[0.02, -0.01], [0.03, -0.015]])
    u.interpolate(lambda x: G @ x[:2])
    eps = 0.5 * (G + G.T)
    mu, lam = 1.0 / (2 * 1.35), 0.35 / (1.35 * (1 - 0.7))
    assert ext.lame_parameters() == pytest.approx((mu, lam))
    expected = 2 * mu * eps + lam * np.trace(eps) * np.eye(2)
    stress = curved.sample(op.stress(u))
    w = curved.sample(op.weight)
    np.testing.assert_allclose(stress, w[..., None, None] * expected, rtol=1e-10, atol=1e-14)


def test_parameter_validation():
    for bad in (dict(stiffening_exponent=-1.0), dict(poisson_ratio=0.5), dict(poisson_ratio=-0.1),
                dict(modulus=0.0), dict(stiffening_exponent=float("nan")), dict(quadrature_degree=0),
                dict(weighting="gaussian"), dict(j_star=-1.0)):
        with pytest.raises(ValueError):
            StiffenedElasticMeshExtension(**bad)


class StandaloneExtension:
    """The elastic extension of a prescribed solid displacement into the fluid (LinearProblem).

    All DOFs of solid cells follow ``solid_motion`` scaled by an amplitude, the
    outer boundary is fixed.
    """

    def __init__(self, path, extension, solid_motion):
        self.problem = build_problem(path, DT, mesh_extension=extension)
        problem = self.problem
        U = problem.U
        self.op = problem.mesh_operator
        u, du = ufl.TrialFunction(U), ufl.TestFunction(U)
        # the zero solid term puts the (Dirichlet) solid DOFs into the sparsity pattern
        a = self.op.volume_form(u, du) + dfx.fem.Constant(problem.mesh, 0.0) * ufl.inner(u, du) * problem.dx_solid
        L = ufl.inner(dfx.fem.Constant(problem.mesh, (0.0, 0.0)), du) * problem.dx_fluid
        solid_dofs = dfx.fem.locate_dofs_topological(U, 2, problem.cell_tags.find(PHYSICAL_MARKERS["solid"]))
        self.motion = dfx.fem.Function(U)
        self.motion.interpolate(solid_motion)
        self.solid_value = dfx.fem.Function(U)
        bcs = [*problem.bcs_u, dfx.fem.dirichletbc(self.solid_value, solid_dofs)]
        self.solution = dfx.fem.Function(U)
        self.linear_problem = dfx.fem.petsc.LinearProblem(
            a, L, bcs=bcs, u=self.solution, petsc_options_prefix="test_mesh_extension_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

    def solve(self, amplitude):
        self.solid_value.x.array[:] = amplitude * self.motion.x.array
        self.linear_problem.solve()
        return self.solution.x.array.copy()


def beam_bending(x):
    """Bending-like beam motion (rotating cross sections), tip deflection 0.062.

    It vanishes where the beam is attached to the cylinder (x = 0.249).
    """
    s = np.maximum(x[0] - 0.249, 0.0)
    return np.vstack((-(x[1] - 0.2) * s, 0.5 * s ** 2))


def global_relative(a, b):
    diff = comm.allreduce(np.sum((a - b) ** 2), op=MPI.SUM)
    norm = comm.allreduce(np.sum(b ** 2), op=MPI.SUM)
    return np.sqrt(diff / norm)


def test_modulus_scale_leaves_extension_unchanged():
    base = StandaloneExtension(CURVED_TRIANGLES, StiffenedElasticMeshExtension(modulus=1.0), beam_bending)
    scaled = StandaloneExtension(CURVED_TRIANGLES, StiffenedElasticMeshExtension(modulus=1e3), beam_bending)
    a, b = base.solve(1.0), scaled.solve(1.0)
    assert global_relative(b, a) < 1e-10
    # the extension is non-trivial, and differs from the unweighted one
    unweighted = StandaloneExtension(CURVED_TRIANGLES, StiffenedElasticMeshExtension(stiffening_exponent=0.0),
                                     beam_bending).solve(1.0)
    assert global_relative(unweighted, a) > 1e-3


def test_extension_is_history_independent():
    """A boundary-displacement cycle returns to the reference state; the geometry is never moved."""
    ext = StandaloneExtension(CURVED_TRIANGLES, StiffenedElasticMeshExtension(), beam_bending)
    x0 = ext.problem.mesh.geometry.x.copy()
    first = ext.solve(1.0)
    states = [ext.solve(amplitude) for amplitude in (2.0, 1.0, 0.0, -1.0, 0.0, 1.0)]
    assert global_relative(states[1], first) < 1e-12
    assert global_relative(states[5], first) < 1e-12
    for zero in (states[2], states[4]):
        assert np.abs(zero).max(initial=0.0) == 0.0
    np.testing.assert_array_equal(ext.problem.mesh.geometry.x, x0)
    # admissible at a moderate amplitude (tip deflection 0.031)
    ext.solve(0.5)
    detF = ext.op.geometry.sample(ufl.det(ufl.Identity(2) + ufl.grad(ext.solution)))
    assert comm.allreduce(detF.min(initial=np.inf), op=MPI.MIN) > 0.0


@pytest.mark.parametrize("path", [CURVED_TRIANGLES, CURVED_QUADRILATERALS])
def test_quadrature_increase(path):
    """Doubling the default quadrature degree hardly changes the volume and interface residuals."""
    results = []
    for factor in (1, 2):
        mesh, _ = read_mesh(path)
        degree = factor * DEFAULT_QUADRATURE_DEGREE[mesh.topology.cell_name()]
        problem = build_problem(path, DT, mesh_extension=StiffenedElasticMeshExtension(quadrature_degree=degree))
        problem.u.interpolate(lambda x: np.vstack((0.01 * np.sin(3 * x[0]) * x[1], 0.02 * np.cos(2 * x[0] + x[1]))))
        du = ufl.TestFunction(problem.U)
        op = problem.mesh_operator
        vectors = []
        for form in (op.volume_form(problem.u, du), op.interface_form(problem.u, du)):
            b = dfx.fem.assemble_vector(dfx.fem.form(form))
            b.scatter_reverse(dfx.la.InsertMode.add)
            vectors.append(b.array[:problem.U.dofmap.index_map.size_local * 2].copy())
        results.append(vectors)
    tolerance = {"triangle": 1e-8, "quadrilateral": 1e-6}[mesh.topology.cell_name()]
    for coarse, fine in zip(*results, strict=True):
        assert comm.allreduce(np.sum(fine ** 2), op=MPI.SUM) > 0.0
        assert global_relative(coarse, fine) < tolerance


def test_interface_measure_is_fluid_sided():
    """Every interface facet is integrated once, from its fluid cell, on every rank."""
    from xfsi_solver.solvers.fsi2_harmonic_diffmesh import interface_fluid_entities

    problem = build_problem(CURVED_TRIANGLES, DT,
                            mesh_extension=StiffenedElasticMeshExtension())
    mesh, cell_tags, facet_tags = problem.mesh, problem.cell_tags, problem.facet_tags
    facets = facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])
    entities = interface_fluid_entities(cell_tags, facets)
    fluid_cells = cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"])
    assert np.all(np.isin(entities[:, 0], fluid_cells))
    n_owned_facets = np.sum(facets < mesh.topology.index_map(1).size_local)
    assert comm.allreduce(entities.shape[0], op=MPI.SUM) == comm.allreduce(int(n_owned_facets), op=MPI.SUM) > 0
    # the facets of the selected cells are the interface facets
    c2f = mesh.topology.connectivity(2, 1)
    selected = np.array([c2f.links(c)[f] for c, f in entities])
    assert set(selected) == set(facets[facets < mesh.topology.index_map(1).size_local])

    # interface length: beam top and bottom from the cylinder (x = 0.249) to x = 0.6, and its tip
    length = comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1.0 * problem.ds_interface_fluid)), op=MPI.SUM)
    assert length == pytest.approx(2 * (0.6 - np.sqrt(0.05 ** 2 - 0.01 ** 2) - 0.2) + 0.02, rel=1e-6)

    # the weight on the interface is the fluid cell's: a DG0 copy that vanishes on solid cells
    op = problem.mesh_operator
    Q = dfx.fem.functionspace(mesh, ("DG", 0))
    w_fluid = dfx.fem.Function(Q)
    n_local = mesh.topology.index_map(2).size_local + mesh.topology.index_map(2).num_ghosts
    fluid_local = fluid_cells[fluid_cells < n_local]
    w_fluid.x.array[fluid_local] = evaluate(op.weight, mesh, fluid_local, np.array([[1 / 3, 1 / 3]]))[:, 0]
    # interface-adjacent fluid cells of this coarse mesh are straight, so the weight is constant there
    exact = comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(op.weight * op.ds)), op=MPI.SUM)
    dg0 = comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(w_fluid * op.ds)), op=MPI.SUM)
    assert exact == pytest.approx(dg0, rel=1e-10)
    assert exact > 0.0


def test_sample_points_include_vertices():
    for cell, n_points in (("triangle", 28), ("quadrilateral", 49)):
        points = sample_points(cell)
        assert points.shape == (n_points, 2)
        for vertex in ([0, 0], [1, 0], [0, 1]):
            assert np.any(np.all(np.isclose(points, vertex), axis=1))


@pytest.mark.skipif(comm.size > 1 or os.environ.get("XFSI_SKIP_MPI_TESTS"), reason="launcher only")
def test_mesh_extension_on_two_ranks():
    env = dict(os.environ, XFSI_SKIP_MPI_TESTS="1")
    cmd = ["mpiexec", "-n", "2", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
