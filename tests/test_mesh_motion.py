from pathlib import Path

import dolfinx
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc
from restart_helpers import run_mpi_python

from xfsi_solver.mesh_motion import (
    biharmonic_c0ip_mesh_motion,
    biharmonic_mixed_mesh_motion,
    default_mesh_motion_solver_options,
    harmonic_mesh_motion,
)

MESH = "data/meshes/fsi2/mesh_sec_coarse.xdmf"
FLUID = 2  # cell tag of the fluid in the FSI mesh
ELEMENT = ("Lagrange", 2, (2,))

MESH_MOTIONS = {
    "harmonic": harmonic_mesh_motion,
    "biharmonic_mixed": biharmonic_mixed_mesh_motion,
    "biharmonic_c0ip": biharmonic_c0ip_mesh_motion,
}

# MUMPS LU on the whole system, against which the default component-wise solve is checked. The mixed
# system is indefinite, so MUMPS pivots; with its default threshold and workspace, the factorization
# runs out of workspace on 3 ranks.
WHOLE_SYSTEM_LU = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_cntl_1": 1e-4,
    "mat_mumps_icntl_14": 100,
}

# The solves agree to round-off amplified by the condition number: the mixed form differs most, by
# 7e-11 of the largest displacement on this mesh. A wrong split of the components gives errors of the
# order of the displacement itself.
RTOL_TO_MAX = 1e-9


def _setup(comm=MPI.COMM_WORLD):
    """The displacement u on the fluid submesh of the FSI mesh, as in the solvers, and a Dirichlet
    condition on its whole boundary with the values of g."""
    with dolfinx.io.XDMFFile(comm, MESH, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name="Cell tags")
    fluid_mesh, _, _, _ = dolfinx.mesh.create_submesh(mesh, 2, cell_tags.find(FLUID))
    fluid_mesh.topology.create_connectivity(1, 2)
    U = dolfinx.fem.functionspace(fluid_mesh, ELEMENT)
    u, g = dolfinx.fem.Function(U, name="u"), dolfinx.fem.Function(U)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(U, 1, dolfinx.mesh.exterior_facet_indices(fluid_mesh.topology))
    return u, g, dolfinx.fem.dirichletbc(g, boundary_dofs)


def _bend(x):
    """A boundary displacement like that of the FSI2 flag: zero far from it, bending the fluid around it."""
    bump = np.exp(-((x[0] - 0.45) ** 2 + (x[1] - 0.2) ** 2) / 0.02)
    return np.vstack([0.01 * bump, 0.05 * bump * np.sin(4 * x[0])])


@pytest.mark.parametrize("kind", ["harmonic", "biharmonic_mixed", "biharmonic_c0ip"])
def test_reproduces_translation(kind):
    """A translation is harmonic and biharmonic, with a zero normal derivative, so every mesh motion
    reproduces it exactly, also in a second solve with new boundary values, with the factorization of
    the first."""
    u, g, bc = _setup()
    mesh_motion = MESH_MOTIONS[kind](u, [bc])
    for shift in ([0.1, -0.2], [-0.3, 0.05]):
        g.interpolate(lambda x, s=shift: np.vstack([np.full_like(x[0], s[0]), np.full_like(x[0], s[1])]))
        mesh_motion.solve()
        np.testing.assert_allclose(
            u.x.array.reshape(-1, 2), np.broadcast_to(shift, (u.x.array.size // 2, 2)), atol=1e-10
        )
    if kind == "biharmonic_mixed":
        z = mesh_motion.unknowns[0]
        np.testing.assert_allclose(z.x.array, 0.0, atol=1e-8)
    mesh_motion.destroy()


def test_harmonic_reproduces_affine_displacement():
    """An affine displacement is harmonic, but has a nonzero normal derivative, which the biharmonic
    mesh motions set to zero."""
    u, g, bc = _setup()
    g.interpolate(lambda x: np.vstack([0.1 * x[0] + 0.2 * x[1], -0.3 * x[0] + 0.05 * x[1]]))
    mesh_motion = harmonic_mesh_motion(u, [bc])
    mesh_motion.solve()
    np.testing.assert_allclose(u.x.array, g.x.array, atol=1e-10)
    mesh_motion.destroy()


def _solve_bend(kind, solver_options):
    u, g, bc = _setup()
    g.interpolate(_bend)
    mesh_motion = MESH_MOTIONS[kind](u, [bc], solver_options=solver_options)
    mesh_motion.solve()
    mesh_motion.destroy()
    return u


@pytest.mark.parametrize("kind", ["harmonic", "biharmonic_mixed", "biharmonic_c0ip"])
def test_default_solve_matches_whole_system_lu(kind):
    """The default solve, by components, gives the solution of MUMPS LU on the whole system."""
    u = _solve_bend(kind, None)
    u_lu = _solve_bend(kind, WHOLE_SYSTEM_LU)
    assert np.abs(u_lu.x.array).max() > 1e-3
    np.testing.assert_allclose(u.x.array, u_lu.x.array, rtol=0, atol=RTOL_TO_MAX * np.abs(u_lu.x.array).max())


@pytest.mark.parametrize("kind", ["harmonic", "biharmonic_mixed", "biharmonic_c0ip"])
def test_quadrature_cap_changes_the_solution_slightly(kind):
    """A cap of 2 is below the degree UFL estimates for every mesh motion on the curved P2 mesh, so it
    changes the solution, but only by integrating the forms inexactly: on this mesh by 7e-6
    (harmonic), 2.8e-3 (mixed) and 2.9e-5 (C0IP) of the largest displacement, so 1e-2 leaves a margin
    of more than 3. A translation is still reproduced exactly, with the residual and the Jacobian
    both capped."""
    u_exact = _solve_bend(kind, None)

    u, g, bc = _setup()
    g.interpolate(_bend)
    mesh_motion = MESH_MOTIONS[kind](u, [bc], max_quadrature_degree=2)
    mesh_motion.solve()
    difference = np.abs(u.x.array - u_exact.x.array).max()
    assert 0.0 < difference < 1e-2 * np.abs(u_exact.x.array).max()

    g.interpolate(lambda x: np.vstack([np.full_like(x[0], 0.1), np.full_like(x[0], -0.2)]))
    mesh_motion.solve()
    np.testing.assert_allclose(
        u.x.array.reshape(-1, 2), np.broadcast_to([0.1, -0.2], (u.x.array.size // 2, 2)), atol=1e-10
    )
    mesh_motion.destroy()


_MPI_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
import numpy as np
from test_mesh_motion import RTOL_TO_MAX, WHOLE_SYSTEM_LU, _solve_bend
for kind in ["harmonic", "biharmonic_mixed", "biharmonic_c0ip"]:
    u = _solve_bend(kind, None)
    u_lu = _solve_bend(kind, WHOLE_SYSTEM_LU)
    np.testing.assert_allclose(u.x.array, u_lu.x.array, rtol=0, atol=RTOL_TO_MAX * np.abs(u_lu.x.array).max())
"""


def test_default_solve_matches_whole_system_lu_on_3_ranks():
    """IdenticalInterleavedBlocksPC splits the rows of each rank into the components, also for the
    block matrix [z | u] of the mixed form."""
    run_mpi_python(_MPI_SCRIPT.format(tests=str(Path(__file__).parent)), 3, timeout=300)


def test_failed_solve_raises():
    u, g, bc = _setup()
    g.interpolate(_bend)
    options = {"ksp_type": "cg", "pc_type": "none", "ksp_max_it": 1}
    mesh_motion = harmonic_mesh_motion(u, [bc], solver_options=options)
    with pytest.raises(RuntimeError, match="mesh_motion_ failed: SNES reason -3, KSP reason -3"):
        mesh_motion.solve()
    mesh_motion.destroy()


def test_solver_options_replace_those_of_an_earlier_problem():
    """The options are kept in the options database for the PCs set up in the first solve; those of
    an earlier problem with the same prefix are removed first, so they cannot reach these PCs."""
    u, _, bc = _setup()
    harmonic_mesh_motion(u, [bc], solver_options=WHOLE_SYSTEM_LU).destroy()
    harmonic_mesh_motion(u, [bc]).destroy()
    kept = {key for key in PETSc.Options().getAll() if key.startswith("mesh_motion_")}
    default = {f"mesh_motion_{key}" for key in default_mesh_motion_solver_options(2)}
    earlier_only = {f"mesh_motion_{key}" for key in WHOLE_SYSTEM_LU} - default
    assert default <= kept
    assert not kept & earlier_only
