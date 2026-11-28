import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.tools.interface_ale import ROWS, InterfaceALE, check_interface_ale_rows, interface_ale_description


@pytest.mark.parametrize("rows", [("momentum",), ("mass",), ("momentum", "mass"), ("mass", "momentum"), ["momentum"]])
def test_rows_accepted(rows):
    check_interface_ale_rows(rows)


@pytest.mark.parametrize("rows", [(), "momentum", ("momentum", "momentum"), ("pressure",), ("momentum", "continuity")])
def test_rows_rejected(rows):
    with pytest.raises(ValueError, match="interface_ale_rows"):
        check_interface_ale_rows(rows)


def test_description():
    assert interface_ale_description(False, None) == "off"
    assert interface_ale_description(True, None) == (
        "on, momentum and mass rows, integrated with the quadrature of the residual"
    )
    # The rows are named in the order of ROWS, whatever the order given.
    assert interface_ale_description(True, 2, ("mass", "momentum")) == interface_ale_description(True, 2, ROWS)
    assert interface_ale_description(True, 2, ("momentum",)) == "on, momentum rows, integrated with quadrature degree 2"


def test_monolithic_jacobian_gets_the_terms_of_the_nest_jacobian():
    # A fluid (x > 0.5) and a solid (x < 0.5) on the unit square, with the fluid displacement on the fluid
    # submesh and a fluid residual that depends on it, in the momentum and mass rows. The terms added to a
    # monolithic Jacobian of (v, p) are those added to the blocks of a nest Jacobian, in the same order.
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 6, 6)
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    fluid_marker, solid_marker, interface_marker = 2, 1, 11
    cell_tags = dolfinx.mesh.meshtags(
        mesh, tdim, cells, np.where(midpoints[:, 0] > 0.5, fluid_marker, solid_marker).astype(np.int32)
    )
    mesh.topology.create_connectivity(tdim - 1, tdim)
    interface = dolfinx.mesh.locate_entities(mesh, tdim - 1, lambda x: np.isclose(x[0], 0.5))
    facet_tags = dolfinx.mesh.meshtags(mesh, tdim - 1, interface, np.full(len(interface), interface_marker, np.int32))
    fluid_mesh, fluid_cell_map, _, _ = dolfinx.mesh.create_submesh(mesh, tdim, cell_tags.find(fluid_marker))
    entity_maps = [fluid_cell_map]

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (2,)))
    P = dolfinx.fem.functionspace(fluid_mesh, ("Lagrange", 1))
    U_f = dolfinx.fem.functionspace(fluid_mesh, ("Lagrange", 2, (2,)))
    v, p, u_f = dolfinx.fem.Function(V), dolfinx.fem.Function(P), dolfinx.fem.Function(U_f)
    v.interpolate(lambda x: np.stack([np.sin(3 * x[1]), x[0] * x[1]]))
    p.interpolate(lambda x: 1.0e3 + x[0] - x[1] ** 2)
    u_f.interpolate(lambda x: np.stack([0.05 * (x[0] - 0.5) * x[1], 0.08 * np.sin(np.pi * x[0])]))
    dv, dp = ufl.TestFunctions(ufl.MixedFunctionSpace(V, P))
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    F = ufl.Identity(tdim) + ufl.grad(u_f)
    J = ufl.det(F)
    residual = ufl.inner(J * ufl.grad(v) * ufl.inv(F), ufl.grad(dv)) * dx(fluid_marker)
    residual += ufl.inner(-p * J * ufl.inv(F).T, ufl.grad(dv)) * dx(fluid_marker)
    residual += ufl.inner(v, dv) * dx(solid_marker)
    residual += ufl.div(J * ufl.inv(F) * v) * dp * dx(fluid_marker)
    residual_blocks = ufl.extract_blocks(residual)
    jacobian = dolfinx.fem.form(dolfinx.fem.forms.derivative_block(residual_blocks, (v, p)), entity_maps=entity_maps)
    left = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bcs = [dolfinx.fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), left, V)]
    terms = InterfaceALE(
        residual_blocks,
        u_f,
        V,
        cell_tags,
        facet_tags,
        fluid_marker,
        interface_marker,
        entity_maps,
        bcs,
        theta=0.5,
        dt=0.01,
        block_spaces=(V, P),
    )

    matrices = {}
    for kind in ("nest", None):
        A = dolfinx.fem.petsc.create_matrix(jacobian, kind=kind)
        dolfinx.fem.petsc.assemble_matrix(A, jacobian, bcs=bcs)
        A.assemble()
        before = A.convert("aij", PETSc.Mat()) if kind == "nest" else A.copy()
        terms.add(A)
        after = A.convert("aij", PETSc.Mat()) if kind == "nest" else A
        after.axpy(-1.0, before)
        matrices[kind] = after
    added = matrices["nest"]
    assert added.norm() > 0.0
    added.axpy(-1.0, matrices[None])
    assert added.norm() < 1e-12 * matrices[None].norm()
