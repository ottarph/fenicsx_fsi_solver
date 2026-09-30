import dolfinx.fem.petsc
import dolfinx.la.petsc
import numpy as np
import scifem
import ufl
from mpi4py import MPI
from petsc4py import PETSc

# mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 12)


def fluid(x, tol=1e-14):
    return x[0] >= 0.5 - tol


def interface(x, tol=1e-14):
    return np.isclose(x[0], 0.5, atol=tol)


def left(x):
    return np.isclose(x[0], 0.0, atol=1e-14)


def right(x):
    return np.isclose(x[0], 1.0, atol=1e-14)


solid_marker = 1
fluid_marker = 2
cm = mesh.topology.index_map(mesh.topology.dim)
vec = dolfinx.la.vector(cm, solid_marker, dtype=np.int32)
vec.array[:] = solid_marker
values = vec.array
values[dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, fluid)] = fluid_marker
vec.scatter_forward()

ct = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim, np.arange(len(values), dtype=np.int32), values
)

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

bcF = dolfinx.fem.dirichletbc(
    dolfinx.fem.Constant(mesh, 1.2), dolfinx.fem.locate_dofs_geometrical(V, right), V
)
bcS = dolfinx.fem.dirichletbc(
    dolfinx.fem.Constant(mesh, 0.3), dolfinx.fem.locate_dofs_geometrical(V, left), V
)
bcs = [bcF, bcS]

interface_facets = scifem.find_interface(ct, solid_marker, fluid_marker)

total_interface_facets_found = MPI.COMM_WORLD.allreduce(
    interface_facets.size, op=MPI.SUM
)

assert total_interface_facets_found > 0, "Interface not found."
dofs_interface = dolfinx.fem.locate_dofs_topological(
    V, mesh.topology.dim - 1, interface_facets
)
bc_deactivate = dolfinx.fem.dirichletbc(
    dolfinx.fem.Constant(mesh, 0.0), dofs_interface, V
)


dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dxF = dx(fluid_marker)
dxS = dx(solid_marker)

f_S = dolfinx.fem.Constant(mesh, 2.0)
f_F = dolfinx.fem.Constant(mesh, -3.0)


bilinform = (
    dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dxS
    + dolfinx.fem.Constant(mesh, 1.0) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dxF
)
linform = (
    dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(f_S, v) * dxS
    + dolfinx.fem.Constant(mesh, 1.0) * ufl.inner(f_F, v) * dxF
)

petsc_options_prefix = "solver_"
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
    "ksp_monitor": None,
}
from xfsi_solver.tools.custom_linear_problem import MyLinearProblem

problem = MyLinearProblem(
    bilinform,
    linform,
    bcs=[bcF],
    petsc_options_prefix=petsc_options_prefix,
    petsc_options=petsc_options,
)


A = problem.A
b = problem.b

problem.assemble_matrix()

for bc in [bc_deactivate]:
    dofs, _ = bc._cpp_object.dof_indices()
    A.zeroRowsLocal(dofs, diag=0)
A.assemble()

dolfinx.fem.petsc.assemble_matrix(
    A, dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dxS), bcs=[bcS]
)
A.assemble()

if MPI.COMM_WORLD.size == 1:
    print("\nA =\n", A[:, :])

    with (
        open("output/logs/restr_test_linprob_matvec.txt", "w") as f,
        np.printoptions(precision=2, linewidth=140),
    ):
        print("\nA =\n", A[:, :], file=f)

problem.assemble_rhs()

if MPI.COMM_WORLD.size == 1:
    with np.printoptions(precision=2, linewidth=140):
        print("\nb =", b.array[:])

bc_deactivate.set(b.array_w, alpha=0.0)
# Clear ghost entries so the second reverse scatter only sends solid contributions
with b.localForm() as b_loc:
    b_loc.array[b.getLocalSize() :] = 0.0

if MPI.COMM_WORLD.size == 1:
    with np.printoptions(precision=2, linewidth=140):
        print("\nb =", b.array[:])

dolfinx.fem.petsc.assemble_vector(
    b,
    dolfinx.fem.form(ufl.inner(f_S, v) * dxS),
)

if MPI.COMM_WORLD.size == 1:
    with np.printoptions(precision=2, linewidth=140):
        print("\nb =", b.array[:])

dolfinx.fem.petsc.apply_lifting(
    b,
    [dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dxS)],
    alpha=1.0,
    bcs=[[bcS]],
)

dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)

if MPI.COMM_WORLD.size == 1:
    with np.printoptions(precision=2, linewidth=140):
        print("\nb =", b.array[:])

bcS.set(b.array_w, alpha=1.0)

if MPI.COMM_WORLD.size == 1:
    with np.printoptions(precision=2, linewidth=140):
        print("\nb =", b.array[:])

        with open("output/logs/restr_test_linprob_matvec.txt", "a") as f:
            print("\nb =", b.array[:], file=f)


ksp = PETSc.KSP().create(mesh.comm)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
ksp.setErrorIfNotConverged(True)
ksp.setOperators(A)
x = dolfinx.fem.Function(V)
ksp.solve(b, x.x.petsc_vec)
x.x.scatter_forward()


if MPI.COMM_WORLD.size == 1:
    with (
        np.printoptions(precision=2, linewidth=140),
        open("output/logs/restr_test_linprob_matvec.txt", "a") as f,
    ):
        print("\nx =", x.x.array[:], file=f)


writer = dolfinx.io.VTXWriter(mesh.comm, "output/pv/restr_test_linprob_solution.bp", [x])
writer.write(0)
writer.close()

if MPI.COMM_WORLD.size == 1 and mesh.topology.dim == 1:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["svg.hashsalt"] = "restricted-test-functions"

    tt = x.function_space.tabulate_dof_coordinates()[:, 0]
    dof_sorting = np.argsort(tt)
    plt.figure()
    plt.plot(tt[dof_sorting], x.x.array[dof_sorting], "k-")

    plt.axvline(
        x=tt[dofs_interface[0]], color="black", alpha=0.4, lw=0.3, label="interface"
    )

    plt.legend()
    plt.savefig("output/figures/restr_test_linprob_solution.svg", metadata={"Date": None})

    plt.figure()
    plt.spy(A[:, :])
    plt.axhline(y=dofs_interface[0] + 0.5, color="black", alpha=0.5, lw=0.2)
    plt.axvline(x=dofs_interface[0] + 0.5, color="black", alpha=0.5, lw=0.2)

    plt.savefig("output/figures/restr_test_linprob_sparsity.svg", metadata={"Date": None})

    # plt.show() # Don't use on headless.
