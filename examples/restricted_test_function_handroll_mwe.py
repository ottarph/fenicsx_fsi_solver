import dolfinx.fem.petsc
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

ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, np.arange(len(values), dtype=np.int32), values)

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

bcF = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 1.2), dolfinx.fem.locate_dofs_geometrical(V, right), V)
bcS = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 0.3), dolfinx.fem.locate_dofs_geometrical(V, left), V)
bcs = [bcF, bcS]

interface_facets = scifem.find_interface(ct, solid_marker, fluid_marker)

total_interface_facets_found = MPI.COMM_WORLD.allreduce(interface_facets.size, op=MPI.SUM)

assert total_interface_facets_found > 0, "Interface not found."
dofs_interface = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, interface_facets)
bc_deactivate = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 0.0), dofs_interface, V)


dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dxF = dx(fluid_marker)
dxS = dx(solid_marker)

kernel = ufl.inner(ufl.grad(u), ufl.grad(v))

kernelS = kernel * dxS
compiled_solid = dolfinx.fem.form(kernelS)
As = dolfinx.fem.petsc.assemble_matrix(compiled_solid, bcs=bcs)
As.assemble()
compiled_fluid = dolfinx.fem.form(kernel * dxF)


Af = dolfinx.fem.petsc.assemble_matrix(compiled_fluid, bcs=bcs, diag=1.0)
Af.assemble()
for bc in [bc_deactivate]:
    dofs, _ = bc.dof_indices()
    Af.zeroRowsLocal(dofs, diag=0)
Af.assemble()
A = As + Af

if MPI.COMM_WORLD.size == 1:
    print("\nAs = \n", As[:, :])
    print("\nAf =\n", Af[:, :])
    print("\nA =\n", A[:, :])

    interface_x = V.tabulate_dof_coordinates()[dofs_interface[0]]
    print(f"{interface_x[0] = :.2f}")

    with (
        open("output/logs/restr_test_hr_mwe_matvec.txt", "w") as f,
        np.printoptions(precision=2, linewidth=140),
    ):
        print("\nA =\n", A[:, :], file=f)


f_S = dolfinx.fem.Constant(mesh, 2.0)
f_F = dolfinx.fem.Constant(mesh, -3.0)
Ls = ufl.inner(f_S, v) * dxS
Lf = ufl.inner(f_F, v) * dxF

bs = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Ls))
dolfinx.fem.petsc.apply_lifting(bs, [compiled_solid], bcs=[bcs])
bs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


bf = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Lf))
dolfinx.fem.petsc.apply_lifting(bf, [compiled_fluid], bcs=[bcs], alpha=1.0)


# Accumulate ghost contributions after lifting, before any bc.set().
bf.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


bcS.set(bs.array_w)

# Zero out the row for disappearing test function.
bc_deactivate.set(bf.array_w, alpha=0.0)

bcF.set(bf.array_w)


b = bs + bf
# Final forward scatter, probably not needed.
# b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

if MPI.COMM_WORLD.size == 1:
    with np.printoptions(precision=2, linewidth=140):
        print("\nbs =", bs.array[:])
        print("\nbf =", bf.array[:])
        print("\nb =", b.array[:])

        with open("output/logs/restr_test_hr_mwe_matvec.txt", "a") as f:
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
        open("output/logs/restr_test_hr_mwe_matvec.txt", "a") as f,
    ):
        print("\nx =", x.x.array[:], file=f)


writer = dolfinx.io.VTXWriter(mesh.comm, "output/pv/restr_test_hr_solution.bp", [x])
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

    plt.axvline(x=tt[dofs_interface[0]], color="black", alpha=0.4, lw=0.3, label="interface")

    plt.legend()
    plt.savefig("output/figures/restr_test_hr_solution.svg", metadata={"Date": None})

    plt.figure()
    plt.spy(A[:, :])
    plt.axhline(y=dofs_interface[0] + 0.5, color="black", alpha=0.5, lw=0.2)
    plt.axvline(x=dofs_interface[0] + 0.5, color="black", alpha=0.5, lw=0.2)

    plt.savefig("output/figures/restr_test_hr_sparsity.svg", metadata={"Date": None})

    # plt.show() # Don't use on headless.
