import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.tools.cahouet_chabard import CahouetChabard

COMM = MPI.COMM_WORLD


def test_python_pc_with_the_context_can_be_destroyed():
    # petsc4py calls a destroy(pc) method of a python PC context when the PC is destroyed, so the
    # context must not define a destroy method with another signature.
    mesh = dolfinx.mesh.create_unit_square(COMM, 4, 4)
    P = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    outflow_dofs = dolfinx.fem.locate_dofs_geometrical(P, lambda x: np.isclose(x[0], 1.0))
    cahouet_chabard = CahouetChabard(P, ufl.dx(domain=mesh), None, outflow_dofs, alpha=1.0, mu=1.0)

    pc = PETSc.PC().create(COMM)
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(cahouet_chabard)
    pc.destroy()

    cahouet_chabard.destroy_solvers()
