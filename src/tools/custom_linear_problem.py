import dolfinx.fem
import dolfinx.fem.petsc
from petsc4py import PETSc

class MyLinearProblem(dolfinx.fem.petsc.LinearProblem):

    @dolfinx.common.timed("MyLinearProblem assemble_matrix")
    def assemble_matrix(self) -> None:

        # Assemble lhs
        self._A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

        return
    
    @dolfinx.common.timed("MyLinearProblem solve")
    def solve(self) -> dolfinx.fem.Function:

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        dolfinx.fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(self._b, self.bcs)

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()

        return self.u
