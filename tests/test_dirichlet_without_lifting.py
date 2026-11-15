import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.tools.dirichlet_without_lifting import DirichletWithoutLifting

COMM = MPI.COMM_WORLD

DIRECT = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-12,
    "snes_rtol": 1e-14,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
# A nest matrix cannot be factorized as a whole, so its blocks are, inside an outer Krylov solve.
NEST = {
    **DIRECT,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-13,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
}
del NEST["pc_factor_mat_solver_type"]


class Setup:
    """A nonlinear diffusion problem on the unit square with an inhomogeneous Dirichlet condition on the
    left, whose values change with ``t``, and a homogeneous one on the right: one unknown u if ``kind`` is
    "single", and otherwise u coupled to a second unknown w with a homogeneous condition at the bottom, as
    a block (``kind=None``) or nest problem."""

    def __init__(self, kind: str, prefix: str):
        mesh = dolfinx.mesh.create_unit_square(COMM, 8, 8)
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
        fdim = mesh.topology.dim - 1

        def dofs(space, marker):
            return dolfinx.fem.locate_dofs_topological(
                space, fdim, dolfinx.mesh.locate_entities_boundary(mesh, fdim, marker)
            )

        self.g = dolfinx.fem.Function(V)
        self.set_time(1.0)
        zero = dolfinx.fem.Function(V)
        self.u = dolfinx.fem.Function(V, name="u")
        self.bcs = [
            dolfinx.fem.dirichletbc(self.g, dofs(V, lambda x: np.isclose(x[0], 0.0))),
            dolfinx.fem.dirichletbc(zero, dofs(V, lambda x: np.isclose(x[0], 1.0))),
        ]
        x = ufl.SpatialCoordinate(mesh)
        f = 10.0 * ufl.sin(3.0 * x[0]) * x[1]
        v0 = ufl.TestFunction(V)
        F0 = (1.0 + self.u**2) * ufl.inner(ufl.grad(self.u), ufl.grad(v0)) * ufl.dx - f * v0 * ufl.dx
        if kind == "single":
            self.unknowns, residual, jacobian = self.u, F0, None
            options = DIRECT
        else:
            W = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
            self.w = dolfinx.fem.Function(W, name="w")
            v1 = ufl.TestFunction(W)
            F0 += self.w * v0 * ufl.dx
            F1 = ufl.inner(ufl.grad(self.w), ufl.grad(v1)) * ufl.dx + (self.w - self.u**2) * v1 * ufl.dx
            self.bcs.append(dolfinx.fem.dirichletbc(dolfinx.fem.Function(W), dofs(W, lambda x: np.isclose(x[1], 0.0))))
            self.unknowns, residual = [self.u, self.w], [F0, F1]
            jacobian = dolfinx.fem.forms.derivative_block(residual, self.unknowns)
            options = NEST if kind == "nest" else DIRECT
        self.problem = dolfinx.fem.petsc.NonlinearProblem(
            residual,
            self.unknowns,
            J=jacobian,
            bcs=self.bcs,
            kind="nest" if kind == "nest" else None,
            petsc_options_prefix=prefix,
            petsc_options=options,
        )
        self.blocks = self.problem.b.getAttr("_blocks")

    def set_time(self, t: float) -> None:
        self.g.interpolate(lambda x: t * (1.0 + np.sin(np.pi * x[1])))
        self.g.x.scatter_forward()

    def bcs_by_block(self) -> list[list[dolfinx.fem.DirichletBC]]:
        """The conditions of each unknown."""
        if not isinstance(self.unknowns, list):
            return [self.bcs]
        return dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(self.problem.F), self.bcs)

    def functions(self):
        return self.unknowns if isinstance(self.unknowns, list) else [self.unknowns]

    def departure(self, x: PETSc.Vec) -> float:
        """The largest |x - g| on the Dirichlet dofs of x, over all ranks."""
        diff = x.duplicate()
        if self.blocks is not None:
            diff.setAttr("_blocks", self.blocks)
        diff.zeroEntries()
        bcs = self.bcs_by_block() if isinstance(self.unknowns, list) else self.bcs
        dolfinx.fem.petsc.set_bc(diff, bcs, x0=x, alpha=-1.0)
        return diff.norm(PETSc.NormType.NORM_INFINITY)

    def residual(self, x_values) -> tuple[float, PETSc.Vec]:
        """The norm of the residual of the SNES at the iterate with the function values ``x_values``, and a
        copy of the residual."""
        for fn, values in zip(self.functions(), x_values, strict=True):
            fn.x.array[:] = values
        dolfinx.fem.petsc.assign(self.unknowns, self.problem.x)
        b, (function, args, kargs) = self.problem.solver.getFunction()
        function(self.problem.solver, self.problem.x, b, *args, **kargs)
        return b.norm(), b.copy()


KINDS = ["single", "block", "nest"]


def random_state(setup: Setup, seed: int, with_dirichlet_values: bool) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    values = []
    functions = setup.functions()
    bcs_of = setup.bcs_by_block()
    for fn, bcs in zip(functions, bcs_of, strict=True):
        fn.x.array[:] = rng.random(fn.x.array.size)
        fn.x.scatter_forward()
        if with_dirichlet_values:
            for bc in bcs:
                bc.set(fn.x.array)
        values.append(fn.x.array.copy())
    return values


@pytest.mark.parametrize("kind", KINDS)
def test_residual_without_lifting_equals_dolfinx_where_x_holds_g(kind):
    lifted = Setup(kind, f"lifted_{kind}_")
    unlifted = Setup(kind, f"unlifted_{kind}_")
    DirichletWithoutLifting(unlifted.problem)

    held = random_state(lifted, 1, with_dirichlet_values=True)
    norm, b_lifted = lifted.residual(held)
    _, b_unlifted = unlifted.residual(held)
    b_unlifted.axpy(-1.0, b_lifted)
    # The lifting J (g - x) is exactly zero, but the Jacobian kernels add it in a different order.
    assert b_unlifted.norm() <= 1e-13 * norm

    # Where x departs from g, the lifting is not zero, and the residuals differ in the other rows.
    free = random_state(lifted, 2, with_dirichlet_values=False)
    norm, b_lifted = lifted.residual(free)
    _, b_unlifted = unlifted.residual(free)
    b_unlifted.axpy(-1.0, b_lifted)
    assert b_unlifted.norm() > 1e-3 * norm


@pytest.mark.parametrize("kind", KINDS)
def test_iterates_hold_g_and_converge_to_the_lifted_solution(kind):
    lifted = Setup(kind, f"lifted_solve_{kind}_")
    unlifted = Setup(kind, f"unlifted_solve_{kind}_")
    guesses = []
    # An initial-guess hook set before, which DirichletWithoutLifting calls after setting g in the guess.
    unlifted.problem.solver.setInitialGuess(lambda snes, x: guesses.append(unlifted.departure(x)))
    DirichletWithoutLifting(unlifted.problem)
    departures = []
    unlifted.problem.solver.setMonitor(
        lambda snes, its, norm: departures.append(unlifted.departure(snes.getSolution()))
    )

    # Two "time steps": in the second, the initial guess holds the values of the first.
    for t in (1.0, 2.0):
        departures.clear()
        for setup in (lifted, unlifted):
            setup.set_time(t)
            setup.problem.solve()
            assert setup.problem.solver.getConvergedReason() > 0
        assert guesses[-1] == 0.0
        assert len(departures) == unlifted.problem.solver.getIterationNumber() + 1
        assert all(departure == 0.0 for departure in departures)
        for fn_lifted, fn_unlifted in zip(lifted.functions(), unlifted.functions(), strict=True):
            np.testing.assert_allclose(fn_unlifted.x.array, fn_lifted.x.array, rtol=0, atol=1e-10)
    assert len(guesses) == 2


@pytest.mark.parametrize("kind", KINDS)
def test_hold_update_makes_the_next_iterate_hold_g(kind):
    setup = Setup(kind, f"hold_{kind}_")
    dirichlet = DirichletWithoutLifting(setup.problem)
    x = setup.problem.x
    random_state(setup, 3, with_dirichlet_values=False)
    dolfinx.fem.petsc.assign(setup.unknowns, x)
    assert setup.departure(x) > 0.0

    # An update with arbitrary values on the Dirichlet dofs, as a linear solve may leave there.
    y = x.duplicate()
    for y_sub in y.getNestSubVecs() if kind == "nest" else [y]:
        y_sub.setRandom()
    dirichlet.hold_update(x, y)
    x_next = x.copy()
    x_next.axpy(-1.0, y)
    # x - (x - g) is g up to the rounding of the two subtractions.
    assert setup.departure(x_next) <= 4 * np.finfo(float).eps * setup.g.x.array.max()

    # Where x holds g, the update is exactly zero on the Dirichlet dofs, and so the next iterate is g.
    dirichlet.set_values(x)
    assert setup.departure(x) == 0.0
    dirichlet.hold_update(x, y)
    x.axpy(-1.0, y)
    assert setup.departure(x) == 0.0


def test_rejects_a_residual_that_is_not_dolfinx():
    setup = Setup("single", "rejects_")
    b, (function, args, kargs) = setup.problem.solver.getFunction()
    setup.problem.solver.setFunction(lambda snes, x, b: function(snes, x, b, *args, **kargs), b)
    with pytest.raises(ValueError, match="assemble_residual"):
        DirichletWithoutLifting(setup.problem)
