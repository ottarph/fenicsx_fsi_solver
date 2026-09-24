# Shared-space iterative solver: implementation log

Plan: `docs/shared-space-iterative-solver-plan.md`. Target:
`src/xfsi_solver/solvers/fsi2_harmonic_diffmesh.py`. Branch `iterative-solver`.

## Environment (inspected 2026-09-24)

- DOLFINx 0.11.0, UFL 2026.1.0, PETSc/petsc4py 3.25.5, real float64, Python 3.14.
- External PETSc packages: hypre, MUMPS, SuperLU_dist (no ML).
- `dolfinx.fem.petsc.NonlinearProblem(F, u, J=..., P=..., kind=None, ...)`: with
  block forms and `kind=None` the Jacobian is one monolithic MPIAIJ matrix whose
  rows are, on each rank, the owned DOFs of `u`, then `v`, then `p` (block
  layout). `J` and `P` are accepted as nested lists of UFL forms; `P` is
  assembled into a separate matrix and passed to the KSP as `Pmat`.
- `derivative_block` does not simplify zero blocks; `jacobian_forms` expands
  derivatives and drops empty forms (so `(u,p)`, `(p,p)` and, in `no_ale`,
  `(p,u)` are never compiled).

## Mesh sizes (P2/P2/P1)

| mesh | cells | dim U (= dim V) |
|---|---|---|
| `mesh_sec_coarse` | 492 | 2 088 |
| `mesh_sec` (default in `main`) | 5 861 | 23 898 |
| `mesh_quad_fine_sec` | 5 343 quads | 43 352 |

## Baseline (`full/direct`, MUMPS), coarse mesh, dt = 0.0025

`tests/test_fsi2_harmonic_diffmesh.py`, 6 steps: Newton converges in 2
iterations per step (quadratic: 1.8e-1, 1.2e-6, 4.5e-12 at t = 0.005),
about 0.08 s per step serial.

## Step 1: configuration and no-ALE Jacobian

- `solve()` split into `build_problem()` (forms, spaces, BCs as an
  `FSIProblem`), `jacobian_forms(problem, mode)` and
  `create_nonlinear_problem(problem, config)`. `solve()` keeps its positional
  signature, takes an optional `SolverConfig`, and returns a `SolveResult`
  with per-step `StepInfo` (Newton and linear iterations, residual history,
  step time, drag, lift, tip displacement) and the final `FSIProblem`.
- `no_ale`: block `(v,u)` is the derivative of the solid-cell integrals of
  the momentum residual (`restrict_to_cells`, which refuses anything but
  single-marker cell integrals so that no term is silently dropped); `(p,u)`
  is dropped. The mesh-velocity term `(u-u_old)/dt` in the fluid momentum is
  among the omitted derivatives.
- Also fixed: `writer_p` was never closed on success; both writers now close in
  a `finally`. QoIs use `allreduce` so every rank has them.
- Tests (`tests/test_fsi2_harmonic_diffmesh_jacobian.py`): central finite
  differences agree with every retained block to 1e-6 relative at a nonzero
  admissible state (min det F > 0.9); the no-ALE displacement column matches
  a frozen-fluid-geometry residual; the omitted part is exactly the
  fluid-domain derivative and is nontrivial.
- Coarse mesh, 20 steps, `snes_atol=1e-10`: `no_ale/direct` takes 2-3 Newton
  iterations (full: 2) and agrees with `full/direct` to 1.5e-10 (u),
  1.3e-11 (v), 8e-13 (p) relative. Time per step 0.075 s vs 0.103 s, because
  the no-ALE Jacobian is cheaper to assemble. This is startup only; developed
  motion is still to be measured.
