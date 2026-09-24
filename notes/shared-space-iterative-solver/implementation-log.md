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

## Step 2: distributed splits and the exact Schur diagnostic

- `xfsi_solver.linalg.fieldsplit`: `field_index_sets` maps DOLFINx's local
  (ghosted) block index sets through the matrix's local-to-global map and keeps
  the owned rows, so the splits follow the actual storage rather than an
  assumed field-contiguous order. `nested_index_sets` renumbers subsets of a
  split into the numbering of the extracted submatrix (rank-wise position in
  the parent IS, offset by the exclusive scan of parent sizes).
- `fsi2_harmonic_diffmesh_fieldsplit.FieldSplitSolver`: FGMRES + PCFIELDSPLIT,
  splits `u` and `vp`, Schur factorization `full`,
  `pc_fieldsplit_diag_use_amat = off_diag_use_amat = false` (all blocks from
  Pmat, so a no-ALE Pmat is never mixed with full-Jacobian blocks).
  - It replaces the SNES Jacobian callback with one that calls
    `dolfinx.fem.petsc.assemble_jacobian` and then `ksp.setUp()`, so PC setup
    is timed separately; SNES's own `KSPSetOperators` with unchanged
    matrices does not repeat the setup.
  - Sub-solver options are inserted into the options database only around
    the first setup. `PetscOptionsClearValue` ignores `prefixPush`, so full
    names are used (a first version left all options behind).
  - KSP pre/post-solve hooks record iterations, reason, the true relative
    residual `||b - A x|| / ||b||` (with the Newton operator `A`) and its
    per-field norms. `FieldResidualMonitor` records per-field nonlinear
    residual norms in every mode.
- PETSc behaviour observed (toy problem and FSI): the sub-KSPs of a Schur
  fieldsplit are created at the outer `PCSetUp`, but their own `PCSetUp`
  happens lazily at the first apply. A sub-PC can therefore still be switched
  to a Python context after the outer setup.
- `exact` variant: MUMPS LU on `A`, `schur_precondition=full` and LU on the
  explicit Schur complement. The explicit Schur complement is dense: MUMPS
  accepted a SEQDENSE matrix but failed in the solve phase (INFOG(1)=-3) in
  standalone use, and ScaLAPACK does not accept MPIDENSE; `PCREDUNDANT` +
  PETSc dense LU works in serial and on 2 ranks.
- Tests (`tests/test_fsi2_harmonic_diffmesh_fieldsplit.py`, rerun on 2 ranks
  via `mpiexec` from the serial run): index sets are disjoint and exhaustive
  and select exactly the entries `dolfinx.fem.petsc.assign` writes for each
  field; nested numbering reproduces `J[v,v]`, `J[v,p]`, `J[p,v]` from the
  extracted `vp` matrix; the exact PCFIELDSPLIT application equals monolithic
  LU (no-ALE Jacobian at the admissible state, random RHS), and so do the
  reduced RHS `r_vp - A10 A00^{-1} r_u` and the recovery
  `A00^{-1}(r_u - A01 x_vp)`.
- Coarse mesh, 8 steps, `snes_atol=1e-10`, `ksp_rtol=1e-6`:

  | mode | Newton/step | FGMRES/Newton | s/step |
  |---|---|---|---|
  | full/direct | 2 | - | 0.106 |
  | full/fieldsplit-exact | 2 | 1 | 0.105 |
  | full J, no-ALE Pmat, exact | 2 | 1-2 | 0.120 |
  | no_ale/fieldsplit-exact | 2 | 1 | 0.053 |

  All agree with full/direct to <= 1.3e-10 relative. With the full Jacobian
  and a no-ALE preconditioner, FGMRES needs 2 iterations to reach 1e-6 on
  most Newton steps: at startup the omitted derivatives are a mild
  perturbation.
