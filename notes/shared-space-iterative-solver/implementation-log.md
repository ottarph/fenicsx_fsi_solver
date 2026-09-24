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
- ~~Coarse mesh timings of the exact variant~~ (withdrawn, see the
  correction below): the table first recorded here was measured after a
  direct solve in the same process, and the fieldsplit runs were in fact
  MUMPS LU on the whole Jacobian.

### Correction: leaked NonlinearProblem options

`dolfinx.fem.petsc.NonlinearProblem` (DOLFINx 0.11.0, PETSc 3.25.5) removes
its `petsc_options` after `setFromOptions` with `prefixPush` + `del opts[k]`,
which does not apply the pushed prefix, so every option stays in the global
database. A later solver with the same prefix therefore picked up
`pc_type=lu` from an earlier direct solve when its PC was set from options.
The solve-level exact-fieldsplit test and the timing table above ran LU
without noticing; the standalone block-factorization test used its own
prefix and was valid. Fixed by deleting the options under their full names
after constructing the `NonlinearProblem`; `FieldSplitSolver` now also fails
if the configured FGMRES/fieldsplit hierarchy was replaced by options. With
the fix the exact-fieldsplit solve test takes 5 s (dense Schur complement)
instead of 1.4 s, and passes: 1 FGMRES iteration per Newton step with the
Jacobian as Pmat.

## Step 3: auxiliary P_vp and approximate subsolvers

`FieldSplitConfig` (in `fsi2_harmonic_diffmesh_fieldsplit`) selects every
level; each approximation can be replaced by LU to isolate its effect.
Production default: `variant="auxiliary"`, `displacement="block_triangular"`
with `displacement_fluid="cholesky"`, `momentum="schur"`, `velocity="hypre"`,
`pressure="selfp"`, `convection=True`.

- `P_vp` is the user Schur preconditioning matrix of the outer split and the
  Pmat of a Python PC (`MomentumPressurePC`) that owns an inner PCFIELDSPLIT
  `v|p` (full Schur) on `P_vp`. `FieldSplitSolver._check_vp_layout` verifies
  that `P_vp` and the `(v,p)` split of the Jacobian number the local DOFs
  identically (they do; otherwise it raises).
- With `convection=True` `H_hat = H + theta dt E_s` and `P_vp` is built
  algebraically: copy the `(v,p)` block of Pmat into `P_vp` (subset-pattern
  AXPY; `P_vp` has a structurally present zero `(p,p)` block and is assembled
  once to fix its pattern) and assemble `theta dt E_s` (solid cells only) on
  top. 16 ms instead of 82 ms (default mesh) for the form-based `P_vp`, and
  equal to the form-assembled `[[J_vv + theta dt E_s, G], [B, 0]]` to 1e-13
  (test). DOLFINx *inserts* the diagonal of Dirichlet rows when assembling
  with BCs, so the added form must use `diag=1.0`; `diag=0.0` zeroed the
  unit diagonal copied from Pmat and FGMRES diverged.
- In this discretization `A^{-1} C = -theta dt` on the solid DOFs up to the
  alpha-scaled interface terms, so `P_vp` with exact subsolves is almost the
  exact reduced Schur complement: 1.0-1.1 FGMRES iterations per Newton step at
  startup, 3.0 (with convection) / 5-5.5 (without) at developed coarse states.
- Displacement (`DisplacementPC`): `I` = DOFs of solid cells (interface
  included), `f` = fluid interior; `M_II y_I = r_I` with the constant solid
  mass (Cholesky, factored once), then `A_ff y_f = r_f - A_fI y_I`.
  `||A_II - M_II|| <= 1e-8 ||M_II||` (test). Found: with a single AMG V-cycle
  for `A_ff` the displacement error stayed at 4e-7 while the total residual
  converged, because the alpha-scaled mesh rows are invisible in the FGMRES
  norm. The fluid-interior solve must therefore be accurate. It is constant
  (linear mesh equation on the reference configuration), so it is factored
  once (Cholesky: 3 ms per solve on the default mesh vs 29 ms for
  BoomerAMG-CG to 1e-8); `displacement_fluid="amg"` keeps the AMG option. The
  AMG hierarchy / factorization is rebuilt only if `A_ff` changes (checked at
  every setup).
- PETSc pitfall: `fieldsplit_<schur split>_inner_` is reserved (KSP for
  `A00^{-1}` inside the Schur complement, created when options with that
  prefix exist), so my first inner prefix `fieldsplit_vp_inner_` silently
  replaced the `u` sub-KSP. The inner split uses `fieldsplit_vp_aux_`.
- PETSc's separate upper KSP (`fieldsplit_u_upper_`) does not save the lower
  `A^{-1}` in the full factorization (it adds a third solve), so the lower
  application, of which only the solid/interface part enters `A10 y`, still
  does the full displacement solve.
- Velocity: `H_hat` is SPD without convection. Default mesh, CG to 1e-8 on
  `H_hat`: BoomerAMG 10 iterations, GAMG + rigid body modes 25, GAMG without
  near-nullspace 34, BoomerAMG nodal 18, Jacobi 236. One BoomerAMG V-cycle is
  used (GAMG + rigid body modes is `velocity="gamg"`).
- Pressure: GMRES iterations to 1e-8 on the inner Schur complement
  `B H_hat^{-1} G` (exact `H_hat` solves), default mesh: Cahouet-Chabard
  (`rho_f/dt K_p^{-1} + theta mu_f M_p^{-1}`, ALE metric `J^2/J_mid`,
  Dirichlet at the outflow, Neumann elsewhere incl. the interface) 25,
  `selfp` (`B diag(H_hat)^{-1} G`) + LU 17, + one BoomerAMG V-cycle 18. The
  coarse developed state gives 17 vs 13-14. `selfp` is the default; it
  inherits boundary conditions and the solid mass at the interface
  algebraically. Pressure preconditioning remains the dominant source of
  outer iterations (below).
- Outer FGMRES iterations per linear solve (no-ALE Jacobian, ksp_rtol 1e-6):

  | preconditioner | coarse startup | default startup | coarse t=3 | coarse t=6 |
  |---|---|---|---|---|
  | u LU, `P_vp` LU | 1.1 | 1.0 | 5.0 (3.0 conv.) | 5.5 (3.0 conv.) |
  | inner Schur, v LU, accurate p | 1.1 | 1.0 | | |
  | inner Schur, v LU, Cahouet-Chabard | 7.9 | 12.9 | | |
  | inner Schur, v LU, selfp | | 8.4 | | |
  | v BoomerAMG, Cahouet-Chabard | | 18.7 | 11.5 | 11.7 |
  | v BoomerAMG, selfp (production) | 10.2 | 12.2 | 7.8-8.0 | 8.2-8.3 |
  | v GAMG, Cahouet-Chabard | 19.5 | 28.2 | | |

- Default mesh, startup (4 steps), s/step: direct no-ALE 0.84, production
  1.02 (Jacobian 0.12 s, auxiliary 0.03 s, linear solve 0.30 s per Newton
  step; displacement PC 5.4 ms and momentum-pressure PC 9.3 ms per
  application).
- Tests (`tests/test_fsi2_harmonic_diffmesh_preconditioners.py`, also on 2
  ranks): algebraic `P_vp` equals the form-assembled operator, form-based
  `H_hat` symmetric, the displacement PC equals the block-triangular solve
  (with LU) and partitions the displacement DOFs, Cahouet-Chabard equals
  `rho/dt K^{-1} + theta mu M^{-1}` including the pressure numbering map,
  the production hierarchy has no LU and no dense matrices (`ksp_view`), and
  production fieldsplit (no-ALE Jacobian, and full Jacobian with no-ALE Pmat)
  matches full/direct over 12 steps (fields 1e-7, QoIs 1e-6) with the
  fluid-interior mesh rows resolved to 1e-6 relative.

## Step 4: checkpoints, benchmarks, results

- `solve(..., initial_state=, checkpoint_dir=, checkpoint_every=)` and
  `save_state`/`load_state` (per-rank npz, same mesh and rank count);
  `scripts/fsi2_harmonic_diffmesh_checkpoints.py` generates developed states
  with `full/direct`; `scripts/fsi2_harmonic_diffmesh_benchmark.py` runs each
  mode in its own process from the same state and writes
  `results.{json,md}`. The direct solver is instrumented like the fieldsplit
  solver (`InstrumentedSolver`), so LU factorization appears as setup time.
- Baseline failure: `full/direct` on `mesh_sec_coarse`, dt 0.0025, failed at
  t = 9.5675 (SNES not converged; tip A_y about -0.035, still growing towards
  the FSI2 amplitude of about 0.08). Developed-state benchmarks therefore use
  the t = 6 and t = 9 coarse states and the t = 5 default-mesh state.
- Tried after the batch: fixed 2-4 BoomerAMG V-cycles (Richardson) for the
  velocity and 2 for the `selfp` pressure. Default mesh startup: FGMRES
  iterations 11.8 -> 8.5 but time per step 0.98 -> 1.45 s; coarse t = 9:
  8.4 -> 4.9 iterations, 0.180 -> 0.194 s/step. Not adopted (not committed).
- Results, commands and conclusions: `docs/shared-space-iterative-solver.md`.
  Summary: all modes agree with `full/direct` to <= 2e-8; FGMRES needs 7-12
  iterations per solve on both meshes and all states; the no-ALE Newton needs
  4-5 iterations at developed motion (full: 2-3), which on the default mesh
  cancels its 2.7x cheaper assembly; MUMPS remains faster at these 2D sizes
  (no-ALE fieldsplit 7-25% slower than no-ALE direct on the default mesh,
  40-50% on the coarse mesh).
