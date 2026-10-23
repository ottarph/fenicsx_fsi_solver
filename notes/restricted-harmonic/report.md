# Restricted harmonic FSI2 solvers: report

Task: `output/prompts/restricted-harmonic-solvers.md`. Branch `restricted-harmonic` from
`jacobian-plotting`; not pushed. Details in `implementation-log.md`.

## Commits

```
(+ the commit adding this report and the log entries after 87709d6)
87709d6 Make the Jacobian plots of the restricted harmonic formulation
13f52ed Document the restricted harmonic FSI2 solvers
e83a4d4 Add the condensed restricted harmonic FSI2 solver with J_0 and fieldsplit
5c7fc9e Add the restricted harmonic FSI2 solver with the full Jacobian
```

## What was added

- `src/xfsi_solver/solvers/fsi2_harmonic_fulljacobian.py`: `fsi2_biharmonic_fulljacobian` with
  harmonic mesh motion `(grad u, grad du)_f`, unknowns (u, v, p), restricted interface rows holding
  the solid kinematic equation, full Jacobian, MUMPS LU (`mesh_motion_scaling` kept).
- `src/xfsi_solver/solvers/fsi2_harmonic_condensed.py`: `fsi2_biharmonic_mixed` with the same
  harmonic mesh motion, built from `fsi2_biharmonic_c0ip` (m = u_I only): u_S condensed out of the
  solid stress, J_0, Jacobian reuse, fieldsplit q_c, u_S, m = u_I with K_m = A_II factorized once by
  MUMPS Cholesky through `IdenticalBlocksPC`, `direct_k_c_solve` and the iterative K_c option.
  Named "condensed" rather than "mixed" because there is no mixed form of the harmonic equation;
  the u_S condensation is what distinguishes it from the full-Jacobian solver.
- Tests: `tests/test_fsi2_harmonic_fulljacobian.py`, `tests/test_fsi2_harmonic_condensed.py`
  (solve, restart, restart on 2→1 and 1→2 ranks, initial state, start from a biharmonic
  checkpoint, matrix plots, and the two harmonic solvers against each other).
- `jobs/fsi2_jacobian_plots.sh`: `FORMULATION=harmonic` (default `biharmonic`), harmonic default
  COMMIT e83a4d4; the biharmonic default stays 66b0885. Since 2026-10-23, one run makes both
  formulations at e83a4d4 into one folder; see `implementation-log.md`.
- README solver table and an implementation line in `docs/restricted-iterative-solver.md`.

## Tests

- Full suite: 189 passed in 543 s.
- Condensed vs full-Jacobian solver after 8 steps from rest on `mesh_sec_coarse`: relative
  difference 2e-10 (u), 9e-10 (v), 2e-11 (p) with the default Jacobian reuse (3e-11 with a fresh J_0
  every iteration). The test tolerance is 1e-6 relative to each field's maximum, as in the restart
  tests of the solvers with Jacobian reuse: the difference comes from where the linearly converging
  J_0 iteration stops below `snes_atol` = 1e-7.
- Side finding: the old `fsi2_harmonic` (unrestricted test functions, alpha_u = 1e-9 with an
  interface flux term) agrees with `fsi2_harmonic_fulljacobian` to about 1e-12 after 8 steps.
  With alpha_u = 1e-9, the fluid contribution to the interface rows is negligible.

## Runs

| run | state | time | Newton | min det F (fluid) at t = 7 |
|---|---|---|---|---|
| `output/matrix_block_runs/fsi2_harmonic_coarse/`, 4 ranks | finished, rest → t = 7, checkpoint at t = 7 | 71 s | 13475 in 2800 steps (≤ 8 per step), 399 assemblies | 0.970 |
| `output/matrix_block_runs/fsi2_harmonic_normal/` (mesh_sec), 8 ranks | finished, rest → t = 7, checkpoint at t = 7 | 191 s | 13982 in 2800 steps (≤ 9 per step), 58 assemblies | 0.603 |
| `FORMULATION=harmonic jobs/fsi2_jacobian_plots.sh` | finished | 17 s | | |
| biharmonic check of the edited script (scratch OUT_DIR) | finished; PNGs byte-identical to `66b0885/figures` | ~1 min | | |
| full test suite | finished, 189 passed | 543 s | | |

The mesh did not degenerate before t = 7: no Newton failures, and min det F > 0 at every checkpoint.
On mesh_sec, min det F fell steadily, from 0.99 at t = 1 to 0.60 at t = 7. At t = 7 the oscillation
has only started: tip |A_y| ≤ 1.5e-2 on mesh_sec (≤ 4.6e-3 on the coarse mesh), against about 7.8e-2
in the biharmonic state at t = 15. So the harmonic plots show a far less deformed state than the
biharmonic ones.

## Figures

`output/figures/jacobian_plots/e83a4d4/`, next to the biharmonic plots `*_biharmonic_*.png`:
`spy_harmonic_{full,condensed,approximate}.png` (coarse mesh, 1 rank) and
`block_norms_harmonic_{full,condensed,approximate}.png` (mesh_sec, 8 ranks), at t = 7.0025, in the
order (u_S, v, p, u_I), with `run_info.txt`. Since 2026-10-23 the script keeps only these; the
per-run plots and logs are left in a temporary directory, see `implementation-log.md`.

What they show (block norms on mesh_sec):

- (v, u_S): 2.68e8 in the full Jacobian (E_S + F_S), 6.60e3 after condensation (F_S only), and
  empty in J_0. The condensation moves the solid stiffness into (v, v), which grows from 6.97e3 (H)
  to 3.37e5 (H_c).
- (u_S, u_S) = A_S = 737 and (u_S, v) = C = 0.926 = θΔt·737 in all three. (u_S, p) and (u_S, u_I) are
  empty.
- The ALE blocks F_I = 5.55e4, D_S = 0.258 and D_I = 9.39 are present in the full and condensed
  Jacobians and gone in J_0.
- The mesh rows are only A_IS = 27.4 and A_II = 690 in all three, with no v or p columns, so J_0 is
  block lower triangular in ((u_S, v, p), u_I). The spy plots show the same structure.

This matches the harmonic versions of eqs. (1), (3) and (6).

## Left undone / notes

- Nothing in the task is left undone.
- Not tested: the iterative K_c option over a long run. It is covered only by the short tests, as
  for the biharmonic solvers. The develop runs used `direct_k_c_solve=True`, like the biharmonic
  matrix-block runs.
- Run helpers (not in git): `output/matrix_block_runs/harmonic_run.py`, `harmonic_watch_detF.py`,
  `harmonic_src/` (source of e83a4d4).
