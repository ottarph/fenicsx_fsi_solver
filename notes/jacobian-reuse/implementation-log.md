# Jacobian reuse parameter sweep: implementation log

Goal (2026-10-31): find the best `gamma_reassemble` of `fsi2_simplified` for the FSI2 benchmark over
t = 0 to 15, with MUMPS LU and the displacement prediction (`predict_displacements=True`), as a function
of the number of MPI ranks. Branch `gamma-reassemble-sweep`, from `restricted` at 129b4d6.

With reuse, the Jacobian J_0 of (v, p) is reassembled (and refactorized by MUMPS) when the last Newton
iteration reduced the residual norm by less than a factor γ, ||R_l|| > γ ||R_(l-1)||, and is otherwise
kept, also across time steps. `gamma_reassemble=None` reassembles in every iteration. A small γ
reassembles often and needs fewer iterations; a large γ keeps J_0 for longer at more iterations.

## Approach

The number of assemblies N_A(γ) and Newton iterations N_it(γ) do not depend on the number of ranks
(up to round-off in MUMPS, which could flip a borderline reuse decision), while the costs do. So:

1. **Counts.** One full run (T = 15, 6000 steps of dt = 0.0025, from rest) per γ, all at once on
   2 ranks each, 16 values over the 32 physical cores, each run pinned to its own two cores. Their
   wall times are disturbed by the shared memory bandwidth and are not used for the costs.
2. **Costs.** Short runs on the otherwise idle machine, one at a time, at 1 to 32 ranks, from a state
   in the periodic regime, with γ = None (every iteration assembles) and γ = 0.2 (most iterations
   reuse). The Newton log gives the wall time of each iteration and whether it assembled, so the
   mean time t_A(p) of an iteration that assembles and t_R(p) of one that reuses, and the time per
   step outside the Newton iterations o(p).
3. **Model.** E(γ, p) = N_A(γ) t_A(p) + (N_it(γ) − N_A(γ)) t_R(p) + S o(p), minimized over γ per p.

Choices:

- MUMPS LU (`DEFAULT_DIRECT_SOLVER_OPTIONS`), the solver's default; the refactorization is what reuse
  saves. The uncommitted toggles in `main()` on `restricted` (the user's) are left alone; the sweep
  calls `solve()` through `xfsi_solver.scripts.jacobian_reuse_sweep`.
- `max_quadrature_degree=6`, as in `jobs/fsi2_simplified.sh` (same QoIs as uncapped,
  notes/restricted-cleanup).
- γ values: None, 0.01, 0.02, 0.05, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0.
  With γ ≥ 1, J_0 is reassembled only when the residual grows; such runs may exceed `snes_max_it`
  = 20 in some step and stop, which counts as infeasible.
- The user asked for the whole workstation, all 32 physical cores, for this sweep, rather than the
  usual two cores left free.

## Log

### 2026-10-31

- `solve()` returns its statistics (elapsed time, steps, Newton iterations, assemblies, and both per
  step); test against the Newton log. Commit 3adbde6.
- `scripts/jacobian_reuse_sweep.py` runs one γ and writes `stats.json` with the statistics and the
  PETSc event times (max over ranks); `jobs/jacobian_reuse_sweep.sh` launches the sweep. MPICH Hydra
  does not bind ranks by default, so each run is pinned with `-bind-to user:<cores>`. Commit f220fd8.
- 14:34: sweep launched, 16 runs × 2 ranks, into `output/jacobian_reuse_sweep/g<γ>/`, logs
  `output/logs/jrs_g<γ>.txt`.
- 14:37: γ = 1.0 stopped at t = 1.5675 (start-up, Newton over `snes_max_it` = 20). Infeasible from
  rest; whether it works from a developed state is open (see below).
- The user also asked for the best γ per stage: from rest until the flow has developed, and from an
  already developed flow (a restart). Every sweep run records the iterations and assemblies of each
  step, so the cost model applies to any window of steps without new runs. `analyze.py` defines
  t_dev from the tip displacement A_y (the first peak from which the amplitude stays within 5% of
  the mean amplitude over the last 2 s), gives the best γ before and after it, the gain from
  switching γ at t_dev over the best single γ, and the best γ per 1 s window. A restart assembles
  once in its first step, which the developed-stage counts (carried over from the start-up) lack;
  negligible over thousands of steps. Since γ = 1.0 failed only in the start-up, a run with it (and
  any other γ that fails) restarted from a developed checkpoint is needed to answer the second
  question fully; to do once a developed checkpoint exists.
- 14:40: γ = 0.6 stopped at t = 6.52 (Newton over 20 iterations), while the oscillation grows.
  Reuse is kept as long as each iteration contracts by at least γ, so with γ ≥ 0.6 a stale J_0 can
  contract by just under γ in every iteration and run out of iterations. The solver has no fallback
  (reassemble and retry a step that fails), so such γ are infeasible for this setup as is.
- 14:41: γ = 0.8 stopped at t = 6.5275 the same way.
- 14:43: γ = 0.5 stopped at t = 8.11 the same way: every γ ≥ 0.5 fails from rest.
- 14:46: copied the checkpoints of the γ = 0.2 run while its latest was t = 10 (the oscillation has
  grown to its full amplitude of about ±80 mm by t ≈ 9) to `output/jacobian_reuse_sweep/developed_t10`.
  (A first copy, at t = 8, was taken too early by a broken wait condition and deleted.) Restarted
  γ = 0.5, 0.6, 0.8, 1.0 from it to T = 15 on the 8 cores the failed runs freed, into
  `output/jacobian_reuse_restart/g<γ>/`, logs `output/logs/jrr_g<γ>.txt`; they start at t = 10.0025.
  `jobs/jacobian_reuse_sweep.sh` got `INITIAL_STATE_DIR` and `LOG_PREFIX` for this.
- 14:49: all four restarts stopped within 0.06 to 0.16 s (t = 10.16, 10.15, 10.06, 10.055 for
  γ = 0.5, 0.6, 0.8, 1.0): γ ≥ 0.5 is infeasible in the developed flow too, and fails there sooner
  than from rest. Restarted γ = 0.2, 0.3, 0.4, 0.45 from t = 10 instead, on the same cores: they
  check that the counts of a restart match those of the continuous runs over t = 10 to 15, and
  narrow the limit of feasibility between 0.4 and 0.5.
- 14:58: the restarts with γ = 0.2, 0.3, 0.4 reached T = 15; γ = 0.45 stopped at t = 10.4325. So
  the limit of feasibility lies between 0.4 and 0.45 in the developed flow, and γ = 0.4 is close
  to it (a margin is advisable). (The monitor on these logs had expanded its glob at the start, so
  it missed this failure; seen in the log.) All feasible sweep runs finished in 1160 to 1408 s on
  2 ranks each; only γ = None is still running.
- 15:12: γ = None finished (2342 s on 2 ranks). The restarts from t = 10 give the same iterations and
  assemblies per step over t = 10 to 15 as the continuous runs, to three digits, so the
  developed-stage counts of the sweep stand for restarts. t_dev = 9.40 by the 5% criterion
  (amplitude 82.7 mm). Calibration launched alone on the idle machine (process group in
  `output/logs/jrc.pid`), 1 to 32 ranks, γ = None and 0.2, 200 steps from t = 15 each.
- 15:23: calibration done, 16 runs, 23 to 130 s each. The model reproduces their measured wall times
  to within 3% (`tables.md`). An iteration that assembles and refactorizes costs 3.7 to 5.4 times
  one that reuses; both stop scaling past 24 ranks (32 ranks is slightly slower than 24).
- `analyze.py` writes `tables.md` and `plots/` (SVG committed; PNG is git-ignored).

## Results

Predicted wall time over t = 0 to 15 (`plots/predicted_time.svg`, `tables.md`):

| ranks | best γ | time (s) | vs γ = 0.2 | vs None | within 2% of best |
|---|---|---|---|---|---|
| 1 | 0.08 | 1245 | −5.5% | −62% | 0.05, 0.08, 0.1 |
| 2 | 0.05 | 843 | −6.5% | −59% | 0.05, 0.08, 0.1 |
| 4 | 0.08 | 594 | −5.8% | −62% | 0.05, 0.08, 0.1 |
| 8 | 0.05 | 389 | −7.9% | −56% | 0.05, 0.08, 0.1 |
| 12 | 0.05 | 343 | −8.9% | −54% | 0.02, 0.05, 0.08 |
| 16 | 0.05 | 312 | −9.3% | −52% | 0.02, 0.05, 0.08 |
| 24 | 0.05 | 284 | −9.6% | −51% | 0.02, 0.05, 0.08 |
| 32 | 0.05 | 292 | −9.3% | −52% | 0.02, 0.05, 0.08 |

- **γ = 0.05 is within 2% of the best at every rank count** (0.08 is best by < 0.5% on 1 and 4
  ranks). The optimum shifts slightly toward smaller γ with more ranks, as t_A / t_R falls from about
  5 to under 4. Any reuse halves the run time compared with reassembling every iteration.
- The minimum is flat between 0.02 and 0.1 and rises slowly above it (+20 to 28% at γ = 0.4).
  γ ≥ 0.45 is infeasible (Newton exceeds `snes_max_it` = 20 in some step): from rest, γ ≥ 0.5 fails
  between t = 1.6 and 8.1; from the developed state at t = 10, γ = 0.45 to 1.0 fail within 0.5 s.
- **Per stage** (start-up t < 9.40, developed after): the best γ is the same in both, 0.05 (0.08 on
  1, 2 and 4 ranks in the developed stage), and switching γ at t_dev gains nothing (≤ 0.1%). A
  restart from a developed state should use the same γ. Per 1 s window, γ = 0.01 is best during the
  quiet first 5 s (few iterations per step, so the reuse gains little and costs extra iterations),
  then 0.05 to 0.1; the gain of following this is negligible.
- 24 ranks is the fastest rank count for this mesh (284 s); 16 ranks is within 10% of it.
- Caveat: t_A, t_R and o come from 200 steps in the developed flow; the start-up steps may differ a
  little in cost per iteration. The counts are from 2-rank runs; on other rank counts MUMPS round-off
  could change borderline reuse decisions, which is not expected to shift the averages.

## threeDfsi_simplified from a developed state

Goal (2026-10-31, the user): instrument `threeDfsi_simplified` as `fsi2_simplified` and test a few
values of γ below its default 0.2 from a developed state. Branch `gamma-reassemble-3d`, from
`restricted` at 36adee8 (which has the 2D sweep merged and `fsi2_simplified` defaulting to 0.05).

Expectation: in 3D a Newton iteration is dominated by the linear solve (FGMRES), the residual and
the mesh motion. With `MUMPS_ON_SOLID`, assembling the Jacobian and setting up its preconditioner
(MUMPS on the solid, ILU on the fluid, BoomerAMG on the Schur complement) is only about a tenth of
an iteration (notes/fsi3d, 2026-10-29). So t_A / t_R is close to 1, unlike about 4 to 5 in 2D, and
the fewer iterations of a small γ should pay off more than in 2D, possibly down to γ = None.

- `threeDfsi_simplified.solve` returns the statistics of `fsi2_simplified.solve`, plus the linear
  iterations and the wall time of each step. The check against the Newton log is added to the slow
  test `test_threeDfsi_simplified_prediction_options` (not run; slow tests need permission) and was
  run by hand on a 2-step coarse-mesh run of the driver. Commit 68fda56.
- `scripts/jacobian_reuse_sweep.py` takes `--solver threeDfsi_simplified`: `MUMPS_ON_SOLID` with
  `requires_v_sub_IS`, the quadrature cap of 6 also on the mesh motion, the medium mesh, dt = 0.004.
  Commit 7b1ad24.
- 16:23: `jobs/jacobian_reuse_3d.sh` launched (process group in `output/logs/jr3d.pid`): γ = 0.2,
  0.1, 0.05, 0.02, 0.01, None, one at a time on 28 ranks on the otherwise idle machine, 88 steps
  (two periods of the beam, which oscillates with period 0.176 s at constant amplitude) from the
  `threeDfsi_mixed_medium` checkpoint at t = 10, into `output/jacobian_reuse_3d/g<γ>/`, logs
  `output/logs/jr3d_g<γ>.txt`. `analyze_3d.py` writes `tables_3d.md`. Commit 1f14883.
- 17:14: all six runs finished (432, 411, 410, 540, 599, 604 s for γ = 0.2, 0.1, 0.05, 0.02, 0.01,
  None), QoIs equal to γ = 0.2 within 2e-7 of their range (drag within 1.5e-5, the digits written).

### Results (`tables_3d.md`)

| γ | Newton it. / step | assemblies / step | FGMRES / solve | s / step, after the first | vs 0.2 |
|---|---|---|---|---|---|
| None | 5.00 | 5.00 | 13.5 | 6.78 | +40% |
| 0.01 | 5.65 | 4.51 | 13.5 | 6.73 | +40% |
| 0.02 | 5.83 | 3.39 | 13.3 | 6.06 | +26% |
| 0.05 | 6.22 | 1.05 | 12.9 | 4.58 | −5.1% |
| 0.1 | 6.26 | 1.00 | 12.9 | 4.58 | −4.9% |
| 0.2 | 6.88 | 0.86 | 12.5 | 4.82 | 0 |

- **γ = 0.05 and 0.1 are best, both 5% faster than the default 0.2; below 0.05 it gets much worse.**
- The expectation above was wrong: an iteration that assembles costs t_A = 1.28 s and one that
  reuses t_R = 0.57 s, a ratio of 2.2. Per assembly: SNESJacobianEval 0.39 s, the factorizations
  (MUMPS on the solid and ILU on the fluid, two MatLUFactorNum) 0.15 s, the rest of the PC setup
  (BoomerAMG on the Schur complement) about 0.15 s. The 0.05 s of the Jacobian in notes/fsi3d was
  per Newton iteration, averaged over 53 iterations with 7 assemblies, so about 0.38 s per assembly,
  consistent with this; I had misread it as per assembly.
- Why the step from 0.05 to 0.02 is so large: the Jacobian is that of (v, p) only, without the
  derivatives through the mesh motion and the update of u_s, so even a fresh Jacobian contracts
  Newton only linearly, by 0.01 to 0.03 per iteration (median 0.012, 0.006, 0.019, 0.022, 0.026 in
  iterations 1 to 5 with γ = None). A γ below that floor reassembles in almost every iteration for
  no faster contraction (γ = 0.01: 4.5 assemblies in 5.65 iterations per step). With γ = 0.05 or
  0.1, J is reassembled about once per step and the reused one contracts by 0.03 to 0.1. γ = 0.2
  saves 0.14 assemblies per step but costs 0.6 iterations more.
- Compared with 2D (best γ = 0.05 at t_A / t_R = 4 to 5), the optimum in 3D is set less by the cost
  ratio than by the contraction floor of the inexact Jacobian: γ should stay above it. 0.1 keeps
  more margin from the floor than 0.05 at the same speed. The default 0.2 of
  `threeDfsi_simplified` is left unchanged for now (the user's decision).
