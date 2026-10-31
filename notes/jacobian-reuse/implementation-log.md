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
