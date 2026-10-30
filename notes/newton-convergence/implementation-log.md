# Newton convergence of the FSI2 solvers: implementation log

Goal: analyse the nonlinear convergence of the solvers with direct linear solves, in particular

- the effect of the approximate Jacobian J_0, by comparing `fsi2_biharmonic_mixed` (J_0) with
  `fsi2_biharmonic_fulljacobian` (the full Jacobian), and
- the effect of the Jacobian reuse (`gamma_reassemble`), in `fsi2_simplified`.

The iterative solvers (FGMRES on K_c) are left out for now.

## 2026-10-30: Newton log

### Built-in monitors or a custom one

PETSc's `snes_monitor` prints only the total residual norm, with no norms per field (that needs a DM
with sections, which dolfinx's blocked and nest vectors do not have), and nothing shows whether the
solver's Jacobian callback reassembled the Jacobian or reused it. Parsing the ASCII monitor output
is also fragile once the solvers print their own lines. So the solvers get a small custom SNES
monitor that writes a table instead.

### `xfsi_solver.tools.newton_log`

`NewtonLog(path, blocks, comm, restart, t, dt)` is a SNES monitor, attached by the solvers with
`solver.setMonitor(newton_log)`, that appends one row per Newton iterate to a tab-separated text
file on rank 0, with the columns

    t  it  assembled  time  residual  residual_<block>...  update  update_<block>...

- `residual*`: l2 norms of the SNES residual vector at the iterate, in total and on the owned rows
  of each block (`field_blocks(problem, names)` gives one block per unknown). The norms of all
  blocks are reduced over the ranks in one `allreduce`.
- `update*`: the same norms of `snes.getSolutionUpdate()`, the Newton step to the iterate (with
  `snes_linesearch_type: none` the full step); `nan` at `it = 0`.
- `assembled`: whether the Jacobian of the step to the iterate was assembled anew. Detected from
  the PETSc object state of the Jacobian matrix (`Mat.stateGet()`), which every assembly
  increases and a reused Jacobian keeps, so the solvers' Jacobian callbacks need no changes.
  Checked in the tests against a callback that assembles only once, and against
  `fsi2_biharmonic_fulljacobian`, which assembles in every iteration.
- `time`: wall time since the residual of the initial guess, so row differences are the time of
  one Newton iteration (Jacobian, linear solve and, in `fsi2_simplified`, the mesh motion in the
  KSP post-solve hook, and the residual).

A row is written as soon as the iterate is reached, so a failing solve leaves its iterations in the
file. The file is set up with `init_qoi_file`, so a restart drops the rows after the checkpoint
time, as from the QoI file. `read_newton_log(path)` reads it into a structured array.

The values are written with 7 significant digits, which is enough for convergence rates. The norms
are unscaled l2 norms over all dofs, so blocks with different units (p against v and u) differ by
orders of magnitude: the pressure update dominates the total update norm. Compare the blocks
separately, or rates per block.

The solvers `fsi2_simplified` (blocks v, p), `fsi2_biharmonic_mixed` and
`fsi2_biharmonic_fulljacobian` (blocks u, v, p, z) take `newton_log_path=None`; their `main()` and
the job scripts `jobs/fsi2_simplified.sh` and `jobs/fsi2_biharmonic_mixed.sh` pass a path next to
the QoI file.

### `gamma_reassemble=None`

With `gamma_reassemble=0`, `fsi2_biharmonic_mixed` and `fsi2_simplified` still reuse the Jacobian
at the first iteration of every time step, from the step before. To compare J_0 with the full
Jacobian, which `fsi2_biharmonic_fulljacobian` assembles in every iteration, and to have a baseline
without reuse, `gamma_reassemble=None` now reassembles in every Newton iteration.

### First comparison on mesh_sec

Runs over 200 steps of dt = 0.0025 from t = 16.0025, with the beam oscillating, all with direct
solves (`direct_k_c_solve=True` for the mixed solver), each on 5 ranks, the five at once on the
32-core workstation. The start state is the `fsi2_simplified` checkpoint at t = 16.0 (from
`mesh_sec`), advanced by one step of `fsi2_biharmonic_mixed`, which checkpoints z, since
`fsi2_biharmonic_fulljacobian` needs z in its start state and `fsi2_simplified` does not checkpoint
it. The first step of each run is left out of the table. The script was a scratch script that
called `solve()` with `initial_state_dir` and `newton_log_path`; the rates are medians over the
steps of ||R_(k+1)|| / ||R_k||.

| run                              | Newton its/step | assemblies | time/step (s) | time/it (s) | rate 0→1 | 1→2    | 2→3    | 3→4    |
|----------------------------------|----------------:|-----------:|--------------:|------------:|---------:|-------:|-------:|-------:|
| fulljacobian                     | 3.00            | 597        | 1.106         | 0.369       | 7.6e-2   | 1.0e-4 | 9.0e-7 |        |
| mixed, gamma_reassemble=None     | 5.21            | 1036       | 0.595         | 0.114       | 1.6e-1   | 3.7e-3 | 6.8e-3 | 6.9e-3 |
| mixed, gamma_reassemble=0.2      | 7.57            | 126        | 0.229         | 0.030       | 1.6e-1   | 1.5e-2 | 3.1e-2 | 6.6e-2 |
| simplified, gamma_reassemble=None | 5.21           | 1036       | 0.468         | 0.090       | 1.6e-1   | 3.7e-3 | 6.8e-3 | 6.9e-3 |
| simplified, gamma_reassemble=0.2 | 7.57            | 126        | 0.197         | 0.026       | 1.6e-1   | 1.5e-2 | 3.1e-2 | 6.6e-2 |

- The full Jacobian converges quadratically, in 3 iterations per step. J_0 converges linearly,
  with a contraction factor of about 7e-3 after the first iteration, in 5.2 iterations per step.
  J_0 still takes half the time per step, since an iteration with it costs a third of one with the
  full Jacobian (the K_m factorization is reused and only K_c is refactorized).
- Reusing J_0 with gamma_reassemble = 0.2 assembles it in 1 of 12 iterations. The rate degrades
  as the reused Jacobian ages (1.5e-2 to 6.6e-2), giving 7.6 iterations per step, but each costs a
  quarter as much, so a step takes 2.6 times less time than without reuse.
- `fsi2_simplified` takes exactly the same Newton iterations as `fsi2_biharmonic_mixed`, in every
  configuration, as its docstring says it should with the mixed biharmonic mesh motion and exact
  solves, at 15-20% less time per step.
- The timings are from five runs sharing the machine, and over one 0.5 s window, so they are
  rough.

## 2026-10-30: Analysis of the first comparison

`xfsi_solver.scripts.plot_newton_logs` plots the Newton logs of several runs: the residual per
iteration (median and interquartile range over the steps), the contraction against the age of a
reused Jacobian, the iterations and assemblies over time, and the cost per step and per iteration.
The analysis is in `report.md`, with the SVG figures in `plots/`.

Two findings changed how the reuse should be read:

- The age of a reused Jacobian is best counted in time steps, not in iterations: J_0 is kept for
  up to 12 steps (145 iterations). Plotted against iterations, the contraction showed a sawtooth
  from the position of the iteration within its step.
- The first iteration of a step contracts by 0.13 to 0.24 independently of the Jacobian, with the
  same values for a fresh J_0 in every iteration, and with the phase of the oscillation (the full
  Jacobian shows the same phases, 0.14 against 0.03). With γ = 0.2 inside that band, most
  assemblies (74 of 126) are triggered right after the first iteration, in the phases where it
  contracts by more than 0.2. The later iterations, which do measure the age of the Jacobian, only
  reach γ after 11 to 12 steps.

## 2026-10-30: Predicted displacements and skipping the first reuse test

Two options of `fsi2_simplified`, off by default:

- `skip_first_reuse_test`: `wrapped_jacobian` tests the rate of the last iteration only from the
  third Newton iteration of a step on (`first_tested = 2`), so the Jacobian is always reused at the
  second.
- `predict_displacements`: `update_displacements()`, split out of the KSP post-solve hook, is also
  called in the time loop right before `problem.solve()`, while v is still v_old. It has to be
  there rather than in a SNES hook: SNES evaluates the residual of the initial guess before any
  of its per-iteration hooks (`SNESSetUpdate` runs after it). The solid stress already uses
  u_s_old + dt (θ v + (1 - θ) v_old), so the prediction only changes u_f, and with it the fluid
  geometry and the mesh velocity of the initial guess.

The coarse-mesh tests cannot show the effect of the prediction: from rest, the structure hardly
moves in the first steps, and the residuals of the initial guesses agree to 7 digits. The test
checks that the states agree and that the initial residual is no larger.

Runs on mesh_sec, from the same state as the first comparison (results in `report.md`): the
prediction cuts the first-iteration contraction from 0.165 to 3e-3, gives 4 iterations in every
step without reuse and 10% fewer iterations with reuse at γ = 0.2 (the times of this batch
are corrected by the rerun below). Skipping the first
reuse test saves 2% of the iterations: the assemblies it triggered move to the fourth iterate of the same steps,
since a Jacobian from the step before is out of date within one step in that phase of the
oscillation. This corrects the reading in the first analysis that the first-iteration test causes
needless assemblies.

## 2026-10-30: The prediction as the default

`predict_displacements=True` is now the default of `fsi2_simplified`, so `main()` and
`jobs/fsi2_simplified.sh` use it. With it, the iterates are no longer those of
`fsi2_biharmonic_mixed`; the test that compares the two solvers and continues each from the
other's checkpoints passes `predict_displacements=False` to keep comparing the same iterates.

## 2026-10-30: The same options in threeDfsi_simplified

`threeDfsi_simplified` updates the displacements in the same way as `fsi2_simplified`, so it gets
the same options: `predict_displacements=True` by default, `skip_first_reuse_test=False`,
`gamma_reassemble=None` for reassembly in every iteration, and `newton_log_path` (blocks v and p),
which `main()` and `jobs/threeDfsi_simplified.sh` pass next to the QoI file.

The 3d solves are slow tests. `test_threeDfsi_simplified_prediction_and_reuse_options` runs two
steps on the coarse 3d mesh with the direct solver: the default prediction without reuse (assembled
in every iteration), and no prediction with the skipped first test (never assembled at the second
iteration), and checks that the QoIs agree. It has not been run yet. The effect of the prediction
in 3d has not been measured.

## 2026-10-30: `skip_first_reuse_test` removed

With the prediction as the default, `skip_first_reuse_test` changed nothing (the first iteration
contracts by 3e-3 to 2e-2, far below γ = 0.2, and the predicted runs with and without it took the
same iterations), and without the prediction it saved only 2% of the iterations. With the prediction, the
first-iteration rate grows with the age of the Jacobian, so it is a fair input to the reuse test.
The option is removed from both simplified solvers, with its 2d test; the 3d slow test, which ran
once with it (passed, 104 s), now compares the prediction on and off only. The results of the runs
with it stay in `report.md`.

## 2026-10-30: Rerun one at a time

All seven configurations again, with `2922423`, one after the other on 5 ranks
(`output/newton_convergence_rerun/`), to time them without sharing the machine. The iterates are
identical to those of the first batches. The times change: the prediction saves 27% of the time
without reuse and 11% with it, not the 31% and 19% from comparing runs of two batches, since an
iteration costs the same with and without it. Skipping the first reuse test saved 2% of the
iterations; its 6% in time was also a comparison between batches. The full-Jacobian run of the
rerun was 18% slower than in the first batch, and a 60-step check alone again matched the first
batch, so the rerun's full-Jacobian time was disturbed by other work on the machine. The docstrings
of the simplified solvers now quote 11% and 27%. Results in the last section of `report.md`.

## 2026-10-30: Report rewritten

`report.md` is rewritten around the current state: the prediction as the default, the rerun
timings, and the full-Jacobian time from the 60-step check. The figures are made again from the
rerun logs by `make_figures.py` next to it (same iterates as the first batches), and replace the
earlier figure set, which is in the history of the branch. The legends of the contraction and
iteration plots of `plot_newton_logs` now sit below the axes, clear of the lines. An HTML version of
the report was published as a claude.ai artifact.

## 2026-10-30: Interface values of u_f straight from v

The residual of the simplified solvers depends on the solid displacement only through u_s_old (the
solid stress is written in terms of v), so the per-iteration u_s served only to carry the interface
values to the mesh motion, through the parent mesh: an interpolation into the parent space, a
reverse scatter to the owners with an average over the ranks that hold a solid cell, a forward
scatter, an interpolation into U_f and another forward scatter. The interface values now come
directly from v: u_f = u_f_old + dt (θ v + (1 - θ) v_old) at the interface dofs of U_f, through the
dofs of V on the fluid cells. That is local to each rank: every U_f dof on a rank lies in one of its
fluid cells, whose V dofs it holds, and v is forward-scattered. u_s is set once per step, after the
solve, for u_s_old, the QoIs and the checkpoints.

A scratch copy of `fsi2_simplified` with both paths, on mesh_sec from the init_mixed state (defaults,
prediction and γ = 0.2):

- Same results: QoIs identical to 7 digits, the same iterations and assemblies in every step, and
  residuals equal to rounding, over 20 steps on 1 and 7 ranks.
- Little time saved: the old transfer took 0.022 s against 0.003 s over 20 steps on 7 ranks (of a
  2.9 s run), and 0.048 s against 0.006 s over 40 steps on 28 ranks (of 3.7 s), so about 1%. The
  change is kept for the simpler code.

The mesh motion solve is the larger cost: over 40 steps (278 solves), the first, which assembles
and factorizes, took 0.10 to 0.18 s, and the later ones 7.0 ms (1 rank), 3.9 ms (7) and 3.4 ms (28) in
the median, 9%, 18% and 26% of the run. The MUMPS solves barely speed up with the ranks, so their
share grows.

## 2026-10-30: The prediction as an initial-guess hook

The prediction now runs in an initial-guess hook of the SNES (`solver.setInitialGuess`, PETSc's
`SNESSetComputeInitialGuess`) instead of in the time loop. A check with a small problem showed that
SNES calls it once per solve, before it evaluates the residual of the initial guess, with the solution
vector already holding the state of the previous step. u_s is set in the same update as the interface
values of u_f again, after each linear solve and in the prediction: it is cheap, local and keeps u_s
and u_f updated together. Over 30 steps on mesh_sec, `fsi2_simplified` with the prediction and γ = 0.2,
the prediction without reuse, and no prediction took the same iterations and assemblies as the rerun
of the analysis, with the same QoIs to 7 digits.

## 2026-10-30: The saving of the prediction depends on the solid velocity

In the aneurism-fsi port of `threeDfsi_simplified` (10 steps of dt = 1e-4 at full load, 8 ranks,
Jacobian reuse), the prediction lowered the first contraction only from 0.04–0.06 to 0.03–0.04 and
saved 0 to 2 of 34 to 37 Newton iterations: the wall moves very little per step. In the FSI2 rerun,
the first-iteration rate without the prediction correlates with the tip speed (0.98); by thirds of
tip speed (0.31–0.54, 0.54–0.86, 0.86–1.06 m/s) it is 0.12, 0.17 and 0.25, and with reuse the
prediction saves 0.1, 0.8 and 1.3 iterations per step. Added as a note to the report: the findings
stay based on FSI2, with the prediction doing much less for small deformations.

## 2026-10-30: Report by topic, with paired figures

The report now describes the three solvers (`fsi2_simplified` as an implementation of
`fsi2_biharmonic_mixed` with Newton only on (v, p)), and is divided by what the figures show
instead of by with and without the prediction: residual per iteration, contraction against the age
of a reused J_0, and iterations over the period, each as a pair on shared axes with the same colors
and labels. The residual pair puts the run without the prediction first, so the full and
approximate Jacobian come right after the results table; the other pairs put the prediction first.
`plot_residual_history`, `plot_contraction_by_age` and `plot_iterations_over_time` take a title
and fixed axis limits for this.

## 2026-10-30: The prediction in the full-Jacobian solver

`fsi2_biharmonic_fulljacobian_initialguess` is `fsi2_biharmonic_fulljacobian` with the displacement
prediction as an initial-guess hook of the SNES (`predict_displacements=True`, the default): u =
u_old + dt v_old on the dofs of the solid cells, and u and z on the fluid cells from the mixed
biharmonic mesh motion on the fluid submesh (MUMPS on the x component, factorized once), with the
interface values taken from u_old and v_old, so that a rank holding an interface dof only through a
fluid cell gets them too. The mesh motion writes into the solver's z, and the hook assigns (u, v, p,
z) to the initial guess.

Unlike in `fsi2_simplified`, the residual of the initial guess is not smaller with the prediction on
the coarse mesh from rest (the solid stress is taken at the predicted displacement, which the old
velocity does not balance), but the residual after the first Newton step is 3 to 20 times smaller.

On the window of the report (200 steps of dt = 0.0025 from t = 16.0025, mesh_sec, 5 ranks, the two
runs one after the other on an idle machine, output/fulljacobian_initialguess/):

| | its/step | rate 0→1 | median ||R_1|| | s/step |
|---|---:|---:|---:|---:|
| full Jacobian | 3.00 | 7.6e-2 | 7.3 | 1.152 |
| full Jacobian, predicted | 2.34 (132 steps in 2, 67 in 3) | 4.8e-4 | 0.047 | 0.893 |

The initial residual is the same (median 97 against 98). An iteration costs the same (0.38 s), so
the prediction saves 22% of the time. The QoIs agree with the run without it to all 6 printed digits,
except the lift, to 4e-9 relative. The control run reproduces the earlier full-Jacobian run exactly
and takes 1.152 s per step, against 1.145 s in the earlier 60-step check.
