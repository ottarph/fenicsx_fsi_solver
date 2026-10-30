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
