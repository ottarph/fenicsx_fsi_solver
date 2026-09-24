# Shared-space iterative solver: usage, results and limitations

Implementation of `docs/shared-space-iterative-solver-plan.md` for
`src/xfsi_solver/solvers/fsi2_harmonic_diffmesh.py`. The detailed record of
decisions and measurements is `notes/shared-space-iterative-solver/implementation-log.md`.

## Usage

```python
from xfsi_solver.solvers.fsi2_harmonic_diffmesh import SolverConfig, solve
from xfsi_solver.solvers.fsi2_harmonic_diffmesh_fieldsplit import FieldSplitConfig

# reference (default): exact Newton, MUMPS LU
solve(mesh_path, T, dt, output_path, output_path_p, qoi_path)

# no-ALE Jacobian with the production FGMRES/fieldsplit solver
solve(mesh_path, T, dt, output_path, output_path_p, qoi_path,
      config=SolverConfig(jacobian_mode="no_ale", linear_solver="fieldsplit"))

# diagnostics: full Newton, no-ALE preconditioning matrix; exact subsolves
SolverConfig(jacobian_mode="full", preconditioner_mode="no_ale", linear_solver="fieldsplit",
             fieldsplit=FieldSplitConfig(displacement="lu", momentum="lu"))
```

`solve()` returns a `SolveResult` with the final `FSIProblem` (the solution
functions) and a `StepInfo` per step: Newton and linear iterations, residual
history, per-field nonlinear residuals (`u_solid`, `u_fluid`, `v`, `p`),
every linear solve (iterations, true relative residual, per-field true
residuals), timings (Jacobian, auxiliary assembly, preconditioner setup,
Krylov solve), preconditioner work counters, drag, lift and tip displacement.
`initial_state`, `checkpoint_dir` and `checkpoint_every` restart from and save
states.

## Solver hierarchy (production defaults)

```text
FGMRES on J_noALE (rtol 1e-6)
+-- PCFIELDSPLIT Schur, full factorization, all blocks from Pmat: u | (v,p)
    +-- u: Python PC, block lower triangular: solid/interface DOFs with the
    |      constant solid mass (Cholesky once), then the fluid-interior
    |      harmonic extension (constant, Cholesky once; AMG optional) with A_fI
    +-- (v,p): preonly, Python PC on the user Schur matrix
           P_vp = [[H + theta dt E_s, G], [B, 0]] (built algebraically)
        +-- inner PCFIELDSPLIT Schur, full factorization: v | p
            +-- v: one BoomerAMG V-cycle on H + theta dt E_s
            +-- p: selfp (B diag(H_hat)^{-1} G), one BoomerAMG V-cycle
```

No monolithic LU and no dense Schur complement is formed in this mode (tested
with `ksp_view`). The constant displacement blocks are factored once per run;
`displacement_fluid="amg"` replaces that factorization by AMG-preconditioned
CG. All levels are selected by `FieldSplitConfig` and each can be replaced by
LU for diagnosis; `variant="exact"` is the exact Schur factorization (dense
Schur complement, coarse meshes only).

## Benchmarks

Commands (states from `xfsi_solver.scripts.fsi2_harmonic_diffmesh_checkpoints`,
which runs `full/direct` and saves restart states):

```bash
conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.scripts.fsi2_harmonic_diffmesh_checkpoints --mesh data/meshes/fsi2/mesh_sec_coarse.xdmf --dt 0.0025 --T 8 --every 0.5 --out output/checkpoints/mesh_sec_coarse_dt0.0025
```

```bash
conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.scripts.fsi2_harmonic_diffmesh_checkpoints --mesh data/meshes/fsi2/mesh_sec_coarse.xdmf --dt 0.0025 --T 14 --every 1.0 --out output/checkpoints/mesh_sec_coarse_dt0.0025 --restart output/checkpoints/mesh_sec_coarse_dt0.0025/state_t8.0000.npz
```

```bash
conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.scripts.fsi2_harmonic_diffmesh_checkpoints --mesh data/meshes/fsi2/mesh_sec.xdmf --dt 0.0025 --T 6 --every 1.0 --out output/checkpoints/mesh_sec_dt0.0025
```

```bash
conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.scripts.fsi2_harmonic_diffmesh_benchmark --mesh data/meshes/fsi2/mesh_sec_coarse.xdmf --dt 0.0025 --steps 20 --state output/checkpoints/mesh_sec_coarse_dt0.0025/state_t9.0000.npz --out output/benchmarks/coarse_t9_dt0.0025
```

The other cases use the same command with `--mesh data/meshes/fsi2/mesh_sec.xdmf`,
`--dt 0.00125`, `--steps 40` (startup), `--state` omitted (startup, t = 0) or
another state, and `--np 4`. The `dt 0.00125` runs from a state continue a
`dt 0.0025` trajectory with the smaller step.

Cases: `coarse` = `mesh_sec_coarse` (2 088 displacement DOFs, 4.4k unknowns),
`default` = `mesh_sec` (23 898 displacement DOFs, 50k unknowns); `t0` =
startup, `t6`/`t9` (coarse) and `t5` (default) = developed motion from a
checkpoint (tip amplitude over the preceding 0.5 time units about 0.001, 0.01
and 0.001; the FSI2 reference amplitude is about 0.08); 20 steps (40 for coarse
startup, 10 from `t5`); serial unless `np4`. `snes_atol 1e-9`, `ksp_rtol 1e-6`,
AMD Ryzen Threadripper 3970X, one process per mode. Times are seconds per time step
(including the first step and its setup); Jacobian, aux, setup and solve are
per step. Errors are against `full/direct` (QoIs: maximum over the steps,
relative to the maximum reference value; fields: final step).

| case | mode | Newton/step | FGMRES/solve (max) | s/step | Jacobian | aux | setup | solve | peak MB | max rel err drag / lift / A_y | rel err u / v / p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coarse_t0_dt0.0025 | full/direct | 1.95 | - | 0.103 | 0.051 | 0.000 | 0.022 | 0.002 | 198 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| coarse_t0_dt0.0025 | no_ale/direct | 2.38 | - | 0.062 | 0.024 | 0.000 | 0.019 | 0.002 | 195 | 2.4e-11 / 1.2e-08 / 1.9e-08 | 7.6e-10 / 3.5e-11 / 7.4e-11 |
| coarse_t0_dt0.0025 | full/fieldsplit(no_ale P) | 1.95 | 9.2 (12) | 0.141 | 0.071 | 0.005 | 0.002 | 0.035 | 204 | 2.5e-11 / 3.2e-09 / 3.8e-11 | 3.4e-12 / 8.1e-13 / 2.7e-12 |
| coarse_t0_dt0.0025 | no_ale/fieldsplit | 2.42 | 9.1 (12) | 0.092 | 0.025 | 0.006 | 0.002 | 0.042 | 202 | 1.1e-10 / 5.0e-08 / 1.3e-08 | 5.1e-10 / 3.4e-11 / 5.6e-11 |
| coarse_t0_dt0.00125 | full/direct | 1.93 | - | 0.102 | 0.051 | 0.000 | 0.022 | 0.002 | 198 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| coarse_t0_dt0.00125 | no_ale/direct | 1.93 | - | 0.051 | 0.020 | 0.000 | 0.016 | 0.001 | 194 | 3.6e-12 / 1.2e-09 / 1.2e-08 | 1.5e-10 / 3.0e-11 / 1.9e-12 |
| coarse_t0_dt0.00125 | full/fieldsplit(no_ale P) | 1.93 | 9.1 (11) | 0.139 | 0.071 | 0.005 | 0.002 | 0.035 | 204 | 2.1e-09 / 1.1e-06 / 8.5e-10 | 1.7e-10 / 6.8e-11 / 3.0e-10 |
| coarse_t0_dt0.00125 | no_ale/fieldsplit | 1.93 | 8.8 (11) | 0.073 | 0.020 | 0.005 | 0.002 | 0.033 | 200 | 2.1e-09 / 1.1e-06 / 1.1e-08 | 2.9e-10 / 1.1e-10 / 3.2e-10 |
| coarse_t6_dt0.0025 | full/direct | 2.90 | - | 0.150 | 0.077 | 0.000 | 0.033 | 0.002 | 198 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| coarse_t6_dt0.0025 | no_ale/direct | 4.00 | - | 0.101 | 0.041 | 0.000 | 0.033 | 0.003 | 195 | 9.3e-13 / 5.8e-10 / 1.4e-11 | 1.5e-11 / 2.3e-12 / 8.8e-12 |
| coarse_t6_dt0.0025 | full/fieldsplit(no_ale P) | 3.00 | 9.6 (11) | 0.214 | 0.110 | 0.008 | 0.003 | 0.056 | 204 | 7.2e-13 / 4.1e-12 / 1.8e-15 | 4.1e-14 / 3.3e-14 / 5.5e-12 |
| coarse_t6_dt0.0025 | no_ale/fieldsplit | 4.00 | 8.2 (9) | 0.144 | 0.041 | 0.010 | 0.003 | 0.064 | 201 | 9.2e-13 / 5.8e-10 / 1.4e-11 | 1.6e-11 / 2.3e-12 / 8.9e-12 |
| coarse_t6_dt0.00125 | full/direct | 2.00 | - | 0.106 | 0.053 | 0.000 | 0.023 | 0.002 | 198 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| coarse_t6_dt0.00125 | no_ale/direct | 4.00 | - | 0.101 | 0.041 | 0.000 | 0.033 | 0.003 | 194 | 2.3e-13 / 9.7e-11 / 6.3e-13 | 6.4e-13 / 2.2e-13 / 1.3e-12 |
| coarse_t6_dt0.00125 | full/fieldsplit(no_ale P) | 2.85 | 8.5 (10) | 0.199 | 0.104 | 0.007 | 0.003 | 0.048 | 204 | 1.9e-11 / 1.5e-09 / 1.4e-13 | 5.5e-13 / 1.9e-12 / 1.2e-10 |
| coarse_t6_dt0.00125 | no_ale/fieldsplit | 4.00 | 7.0 (8) | 0.136 | 0.041 | 0.010 | 0.003 | 0.057 | 200 | 2.2e-13 / 9.2e-11 / 6.2e-13 | 6.3e-13 / 2.2e-13 / 1.3e-12 |
| coarse_t9_dt0.0025 | full/direct | 3.00 | - | 0.153 | 0.079 | 0.000 | 0.034 | 0.002 | 199 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| coarse_t9_dt0.0025 | no_ale/direct | 4.75 | - | 0.118 | 0.049 | 0.000 | 0.038 | 0.003 | 195 | 2.2e-12 / 1.0e-10 / 8.9e-14 | 2.3e-13 / 1.9e-12 / 3.5e-11 |
| coarse_t9_dt0.0025 | full/fieldsplit(no_ale P) | 3.00 | 9.0 (11) | 0.211 | 0.110 | 0.008 | 0.003 | 0.053 | 203 | 7.2e-13 / 9.1e-13 / 5.9e-16 | 1.4e-15 / 7.1e-15 / 1.0e-12 |
| coarse_t9_dt0.0025 | no_ale/fieldsplit | 4.80 | 8.4 (9) | 0.174 | 0.050 | 0.012 | 0.004 | 0.079 | 200 | 2.2e-12 / 9.1e-11 / 4.0e-14 | 1.1e-13 / 1.4e-12 / 2.3e-11 |
| coarse_t9_dt0.00125 | full/direct | 3.00 | - | 0.154 | 0.079 | 0.000 | 0.034 | 0.002 | 198 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| coarse_t9_dt0.00125 | no_ale/direct | 4.00 | - | 0.101 | 0.041 | 0.000 | 0.033 | 0.003 | 194 | 5.0e-12 / 6.4e-11 / 5.1e-13 | 6.0e-13 / 3.1e-12 / 4.5e-11 |
| coarse_t9_dt0.00125 | full/fieldsplit(no_ale P) | 3.00 | 8.3 (10) | 0.209 | 0.110 | 0.008 | 0.003 | 0.050 | 204 | 9.6e-14 / 2.0e-12 / 3.0e-16 | 3.1e-16 / 5.6e-15 / 5.7e-13 |
| coarse_t9_dt0.00125 | no_ale/fieldsplit | 4.00 | 7.3 (9) | 0.138 | 0.041 | 0.010 | 0.003 | 0.059 | 200 | 5.0e-12 / 5.8e-11 / 5.2e-13 | 5.9e-13 / 3.1e-12 / 4.4e-11 |
| default_t0_dt0.0025 | full/direct | 1.90 | - | 1.344 | 0.594 | 0.000 | 0.550 | 0.037 | 392 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| default_t0_dt0.0025 | no_ale/direct | 1.90 | - | 0.748 | 0.220 | 0.000 | 0.384 | 0.029 | 333 | 5.1e-12 / 5.6e-09 / 2.5e-08 | 2.4e-10 / 3.2e-11 / 4.0e-12 |
| default_t0_dt0.0025 | full/fieldsplit(no_ale P) | 1.90 | 11.7 (15) | 1.567 | 0.812 | 0.049 | 0.020 | 0.524 | 370 | 6.7e-12 / 2.4e-09 / 5.2e-10 | 2.1e-12 / 1.3e-12 / 2.3e-12 |
| default_t0_dt0.0025 | no_ale/fieldsplit | 1.90 | 11.4 (15) | 0.908 | 0.221 | 0.049 | 0.020 | 0.504 | 347 | 3.6e-11 / 6.1e-08 / 4.6e-08 | 2.2e-10 / 7.5e-11 / 9.8e-12 |
| default_t0_dt0.00125 | full/direct | 1.85 | - | 1.310 | 0.580 | 0.000 | 0.534 | 0.036 | 391 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| default_t0_dt0.00125 | no_ale/direct | 1.85 | - | 0.729 | 0.215 | 0.000 | 0.374 | 0.029 | 332 | 9.2e-13 / 1.1e-09 / 1.4e-09 | 3.7e-11 / 1.1e-11 / 1.0e-12 |
| default_t0_dt0.00125 | full/fieldsplit(no_ale P) | 1.85 | 10.7 (13) | 1.486 | 0.797 | 0.048 | 0.019 | 0.460 | 371 | 3.0e-10 / 4.4e-07 / 3.7e-08 | 9.0e-11 / 5.7e-11 / 3.1e-11 |
| default_t0_dt0.00125 | no_ale/fieldsplit | 1.85 | 10.5 (13) | 0.839 | 0.214 | 0.048 | 0.019 | 0.445 | 346 | 3.0e-10 / 4.4e-07 / 3.0e-08 | 1.1e-10 / 6.0e-11 / 3.0e-11 |
| default_t5_dt0.0025 | full/direct | 2.00 | - | 1.417 | 0.626 | 0.000 | 0.583 | 0.039 | 392 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| default_t5_dt0.0025 | no_ale/direct | 4.00 | - | 1.526 | 0.463 | 0.000 | 0.802 | 0.062 | 332 | 2.3e-13 / 2.9e-12 / 9.8e-15 | 3.4e-14 / 1.4e-14 / 3.3e-12 |
| default_t5_dt0.0025 | full/fieldsplit(no_ale P) | 2.90 | 11.1 (15) | 2.347 | 1.247 | 0.075 | 0.030 | 0.774 | 372 | 3.3e-11 / 6.5e-09 / 3.2e-14 | 6.6e-13 / 2.9e-12 / 8.4e-10 |
| default_t5_dt0.0025 | no_ale/fieldsplit | 4.00 | 9.9 (11) | 1.762 | 0.464 | 0.103 | 0.042 | 0.954 | 347 | 2.3e-13 / 3.1e-12 / 1.0e-14 | 3.4e-14 / 1.3e-14 / 3.3e-12 |
| default_t5_dt0.00125 | full/direct | 2.00 | - | 1.418 | 0.626 | 0.000 | 0.582 | 0.039 | 400 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| default_t5_dt0.00125 | no_ale/direct | 4.00 | - | 1.524 | 0.462 | 0.000 | 0.802 | 0.062 | 332 | 1.5e-14 / 4.4e-12 / 2.3e-15 | 3.0e-15 / 2.4e-15 / 7.3e-14 |
| default_t5_dt0.00125 | full/fieldsplit(no_ale P) | 2.00 | 8.5 (9) | 1.508 | 0.852 | 0.051 | 0.021 | 0.416 | 370 | 2.6e-10 / 7.3e-09 / 1.4e-13 | 4.5e-12 / 2.1e-12 / 1.1e-09 |
| default_t5_dt0.00125 | no_ale/fieldsplit | 4.00 | 8.7 (10) | 1.631 | 0.463 | 0.103 | 0.042 | 0.824 | 346 | 1.2e-14 / 2.2e-12 / 2.1e-15 | 2.9e-15 / 2.4e-15 / 1.3e-13 |
| default_t0_dt0.0025_np4 | full/direct | 1.90 | - | 0.543 | 0.165 | 0.000 | 0.299 | 0.021 | 273 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| default_t0_dt0.0025_np4 | no_ale/direct | 1.90 | - | 0.321 | 0.064 | 0.000 | 0.205 | 0.016 | 259 | 5.1e-12 / 5.6e-09 / 2.5e-08 | 2.4e-10 / 3.2e-11 / 4.0e-12 |
| default_t0_dt0.0025_np4 | full/fieldsplit(no_ale P) | 1.90 | 11.8 (15) | 0.597 | 0.230 | 0.023 | 0.006 | 0.280 | 248 | 1.9e-11 / 2.3e-09 / 2.7e-10 | 2.7e-12 / 1.1e-12 / 8.9e-13 |
| default_t0_dt0.0025_np4 | no_ale/fieldsplit | 1.90 | 11.4 (15) | 0.400 | 0.064 | 0.023 | 0.006 | 0.270 | 241 | 4.2e-11 / 3.8e-08 / 3.6e-08 | 2.4e-10 / 6.5e-11 / 1.1e-11 |


## Conclusions

- All modes solve the same nonlinear equations: every mode agrees with
  `full/direct` in drag, lift, tip displacement and fields to at most 2e-8
  relative, typically 1e-10 or better.
- The no-ALE Jacobian is 2.5-3x cheaper to assemble, but Newton loses
  quadratic convergence: 2-2.4 iterations per step at startup, 4-5 at
  developed motion (full Newton: 2-3). On the default mesh at developed motion
  this cancels the assembly savings: `no_ale/direct` 1.52 s/step vs
  `full/direct` 1.42 s/step.
- The iterative solver's FGMRES iteration count is 7-12 per linear solve,
  nearly independent of the mesh (coarse vs default), the time step and the
  amount of motion; with the full Jacobian and the no-ALE Pmat it retains
  full Newton's iteration counts. The auxiliary operator is almost exact:
  with exact subsolves FGMRES needs 1-3 iterations even at t = 9, so the
  remaining iterations come from the single AMG V-cycles for velocity and
  pressure (more V-cycles reduce iterations but not time).
- For these 2D problems (up to 50k unknowns) MUMPS is faster. On the default
  mesh `no_ale/fieldsplit` is 7-21% slower than `no_ale/direct` in serial and
  25% slower on 4 ranks, and 15-24% slower than `full/direct` at developed
  motion; on the coarse mesh it is 40-50% slower than `no_ale/direct`. It
  uses about 11% less memory than `full/direct` on the default mesh. The
  Krylov solve dominates its cost: about 23 ms per FGMRES iteration on the
  default mesh, of which about 20 ms are the displacement (twice) and
  momentum-pressure preconditioner applications. It beats
  `full/direct` at startup on both meshes (0.91 vs 1.34 s/step on the
  default mesh) and in two of the four coarse developed cases, but not at
  developed motion on the default mesh. No speedup is claimed.

## Limitations and open issues

- Robustness beyond the reference solver is untested: `full/direct` itself
  fails on the coarse mesh at t = 9.57 (SNES divergence at a tip displacement
  of about -0.035; the default mesh is noted to fail near t = 7), so no mode
  has been run through fully developed FSI2 oscillation.
- Scalability has been checked on 4 ranks of a 2D problem only. The
  factorization of the constant fluid-interior displacement block is cheap in
  2D but not in 3D, where `displacement_fluid="amg"` (7 CG iterations,
  about 10x slower per solve here) should be used.
- The pressure approximation (`selfp`, or Cahouet-Chabard with Neumann
  conditions on the moving interface) has not been tested for other density
  ratios, Reynolds numbers or time steps than FSI2's; parameter robustness is
  not claimed. A PCD variant was not needed at these parameters.
- Preconditioner lagging is not implemented: setup is 2-3% of the step time,
  while the no-ALE Newton iteration count, not setup, limits the method.
- `P_vp` and the Jacobian's `(v,p)` split must number the DOFs identically
  (checked; the solver raises otherwise). States are saved per rank and only
  restart on the same number of ranks.
- The Krylov-level overhead of the Python PC contexts could be reduced by
  moving them to PETSc-native components (e.g. a `PCFIELDSPLIT` multiplicative
  split for the displacement block) if the iterative solver is pursued for
  larger problems.
