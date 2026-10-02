# Checkpointing with io4dolfinx: implementation log

Goal: restart checkpoints for the monolithic FSI2 solvers, so a long
`mesh_sec` run that crashes or is stopped can continue, possibly on a
different number of MPI ranks. Started with `solvers/fsi2_harmonic.py`.

## Feasibility (2026-10-02)

- io4dolfinx 1.4.0 (conda-forge and PyPI) only lists dolfinx 0.9, 0.10 and
  nightly as supported, but works with this environment (dolfinx 0.11.0,
  Python 3.14, adios2 2.12, default adios2 backend).
- Functions on the full mesh round-trip exactly N-to-M (written on 2 ranks,
  read on 1 and 3; max error ~1e-15) with both
  - `write_mesh_input_order` + `write_function_on_input_mesh`, reading the
    mesh from the original XDMF on restart, and
  - `write_mesh` / `write_meshtags` / `write_function`, reading the mesh from
    the checkpoint.

  The first is used: the solver keeps loading its mesh and tags from the XDMF
  file, so a restart only adds a `read_function` per field.
- Functions on submeshes (`create_submesh`) cannot be read back:
  `RuntimeError: Mixed topology unsupported`, raised by
  `mesh.topology.original_cell_index`, which submeshes don't have. Workaround
  for the diffmesh/lagrange solvers, tested exactly N-to-M: copy codim-0
  submesh fields to a parent-mesh function with
  `Function.interpolate(f_sub, cells0=..., cells1=...)` through the entity
  map before writing (and back after reading); map CG1 fields on the
  codim-1 interface submesh to parent CG1 dofs through the submesh vertex map.

## State needed by `fsi2_harmonic`

`u`, `v`, `p` at the end of a step, plus `t` and the step counter. `u_old`
and `v_old` are copied from `u`/`v` at the start of every step, `p` has no
old value (it only seeds Newton), the inflow BC is recomputed from `t`, and
neither SNES nor MUMPS keep state between steps. `t` is restored exactly
(stored as float64), so the time stepping accumulates `t += dt` identically.

## Design: `tools/checkpoint.py`

- First version: one checkpoint file per run, every checkpoint appended as a
  new time stamp, `t`/`step` stored with `write_attributes` after the
  functions as a commit marker. Serial restart was bitwise identical to a
  continuous run, but **writing the first part on 2 ranks and restarting on
  1 gave garbage** (differences as large as the solution). Cause, in
  io4dolfinx's adios2 `write_function`: `{name}_dofmap`, `{name}_XDofmap` and
  `CellPermutations` are written only if not already in the file. The
  dofmap holds global dof indices in the writing run's dolfinx numbering,
  which depends on the partition, so values appended by a run with another
  partition are paired with the first run's dofmap. Appending to one file is
  only safe within a single run.
- Current version, `Checkpointer`: every checkpoint is a self-contained file
  (mesh + functions + attributes, written in `FileMode.write`), alternating
  between `checkpoint_0.bp` and `checkpoint_1.bp` in a checkpoint directory.
  The attributes (`t`, `step`, `dt`, `num_cells`) are written last; `read()`
  takes the file with the largest `step` among those with complete
  attributes, so a run killed while writing one file restarts from the
  other. `read_attributes` on a file without them returns an empty dict
  rather than raising (found by `test_skips_incomplete_checkpoint`), so
  completeness is checked by the attribute keys. The next write after a
  restart goes to the file not read from.
- `clear()` removes both files at the start of a fresh run; otherwise a
  restart after an early crash could pick up a later checkpoint from an
  earlier, longer run in the same directory.
- `read()` rejects a checkpoint written with another `dt` or on a mesh with
  another number of cells.

## Solver integration (`fsi2_harmonic.solve`)

- New arguments `checkpoint_dir`, `checkpoint_every` (in time steps, like
  `save_every`) and `restart`. A checkpoint is written after step `step`
  when `(step + 1) % checkpoint_every == 0`, after its QoI row.
- On restart: QoI rows with `t` later than the checkpoint are dropped from
  `qoi_path` (they were written between the checkpoint and the crash), and
  VTX output goes to `<name>_from_t<t>.bp`, since `VTXWriter` can't append
  and would otherwise overwrite the first part of the series.
- `main()` checkpoints to `output/checkpoints/fsi2_harm` every 100 steps
  (0.25 s); `--restart` continues from the latest checkpoint.

## Verification

- `tests/test_fsi2_harmonic.py`: 6-step run (checkpoint after step 4, two
  extra QoI rows) + restart to 8 steps against a continuous 8-step run, on
  `mesh_coarse`: QoI rows and final `u`, `v`, `p` agree (serial restart is
  bitwise identical); restart with another `dt` raises.
- `tests/test_checkpoint.py`: latest checkpoint, incomplete-file fallback,
  `clear()`, `dt` check, missing checkpoint, and a serial run restarted on 2
  ranks via `mpiexec` and read back in serial.
- Solver level, `mesh_coarse`, 12 steps, checkpoint every 4, restart from
  step 4 after a run stopped at step 6, compared with a continuous serial run
  (max abs difference at the end; max |u| 2.1e-6, |v| 1.1e-3, |p| 90):

  | first part | restart | u       | v       | p       |
  |------------|---------|---------|---------|---------|
  | 1 rank     | 1 rank  | 0       | 0       | 0       |
  | 2 ranks    | 1 rank  | 7.0e-19 | 1.4e-15 | 7.7e-12 |
  | 1 rank     | 2 ranks | 3.9e-19 | 1.1e-15 | 1.0e-11 |

- Cost on `mesh_sec` (5861 cells), 2 ranks: 12–25 ms per checkpoint write,
  13 ms per read, 2.3 MB per checkpoint file.

## Submesh fields (2026-10-02)

- `Checkpointer(directory, mesh, submeshes=[(submesh, entity_map), ...])`:
  a function on a listed cell submesh is copied to a function with the same
  element on the parent mesh (`interpolate` with `cells0`/`cells1` from the
  entity map, over owned and ghost cells), and that is what is written; on
  read it is copied back. Parent dofs outside the submesh are zero and
  unused. Submeshes are matched by identity (`f.function_space.mesh is
  submesh`), since the `EntityMap` topologies are new Python wrappers on each
  access and don't compare equal to `submesh.topology`.
- Facet submeshes (`fsi2_harmonic_lagrange`'s interface multipliers) raise
  `NotImplementedError` for now; the vertex-map transfer from the
  feasibility check would cover CG1 fields there.
- `restart_output_path` and `truncate_qoi_file` moved from `fsi2_harmonic`
  into `tools/checkpoint.py` to be shared by the solvers.
- `tests/test_checkpoint.py` now carries a CG2 field on a cell submesh
  (`x <= 0.5`) through every test, including the 2-rank restart.

## Diffmesh solvers (2026-10-02)

- `fsi2_harmonic_diffmesh` checkpoints `u`, `v`, `p` and
  `fsi2_biharmonic_diffmesh` `u`, `v`, `p`, `z`, with `p` and `z` on the
  fluid submesh passed to `Checkpointer` as `(fluid_mesh, fluid_cell_map)`.
- Their loop is shifted from `fsi2_harmonic`'s: the first solve is at
  `t = t0`, and `step`/`t` are incremented at the end of the iteration. The
  checkpoint is written after the increment, so it stores the loop-carried
  `(t, step)` of the next step to solve, and on restart the QoI file keeps
  rows up to `t - dt`. `save_every` alignment and the inflow BC follow from
  the restored values as before.
- "Time per step" in all three solvers now divides by the steps solved in
  this run rather than the total step counter, which is wrong after a
  restart.
- The restart-equivalence test is shared in `tests/restart_helpers.py`; it
  reads the checkpoints back as full-mesh functions, which is how submesh
  fields are stored.
- Findings from mutation checks (not restoring a field on restart):
  - Leaving out `v` fails the test (relative error ~3).
  - Leaving out `p` (harmonic diffmesh) or `z` (biharmonic) changes nothing
    measurable: final-state differences stay at ~1e-11 in `p` (|p| ~ 90) and
    the QoI files are identical. Both only seed Newton (no `_old` values),
    and the converged solution doesn't remember the initial guess beyond
    round-off. They are still checkpointed so the Newton iterations are the
    same as in a continuous run.
  - A correct serial restart of the diffmesh solvers is not bitwise
    identical (unlike `fsi2_harmonic`): the submesh -> parent -> submesh
    `interpolate` round trip perturbs values at round-off (~1e-14 relative
    in `p`).
- Solver level, `mesh_sec_coarse`, 12 steps, checkpoint every 4, restart
  from step 4 after a run stopped at step 6, against a continuous serial run
  (max abs difference at the end; max |u| 1.8e-6, |v| 9.0e-4, |p| 84,
  |z| 1.2e-2; QoI files identical in all cases):

  | solver       | first part | restart | u       | v       | p       | z       |
  |--------------|------------|---------|---------|---------|---------|---------|
  | harmonic dm  | 2 ranks    | 1 rank  | 4.3e-19 | 1.6e-15 | 7.8e-12 |         |
  | harmonic dm  | 1 rank     | 2 ranks | 3.8e-19 | 1.9e-15 | 1.2e-11 |         |
  | biharmonic dm| 2 ranks    | 1 rank  | 5.6e-19 | 1.3e-15 | 1.0e-11 | 3.6e-14 |
  | biharmonic dm| 1 rank     | 2 ranks | 4.9e-19 | 1.8e-15 | 1.7e-11 | 3.5e-14 |

## Lagrange solver (2026-10-02)

- `fsi2_harmonic_lagrange` checkpoints `u_f`, `v_f` (fluid submesh), `u_s`,
  `v_s` (solid submesh) and `p` (fluid submesh). Its loop starts at
  `step = 0`, so the checkpoint is written when `step % checkpoint_every ==
  0` after the increment; otherwise as in the diffmesh solvers. The solid
  VTX file gets the restart suffix too (`<name>_solid_from_t<t>.bp`), and
  the QoI file's parent directory is now created like in the other solvers.
- The interface multipliers `lambda_u`, `lambda_v` (facet submesh) are left
  out by decision, and restart from zero; documented at the checkpointer set-up
  in the solver and in its `solve` docstring.
- In exact arithmetic that is harmless: they enter the residual linearly
  with constant coefficients, so the Newton iterates after the first update
  don't depend on their initial value. Measured, however (serial,
  `mesh_sec_coarse`, 12 steps, relative to each field's max):
  - continuous vs zeroing the multipliers in place at step 4 (no restart):
    `u_f` 1.5e-6, `v_f` 1e-10, `u_s` 1.5e-11, `v_s` 4.9e-10, `p` 6.9e-12;
  - continuous vs restart: the same (`u_f` 1.5e-6, others <= 1.8e-10);
  - continuous vs continuous with Newton `atol` 1e-11 instead of 1e-7:
    `u_f` 1.1e-6, `v_s` 3.2e-8, others ~1e-9.

  So `u_f` is only resolved to ~1e-6 by the solver anyway (its equation is
  scaled by `alpha_u = 1e-9`, so its errors barely show in the residual),
  and the restart differences are at or below solver accuracy. The largest
  `u_f` differences are on the interface (1.5e-6 relative, 6.8e-13
  absolute; interior 2.2e-7), far below the solver's weak continuity
  mismatch `u_f - u_s` there (`interface_u_gap` ~1e-8), while `u_s` agrees
  to ~1e-11. `u_f` is not just a change of coordinates -- the mesh velocity
  `(u_f - u_f_old) / dt` enters the fluid momentum equation -- so the
  acceptance rests on the measured effect on the physical fields, which
  includes that term: `v_f` ~1e-10 and `p` ~7e-12 relative, QoIs unchanged.
  If tighter reproducibility is wanted, the multipliers can be checkpointed
  with the CG1 vertex-map transfer from the feasibility check. Drag, lift
  and A_x/A_y match to the printed 6 digits in serial, 2 -> 1 and 1 -> 2
  rank restarts; only the `interface_u_gap` diagnostic (~1e-9) differs, by
  up to 1.8e-12. Newton iteration counts after a restart are unchanged (2).
- That `u_f` difference passed the shared test's elementwise
  `atol = 1e-12` with only ~1.5x margin, and that `atol` was loose for small
  fields (5e-7 relative for `u` ~ 2e-6). `restart_helpers` now compares
  each field norm-wise against `rtol * max|field|` (default 1e-10, which the
  other solvers pass with margin); the lagrange test uses 1e-8 and 1e-5 for
  `u_f`. Leaving `v_s` out of the restart still fails it (abs diff 8.8e-3).
- The lagrange solver had no test; `tests/test_fsi2_harmonic_lagrange.py`
  adds a plain solve test besides the restart tests.
