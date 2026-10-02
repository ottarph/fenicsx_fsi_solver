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

## Next

Roll out to `fsi2_harmonic_diffmesh`, `fsi2_biharmonic_diffmesh` (`p`, `z` on
the fluid submesh) and `fsi2_harmonic_lagrange` (fields on fluid/solid
submeshes, multipliers on the interface submesh) with the submesh transfer
above.
