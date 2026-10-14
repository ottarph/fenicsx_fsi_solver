# Restricted branch cleanup: implementation log

Goal: put the descendants of `restricted-test-functions` (`restricted-iterative-solver`, `3d` and
`restricted-c0ip-mm`) on one branch `restricted`, and clean it up. The cleanup consolidates the solver scripts,
removes options that the earlier studies showed are not needed, and harmonizes output and naming.

## Plan (agreed 2026-10-14)

Solver scripts at the end. Every solver is based on the submesh ("diffmesh") version, but `diffmesh` is
dropped from the names:

| Script | From |
|---|---|
| `dfg_2d_3.py` | unchanged |
| `fsi2_harmonic.py` | `fsi2_harmonic_diffmesh.py` (the single-mesh `fsi2_harmonic.py` is deleted) |
| `fsi2_harmonic_lagrange.py` | unchanged |
| `fsi2_biharmonic_simple.py` | `fsi2_biharmonic_diffmesh.py` |
| `fsi2_biharmonic_fulljacobian.py` | `fsi2_biharmonic_diffmesh_restricted.py`, with `_scaled` folded in as `mesh_motion_scaling` |
| `fsi2_biharmonic_mixed.py` | `fsi2_biharmonic_diffmesh_restricted_split_condensed.py` |
| `fsi2_biharmonic_c0ip.py` | rebuilt on `fsi2_biharmonic_mixed.py`, so the two differ only in the mesh motion |
| `threeDfsi_mixed.py` | `threeDfsi_iterative.py` |
| `threeDfsi_c0ip.py` | new, on `threeDfsi_mixed.py`, if the machine is free after the cleanup |

Deleted: `_restricted_split.py` (superseded by the condensed solver) and `_restricted_scaled.py` (folded in).

Other decisions:

- Remove Cahouet-Chabard completely, along with the full-Schur K_c path and the auxiliary A_0 preconditioner.
  Use SOR damping omega = 0.5 in 2d as in 3d.
- Keep `ReuseKSP` (the companion of `ReusePC`), `nu_s`, `bcs_post`, the no-op hooks, and all commented-out
  lines. The commented-out lines are how options are switched for runs.
- `log_view` becomes a commented-out line.
- Every solver gets `vtx_save_every` (None: no VTX writers) and one `write_output(t, step)` function.
- Every solver that plots matrix blocks gets `plot_matrices` and the shared `tools/matrix_blocks.py`.
- Figures go to one folder per solver script: `output/figures/matrix_blocks/<script>/` and
  `output/figures/qoi/<script>/`.
- QoI files are named after the solver script, without `_qoi`.
- The 2d and 3d solvers stay separate scripts.
- Reference QoIs are kept outside git.

## Phase 0: reference QoIs (2026-10-14)

- `output/cleanup-reference/` (gitignored) holds a driver, a batch script and a comparison script, plus the QoIs
  and logs of the code before the cleanup. Most setups come from `3d` e52146a; C0IP comes from
  `restricted-c0ip-mm` 51d62d8. Its `README.md` says how to rerun the setups.
- Each setup runs 20 steps. 2d runs use 4 ranks on `mesh_sec`; 3d runs use 14 ranks on `mesh_sec_coarse`.
  The restricted solvers also restart from copies of developed-flow checkpoints: FSI2 at t = 15.5 or 16, 3d at
  t = 10. Starting from rest, the condensed solvers assemble the Jacobian once in 20 steps, so neither the
  reassembly nor the iterative K_c solve gets exercised.
- Two identical passes give bitwise identical QoIs for every setup except `fsi2_harmonic_lagrange`, which varies
  by about 2e-6 relative between runs. The renames and the output refactor can therefore be checked for exact
  equality, with lagrange checked against a tolerance.
- Developed 2d flow, iterative K_c:
  - Cahouet-Chabard costs 0.53 s/step, against 0.45 s/step for selfp.
  - Summed over the 20 steps, its Krylov solves need about 3,070 iterations, against 2,040 for selfp. Both use
    the same 138 Newton iterations; the sum covers all fieldsplit solves.
  - This is consistent with the 3d results in `notes/fsi3d/implementation-log.md`.

## Phase 1: merge (2026-10-14)

- `restricted` starts at `3d`, which contains `restricted-iterative-solver`, as a fast-forward. Then
  `restricted-c0ip-mm` is merged in.
- The only conflict is `scripts/plot_qois.py`. Resolved by keeping `3d`'s version (`_load_qois`, which handles
  missing and partial files, and `plot_3d_qois`) and adding `plot_c0ip_comparison` in the same style.
- `tools/petsc_reuse.py` and its test are byte-identical on both branches, so they merged cleanly.
- Test suite on the merge: 111 passed.
