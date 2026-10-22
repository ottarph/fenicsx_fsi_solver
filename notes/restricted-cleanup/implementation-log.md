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

## Phase 2: renames and deletions (2026-10-14)

- Each rename is its own commit. The script and its test move together, and the commit also updates the
  defaults that carry the name: the output, checkpoint, QoI and log paths, and the PETSc prefix where it
  was the script name. The three condensed solvers keep the prefix `solver_`. These edits are small, so git
  still detects the renames.
- QoI, VTX and checkpoint names follow the pattern `<script>[_<mesh>]`: `<script>.txt`, `<script>.bp`,
  `<script>_p.bp`, `output/checkpoints/<script>`. The mesh suffix appears only where `main()` runs a
  non-default mesh: `fsi2_biharmonic_mixed_fine`, `threeDfsi_mixed_medium`.
- `fsi2_biharmonic_fulljacobian` gets the scaled factorization as `mesh_motion_scaling` (None or 1e-9).
  The unscaled path is unchanged and still uses MUMPS `cntl_1 = 1e-4`. The scaled path is the `ScaledLU` of
  the old `_scaled` solver, which did not set `cntl_1`. Both paths give QoIs bitwise identical to the two old
  solvers, from rest and from developed flow. Both are tested, through a new `solve_kwargs` argument of
  `check_restart_reproduces_continuous_run`.
- `create_mesh_DFG2d_alt.py` replaces `create_mesh_DFG2d.py`. It still writes to `data/meshes/dfg2d_alt/`,
  so existing meshes and paths keep working. The commented-out mesh option in `dfg_2d_3.py` that points to
  the old script's mesh stays, since that mesh is still on disk.
- The existing files in `output/` were renamed to the new names with `mv -n`: QoIs, checkpoints and VTX.
  Output of deleted solvers, logs and figures stay as they were.
- Check: all setups were rerun with the renamed code (tag `phase2`). They are bitwise identical to the
  reference, except `fsi2_harmonic_lagrange`, which differs by 1.6e-6, the same as between two identical
  runs.

## Phase 3: VTX output option (2026-10-14)

- Every FSI2 solver, `dfg_2d_3` and the six time-dependent component solvers take `vtx_save_every`, following
  the pattern the mixed solvers already had:
  - With None (the default in `solve()`), no VTX writers are opened. Otherwise a snapshot is written every
    `vtx_save_every` steps.
  - A local `write_output(t, step)` writes the QoIs and the snapshot.
  - Each `main()` passes the interval that used to be hard-coded: 4, 8 for lagrange, 10 for DFG, 1 for the
    component solvers.
  - The five static component solvers write their single solution as before.
- The restart check and the solve tests ask for VTX output, and a test checks that none is written by
  default. QoIs are bitwise identical (tag `phase3`).

## Phase 4: options removed from the mixed solvers (2026-10-14)

- Removed from `fsi2_biharmonic_mixed` and `threeDfsi_mixed`:
  - Cahouet-Chabard, with `tools/cahouet_chabard.py` and its test;
  - the full-Schur K_c solve;
  - the auxiliary A_0 preconditioning matrix.

  The iterative K_c solve is now the block-preconditioned FGMRES with selfp, the only setting of the
  production runs. Selfp, direct and 3d runs give bitwise identical QoIs and iteration counts (tag
  `phase4`).
- SOR damping in 2d: not adopted. ω = 0.5 against undamped SOR on developed flow (`mixed_iterative_dev`,
  t = 16, 20 steps, 138 Newton iterations in both cases):

  | ranks | Krylov its, ω = 1 | Krylov its, ω = 0.5 | s/step, ω = 1 | s/step, ω = 0.5 |
  |---|---|---|---|---|
  | 4 | 2,043 | 2,370 | 0.40 | 0.43 |
  | 16 | 2,437 | 2,366 | 0.26 | 0.26 |
  | 28 | 2,137 | 2,280 | 0.235 | 0.247 |

  The Krylov counts are the sums over all fieldsplit solves. Undamped SOR converged on every rank count. In
  2d, damping costs up to 15% more iterations and 9% more time and does not help robustness, unlike in 3d,
  where ω = 1 diverged on 26 to 30 ranks. The 2d solver therefore keeps undamped SOR.

## Phase 5: C0IP on the mixed solver (2026-10-14)

- `fsi2_biharmonic_c0ip` is now `fsi2_biharmonic_mixed` with only the mesh motion replaced:
  - the C0IP form and its facet measures, with penalty `c0ip_a_val`, now an option;
  - no z;
  - MUMPS Cholesky on K_m.

  It was a copy of the restricted split solver: direct solve of (u_S, v, p), solid stress in terms of u.
  It gets the condensation, the iterative K_c option, `vtx_save_every` and `plot_matrices` from the mixed
  solver. The CG option for K_m and the separate first-solve timing were for comparing CG with Cholesky,
  and are removed.
- Check against the old solver, 20 steps on 4 ranks, `mesh_sec`:
  - From rest: QoIs within 6e-8 relative, with the direct K_c solve.
  - From the developed flow at t = 16: identical in all printed digits, with the same 138 Newton iterations
    and 21 Jacobian assemblies.
  - The iterative K_c solve agrees to 7e-7 from rest and to all printed digits from t = 16.
- `component_solvers/c0ip_mesh_motion.py` keeps `direct_component`, `cg_gamg` and `cg_laplacian_squared`
  (BoomerAMG or LU). Brenner-Sung, the full LU and the GAMG V-cycles in `LaplacianSquaredPC` are removed.
- Tests for `fsi2_biharmonic_mixed` and `fsi2_biharmonic_c0ip`:
  - direct and iterative K_c;
  - restart checks, with a tolerance of 1e-6. A restart assembles the approximate Jacobian where the
    continuous run reuses it, so the states differ by about 1e-8 relative.

## Phase 6: plots, tests and docs (2026-10-14)

- `tools/matrix_blocks.py` holds the spy and block-norm plots, gathered on rank 0. `fsi2_biharmonic_mixed`
  and `fsi2_biharmonic_c0ip` use it with `plot_matrices=True`, and save to
  `output/figures/matrix_blocks/<script>/`. The 3d solver does not plot matrices, since gathering the
  Jacobian on one rank is too large there.
- `plot_qois.plot_comparison(runs)` replaces the three comparison functions:
  - A run is named by its QoI file. The folder is `output/figures/qoi/<script>/`, where the script is the
    longest solver module name the run starts with. The rest of the name, such as `fine_` or
    `scaled_T10_`, prefixes the file names, then `compare_<other runs>_`.
  - `dfg_2d_3` saves to `output/figures/qoi/dfg_2d_3/`, and the reference plot to
    `output/figures/qoi/reference/`.
- `create_test_meshes` also makes the coarse 3d mesh. It restores gmsh's default options first, since
  `gmsh.clear()` keeps the quad recombination of the DFG quad mesh, which fails on the periodic surfaces
  of the 3d mesh.
- The README figure moved from `output/figures/` to `docs/figures/`.

## Old and new names

| Old | New |
|---|---|
| `solvers/fsi2_harmonic.py` (single mesh) | removed |
| `solvers/fsi2_harmonic_diffmesh.py` | `solvers/fsi2_harmonic.py` |
| `solvers/fsi2_biharmonic_diffmesh.py` | `solvers/fsi2_biharmonic_simple.py` |
| `solvers/fsi2_biharmonic_diffmesh_restricted.py` | `solvers/fsi2_biharmonic_fulljacobian.py` |
| `solvers/fsi2_biharmonic_diffmesh_restricted_scaled.py` | `fsi2_biharmonic_fulljacobian`, `mesh_motion_scaling=1e-9` |
| `solvers/fsi2_biharmonic_diffmesh_restricted_split.py` | removed |
| `solvers/fsi2_biharmonic_diffmesh_restricted_split_condensed.py` | `solvers/fsi2_biharmonic_mixed.py` |
| `solvers/fsi2_biharmonic_diffmesh_restricted_split_condensed_cahouet_chabard.py` | removed (on `3d`) |
| `solvers/threeDfsi_iterative.py` | `solvers/threeDfsi_mixed.py` |
| `scripts/create_mesh_DFG2d.py` | removed |
| `scripts/create_mesh_DFG2d_alt.py` | `scripts/create_mesh_DFG2d.py` |
| `component_solvers/restricted_test_function_*_mwe.py` | `examples/` |
| `tools/cahouet_chabard.py` | removed |
| `restricted-test-functions-solver-time.md` | `notes/restricted-test-functions/solver-time.md` |
| `output/figures/y-disp.svg` | `docs/figures/y-disp.svg` |
| QoI files `fsi2_harm_dm_qoi.txt`, `fsi2_biharm_qoi.txt`, `fsi2_biharm_qoi_restr*.txt`, `fsi2_biharm_c0ip.txt`, `fsi3d_iterative_qoi_*.txt` | `<script>[_<mesh or variant>].txt` |

## Phase 7: threeDfsi_c0ip (2026-10-14)

- The machine was free after the cleanup, so `threeDfsi_c0ip` was built. It is `threeDfsi_mixed` with the
  mesh motion of `fsi2_biharmonic_c0ip`. The penalty is `get_dg_penalty_parameters_tetrahedron`, with the
  facet diameters and areas of `compute_facet_diameters_and_areas`.
- Coarse 3d mesh, 14 ranks, iterative K_c, 20 steps:

  | start | max rel. QoI difference to `threeDfsi_mixed` | Newton its | s/step, c0ip | s/step, mixed |
  |---|---|---|---|---|
  | rest | 9e-5 | 50 / 50 | 3.97 | 1.23 |
  | t = 10, from the mixed checkpoint | 2e-3 | 74 / 71 | 5.42 | 1.65 |

  The QoIs differ because the mesh motion differs. Both solvers converge with similar iteration counts.
- Profile (`-log_view`, from rest): `SNESFunctionEval` takes 66 of 77 s, about 0.95 s per residual. The
  K_c solves take 8.5 s, the Jacobian 1.3 s, and the Cholesky factorization of K_m 0.09 s. The C0IP facet
  integrals on the curved tetrahedra are reassembled in every residual evaluation, although they are
  linear in u.
- Possible next step: assemble the C0IP block once and add its product with u to the residual, instead of
  integrating it again each time. Not done here, since it changes the formulation code and needs its own
  check. In 2d, the C0IP solver is about 25% slower per step than the mixed one (0.37 vs 0.30 s/step from
  t = 16).
- No production run of `threeDfsi_c0ip` has been made.

## Jacobian plots from a developed state (2026-10-22)

- `fsi2_biharmonic_fulljacobian` gets `plot_matrices`, with the same storage and eq. (1) orderings as
  `fsi2_biharmonic_mixed`. The mixed solver's full Jacobian is now titled "Condensed Jacobian", since u_S is
  condensed out of its solid stress.
- `jobs/fsi2_jacobian_plots.sh` plots the full, condensed and approximate Jacobians in the order
  (u_S, v, p, z, u_I): spy plots on `mesh_sec_coarse` and block norm plots on `mesh_sec`. Each solver takes
  one step from a checkpoint (`initial_state_dir`) with `plot_matrices=True`. The script exports `src` of a
  pinned commit into its output folder and runs from there, so the plots stay tied to that commit.
- The checkpoints are not in git. The defaults are the t = 15 states that `fsi2_biharmonic_mixed` wrote at
  c3c6d01 in `output/matrix_block_runs/` (2026-10-20). Checkpoints from before the start time was stored
  still read, as t0 = 0.
- The spy plots run on one rank, since the ranks set the dof numbering within each block. From the coarse
  t = 15 state, the full Jacobian Newton converges in 3 iterations, and the three matrices have 160494,
  157485 and 116564 nonzeros on 6299 dofs. The whole script takes about 20 s (mesh_sec on 8 ranks).
- The first try hung on mesh_sec on 8 ranks after the full Jacobian was plotted: all ranks at 100% CPU,
  with the plots written. `fsi2_biharmonic_fulljacobian` left its MUMPS factorization to garbage
  collection, and after rank 0 had plotted, the ranks reached that collective destroy at different points.
  It now destroys its solver at the end of `solve()`, as `fsi2_biharmonic_mixed` already did.
- Block norms on mesh_sec at t = 15.0025: the (v, u_S) block is 2.7e8 in the full Jacobian (E_S + F_S)
  and 8.8e3 in the condensed one (F_S), where (v, v) grows from 7.0e3 (H) to 3.4e5 (H_c). The approximate
  Jacobian drops (v, u_S), (v, u_I), (p, u_S) and (p, u_I).
