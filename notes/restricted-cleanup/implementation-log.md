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
  pinned commit into a temporary directory and runs from there, so the plots stay tied to that commit. The
  directory is removed when the script exits, also when it fails or is stopped, and the checkout is not changed.
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

## Simplified mixed solver with Python PCs for u (2026-10-26)

- `fsi2_biharmonic_mixed_simplified.py` (sketched by Ottar, finished here) drops all rows of u from the
  residual of `fsi2_biharmonic_mixed`, apart from the Dirichlet rows. It keeps the multiplicative
  fieldsplit q_c = (v, p), u_S, m = (z, u_I), with MUMPS on K_c, and solves the two displacement splits by
  PCPYTHON contexts that set u of the next Newton iterate directly:
  - `NodalUpdatePC` (u_S): y_S = u - u_old - dt (theta v + (1 - theta) v_old) + theta dt y_v, node by
    node, with y_v read from the q_c sub-KSP (`ksp_q_c.getSolution()`), which the multiplicative
    fieldsplit solves first. Without rows of u, the split's right-hand side holds nothing but the
    Dirichlet rows, so the kinematic defect is taken from the iterate (the fields hold it since the
    residual was assembled there). U and V must have the same element, which is checked.
  - `SeparateSolverPC` (m): interpolates the next displacement onto the fluid submesh, extends it from the
    whole fluid boundary (which includes the interface) with `component_solvers/biharm.py` (mixed
    biharmonic, MUMPS LU on both components at once, assembled once), and returns u - u_new on u_I and
    z - z_new on z. The extension's auxiliary field solves the same equation as z.
  - At Dirichlet rows both return the right-hand side x - g, like the unit diagonal of dolfinx's Newton
    system. Neither PC is linear in its right-hand side, so the outer KSP must stay `preonly`.
- Fixed in the sketch: `NonlinearProblem` got the monolithic residual instead of `residual_blocks`, the
  Jacobian blocks were indexed `[i, j]`, and the trial functions were `TestFunctions`. The
  `IdenticalBlocksPC` component index sets are gone with the K_m solve they served.
- Validation against `fsi2_biharmonic_mixed` (direct K_c, dt = 0.0025):
  | start | mesh, ranks | steps | Newton its, mixed / simplified | max rel. field difference |
  |---|---|---|---|---|
  | rest | mesh_sec_coarse, 1 and 4 | 40 | 79 / 79 | QoIs equal to the 7 printed digits |
  | t = 16, `fsi2_biharm_dm_restr_split_approx` checkpoint | mesh_sec, 4 | 40 | 294 / 294 | u 6e-11, v 2e-11, p 2e-12, z 1e-8 |
  With exact solves in both displacement splits, the PCs give the same corrections as the rows of u in the
  mixed solver, so the iterates agree to round-off. Time per step from t = 16 on 4 ranks: 0.270 s against
  0.280 s for the mixed solver.
- `tests/test_fsi2_biharmonic_mixed_simplified.py` runs both solvers for six steps on mesh_sec_coarse and
  compares their QoIs.
- Benchmark run (c9bff97, `main()`: mesh_sec, dt = 0.0025, from rest to T = 16.025, 16 ranks): finished all
  6410 steps without a failed solve, in 589 s (0.092 s/step), with 38067 Newton iterations (at most 12 in a
  step). Over t in [14, 16.025]: drag 217.9 +- 80.7, lift -0.51 +- 244.2, u_x -1.523e-2 +- 1.302e-2,
  u_y 1.28e-3 +- 8.27e-2 (mean +- half the peak-to-peak range). QoIs in
  `output/qoi/fsi2_biharmonic_mixed_simple.txt`, log in `output/logs/fsi2_biharmonic_mixed_simple.txt`.

## Quadrature cap in the twice simplified solver (2026-10-26)

- `fsi2_biharmonic_mixed_twice_simplified.solve` takes `max_quadrature_degree` (default None, no cap),
  applied with `cap_quadrature_degree` to the residual blocks before `derivative_block`, so the Jacobian
  keeps it, and to the drag and lift forms. On mesh_sec (quadratic geometry, P2/P1), UFL estimates the
  degrees 6, 14, 14, 8, 12, 9, 12, 12, 12, 12 for the integrals of the v residual and 13 for the p
  residual. The default triangle rules have 42 points at degree 14, 25 at 10, 16 at 8, 12 at 6, 7 at 5,
  and 6 at both 4 and 3.
- `jobs/fsi2_quadrature_cap.sh` runs the solver from a developed state (a copy of its benchmark run's
  t = 16 checkpoint in `output/quadrature_cap/initial_state`) to t = 18, 800 steps of dt = 0.0025, once
  per cap, one after the other on 16 ranks. Errors are the largest deviation from the uncapped run over
  the 800 steps, relative to half the peak-to-peak range of the uncapped QoI (drag 80.65, lift 244.2,
  u_x 1.302e-2, u_y 8.274e-2):

  | cap | s/step | speed-up | Newton its (max/step) | drag | lift | u_x | u_y |
  |---|---|---|---|---|---|---|---|
  | none | 0.099 | 1.00 | 6002 (12) | - | - | - | - |
  | 10 | 0.095 | 1.04 | 6002 (12) | 0 | 0 | 0 | 0 |
  | 8 | 0.084 | 1.18 | 6002 (12) | 0 | 0 | 0 | 0 |
  | 6 | 0.081 | 1.22 | 6002 (12) | 0 | 4e-8 | 8e-7 | 1e-7 |
  | 5 | 0.074 | 1.34 | 6002 (12) | 3e-5 | 2e-5 | 3e-5 | 9e-6 |
  | 4 | 0.076 | 1.30 | 6000 (12) | 4e-4 | 3e-4 | 4e-4 | 1e-4 |
  | 3 | 0.073 | 1.36 | 6030 (12) | 1.6e-2 | 1.2e-2 | 1.2e-2 | 5e-3 |

  0 means equal in the 7 printed digits. A repeat of the uncapped and cap 6 runs gave the same QoIs and
  0.097 and 0.081 s/step, so the timings vary by about 2%.
- Down to degree 6 the QoIs do not change, with 22% less time per step; at degree 5 they change by
  3e-5 of their range. The Newton iterations stay the same down to degree 4. The speed-up levels off at
  about 1.35, since the rest of a step (MUMPS solves of K_c and the mesh motion) does not depend on the
  quadrature. Degree 3 has the same 6 points as degree 4 and is not faster, but is 40 times less
  accurate. A cap of 6 is a free 20%; 5 gives 34% for a change of 3e-5 of the range, which was not
  compared to the space or time discretization error. The default stays None, so the benchmark runs
  are unchanged.
- In the benchmark quantities (mean, amplitude and frequency over the last full period of u_y, as in
  `compare_fsi_qois.py`), cap 5 changes every mean, amplitude and frequency by at most 5e-5 relative to
  the uncapped run (the mean lift, 0.57 against an amplitude of 244, by 1e-3 N). For scale: the earlier
  mesh_sec benchmark run, one period near t = 15 instead of t = 17.2, differs by up to 2e-4 (u_x mean,
  1e-3 in the u_y mean); mesh_fine_sec (`fsi_2.txt`, same dt) by 3e-4 to 4e-3 (drag mean +0.43%, lift
  amplitude -0.30%, frequency -0.05%); and the 2006 reference of Turek & Hron by 2-9% (drag amplitude
  -8.5%). Their level 4 results with dt = 0.002 differ from their reference by up to 8% (drag amplitude
  67.61 against 73.75). The quadrature error of cap 5 is 20 to 100 times below the mesh difference and
  three orders of magnitude below the spread of the published values.
- Cap 5 from rest (Ottar's run of `main()` with `max_quadrature_degree=5`, `restart=False`, to t = 16.25;
  QoIs in `output/qoi/fsi2_biharmonic_mixed_twice_simplified.txt`, which replaced those of the uncapped
  run), compared with the uncapped run from rest, `output/qoi/fsi2_biharmonic_mixed_simple.txt` (whose
  last rows equal those of the uncapped twice simplified run). The largest pointwise deviation, relative
  to half the peak-to-peak range over t >= 14, is 1e-6 up to t = 6, grows to 6e-5 while the oscillation
  builds up (t = 8 to 10; |u_y| first exceeds half its final amplitude at t = 8.07 in both runs), and
  settles at 3e-5 from t = 12. The quantities over the last period before t = 16.025 (t = 15.10 to 15.62,
  the same in both runs) change by at most 9e-6 relative, apart from the u_y mean (1.29 mm, 5e-5) and the
  lift mean (0.59 against an amplitude of 244; 1.1e-3, 4e-6 of the amplitude). This is the same as from
  the developed state: the onset of the oscillation does not amplify the quadrature error, and it stays
  20 to 100 times below the difference to mesh_fine_sec and three orders of magnitude below that to the
  2006 reference.
  The run took 392 s on 16 ranks, 0.060 s per step over the 6500 steps (7 Newton iterations per step at
  the end). There is no uncapped run of this solver from rest to compare it with; the simplified solver
  took 0.092 s per step from rest on 16 ranks.

## Quadrature cap and QoI plots in every solver (2026-10-27)

- `cap_quadrature_degree(form, None)` returns the form unchanged, and `quadrature_description` gives
  the line `Quadrature degree: ...` that every solver now prints in its end-of-run summary.
- `plot_run_qois` in `scripts/plot_qois.py` generalises the plots at the end of `main()` of the twice
  simplified solver: `full_<qoi>.pdf` and `closeup_<qoi>.pdf` per QoI in
  `output/figures/qoi/<qoi file stem>/`, with the FeatFlow reference for the QoIs that are a column of
  it. Label sets: `FSI2_QOI_LABELS`, `LAGRANGE_QOI_LABELS`, `INCOMPRESSIBLE_QOI_LABELS` (no reference,
  closeup over the last 2 s) and `QOI_LABELS_3D` (no reference file, closeup over t in [9, 10]).
- Every FSI2 solver has `solve(..., max_quadrature_degree=None)`: the residual blocks, any separately
  built kinematic residual (`residual_post_u`, `res_miss_ufl`) and the drag and lift are capped, before
  differentiation. Diagnostics (interface gaps, det F violation) are not capped. Every `main()` binds
  its run settings to local variables, with `max_quadrature_degree = None`, and plots with
  `plot_run_qois` on rank 0. `dfg_2d_3` has the cap option but keeps its own plots in `solve()`.
- The 3d solvers keep the default cap 8 and raise `ValueError` for `None`: UFL's estimated degrees
  (19 in the residual, 25 in the Jacobian) are far too expensive on tetrahedra. An attempted test of
  None against 8 on `mesh_sec_coarse` took 4.5 to 8 minutes per pair and gave QoIs identical at the
  `%.6e` precision of the QoI file over 2 steps; it was dropped.
- On the FSI2 coarse test mesh, a cap of 4 changes the QoIs over 2 steps by at most 3e-5 relative
  (5e-5 for the Lagrange solver, 7e-5 for det_F_max of the incompressible one); each solver has a test
  of this, with rtol = 1e-3. For dfg_2d_3, a cap of 4 (UFL estimates 5 for the convection) changes the
  lift by up to 6e-5.
- The 3d solve, initial-state and K_c-option tests are marked `slow` and left out by default
  (`pytest -m slow` runs them).

## Fieldsplit on a nest Jacobian in the twice simplified solver (2026-10-27)

- `fsi2_biharmonic_mixed_twice_simplified.solve` takes `solver_options` (default
  `DEFAULT_DIRECT_SOLVER_OPTIONS`, MUMPS LU); with `pc_type = fieldsplit`, as in
  `DEFAULT_ITERATIVE_SOLVER_OPTIONS`, the `NonlinearProblem` is built with `kind="nest"`.
- Three things kept the fieldsplit options from taking effect:
  - `NonlinearProblem` passes the nest index sets to the PC only when it has a preconditioner form `P`
    (`LinearProblem` always does), and then names the splits `<u.name>_<i>`, here `v_0` and `p_1`, so
    `fieldsplit_v_*` would not match. The dolfinx Stokes demo relies on unnamed functions, whose
    splits are `0` and `1`. The solver now sets `("v", is_v), ("p", is_p)` from `A.getNestISs()`
    itself, with block size 2 and the rigid body modes composed on `is_v`, as in
    `fsi2_biharmonic_mixed`.
  - dolfinx deletes `petsc_options` from the options database right after `SNESSetFromOptions`, but
    PCFIELDSPLIT reads its sub-KSP options only in `PCSetUp`, in the first solve. The `fieldsplit_*`
    options are set again under the `solver_` prefix after the problem is built.
  - The KSP post-solve hook read `x.getArray()`, which needs a contiguous vector; with a nest `x` it
    reads the first nest block.
- Check on `mesh_sec`, 4 ranks, 5 steps from the t = 16 checkpoint: FGMRES takes 13 to 16 iterations
  per Newton step, the Newton counts (6 per step) and the QoIs match the MUMPS run at the `%.6e`
  precision of the QoI file, at 0.26 s per step against 0.19 s for MUMPS. `ksp_view` shows GAMG with 3
  near-nullspace vectors (coarse `bs=3`) on `fieldsplit_v` and BoomerAMG on `fieldsplit_p`.
- The end-of-run `Solver setup` summary names the linear solve, from `solver_options` (`MUMPS LU`, or
  the FGMRES/Schur/GAMG/BoomerAMG setup in the style of `fsi2_biharmonic_mixed`), plus the solid
  stress, the (v, p)-only Newton system and the mesh motion solve. The description is built before
  the time loop, so options without the keys it reads fail at the start of a run.
- The VTX output of the twice simplified solver holds u, the displacement on the parent mesh, with v,
  as `fsi2_biharmonic_mixed` writes `[u, v]`; u is gathered from u_s and u_f before each snapshot.

## Checkpoint start-up helpers in tools (2026-10-27)

- `tools/checkpoint.load_start_state` holds the restart / initial-state block that every solver
  repeated: it sets up the run's `Checkpointer` (clearing an earlier run's checkpoints on a new run
  with `checkpoint_every`), reads the start state into the checkpoint functions, raises when no step
  is left to solve or the initial state changed while it was read, renames the VTX output paths on a
  restart, and returns `StartState(t, step, checkpointer, output_paths)`.
- `tools/submesh.SubmeshComposite(parent, [(f, entity_map), ...])` replaces `gather_displacement` and
  `scatter_displacement` of the twice simplified solver: `gather()` pieces the parent-mesh function
  together from the submesh functions cell by cell, `scatter()` copies it back. Where the parts share
  dofs, the one listed last wins (the fluid, for u on the interface). `tools/submesh.all_cells` is also
  used by `Checkpointer`. The copies are exact up to round-off from the interpolation (about 1e-16).
- The twice simplified solver uses both, and scatters u unconditionally after loading the start state
  (u is 0 on a new run from rest). A 5-step start from the t = 16 `mesh_sec` checkpoint on 4 ranks gives
  byte-identical QoIs to before. The other 15 solvers with the same block still have their own copy.

## load_start_state in the other solvers (2026-10-27)

- The other 15 solvers with the restart / initial-state block (the 12 in `solvers/` and `cfd_benchmark`,
  `csm_benchmark`, `fsi2D_benchmark`) call `load_start_state` with their own checkpoint functions,
  submeshes and VTX output paths. `cfd_benchmark` and `csm_benchmark` have no initial-state option and
  pass `initial_state_dir=None`; they keep their local import of the helper.
- The restart message lists three or more output paths as `a, b and c` (`_join_and`), as
  `fsi2_harmonic_lagrange` did; with `' and '.join` it would read `a and b and c`.
  `fsi2_biharmonic_mixed_incompressible` loses the comma before its `and`. The one- and two-path
  messages are unchanged. Solvers that printed with `comm.rank == 0` or `PETSc.Sys.Print` now print on
  `mesh.comm.rank == 0`, which is the same rank since every mesh is read on `COMM_WORLD`.
- `SubmeshComposite` replaces no other code: only `SeparateSolverPC` in `fsi2_biharmonic_mixed_simplified`
  copies between the parent mesh and a submesh by hand, and it copies between different functions in
  each direction (u_next -> u_bc, u_h -> u_new), without the `scatter_forward` of `gather()`.

## Mesh motion package (2026-10-27)

- `src/xfsi_solver/mesh_motion/` sets up the mesh motion of a displacement u on the mesh it lives on
  (in the FSI solvers, the fluid submesh), with Dirichlet conditions on the whole boundary:
  `harmonic_mesh_motion` (-div grad u = 0, no scaling), `biharmonic_mixed_mesh_motion` (with z = div
  grad u as a second unknown; optional `z` for checkpointing) and `biharmonic_c0ip_mesh_motion`
  (penalties of `tools/dg_optpar.py`, `a_val = 2.0`, triangles, quadrilaterals and tetrahedra). The
  biharmonic forms set the normal derivative of u to zero on the boundary, naturally (mixed) or by
  Nitsche (C0IP).
- Each returns a `MeshMotion`: a `NonlinearProblem` solved by one `ksponly` step with an explicitly
  lagged Jacobian (`snes_lag_jacobian -2`, persisting), so the matrix is assembled and factorized in
  the first solve only. `solve()` raises on a failed SNES, KSP, PC or block PC; `destroy()` is
  collective. The default `solver_options` select `IdenticalInterleavedBlocksPC` from the options
  alone, with MUMPS LU (`cntl_1 = 1e-4`) on the first component applied to all by MatMatSolve; its
  interleaved split also fits the block matrix [z | u] of the mixed form, since every rank's z and u
  rows are whole (x, y) pairs. Options passed replace the defaults.
- A `LinearProblem` version (`MyLinearProblem`, matrix assembled once, right-hand side by lifting)
  solved the mesh motion alone to round-off of the old one (1.2e-10), but in the twice simplified
  solver it moved u and z in the checkpoints by 2.2e-6 relative after 4 steps from rest, while v and p
  stayed at 1e-12; the cause was not found. The NonlinearProblem version reproduces the old u
  bit for bit, and the 5-step `mesh_sec` start from t = 16 on 4 ranks gives byte-identical QoIs at the
  same 0.186 s per step.
- dolfinx's tidy-up of `petsc_options` (`opts.prefixPush(prefix)`, then `del opts[k]`) has no effect
  with petsc4py 3.25: deleting under a pushed prefix does nothing, while setting under one works. So
  the options stay in the database. `MeshMotion` sets its KSP options again after the problem is
  built anyway, and first removes the options left by an earlier problem with the same prefix.
- The twice simplified solver takes `mesh_motion_options` (default None, the MUMPS default above) and
  names the mesh motion setup in its end-of-run summary.

## Twice simplified solver: checkpoints without z, selectable mesh motion (2026-10-27)

- The checkpoints of the twice simplified solver hold (u, v, p) only. z, the Laplacian of u_f in the
  mixed mesh motion, is an auxiliary unknown that every mesh motion solve computes again from the
  boundary values, so it needs no checkpoint. Restart and initial-state tests pass at their earlier
  tolerances, and the 5-step `mesh_sec` start from t = 16 on 4 ranks gives byte-identical QoIs
  without reading z.
- `solve(..., mesh_motion_type=...)` takes any function of `xfsi_solver.mesh_motion`, by default
  `biharmonic_mixed_mesh_motion`; the summary names it. Over 3 steps from rest on `mesh_sec_coarse`,
  C0IP and harmonic mesh motion change the QoIs by up to 7e-7 relative. A run can start from a state
  written with another mesh motion. The unused `"Z"` element left the default `elements`.
- `Checkpointer.read(..., optional=names)` and `load_start_state(..., optional=names)` leave the named
  functions as they are when the checkpoint does not hold them (checked by
  `io4dolfinx.read_timestamps`, which returns no times for an absent function; `read_function_names`
  is not implemented for ADIOS2). `fsi2_biharmonic_mixed`, where z is a Newton unknown, reads z as
  optional, so it can still start from the twice simplified solver's states, with z = 0 as Newton's
  initial guess for the linear mesh motion equations; its QoIs then agree with a start from its own
  state within the test's 1e-6. The other solvers that checkpoint z still require it.

## Solver renames (2026-10-27)

- `solvers/fsi2_biharmonic_mixed_twice_simplified.py` is now `solvers/fsi2_simplified.py`, and
  `solvers/fsi2_biharmonic_mixed_simplified.py` is now
  `solvers/fsi2_biharmonic_mixed_intermediate_simplified.py`, with their tests renamed to match
  (`tests/test_fsi2_simplified.py`, `tests/test_fsi2_biharmonic_mixed_intermediate_simplified.py`).
  The entries above use the old names. `jobs/fsi2_quadrature_cap.sh` imports the new module.
- `fsi2_simplified.main()` names its outputs `fsi2_simplified` (`output/checkpoints/fsi2_simplified`
  and so on). The outputs of earlier runs keep the old name, `fsi2_biharmonic_mixed_twice_simplified`,
  and were not moved. The intermediate solver's `main()` keeps its outputs `fsi2_biharmonic_mixed_simple`.

## Interface values through the parent mesh (2026-10-28)

- `fsi2_simplified` and `threeDfsi_simplified` set the Dirichlet values of u_f from u_s through
  `u_s_parent`, a function on the parent mesh that is u_s on the solid cells and 0 on the rest,
  instead of `interpolate_nonmatching` from U_s. The cells and dofs match, so the transfer is exact,
  needs no point search, and gives 0 at the fluid dofs off the interface by itself. The reset of those
  dofs (`fluid_reset_nodes`), added because the padded bounding boxes of `interpolate_nonmatching`
  let it extrapolate u_s into the fluid cells next to the interface, is gone.
- Interpolating u_s into the parent space and calling `scatter_forward` is not enough in parallel: the
  owner of an interface dof need not hold a solid cell with it. On `fsi3d/mesh_sec_coarse` with 2 and
  30 ranks and `mesh_sec_medium` with 13 and 30, some interface dofs came out 0; `fsi2/mesh_sec` and
  `fsi3d/mesh_hex_sec_coarse` were right at every rank count tried (1 to 30). The ranks with a solid
  cell at the dof therefore add their values into the owner (`scatter_reverse` with
  `InsertMode.add`), which divides by the number of such ranks, counted once at set-up. With that the
  transfer matches the exact interface values to 5e-15 on all these meshes and rank counts.
- The QoIs of 5 steps of `fsi2_simplified` on `mesh_sec` with 13 ranks (dt = 0.001), and of 3 steps
  of `threeDfsi_simplified` on `fsi3d/mesh_sec_coarse` with 2 ranks (dt = 0.004), are the same as
  before to all printed digits, with the same Newton iterations.
