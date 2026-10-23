# Restricted harmonic FSI2 solvers: implementation log

Goal: two FSI2 solvers with harmonic mesh motion and restricted displacement test functions,
with matrix plotting, and harmonic plots from `jobs/fsi2_jacobian_plots.sh`. Task:
`output/prompts/restricted-harmonic-solvers.md`. Branch `restricted-harmonic`, from
`jacobian-plotting` (4488596).

## Solvers (2026-10-22)

- `solvers/fsi2_harmonic_fulljacobian.py`: copied from `fsi2_biharmonic_fulljacobian.py`, with
  `z` removed and the mesh equation `(grad u, grad du)_f` on `dx_fluid`. No alpha scaling and no
  interface flux term; the interface u rows are zeroed by `zeroRows` and hold the solid kinematic
  equation (`res_post`/`jac_post`) exactly as in the biharmonic solver. The biharmonic solver's
  zero `0 * (u, du)_f` sparsity term is dropped: the Laplacian already fills the fluid u-u block.
- Kept `mesh_motion_scaling` (the `ScaledLU` shell PC) in the full-Jacobian solver, so that it
  keeps the options of its biharmonic twin; the scaled rows are the fluid-only u rows, without the
  z block. Tested with and without scaling, as for the biharmonic solver.
- `solvers/fsi2_harmonic_condensed.py`: copied from `fsi2_biharmonic_c0ip.py`, which already has
  m = u_I only and MUMPS Cholesky on K_m by `IdenticalBlocksPC`, with the C0IP measures and
  penalty parameters removed and the same harmonic mesh equation. Everything else (u_S condensed
  out of the solid stress, J_0, `gamma_reassemble`, the fieldsplit q_c, u_S, m, `direct_k_c_solve`
  and the iterative K_c option, checkpointing, QoIs, `solver.destroy()` and
  `mesh_motion_pc.destroy_solvers()` at the end) is unchanged. `plot_matrices` titles are
  "Condensed Jacobian" and "Approximate Jacobian", as in `fsi2_biharmonic_mixed`.
- K_m = A_II is the vector Laplacian on the fluid-interior dofs, with the outer fluid boundary
  Dirichlet rows set symmetrically by dolfinx; it does not couple the components and both
  components have the same Dirichlet set, so `IdenticalBlocksPC` with Cholesky applies.
- Name: `fsi2_harmonic_condensed` rather than `fsi2_harmonic_mixed`. "mixed" in
  `fsi2_biharmonic_mixed` names the mixed form of the biharmonic equation (the auxiliary z),
  which the harmonic mesh motion does not have. What distinguishes this solver from
  `fsi2_harmonic_fulljacobian` is the condensation of u_S out of the solid stress (with J_0 and the
  fieldsplit built on it), and its full Jacobian is the one plotted as "Condensed Jacobian".
- Default `main()` of both: mesh_sec, T = 7 (see the developed state below), own output paths.
- The functions are named `u`, `v`, `p`, so a run can start from a biharmonic checkpoint; the
  checkpoint reader reads by name and ignores `z`. Tested
  (`test_fsi2_harmonic_condensed_starts_from_biharmonic_checkpoint`).

## Tests

- `tests/test_fsi2_harmonic_fulljacobian.py`: the biharmonic full-Jacobian tests with z removed
  (solve, restart, restart on 2→1 and 1→2 ranks, dt and T checks, initial state, matrix plots),
  each with and without `mesh_motion_scaling`.
- `tests/test_fsi2_harmonic_condensed.py`: as `test_fsi2_biharmonic_c0ip.py`, plus restart on
  2→1 and 1→2 ranks, start from a biharmonic checkpoint, matrix plots (direct and iterative K_c),
  and the comparison with the full-Jacobian solver.
- Comparison of the two harmonic solvers, 8 steps from rest on `mesh_sec_coarse`, serial:

  | condensed solver setting | u | v | p |
  |---|---|---|---|
  | `gamma_reassemble=0` (fresh J_0 each iteration), 4 steps | 3.5e-11 | 2.8e-11 | 2.3e-12 |
  | default `gamma_reassemble=0.2`, 8 steps | 2.1e-10 | 8.9e-10 | 1.9e-11 |

  (max abs difference / max abs value of the full-Jacobian field). The QoI rows agree to all
  printed digits. The test uses the default reuse and a tolerance of 1e-6 relative to each
  field's (and QoI column's) largest value, the tolerance of the restart tests of the solvers with
  Jacobian reuse: the difference is set by where the linearly converging J_0 iteration stops below
  `snes_atol` 1e-7, and 1e-6 leaves margin for other partitions and MUMPS versions.
- Side check: the old `fsi2_harmonic` (unrestricted test functions, alpha_u = 1e-9 with the
  interface flux term) agrees with `fsi2_harmonic_fulljacobian` to 1e-12 (u), 1e-11 (v) and 5e-13
  (p) after 8 steps on the coarse mesh: with alpha_u = 1e-9 the fluid contribution to the
  interface rows is negligible. So the comparison test does not distinguish the two harmonic
  formulations; it checks that the condensed and full solvers agree.

## Developed harmonic state (2026-10-22)

Runs from rest to T = 7 with `fsi2_harmonic_condensed` (dt = 0.0025, `direct_k_c_solve=True`,
`gamma_reassemble=0.2`, `checkpoint_every=400`), from the source of e83a4d4 exported to
`output/matrix_block_runs/harmonic_src/` (`harmonic_run.py`), in
`output/matrix_block_runs/fsi2_harmonic_{coarse,normal}/`, logs in `output/matrix_block_runs/logs/`.
A serial watcher (`harmonic_watch_detF.py`) logged min det F = det(I + grad u) over the fluid cells
(degree-6 quadrature points and vertices) at each new checkpoint, since only the two latest
checkpoints are kept and a mesh can fold without Newton failing.

- `mesh_sec_coarse`, 4 ranks: finished, 70.8 s (0.025 s/step), 13475 Newton iterations in 2800 steps
  (mostly 4–5 per step, at most 8), 399 Jacobian assemblies, no failures.
  min det F: 0.995 (t = 2), 0.977 (t = 4), 0.954 (t = 6), 0.970 (t = 7).
- The mesh does not degenerate before t = 7; the state at t = 7 is used. It is only the start of
  the oscillation: tip A_y in (6, 7] is -3.8e-3 .. 4.6e-3 (biharmonic coarse run: -4.0e-3 .. 4.8e-3
  over the same interval, and -7.6e-2 .. 7.8e-2 in (14, 15]). So the harmonic plots show a much
  less deformed state than the biharmonic plots at t = 15.
- `mesh_sec`, 8 ranks: finished, 190.6 s (0.068 s/step), 13982 Newton iterations in 2800 steps
  (2–9 per step, mostly 4–6), 58 Jacobian assemblies, no failures. min det F: 0.991 (t = 1),
  0.959, 0.938, 0.906, 0.827, 0.686, 0.603 (t = 7): falling, but far from folding at t = 7. Tip
  A_y in (6, 7] is -1.46e-2 .. 1.39e-2.

## Plot script and harmonic plots (2026-10-22)

- `jobs/fsi2_jacobian_plots.sh`: `FORMULATION=biharmonic|harmonic` (default biharmonic) selects
  the solvers, the default commit, the default checkpoints, the OUT_DIR suffix (`_harmonic`) and
  the figure names (`spy_harmonic_full.png`, ...). The biharmonic default COMMIT stays 66b0885;
  the harmonic default is e83a4d4, the commit of the solvers. A per-formulation default keeps the
  biharmonic plots tied to the commit they were made from.
- Biharmonic behaviour checked: the default run, into a scratch OUT_DIR, gave PNGs byte-identical
  to `output/figures/jacobian_plots/66b0885/figures/`.
- `FORMULATION=harmonic jobs/fsi2_jacobian_plots.sh`: 17 s, figures in
  `output/figures/jacobian_plots/e83a4d4_harmonic/figures/`. One step from t = 7, plots at
  t = 7.0025. The condensed run on mesh_sec took 9 Newton iterations in that step (first step
  after a restart, with a fresh J_0).
- Block norms on mesh_sec, order (u_S, v, p, u_I), against the harmonic eqs. (1), (3), (6):

  | block | full | condensed | J_0 | expected |
  |---|---|---|---|---|
  | (u_S, u_S) A_S | 7.37e2 | 7.37e2 | 7.37e2 | rho_s M_S / dt |
  | (u_S, v) C | 0.926 | 0.926 | 0.926 | theta dt A_S R_S: 0.5025 * 0.0025 * 737 = 0.926 |
  | (v, u_S) | 2.68e8 | 6.60e3 | — | E_S + F_S; F_S only; dropped |
  | (v, v) | 6.97e3 | 3.37e5 | 3.37e5 | H; H_c = H + theta^2 dt K_S |
  | (v, p) G, (p, v) B | 0.874 | 0.874 | 0.874 | unchanged |
  | (v, u_I) F_I | 5.55e4 | 5.55e4 | — | dropped in J_0 |
  | (p, u_S) D_S, (p, u_I) D_I | 0.258, 9.39 | same | — | dropped in J_0 |
  | (u_I, u_S) A_IS, (u_I, u_I) A_II | 27.4, 690 | same | same | the only mesh rows |

  The mesh rows have no v or p columns, (u_S, p) and (u_S, u_I) are empty, and J_0 is block lower
  triangular in ((u_S, v, p), u_I). The spy plots on the coarse mesh show the same structure
  (121372, 118360 and 77439 nonzeros in the full, condensed and approximate Jacobians, 4433 dofs).
  In Frobenius norm, F_I is about 8 times larger than F_S here.

## Test suite (2026-10-22)

Full suite at 13f52ed (solvers, tests and docs; the job script is not covered by tests), serial
pytest with the restart tests spawning 2 ranks, while the plot script and the biharmonic check ran:
189 passed in 542.8 s (`output/logs/restricted_harmonic_tests.txt`).

## Plot script output (2026-10-23)

- `jobs/fsi2_jacobian_plots.sh` now leaves only `run_info.txt` and the six figures in OUT_DIR, which is
  `output/figures/jacobian_plots/<short hash of COMMIT>/` for both formulations (no `_harmonic` suffix
  and no `figures/` subfolder). The exported source, the runs with their logs, and the other plots of
  the solvers are in one temporary directory, removed when the script exits. OUT_DIR is made only after
  all four runs succeeded; a failed run prints the end of its log instead.
- If both formulations are ever pinned to the same commit, the second run stops because OUT_DIR exists;
  pass another OUT_DIR then.
- Regenerated `66b0885/` (biharmonic) and `e83a4d4/` (harmonic) with it, replacing the earlier
  folders `66b0885/` and `e83a4d4_harmonic/`.
- Both formulations now come from one commit and one run of the script: e83a4d4, the first commit with
  the solvers of both, differs from 66b0885 under `src/` only by the two harmonic solver files.
  `FORMULATIONS` (default `biharmonic harmonic`) replaces `FORMULATION`, the checkpoints are set per
  formulation and mesh (`BIHARMONIC_COARSE_CHECKPOINTS` and so on), and the figures are named
  `spy_<formulation>_<jacobian>.png` and `block_norms_<formulation>_<jacobian>.png`.
  `output/figures/jacobian_plots/e83a4d4/` now holds `run_info.txt` and all 12 figures, replacing
  `66b0885/` and the harmonic-only `e83a4d4/`. The biharmonic block norms are the same as at 66b0885.
