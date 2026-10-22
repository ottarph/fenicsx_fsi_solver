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
