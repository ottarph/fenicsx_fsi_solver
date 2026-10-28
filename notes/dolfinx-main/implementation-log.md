# DOLFINx main: implementation log

Goal: run xfsi_solver on the DOLFINx main branch as well as on the 0.11 release, for the partitioning
experiments (`mwe_partition_*.py`), which need the edge weights of the graph partitioners that are on main
only. The code in `src/` keeps working on both versions. The setup of the `fenicsx-dev` env and `.venv` is
described in `CLAUDE.md` and `AGENTS.md`.

## Environment (2026-10-28)

`fenicsx-dev` rebuilt from scratch with https://github.com/ottarph/conda-dolfinx-dev-env at DOLFINx main
`b1aad92` (2026-10-26), with UFL `0d23dd3`, Basix `737b7b0` and FFCx `feb3cc4`, all optional parts on
(PETSc, SLEPc, ParMETIS, PT-SCOTCH, KaHIP, ADIOS2, SuperLU_DIST). PETSc 3.26.0, where the `xfsi_solver` env
has 3.25.5. Added before the build: `h5py` and `adios2` (MPICH builds), `tqdm`, `ruff`; after it, with pip
and `--no-deps`: `scifem` 0.27.0 (compiled against the new DOLFINx), `io4dolfinx` 1.5.0 and
`fenicsx-compat` 0.4.0, which both need. `.venv` is a virtualenv on `fenicsx-dev` with xfsi_solver installed
editable.

Activating `.venv` alone puts `/usr/bin/mpirun` (system MPICH 4.2) on the path, not the env's MPICH 5.0.1;
MPI runs go through `conda run -n fenicsx-dev mpirun ... .venv/bin/python`.

## Fast test suite on main before any change: 45 of 281 failed

- 38 from `dolfinx.mesh.transfer_meshtags_to_submesh`, whose last two arguments swapped on main:
  `(entity_tag, submesh, cell_to_parent, vertex_to_parent)`, in the order `create_submesh` returns them,
  where 0.11 has `(..., vertex_to_parent, cell_to_parent)`. The parameter names are the same in both, so
  the 21 calls in `src/` now pass the maps by keyword. With that, all 281 pass on 0.11, and 7 fail on main.
- The 7 left all compare a solver with a reference to a tolerance.

## The 7 tolerance failures on main

Comparing 6 time steps of `fsi2_biharmonic_mixed`, `fsi2_biharmonic_mixed_intermediate_simplified` and
`fsi2_simplified` on both versions: the latter two give QoI files identical to those on 0.11, but
`fsi2_biharmonic_mixed` takes 8-11 Newton iterations per step instead of 1-2 and its lift differs by up to
4e-5 relative. So not a tolerance, but a wrong Jacobian on main.

Cause: the solvers with a restricted solid kinematic equation assemble a second set of forms
(`jacobian_post`) with the Dirichlet conditions into the Jacobian that the `NonlinearProblem` assembled
(`post_jacobian`). Both assemblies put the unit diagonal on the Dirichlet rows. DOLFINx 0.11 inserts it
after a flush (`insert_diagonal`), so it stays 1; main adds it (`set_diagonal(..., ADD)`, "Adding to zeroed
rows sets the diagonal without a flush"), so it becomes 2, and Newton takes half steps on the Dirichlet
dofs, converging linearly to `snes_atol` instead of quadratically. The same pattern is in nine solvers
(`fsi2_biharmonic_mixed`, `_mixed_incompressible`, `_c0ip`, `_fulljacobian`, `fsi2_harmonic_condensed`,
`fsi2_harmonic_fulljacobian`, `fsi2D_benchmark`, `threeDfsi_mixed`, `threeDfsi_c0ip`); the tests of most
passed only because they converge to the same tolerance anyway.

Fix, the same on both versions: after `J.assemble()` in `post_jacobian`, `J.zeroRows(rows_bc, diag=1.0)` on
the owned Dirichlet rows (`tools.blocked_bcs.owned_dirichlet_rows`). These rows are zero apart from the
diagonal after both assemblies, so this sets the diagonal to 1 and changes nothing else. With it, the
mixed solver takes 1-2 iterations on main and its QoIs are identical to those on 0.11, and on 0.11 the QoIs
of the three solvers are identical to those before the fix.

`test_mesh_motion::test_default_solve_matches_whole_system_lu[biharmonic_mixed]` is a real tolerance case:
the default (component-wise MUMPS LU) and the whole-system MUMPS LU of the indefinite mixed system differ by
1.1e-9 of the largest displacement on main, and 7e-11 on 0.11, against `RTOL_TO_MAX = 1e-9`. Compared with
a solve with stronger pivoting (`cntl_1 = 0.5`) and iterative refinement, the default solve is 8e-10 off on
main and 7e-11 on 0.11, the whole-system LU 3e-10 and 1e-11: round-off larger on the new stack (DOLFINx main,
PETSc 3.26, OpenBLAS 0.3.34), not a wrong split, which gives errors of the order of the displacement.
`RTOL_TO_MAX` raised to 1e-8.

The three restart tests of `fsi2_harmonic_lagrange` are tolerance cases too. Its continuous runs give QoI
files identical on both versions, with the same Newton iterations; only the runs from a checkpoint, where
the Lagrange multipliers start from zero, differ more on main, with the same iteration counts and final
residual norms (5e-12) on both. Not from io4dolfinx: 0.11 with io4dolfinx 1.5.0 gives the deviations of
0.11 with 1.4.0 exactly (1.4.0 does not run on main).

- Same ranks, restart and initial state: `interface_u_gap` changes in its last printed digit, 1.4e-6 of its
  largest value, against the default `qoi_rtol = 1e-10`. The gaps are measured on u_f, which may change by
  1e-5 on restart, so they now get that tolerance.
- Initial state: u_f, still small (largest 8e-8), differs between the two runs from the initial state by
  2.8e-5 relative (2.3e-12, at the final residual) on main and 1.9e-7 on 0.11; the other fields by
  1e-9 or less on main, 100 times more than on 0.11. u_f compared to 1e-4, as across partitions.
- 2 to 1 ranks: the u gap after the first step is 1.1e-10 on one rank and 2.4e-10 on two, 3e-2 of its
  largest value, against 1e-2; zero up to the accuracy of the solve on both. Gaps compared to 1e-1.

Not run: the slow 3D tests, which cover the same fix in `threeDfsi_mixed` and `threeDfsi_c0ip`.
