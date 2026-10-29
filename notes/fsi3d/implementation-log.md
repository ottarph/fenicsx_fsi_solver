# 3d FSI benchmark of Failer & Richter: implementation log

Goal: the 3d benchmark of Failer & Richter (J. Sci. Comput. 82:28, 2020,
Sect. 5.1.2, Table 1, Fig. 1 and Fig. 3), solved with the formulation and the
solver of `solvers/fsi2_biharmonic_diffmesh_restricted_split_condensed.py`.
The paper is in `references/` (not committed).

## Benchmark (2026-10-09)

- Channel 0 < x < 2.8, 0 < y < 0.41, 0 < z < 0.41, with the cylinder
  (x - 0.5)^2 + (y - 0.2)^2 < 0.05^2 cut out over its full height. Elastic
  beam 0.5 < x < 0.9, 0.19 < y < 0.21, 0.1 < z < 0.3, without the cylinder,
  clamped where it touches the cylinder.
- rho_f = rho_s = 1000, nu_f = 1e-3, mu_s = 2e6, lambda_s = 8e6 (Table 1;
  the 2d FSI2 values of the existing solvers differ).
- Inflow v_x = U_bar * 36 y (H - y) z (H - z) / H^4, U_bar = 1.75, H = 0.41,
  ramped by 1/2 - 1/2 cos(pi t) for t < 1 (the 2d FSI2 ramp is over t < 2).
- QoIs: displacement of B = (0.9, 0.2, 0.3) and drag and lift on the cylinder
  and the beam; reference values as mean ± amplitude over t in [9, 10] in
  Fig. 3, for k = 0.004, 0.002 and 0.001 on three mesh levels.
- The paper uses hexahedral Q2 elements for all unknowns with LPS
  stabilization (63,826 dofs on level 1). Here: tetrahedra with P2 u, v, z and
  P1 p (Taylor-Hood), as in 2d.

## Mesh: `scripts/create_mesh_FSI3D.py`

- gmsh OCC: cut the cylinder out of the channel and the beam, fragment them so
  that the interface is conforming. The cell and facet tags are those of the
  2d FSI2 mesh, so the solver's `PHYSICAL_MARKERS` are unchanged; the four
  walls are `channel_side`.
- B lies in the middle of the beam edge x = 0.9, z = 0.3, which is not a mesh
  node in general. Fragmenting the point B into the geometry gives a vertex
  there, so `find_point_dof` finds it as in 2d.
- OCC pads bounding boxes by 1e-7, so the wall test uses `atol=1e-6`.
- Size field: Threshold on the distance to the beam and the cylinder,
  `resolution_close` on them up to `resolution_far` at a distance of 0.2.
- Second order geometry with gmsh's high-order optimization. The coarse mesh
  (`data/meshes/fsi3d/mesh_sec_coarse.xdmf`, 0.02 / 0.08) has 15,142 tets
  (1,624 in the beam, about one cell through its thickness) and 3,386
  vertices. Checked: positive det J at degree-6 quadrature points in all
  cells, beam volume 0.0014013 and fluid volume 0.46606 as computed by hand,
  obstacle area 0.12478.

## Solver: `solvers/threeDfsi_iterative.py`

A copy of the 2d solver with the 3d changes only: function spaces with 3
components, the 3d inflow and parameters, 6 rigid body modes (3 translations,
3 rotations) for the GAMG near-nullspace, block size 3, the point B and a
QoI header with B_x, B_y, B_z (`QOI_HEADER_3D`). `drag_lift_forms` now takes
the drag along -e_x and the lift along e_y in any dimension, unchanged in 2d.

### Quadrature degree

The first runs took 42 s per step on 4 ranks, 17 s on 8 ranks, with
2 Newton iterations per step and a single Jacobian assembly for the whole run.
`-log_view` (3 steps, 8 ranks, iterative K_c): 50 s of 71 s in 9 residual
evaluations, 7 s in one Jacobian assembly, 3 s in all linear solves. UFL
estimates degree 19 for the nonlinear ALE and St. Venant-Kirchhoff terms of
the momentum and continuity residuals (25 in the v-v Jacobian block), while the
u and z blocks stay at 5 to 7. Quadrature points per tetrahedron: degree 8:
45, degree 13: 146, degree 16: 729, degree 19: 1000, degree 25: 2197.

`tools/quadrature.py`: `cap_quadrature_degree(form, max_degree)` sets the
quadrature degree to `max_degree` in the integrals whose estimate, computed
with `ufl.algorithms.compute_form_data` as in
https://jsdokken.com/FEniCS-workshop/src/unified_form_language/ufl_forms.html,
is higher, and leaves the others alone. The solver caps the residual blocks
before `derivative_block`, so the Jacobian blocks inherit the cap (checked: no
block above 8), plus `residual_post_u`, A_0 and the drag/lift forms.
`max_quadrature_degree=8` by default.

With the cap, 8 ranks, 5 steps of dt = 0.004:

| K_c solve | first step | later steps |
|---|---|---|
| MUMPS LU | 15 s | 1.2 s |
| FGMRES, block upper Schur, GAMG + Cahouet-Chabard | 12 s | 1.5 s |

The first step includes the MUMPS factorization of K_m (and of K_c in the
direct solve). The iterative K_c solve takes about 22 FGMRES iterations in the
first Newton iteration of a step and about 90 to 110 in the second. On this
coarse mesh the direct K_c solve is as fast; the iterative one should pay off
on finer meshes. The QoIs over these 5 steps agree with the uncapped run to all 6
printed digits, but the deformation is still tiny (|u_B| ~ 1e-6), so this
says little about the effect of the cap at large deformations.

### Teardown error (fixed)

`CahouetChabard.destroy()` clashed with petsc4py's `destroy(pc)` hook for
python PC contexts, which raised an ignored `TypeError` when the PC was
deallocated (in 2d too). Harmless, since the explicit cleanup had run by then.
Fixed by renaming the explicit cleanup to `destroy_solvers()`, so petsc4py
finds no hook; `tests/test_cahouet_chabard.py` destroys a python PC with the
context.

## Non-convergence on 26 or more ranks (2026-10-09)

Block-preconditioned FGMRES on K_c with GAMG on A (built from A, Richardson +
SOR smoothing) and BoomerAMG on selfp, coarse mesh, first step:

| ranks | 1 | 4 | 8 | 16 | 24 | 26 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|
| FGMRES its (Newton 1, 2) | 30, 26 | 28, 25 | 42, 30 | 70, 56 | 54, 43 | > 200 | > 200 | > 200 |

The FGMRES limit (200) is reached in all 20 Newton iterations from 26 ranks
on. Diagnosis:

- No rank is empty: all ranks have fluid cells and pressure dofs.
- MUMPS LU on A instead of GAMG: 7 and 10 FGMRES iterations on 26 ranks.
  MUMPS LU on the pressure part instead of BoomerAMG: still > 200. So GAMG on
  A is at fault.
- GMRES on A alone, preconditioned by GAMG (rtol 1e-8): 44 iterations on 24
  ranks, no convergence in 300 on 26 ranks.
- `-info :pc`: the GAMG hierarchies on 24 and 26 ranks are similar (6 levels,
  coarsest 36 and 24 unknowns), and `pc_gamg_coarse_eq_limit 1000` does not
  help, so the coarse grid is not the cause.
- Smoothing: PETSc's parallel SOR is Gauss-Seidel within each rank and Jacobi
  between ranks, which can diverge where the coupling across a rank boundary
  is strong. The beam is about one cell thick, its rows are dominated by the
  solid stiffness mu_s theta^2 dt, and on 26 ranks it is split across about
  12 ranks. With a smoother that does not depend on the partition, or with
  damped SOR, the problem goes away:

| ranks | 1 | 8 | 16 | 20 | 24 | 26 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|
| Chebyshev + Jacobi on A | 47, 40 | 48, 43 | 46, 44 | 46, 41 | 47, 41 | 48, 46 | 43, 37 | 45, 40 |
| Richardson (scale 0.5) + SOR | 44, 36 | 46, 39 | 46, 40 | 43, 35 | | 49, 41 | 43, 33 | 43, 36 |
| `auxiliary_vv_preconditioner=True` | 47, 40 | | | | 47, 41 | 48, 46 | | 45, 40 |

`auxiliary_vv_preconditioner=True` builds GAMG from the SPD A_0 with
Chebyshev + Jacobi smoothing (in the block-preconditioned solve). That smoother
does not depend on the partition and does not have Chebyshev's problem with
the nonsymmetric A. Its iteration counts match Chebyshev on A here, since A_0 ≈ A
while the flow is still slow. With Cahouet-Chabard as well, 5 steps on 26 and
30 ranks converge, with about 25 FGMRES iterations in the first Newton
iteration of a step and 100 to 110 in the second (selfp + BoomerAMG: about 45).

The 2d solver uses the same undamped SOR in its iterative K_c solves, so the
same partition dependence may appear there.

## Rank scaling of the K_c solves (2026-10-09)

Coarse mesh, 8 steps of dt = 0.004 from t = 0, one Jacobian assembly per run.
Time per step after the first, which also holds the MUMPS factorization of K_m
(and of K_c in the direct solve) and the AMG setups:

| ranks | MUMPS LU on K_c | A_0 + selfp/BoomerAMG | A_0 + Cahouet-Chabard |
|---|---|---|---|
| 4 | 1.84 s | 2.51 s | 3.68 s |
| 8 | 1.04 s | 1.44 s | 1.99 s |
| 16 | 0.64 s | 0.93 s | 1.47 s |
| 24 | 0.50 s | 0.79 s | 1.33 s |
| 30 | 0.43 s | 0.73 s | 1.28 s |

First step: 24 s on 4 ranks, 8 to 10 s on 16 to 30 ranks, for all three. FGMRES
iterations on 30 ranks (Newton 1, 2 of a step): about 23, 45 with selfp and about
25, 100 with Cahouet-Chabard, which also needed a third Newton iteration in one
step. The direct K_c solve does not use GAMG, so it does not have the smoother
problem above. These are the first steps only, with small deformations, on the
coarse mesh. On finer meshes the cost and memory of the MUMPS factorizations grow
faster than those of the iterative solves.

## Coarse run to T = 10, and finer meshes (2026-10-09)

The full coarse run (30 ranks, direct K_c, dt = 0.004) finished in 2331 s
(0.93 s per step, 11,606 Newton iterations and 46 Jacobian assemblies in 2500
steps), but does not reproduce the benchmark: the beam settles to a steady
state by t = 6 (over [9, 10]: drag 154.7, B_y 4.28e-3, both with no
oscillation; paper, level 3, k = 0.001: drag 185.5 ± 3.5, B_y 2.70e-3 ±
25.6e-3). Even the paper's level 1 oscillates. Not investigated; the beam has
about one cell through its thickness, and the channel about 5 cells across.

Finer meshes from `create_mesh_FSI3D.py` (second order, new suffixes):

| mesh | close / far | cells | beam cells | K_c (v, p) | K_m (z, u_f) | total |
|---|---|---|---|---|---|---|
| coarse | 0.02 / 0.08 | 15,142 | 1,624 | 73,679 | 132,792 | 210,440 |
| medium | 0.01 / 0.04 | 91,989 | 7,133 | 421,968 | 769,110 | 1,210,494 |
| fine | 0.0067 / 0.03 | 233,344 | 17,605 | 1,044,697 | 1,902,378 | 2,997,088 |

Meshing took 28 s and 86 s. The paper's levels have 9.1k, 66k and 504k Q2
nodes with 7 unknowns each; medium has about 120k and fine about 300k P2
nodes, with 10 unknowns in the fluid.

Medium mesh, 30 ranks, 3 steps (the machine has 125 GB):

- Direct K_c: stopped by hand in the first step, with 95 GB in the solver
  ranks and 16 GB left on the machine, during the MUMPS factorizations of K_c
  and K_m.
- A_0 + selfp/BoomerAMG K_c: first step 122 s, of which 107 s is the MUMPS
  factorization of K_m (`MatLUFactorNum`), later steps 5.5 s with 2 Newton
  iterations. FGMRES iterations (Newton 1, 2): 52, 48, then about 30, 55,
  close to the coarse mesh. Peak memory, summed over the ranks: 92 GB (at most
  4.3 GB on a rank).

So on this machine the medium mesh only fits with the iterative K_c solve.

### K_m factorization: mixed vs C0IP, vector vs one component

Measured standalone (scratch script, not in the repository): K_m assembled on
the fluid submesh with u = 0 on its whole boundary, as the mixed biharmonic of
this solver or as the C0IP biharmonic of `fsi2_biharmonic_c0ip.py` (branch
`restricted-c0ip-mm`, with a constant penalty 30 / 0.02 instead of the
triangle penalty parameters, which have no tetrahedron version yet; the
penalty does not change the sparsity), and factorized by MUMPS on 30 ranks
with the solver's options. Memory is MUMPS's INFOG(22), summed over the ranks.

| medium mesh | unknowns | nnz/row | MUMPS memory | factorization |
|---|---|---|---|---|
| mixed, vector, LU (now) | 769,110 | 160 | 41.7 GB | 97 s |
| C0IP, vector, LU | 384,555 | 185 | 30.9 GB | 68 s |
| C0IP, vector, Cholesky | 384,555 | 185 | 20.4 GB | 31 s |
| mixed, one component, LU | 256,370 | 53 | 7.1 GB | 12.6 s |
| C0IP, one component, LU | 128,185 | 62 | 5.7 GB | 8.2 s |
| C0IP, one component, Cholesky | 128,185 | 62 | 4.4 GB | 3.8 s |

On the coarse mesh the overhead of MUMPS on 30 ranks (about 2.5 GB) hides
the differences; factor entries there: mixed vector LU 0.27e9, C0IP vector LU
0.17e9, C0IP vector Cholesky 0.09e9.

- K_m is 42 GB of the 92 GB peak of the medium run with the iterative K_c
  solve; the other 50 GB are the rest of the solver (the Jacobian, the
  separate A_0 preconditioning matrix, the fieldsplit submatrices, GAMG, and
  the per-rank overhead), not split up further.
- The biharmonic does not couple the x, y and z components, and the boundary
  conditions are the same for all three, so one scalar factorization serves
  all three components. The blocked sparsity pattern (block size 3) stores
  the zero couplings, which MUMPS factorizes too: the vector matrices have
  three times the nonzeros per row of the scalar ones.

### Medium mesh without the auxiliary preconditioner

30 ranks, 3 steps, iterative K_c without A_0 (GAMG built from A, Richardson +
SOR smoothing), compared with the A_0 run above (92 GB):

- Undamped SOR (the default): the K_c FGMRES reaches its limit of 200 in every
  Newton iteration of the first step, as on the coarse mesh from 26 ranks on.
- SOR damped by `fieldsplit_v_mg_levels_ksp_richardson_scale 0.5` (scratch
  copy of the solver): converges, FGMRES iterations 50, 47, then about 28,
  51, with 2 Newton iterations per step; 134 s for the 3 steps, as with A_0.
  Peak memory summed over the ranks: 83 GB, so dropping the separate A_0
  matrix saves about 9 GB.

Runtime estimate for medium to T = 10 with dt = 0.004 on 30 ranks (iterative
K_c, MUMPS K_m): later steps take 5.5 s with 2 Newton iterations. Over the
full coarse run, a step took 2.2 times as long as the early steps (4.6 Newton
iterations per step on average, 46 Jacobian assemblies). With the same
factor: about 12 s per step, 8 to 9 hours for 2500 steps. If the beam
oscillates on the medium mesh, which it did not on the coarse one, Newton and
FGMRES iterations may grow further.

### Damped SOR and K_m by component (implemented)

- GAMG's SOR smoothing on the velocity block is damped by 0.5 in both K_c
  solves (`sor_smoothing_options`). Checked: coarse mesh on 26 ranks, which
  failed undamped, converges in the block-preconditioned solve (FGMRES 49, 41,
  then about 25, 47) and in the full Schur solve (GMRES on A about 35
  iterations).
- K_m is solved by an additive fieldsplit over the x, y and z components, with
  one MUMPS factorization each. The component of an owned dof is its position
  in its block modulo 3; the index sets are numbered within the m submatrix.
  Exact: the first 5 coarse steps with the direct K_c solve reproduce the QoIs
  of the earlier run to all printed digits. `MatLUFactorNum` count 4 (K_c and
  the three components).
- Medium mesh, 30 ranks, iterative K_c without A_0, 3 steps: steps of 45 s,
  5.8 s and 5.7 s, 2 Newton iterations each, FGMRES 50, 47, 28, 52, 27, 50.
  Peak memory summed over the ranks 41 GB (at most 1.8 GB on a rank), against
  92 GB with A_0 and K_m factorized as a whole; at least 71 GB stayed free on
  the machine.

## Medium run to T = 10 (2026-10-12)

28 ranks, iterative block-preconditioned K_c without A_0 (damped SOR),
K_m by component, dt = 0.004: 49,003 s (13.6 h, 19.6 s per step), no failed
solves, 17,558 Newton iterations (7.0 per step; most steps 7 or 8, at most 10)
and 1762 Jacobian assemblies over 2500 steps. FGMRES: 35.1 iterations per
Newton iteration on average, at most 45.

The beam oscillates now, at 5.68 Hz in B_y. Over [9, 10]:

| | medium, dt = 0.004 | paper, level 3, k = 0.001 |
|---|---|---|
| drag | 171.3 ± 3.34 | 185.5 ± 3.5 |
| lift | 1.56 ± 40.3 | -0.72 ± 41.3 |
| B_x | -1.92e-3 ± 2.16e-3 | -2.14e-3 ± 2.38e-3 |
| B_y | 0.85e-3 ± 26.1e-3 | 2.70e-3 ± 25.6e-3 |
| B_z | 0.25e-3 ± 0.69e-3 | 0.49e-3 ± 0.88e-3 |

### log_view

2.79 s per Newton iteration over the run, the same as in the first steps
(5.7 s per step with 2 Newton iterations), so the slowdown from the first
steps is the number of Newton iterations. Events (max over ranks; they
overlap, so they do not add up):

| event | time | share |
|---|---|---|
| KSPSolve (whole linear solve) | 35,990 s | 73% |
| MatMult | 16,030 s | 33% |
| MatSOR | 13,272 s | 27% |
| VecScatterEnd (waiting for messages, ratio 4.4) | 11,019 s | 22% |
| SNESFunctionEval (20,058 calls, 0.45 s each) | 9,033 s | 18% |
| MatResidual + MatMultAdd + MatMultTranspose (multigrid) | 5,626 s | 11% |
| MatSolve (K_m by MUMPS, and the GAMG coarse grid) | 3,765 s | 8% |
| SNESJacobianEval (1.15 s per assembly) | 2,032 s | 4% |
| PCSetUp_GAMG+ + MatCreateSubMat + PCSetUp (1.3 s per assembly) | 2,261 s | 5% |
| KSPSolve_FS_Schu (BoomerAMG on selfp) | 1,548 s | 3% |
| MatLUFactorNum (K_m and GAMG coarse grid) | 26 s | 0.05% |

10.7 Gflop/s in total, 0.38 Gflop/s per rank: bound by memory bandwidth,
as sparse matrix work usually is, which fits the flat rank scaling on the
coarse mesh. MatMult time ratio 1.2 between ranks.

The damped Richardson smoother needs a separate MatMult for its residual in
every smoothing step: per V-cycle (one per FGMRES iteration), undamped SOR
(coarse, 24 ranks) makes about 9 MatMult and 10 MatSOR calls, damped (coarse,
28 ranks) about 20 and 16. MatMult and MatSOR together are 60% of the medium
run. Damping SOR itself (`pc_sor_omega`) instead of the Richardson iteration
might keep PETSc's combined SOR smoothing; not tested, and its robustness
across partitions would have to be checked again.

### Scaling to the fine mesh

Cost per call of the same events, 3 coarse steps (28 ranks, same settings)
against the medium run, as cost ~ unknowns^alpha (coarse to medium: 5.75x
the unknowns; medium to fine: 2.48x):

| event | alpha | fine / medium |
|---|---|---|
| MatMult | 0.97 | 2.40 |
| MatSOR | 1.05 | 2.58 |
| MatResidual | 1.11 | 2.74 |
| MatSolve | 1.00 | 2.47 |
| SNESFunctionEval | 0.91 | 2.28 |
| Jacobian assembly (per assembly) | 1.00 | 2.48 |
| KSPSolve_FS_Schu | 0.44 | 1.48 |

FGMRES iterations per Newton iteration in the first steps: coarse about 34,
medium about 42 (alpha 0.13, so about 13% more on fine). Estimate for fine
on 28 ranks: about 2.7 times the cost per Newton iteration, about 7.6 s; with
7 to 8 Newton iterations per step, 53 to 61 s per step, 37 to 42 h for 2500
steps at dt = 0.004. The Newton iterations per step on fine are not known:
they depend on the dynamics, which changed from coarse (no oscillation, 4.6)
to medium (oscillation, 7.0). Memory is not in the log; the medium mesh used
41 GB (3 steps, 30 ranks). With the parts that grow linearly (x2.5) and the
MUMPS factors of K_m (about unknowns^(4/3) in 3d, x3.4), fine needs about
100 to 115 GB of the 125 GB, to be measured.

### SOR damping by omega instead of Richardson scaling (2026-10-12)

`fieldsplit_v_mg_levels_ksp_richardson_scale 1.0` with
`fieldsplit_v_mg_levels_pc_sor_omega` 0.5 or 0.7, against the current
Richardson scale 0.5 (scratch copy of the solver, iterative K_c without A_0).
First steps only (2 Newton iterations per step); s/Newton is the time per
Newton iteration after the first step; MG ms/V is the time of MatMult,
MatSOR, MatResidual, MatMultAdd and MatMultTranspose per V-cycle.

| mesh, ranks | smoother | converged | FGMRES iterations | MatMult / MatSOR per V-cycle | MG ms/V | s/Newton |
|---|---|---|---|---|---|---|
| coarse, 24 | Richardson 0.5 | yes | 43 36 26 43 | 23.7 / 20 | 6.29 | 0.43 |
| coarse, 24 | omega 0.5 | yes | 37 29 22 37 | 9.0 / 10 | 4.60 | 0.36 |
| coarse, 24 | omega 0.7 | yes | 45 33 26 44 | 8.7 / 10 | 4.67 | 0.39 |
| coarse, 26 | Richardson 0.5 | yes | 49 41 26 50 | 23.5 / 20 | 6.21 | 0.44 |
| coarse, 26 | omega 0.5 | yes | 40 33 22 41 | 8.9 / 10 | 4.85 | 0.37 |
| coarse, 26 | omega 0.7 | yes | 52 42 30 51 | 8.5 / 10 | 4.79 | 0.41 |
| coarse, 28 | Richardson 0.5 | yes | 43 33 26 39 | 19.7 / 16 | 6.37 | 0.39 |
| coarse, 28 | omega 0.5 | yes | 32 29 20 35 | 8.1 / 8 | 4.67 | 0.32 |
| coarse, 28 | omega 0.7 | yes | 90 90 55 106 | 6.7 / 8 | 4.47 | 0.60 |
| coarse, 30 | Richardson 0.5 | yes | 43 36 26 43 | 23.7 / 20 | 6.02 | 0.39 |
| coarse, 30 | omega 0.5 | yes | 37 30 22 37 | 9.0 / 10 | 4.49 | 0.33 |
| coarse, 30 | omega 0.7 | **no** | 200 200 | | | |
| medium, 28 | Richardson 0.5 | yes | 42 34 27 38 25 36 | 31.8 / 28 | 58.8 | 2.53 |
| medium, 28 | omega 0.5 | yes | 35 28 22 33 19 32 | 11.2 / 14 | 57.0 | 2.23 |
| medium, 28 | omega 0.7 | yes | 33 29 24 32 21 31 | 11.2 / 14 | 57.2 | 2.24 |
| medium, 30 | Richardson 0.5 | yes | 50 47 28 52 27 50 | 27.4 / 24 | 58.9 | 2.87 |
| medium, 30 | omega 0.5 | yes | 41 36 23 43 21 41 | 9.7 / 12 | 56.7 | 2.46 |
| medium, 30 | omega 0.7 | yes | 57 43 36 50 32 51 | 9.3 / 12 | 56.9 | 2.95 |

- With the Richardson scale at 1, PETSc applies SOR directly: the MatMult
  calls per V-cycle drop from 20 to 32 to 7 to 11.
- On the coarse mesh this makes a V-cycle 25% cheaper. On the medium mesh a
  V-cycle costs about the same (57 against 59 ms); the remaining work moved
  into MatSOR, whose calls now hold two sweeps each and their parallel
  updates. The gain there comes from fewer FGMRES iterations with omega 0.5.
- omega 0.5: converged everywhere, 12 to 14% less time per Newton iteration
  on the medium mesh, 16 to 18% on the coarse mesh.
- omega 0.7: not robust. Diverged on 30 ranks and needed 90 to 106 FGMRES
  iterations on 28 ranks on the coarse mesh, and more iterations than the
  current damping on 30 ranks on the medium mesh.
- Only the first steps; the developed oscillation is not tested.

#### With the beam oscillating: restart from t = 9

The same comparison over 10 steps from the medium run's checkpoint at t = 9
(step 2250, copied to a scratch directory and restarted with
`checkpoint_every=None`, so neither checkpoint was changed). Time per step
over steps 2 to 10:

| ranks | smoother | Newton per step | FGMRES per Newton | s/step | s/Newton |
|---|---|---|---|---|---|
| 28 | Richardson 0.5 | 8 7 9 7 7 8 8 8 7 7 | 35.1 | 21.0 | 2.78 |
| 28 | omega 0.5 | same | 27.3 | 17.7 | 2.34 |
| 28 | omega 0.7 | same | 30.3 | 18.9 | 2.51 |
| 30 | Richardson 0.5 | same | 39.7 | 22.2 | 2.93 |
| 30 | omega 0.5 | same | 32.0 | 19.0 | 2.51 |
| 30 | omega 0.7 | same | 40.8 | 22.1 | 2.92 |

- All runs converged, with the same Newton iterations in every step, and
  reproduce the QoIs of the full run to the 6 printed digits.
- 28 ranks restart in 2.78 s per Newton iteration, the average of the full
  run (2.79 s), so the window is representative.
- omega 0.5: 22% (28 ranks) and 19% (30 ranks) fewer FGMRES iterations, 16%
  and 14% less time per step. MatMult per V-cycle 9.9 against 27.5; the
  V-cycle itself costs the same (57 ms), so the gain is the iterations.
- omega 0.7: 10% faster on 28 ranks, no faster on 30 ranks.

#### Switched to omega = 0.5

`sor_smoothing_options` now uses `pc_sor_omega 0.5` with the Richardson
scale at 1 (the default). Checked: the coarse mesh on 30 ranks reproduces the
FGMRES iterations of the scratch experiment (37 30 22 37); the full Schur
solve, which uses the same smoother, converges on 26 and 30 ranks (25 to 36
GMRES iterations on A).

## One K_m factorization for all components (2026-10-12)

K_m does not couple the displacement components. Every component also has the
same matrix: the forms are component-wise vector Laplacians and every boundary
condition on the mesh motion (`u_f_bc`, `u_s_bc`, `bc_deactivate`) constrains
all components. The component index sets keep the dofs of each rank in node
order, so each component's submatrix also has the same parallel layout. Only the
x split factorizes now (MUMPS LU). The y split, and in 3d the z split, applies
that factorization through `tools/petsc_reuse.py:ReusePC`, a PCPYTHON context
whose `apply` calls the x split's PC. The context only holds a reference, so the
x split still owns and destroys the factorization. The 2d solver
(`fsi2_biharmonic_diffmesh_restricted_split_condensed.py`) previously
factorized K_m as a whole; it now uses the same component split and reuse.
`ICNTL(14) = 80` is left out (commented in the options), so MUMPS uses its
default workspace margin.

Measured from checkpoints with `checkpoint_every=None` (scratch copies, so
neither checkpoint changed), 10 steps per run. The other variants come from a
scratch copy of the solver with the reuse switched off: "whole" is the
committed 2d solver (K_m as one MUMPS LU, `ICNTL(14) = 80`), and "split"
factorizes every component (same MUMPS options as the x split). Peak RSS is
summed over the ranks. MUMPS memory is INFOG(22), summed over the K_m
factorizations; on 16 ranks it is mostly a fixed amount per rank (about 86 MB),
so peak RSS is the better measure there.

| case | ranks | K_m | runs | total (s) | step 1 (s) | steps 2-10 (s/step) | peak RSS (GB) | MUMPS K_m (MB) |
|---|---|---|---|---|---|---|---|---|
| 2d `mesh_sec`, t = 16 | 16 | whole | 3 | 1.52 | 0.431 | 0.120 | 3.53 | 1465 |
| | | split | 3 | 1.57 | 0.469 | 0.122 | 3.50 | 2752 |
| | | reuse | 3 | 1.45 | 0.382 | 0.118 | 3.44 | 1376 |
| 2d `mesh_fine_sec`, t = 16 | 16 | whole | 2 | 5.38 | 1.754 | 0.402 | 4.81 | 1875 |
| | | split | 2 | 5.38 | 1.888 | 0.388 | 4.63 | 2981 |
| | | reuse | 2 | 4.92 | 1.517 | 0.378 | 4.47 | 1489 |
| 3d coarse, t = 9 | 28 | split | 2 | 14.23 | 3.40 | 1.203 | 9.37 | 7327 |
| | | reuse | 2 | 13.16 | 2.29 | 1.206 | 8.32 | 2442 |
| 3d medium, t = 9 | 28 | split | 1 | 223.7 | 55.3 | 18.71 | 37.05 | 19417 |
| | | reuse | 1 | 200.1 | 31.8 | 18.69 | 25.20 | 6812 |

- Every variant gives the same Newton iterations in every step and the same
  QoIs to all printed digits.
- 2d fine: these runs reproduce the user's own runs on 16 ranks, which took
  4.90 s with reuse and 5.43 s without.
- 3d: the savings are the two extra factorizations, as setup time and memory.
  Reuse saves 1.1 s (coarse) and 23.5 s (medium) in the first step:
  MatLUFactorSym + MatLUFactorNum take 11.5 s instead of 35.0 s on the medium
  mesh. Later steps cost the same, since every component still needs its own
  triangular solves. Peak memory drops by 1.05 GB (11%, coarse) and 11.9 GB
  (32%, medium).
- Fine 3d mesh (estimate, not measured): with factor memory growing as about
  N^(4/3) for 2.48 times the unknowns, one component needs about 23 GB of MUMPS
  memory there. Reuse would then save about 40 GB of the 100 to 115 GB
  estimated above, and a few minutes of setup.
- 2d: the totals are small. On `mesh_sec`, reuse saves 0.05 s in the first
  step against "whole", 1 to 2% per later step and 0.09 GB. On the fine mesh
  it saves 0.24 s in the first step, 6% per later step against "whole" (3%
  against "split") and 0.34 GB (7%). Over a full 2d run, the gain is the per-step
  part; the per-step gain against "whole" comes mostly from the smaller
  component solves, without the zero couplings between the components.

## C0IP mesh motion assembled once (2026-10-20)

`threeDfsi_c0ip` (Phase 7 of `notes/restricted-cleanup/implementation-log.md`)
took 3.2 times as long per step as `threeDfsi_mixed` on the coarse mesh, with
a residual evaluation of 0.94 s. Timing the assembly calls showed that the
residual forms take 0.035 s of that; the rest is the lifting by the boundary
conditions in dolfinx's `assemble_residual`, which integrates the Jacobian
forms, including the C0IP facet terms of the u-u block, in every residual
evaluation. Removing the C0IP terms from the residual forms alone changed
nothing.

The C0IP mesh motion is linear in u and integrated on the reference mesh, so
it is now assembled once, as B_m, the u-u block of a matrix with the layout of
the Jacobian (rows and columns of `u_f_bc` zeroed, zero diagonal there):

- Residual: the residual forms and the lifting get a zero form in the u-u
  block, and `post_residual` adds B_m x on the u rows that no boundary
  condition or test function restriction sets. `u_f_bc` is homogeneous, so the
  zeroed columns hold its lifting.
- Jacobian: the first assembly integrates the full forms, since the matrix
  only keeps the entries of its first assembly and the facet couplings are in
  no other block. Reassemblies use the zero form and add B_m by `MatAXPY`
  (`SUBSET_NONZERO_PATTERN`).

Coarse mesh, 14 ranks, iterative K_c, against the committed version and
`threeDfsi_mixed` in the same session:

| case | before | B_m | mixed |
|---|---|---|---|
| from rest, 20 steps (s/step) | 3.82 | 1.07–1.14 | 1.13 |
| from t = 10 (mixed checkpoint), 10 steps (s/step) | 4.95 | 1.50–1.59 | 1.47 |
| from t = 10, `gamma_reassemble=0`, 24 Jacobian assemblies (s/step) | 8.26 | 2.35 | |
| residual evaluation (s) | 0.94 | 0.155 | 0.172 |
| Jacobian reassembly (s) | 1.32 | 0.23 | 0.36 |

- Exact up to round-off: the same Newton iterations and Jacobian assemblies,
  the residual norms agree to 7 to 10 digits where they are above 1e-6, and
  the QoI files are identical, in all three cases.
- The C0IP solver now costs the same per step as the mixed one, slightly less
  in the residual and the Jacobian, which have no auxiliary field.
- B_m is a second copy of the u-u block's nonzeros: about 0.9 GB summed over
  the ranks on the medium mesh, estimated from 185 nonzeros per row. On the
  medium mesh the peak memory went down instead (below).
- The remaining lifting (0.12 of the 0.155 s per residual) is by the velocity
  boundary conditions, with the Jacobian blocks of the v columns. The mixed
  solver lifts the same way; not changed here.

### Medium mesh

28 ranks, 10 steps from the medium mixed checkpoint at t = 10 (scratch copies,
so the checkpoint did not change), with the beam oscillating:

| | before | B_m | mixed |
|---|---|---|---|
| s/step | 38.1 | 18.9 | 21.1 |
| residual evaluation (s) | 2.28 | 0.41 | 0.45 |
| Jacobian assembly (s, 11 assemblies) | 4.27 | 1.06 | 1.16 |
| K_m factorization (s) | 1.3 | 1.4 | 9.9 |
| MatSolve, all calls (s) | 8.8 | 9.1 | 16.6 |
| peak RSS, summed over the ranks (GB) | 21.5 | 19.9 | 26.3 |

- All three: 74 Newton iterations and 11 Jacobian assemblies, 32.4 FGMRES
  iterations per K_c solve on average.
- B_m against before: the QoI files are identical, and the residual norms
  agree to 1.3e-6 relative where they are above 1e-6. The mixed QoIs differ
  by up to 2e-4 relative, since the mesh motion differs.
- The Jacobian assembly with B_m is an average over the first assembly, which
  still integrates the C0IP forms (about 4.3 s), and 10 reassemblies (about
  0.74 s each).
- The C0IP solver is now 10% faster per step than the mixed one, and uses
  6.4 GB (24%) less memory: the K_m Cholesky factor of one component is
  smaller than the LU factor of the mixed form, and the auxiliary field is
  gone. MatSolve also includes GAMG's coarse grid solves, which are the same
  in both.
- Peak memory dropped by 1.6 GB against before, despite the extra B_m;
  probably the coefficient packing of the C0IP facet forms in every lifting,
  not checked.
- MatMult took 34.2 s with B_m, against 28.4 s before and 27.5 s mixed. The
  B_m products (84, one per residual) are part of it; not split up further.

## K_m components solved at once by MatMatSolve (2026-10-20)

`threeDfsi_c0ip` solved the three components of K_m one after the other with
the same MUMPS factorization (`ReusePC`), and each triangular solve reads the
whole factor. `tools/petsc_reuse.py:IdenticalBlocksPC` now factorizes the x
component and solves for all three at once: it copies the components of the
right-hand side into the columns of a dense matrix and calls `MatMatSolve`,
which reads the factor once for the three columns. The components of the owned
dofs on a rank are the same nodes in the same order, so the copies are local.
The m split is a PCPYTHON with this context instead of the additive fieldsplit
over the components; the factorization is set up from the options with the
prefix `<m prefix>block_`.

10 steps from the mixed checkpoint at t = 10, against the B_m version above:

| mesh, ranks | | K_m solves per Newton iteration | s/step | peak RSS (GB) |
|---|---|---|---|---|
| coarse, 14 | 3 MatSolve | 14.8 ms | 1.48 | |
| | MatMatSolve | 8.5 ms | 1.47 | |
| medium, 28 | 3 MatSolve | 123 ms | 18.9 | 19.9 |
| | MatMatSolve | 51 ms | 18.4 | 19.6 |

- The same Newton iterations and Jacobian assemblies, identical QoI files,
  and residual norms agreeing to 1.3e-6 relative where they are above 1e-6.
- 43% (coarse) and 58% (medium) less time in the K_m solves, against the 67%
  of reading the factor a third as often; the rest is per-solve overhead,
  which weighs more on the small coarse factor.
- 3% per step on the medium mesh, since K_m solves were 5% of it. The share
  grows with the factor, about N^(4/3), so the gain is larger on finer meshes.
- `tests/test_petsc_reuse.py` checks the context on three interleaved 1d
  Laplacians, in serial and on 3 ranks.
- Generalized afterwards: the context applies the PC of the first block by
  `PCMatApply` instead of calling `MatMatSolve` on its factor, so it takes any
  PC. For Cholesky that is the same single `MatMatSolve` (checked: the same
  event counts and identical QoIs on the coarse mesh); GAMG runs one V-cycle
  on all columns, with a `MatMatSolve` on its coarse grid; PCs without their
  own `PCMatApply`, such as Jacobi, are applied column by column. The test also
  covers a PCKSP with CG and Jacobi.
- All solvers that split K_m into components now use `IdenticalBlocksPC`:
  `fsi2_biharmonic_c0ip` (Cholesky), `fsi2_biharmonic_mixed`,
  `fsi2_biharmonic_mixed_incompressible`, `component_benchmarks/fsi2D_benchmark`
  and `threeDfsi_mixed` (LU). Checked against the previous version on the
  coarse meshes (6 steps in 2d, 3 in 3d): the same Newton iterations, residual
  norms agreeing to 1e-7 relative or better, and identical QoIs, apart from the
  round-off (1e-17) of the incompressible solver's solid area change, which is
  itself about 3e-16. `component_solvers/c0ip_mesh_motion.py` keeps the
  fieldsplit with `ReusePC`: its CG variants run a Krylov method per
  component, and its logged timings were measured with it.

## K_m on fewer ranks by PCTELESCOPE (2026-10-20)

PCTELESCOPE has no `PCMatApply` (PETSc 3.25.5, also not on `main`): PETSc's
`PCMatApply` applies a PC without one column by column, so a telescope as the
block PC of `IdenticalBlocksPC` would solve the components one by one again.
The other way around works: the m split is a PCTELESCOPE, which moves the
right-hand side with all three components as one vector to fewer ranks, and
its inner PC is `tools/petsc_reuse.py:IdenticalInterleavedBlocksPC`, which
makes the one `MatMatSolve` there. petsc4py gives no access to the inner PC,
so PETSc creates the context from `telescope_pc_python_type`, and it finds the
blocks from the interleaving (row i is component i % 3). PCTELESCOPE splits
the rows over the fewer ranks in whole blocks of the matrix's block size, so
the m index set gets block size 3; with block size 1 the ranks got 8672 rows,
which split nodes. `threeDfsi_c0ip` has it as
`mesh_motion_telescope_reduction_factor`.

Medium mesh, 28 ranks, 10 steps from the mixed checkpoint at t = 10:

| reduction factor | MUMPS ranks | s/step | Cholesky symbolic / numerical (s) | K_m solve per Newton iteration | peak RSS (GB), at most on a rank |
|---|---|---|---|---|---|
| none | 28 | 18.34 | 2.31 / 1.45 | 53 ms | 19.5, 0.76 |
| 2 | 14 | 18.69 | 2.30 / 2.34 | 71 ms | 20.3, 0.97 |
| 4 | 7 | 19.05 | 2.33 / 4.66 | 87 ms | 20.0, 1.10 |
| 7 | 4 | 19.41 | 2.35 / 4.91 | 123 ms | 19.7, 1.33 |
| 14 | 2 | 19.92 | 2.33 / 6.50 | 179 ms | 19.5, 1.89 |
| 28 | 1 | 20.84 | 2.20 / 8.11 | 270 ms | 19.2, 2.88 |

- All runs: 74 Newton iterations and QoI files identical to the run without
  telescope.
- On one socket, every reduction makes K_m slower: the solves still speed up
  from 14 to 28 ranks (71 to 53 ms), so a few cores do not use the socket's
  memory bandwidth, and MUMPS's communication on one node is cheap. From 28
  ranks to 1, the solves take 5.1 times as long and the numerical
  factorization 5.6 times. The analysis does not change.
- The step grows by 0.35 s (2%) at factor 2 and 2.5 s (14%) at factor 28; K_m
  solves are 2 to 10% of a step here.
- The memory moves to the MUMPS ranks (at most 0.76 to 2.88 GB on a rank), with
  the sum unchanged.
- The gain the telescope is meant for, fewer MUMPS ranks when hundreds of ranks
  over several nodes make its solves latency-bound, cannot show on one node.
  These numbers say to keep many MUMPS ranks per socket, at least half of them,
  rather than a few.

## K_c iterations on the fine mesh (2026-10-21)

The fine run on eX3 (`jobs/threeDfsi_c0ip.sh`, one defq node, 64 ranks,
fenicsx-v011 Spack environment) takes about 51 s per step, with 7 to 8 Newton
iterations per step and 50 to 67 FGMRES iterations per K_c solve around
t = 4.1, about 58 on average. The medium mesh needed about 27 to 32 on 28 and 30
ranks.

Two steps from its checkpoint at t = 4 (written on 64 ranks):

| | ranks | Newton iterations | FGMRES per Newton iteration |
|---|---|---|---|
| eX3 (defq, Spack, PETSc 3.24) | 24 | 7, 10 | 34.7, 33.9 |
| workstation (conda, PETSc 3.25) | 24 | 7, 10 | 32.3, 29.6 |

Both take the same Newton iterations and Jacobian assemblies. So the doubling on
eX3 comes from the 64 ranks, not from the mesh or the software. Fewer ranks are
still slower: 107 s per step on 24 ranks on a shared defq node, against 51 s on
64.

Smoother experiments on the workstation, from the same checkpoint (scratch copy
of the solver with the K_c options overridden, 24 ranks, 2 steps, Newton
iterations 7 and 10 where it converged). Step time is the second step, which
has no setup apart from Jacobian reassemblies:

| variant (GAMG on the velocity block unless noted) | FGMRES per Newton iteration | second step |
|---|---|---|
| Richardson + SOR, omega 0.5 (current) | 32.3, 29.6 | 58.9 s |
| the same with 4 smoothing steps instead of 2 | 24.3, 22.4 | 69.1 s |
| SOR as bjacobi with 3 blocks per rank (72 in all) | 37.6, 34.6 | 72.2 s |
| `pc_gamg_repartition` | identical to current | 58.7 s |
| Chebyshev + Jacobi | 200 (limit) in every Newton iteration | failed |
| Chebyshev + SOR | 200 (limit) | stopped |
| no aggressive coarsening | PC setup failed (`SUBPC_ERROR`) | failed |
| BoomerAMG, nodal coarsening 6, interpolation from the rigid body modes | 100 (limit) | failed |

- All converged variants give identical QoIs.
- Chebyshev fails here although it converged in the first coarse steps: its
  eigenvalue estimate assumes a symmetric operator, and A is far from symmetric
  once the flow has developed.
- 72 SOR blocks on 24 ranks give only 15% more iterations, far from the doubling
  on 64 ranks. So the weaker SOR coupling between ranks explains little of it;
  GAMG's coarsening, which aggregates within each rank, probably explains more.
  That cannot be tested here without 64 ranks.
- More smoothing steps lower the iterations by a quarter but make a step 17%
  slower on 24 ranks. With 58 iterations on 64 ranks the trade may differ, to be
  measured there.

`threeDfsi_c0ip` now takes `k_c_options`, a dict of PETSc options that replace
those of the K_c solve, and prints the time and the K_c iterations of every
step, with the mean time per step after the first. `KSPConvCheck` records the
iterations of every solve for that. `jobs/threeDfsi_c0ip.sh` passes
`K_C_OPTIONS` on, so the experiments above can be repeated on eX3 with 64
ranks from a checkpoint (`INITIAL_STATE_RUN`), without editing the solver.

### K_c options on 64 ranks (2026-10-22)

`jobs/threeDfsi_c0ip_kc_experiments.sh`, job 1439820 on n002 (defq, 64 ranks,
the whole node: no other job ran there meanwhile), two steps from the t = 4
checkpoint of `fine_rank_test`, one variant after the other. Every variant
converged. Time per step after the first, which leaves out the setup:

| variant (options replacing the solver's) | K_c iterations per solve | s per step |
|---|---|---|
| none | 47.8 | 42.8 |
| `mg_levels_ksp_max_it 3` (3 smoothing steps instead of 2) | 31.6 | 39.0 |
| `mg_levels_ksp_max_it 4` | 31.4 | 44.5 |
| `pc_gamg_threshold 0.05` (instead of 0.01) | 34.2 | 91.8 |
| `pc_gamg_threshold 0.1` | 23.9 | 101.2 |
| `pc_gamg_process_eq_limit 1000` (instead of 50) | 47.9 | 72.8 |
| `pc_gamg_coarse_eq_limit 5000` | 47.5 | 80.9 |

(all on `fieldsplit_v_`, the GAMG on the velocity block)

- The coarse levels are not where 64 ranks lose: fewer ranks on them or a
  larger coarsest grid leave the iterations unchanged, and only cost time
  (redistribution at every setup, and a serial coarse solve in every V-cycle).
- A third smoothing step recovers most of it: 34% fewer iterations and 9% less
  time per step. A fourth gains no iterations and costs time.
- Higher thresholds cut the iterations but coarsen more slowly, into more and
  denser levels, which are more expensive in every V-cycle and in the setup at
  every Jacobian reassembly.
- With 24 ranks on the workstation, 4 smoothing steps cost 17% more time per
  step; on 64 ranks the smoother has more to make up for.
- To confirm over 10 steps before making 3 smoothing steps the default.

10 steps from the same checkpoint (to t = 4.04), defq, 64 ranks, job of
`jobs/threeDfsi_c0ip_kc_experiments.sh`:

| smoothing steps per level | K_c iterations per solve | s per step after the first |
|---|---|---|
| 2 (PETSc's default) | 53.2 | 61.9 |
| 3 | 35.9 | 57.6 |

33% fewer iterations and 7% less time per step, so `threeDfsi_c0ip` now
smooths with 3 SOR steps per level (`fieldsplit_v_mg_levels_ksp_max_it`), and
the solver summary prints the number. On one rank on the coarse mesh (the
test), the iterations drop from 30.8 to 26.5 per solve and the second step
takes 8.8 s instead of 8.1 s: on few ranks the third step costs more than it
saves. `threeDfsi_mixed` smooths with 3 steps too: its K_c and the K_c
solve are the same as in `threeDfsi_c0ip`, which only differs in the mesh
motion.

### SOR omega with 3 smoothing steps (2026-10-23)

Job 1441099, defq, 64 ranks, two steps from the t = 4 checkpoint of
`fine_rank_test`, 3 smoothing steps per level:

| omega | K_c iterations per solve | s per step after the first |
|---|---|---|
| 0.4 | 28.2 | 37.0 |
| 0.5 (current) | 31.6 | 39.7 |
| 0.6 | 48.4 | 51.8 |
| 0.7 | failed | |

- 0.5 with 3 steps reproduces the earlier run (31.6 iterations, 39.0 s).
- On 64 ranks, 0.5 sits close to where the smoother breaks down: 0.6 already
  takes half again as many iterations, and 0.7 does not converge. On 28 to 30
  ranks, with 2 steps, 0.5 was better than 0.7 by less (and 0.7 failed only on
  30 ranks of the coarse mesh). The Jacobi coupling between ranks grows with the
  ranks, and with it the damping SOR needs.
- 0.4 is 11% fewer iterations and 7% less time per step than 0.5, and further
  from the breakdown, which matters for more ranks over several nodes.

Over 10 steps (to t = 4.04, job 1442868, defq, 64 ranks, 3 smoothing steps):

| omega | K_c iterations per solve | s per step after the first |
|---|---|---|
| 0.5 | 35.9 | 50.0 |
| 0.4 | 30.8 | 42.4 |
| 0.3 | 32.5 | 46.1 |

0.4 takes 14% fewer iterations and 15% less time per step than 0.5, so
`threeDfsi_c0ip` and `threeDfsi_mixed` now under-relax with omega = 0.4. The
0.5 row repeats the iterations of the earlier 10-step run with 3 smoothing
steps (35.9) but not its time (57.6 s), so the step times of different jobs
differ by about 15% even on the same partition; within a job they compare.

## Fine mesh over [8, 10] with three time steps (2026-10-25)

As in Failer & Richter (Sect. 5.4): the fine run to t = 8 with dt = 0.004
(`threeDfsi_c0ip_fine_t8`), then three runs from its checkpoint at t = 8 with
dt = 0.004, 0.002 and 0.001 (`INITIAL_STATE_RUN`), all on one defq node with 64
ranks, with 3 SOR smoothing steps and omega = 0.4 (commit 0f8c678).

| dt | steps | Newton iterations per step | K_c iterations per solve | Jacobian assemblies | s per step | total |
|---|---|---|---|---|---|---|
| 0.004 | 500 | 7.3 | 32.0 | 493 | 31.9 | 4.4 h |
| 0.002 | 1000 | 6.9 | 25.5 | 824 | 34.0 | 9.5 h |
| 0.001 | 2000 | 6.3 | 22.7 | 1229 | 29.9 | 16.6 h |

The run to t = 8 with the old smoother (2 steps, omega 0.5) took about 51 s per
step with dt = 0.004.

Mean ± amplitude over [9, 10] (eq. (16) of the paper), displacements in mm,
and the frequency of B_y:

| run | drag | lift | B_x | B_y | B_z | f(B_y) [Hz] |
|---|---|---|---|---|---|---|
| fine, dt = 0.004 | 181.2 ± 3.99 | 0.69 ± 43.36 | -2.223 ± 2.490 | 2.51 ± 26.51 | 0.385 ± 0.784 | 5.751 |
| fine, dt = 0.002 | 181.0 ± 3.86 | 0.67 ± 43.34 | -2.221 ± 2.481 | 2.54 ± 26.48 | 0.390 ± 0.777 | 5.759 |
| fine, dt = 0.001 | 180.9 ± 3.78 | 0.64 ± 43.28 | -2.217 ± 2.475 | 2.55 ± 26.46 | 0.390 ± 0.774 | 5.761 |
| medium, dt = 0.004 (`threeDfsi_mixed`) | 171.3 ± 3.34 | 1.56 ± 40.30 | -1.923 ± 2.155 | 0.85 ± 26.07 | 0.246 ± 0.689 | 5.677 |
| paper, level 3, k = 0.004 | 185.7 ± 3.6 | -0.70 ± 41.35 | -2.176 ± 2.419 | 2.77 ± 25.69 | -0.196 ± 0.196 | |
| paper, level 3, k = 0.002 | 185.7 ± 3.5 | -0.73 ± 41.41 | -2.161 ± 2.401 | 2.75 ± 25.64 | 0.490 ± 0.881 | |
| paper, level 3, k = 0.001 | 185.5 ± 3.5 | -0.72 ± 41.30 | -2.143 ± 2.383 | 2.70 ± 25.59 | 0.486 ± 0.877 | |

- The time step changes the means and amplitudes by less than 0.5%, except the
  drag amplitude (5% from 0.004 to 0.001), as in the paper. Pointwise, the runs
  drift apart in phase, since the frequency rises slightly with smaller dt: at
  t = 10 the dt = 0.004 run is off by up to 35% of the amplitude, and the
  dt = 0.002 run by about a quarter of that, so second order in dt.
- The mesh matters much more. From medium to fine, the drag mean rises from
  171.3 to 181.2, still 2.5% below the paper's 185.5; B_x, B_z and the
  frequency move toward the paper. The lift and B_y amplitudes overshoot the
  paper's by 5% and 3%; the paper's own amplitudes fell by 12 to 17% from its
  level 2 to level 3, so they are not converged either.
- Plots and the table: `python -m xfsi_solver.scripts.plot_fsi3d_time_steps`
  with the three QoI files copied from eX3 as `--run` and the medium run as
  `--compare`, into `output/figures/plot_fsi3d_time_steps/`.

## Hexahedral meshes (2026-10-25)

`create_hex_mesh` in `scripts/create_mesh_FSI3D.py` writes
`data/meshes/fsi3d/mesh_hex_sec<suffix>.xdmf`, with the geometry, tags and
size parameters of the tet meshes:

- gmsh meshes the cross-section z = 0 with quadrilaterals, and numpy extrudes
  it in z into hexahedra (Q2 geometry: the quadrilaterals' 9 nodes at the z of
  the vertex layers and their midpoints). gmsh's own extrusion does not work:
  Blossom recombination leaves triangles (so prisms), and neither its all-quad
  subdivision nor the quasi-structured quad algorithm can mesh extruded curves.
  The quasi-structured algorithm also ignores the size field, and, in a 2d
  model, gave 3 to 8 times too many quadrilaterals around the beam.
- Cross-section: Frontal-Delaunay with twice the target cell size, recombined
  and split into four quadrilaterals per cell (`Mesh.SubdivisionAlgorithm 1`),
  then `Laplace2D`. The beam is a structured strip; unstructured, its cells
  had minSICN down to 0.2. The cylinder's seam is rotated upstream, so that
  the arc of the beam is one curve and the strip has four corners. After the
  split the strip is two cells thick (at these resolutions), and the middle of
  its end x = 0.9 is a vertex, so B is one at z = 0.3. Coarse cross-section:
  499 quadrilaterals, minSICN 0.52, 1% quantile 0.64.
- z layers: the size on the beam along it (0.1 < z < 0.3), and below and above
  it the size the threshold field gives at the distance to the beam's end.
  Every layer runs the whole channel length, so the far field is finer in z
  than in the tet meshes.
- Tags from the vertex coordinates (`locate_entities`), not from gmsh.
- Checked on all three levels: volumes and tag areas as for the tet meshes (to
  the arc error of the Q2 cylinder, at most 1.7e-5), positive det J at degree-6
  quadrature points, B a vertex. min/max det J per cell is down to 0.28 (tets:
  0.68).

| mesh | close / far | cells | beam cells | z layers | vertices | Q2 nodes | tet P2 nodes |
|---|---|---|---|---|---|---|---|
| hex coarse | 0.02 / 0.08 | 7,984 | 360 | 16 | 9,384 | 69,366 | 23,455 |
| hex medium | 0.01 / 0.04 | 56,608 | 1,440 | 32 | 61,710 | 473,070 | 134,657 |
| hex fine | 0.0067 / 0.03 | 154,800 | 3,120 | 48 | 164,738 | 1,277,878 | 333,734 |

At the same cell sizes the hex meshes have 3 to 3.8 times the degree-2 nodes
of the tet meshes, mainly from the layers through the whole channel. The
coarse and medium hex meshes are close to the paper's levels 2 and 3 (66k and
504k Q2 nodes). Meshing takes 1 to 3 s.

### Hex meshes with the node counts of the tet meshes

Suffix `_<level>_tetnodes`: close and far of the tet level scaled by a common
factor, picked by scanning, so that the Q2 nodes are about the tet P2 nodes.
The count jumps with the rounded numbers of cells along the beam and of the z
layers, so it cannot be matched exactly (coarse: 26,358 at x1.48, 22,034 at
x1.49). Same checks as above, all passed (min/max det J per cell down to 0.27).

| mesh | scale | close / far | cells | beam cells (thick x long x wide) | z layers | Q2 nodes | tet P2 nodes |
|---|---|---|---|---|---|---|---|
| hex coarse_tetnodes | 1.49 | 0.0298 / 0.1192 | 2,431 | 168 (2 x 12 x 7) | 11 | 22,034 | 23,455 |
| hex medium_tetnodes | 1.63 | 0.0163 / 0.0652 | 15,980 | 528 (2 x 22 x 12) | 20 | 136,366 | 134,657 |
| hex fine_tetnodes | 1.62 | 0.010854 / 0.0486 | 39,121 | 1,152 (2 x 32 x 18) | 29 | 328,866 | 333,734 |

The script now also prints the number of geometry (Q2) nodes.

### Cost of `threeDfsi_c0ip` on the hex mesh (estimate)

`threeDfsi_c0ip` refuses hexahedra (no C0IP penalty parameters for them). For
timing only, a scratch copy used the tetrahedron formula on the hexes, with
Q2, Q2, DPC1 on `mesh_hex_sec_fine_tetnodes`, against the default P2, P2, P1
on `mesh_sec_fine`: 8 steps from rest, dt = 0.004, 24 ranks on the
workstation, `-log_view`. K_c has 1.14M unknowns on the hex mesh (DPC1: 152k
pressure dofs) and 1.04M on the tet mesh.

| | tet | hex | hex / tet |
|---|---|---|---|
| s per step after the first (2 Newton iterations) | 12.6 | 26.3 | 2.08 |
| K_c iterations per solve | 25.2 | 26.2 | 1.04 |
| K_c solve per FGMRES iteration (s) | 0.20 | 0.43 | 2.14 |
| residual evaluation (s) | 0.94 | 1.43 | 1.52 |
| first Jacobian assembly, with the C0IP terms (s) | 12.3 | 40.2 | 3.3 |
| K_m Cholesky, symbolic + numerical, once (s) | 15.5 | 40.0 | 2.6 |
| K_m MatMatSolve per call (s) | 0.16 | 0.39 | 2.4 |
| GAMG setup (s) | 6.3 | 7.5 | 1.2 |

MatMult and MatSOR cost about twice as much on the hex mesh (Q2 rows couple
125 nodes at a vertex). The K_c solve is about 70% of a tet step with
developed flow, so weighting the ratios by such a step (7.3 Newton iterations,
about one Jacobian reassembly) also gives about 2.1. The tet fine run to t = 8
would take about 32 to 35 s per step on one eX3 defq node with the current K_c
options, so the hex run would take about 70 s per step, 2000 steps in about
40 h (one node, 64 ranks), assuming the iterations stay as on the tet mesh,
which was only checked from rest.

## threeDfsi_simplified: the simplified solver in 3d (2026-10-27)

- `solvers/threeDfsi_simplified.py` ports `fsi2_simplified` to the 3d benchmark: Newton on (v, p)
  only, with u_s set node by node from the kinematic relation and u_f by a `mesh_motion` solve in a
  post-solve hook of the KSP. The benchmark set-up is that of `threeDfsi_c0ip`: Table 1 constants
  (`lambda_s = 8.0e6` by default), bi-parabolic inflow ramped over t < 1, QoIs at B = (0.9, 0.2, 0.3)
  with `QOI_HEADER_3D`, and a required quadrature cap, by default 8.
- `DEFAULT_ITERATIVE_SOLVER_OPTIONS` are the K_c options of `threeDfsi_c0ip` (FGMRES, upper Schur
  factor with selfp, one GAMG V-cycle with three SOR steps per level at omega = 0.4, one BoomerAMG
  V-cycle), applied to the nest (v, p) Jacobian. They are the default here, unlike in
  `fsi2_simplified`, which defaults to MUMPS LU; `DEFAULT_DIRECT_SOLVER_OPTIONS` selects LU.
- The default mesh motion is `biharmonic_c0ip_mesh_motion`, as in `threeDfsi_c0ip`, factorized by
  MUMPS Cholesky on the x component (`default_mesh_motion_solver_options(3)` with `block_pc_type
  cholesky`); the other mesh motions keep the package default, LU.
- `mesh_motion.default_solver_options` is renamed `default_mesh_motion_solver_options`, since the
  solvers that import it have a `DEFAULT_SOLVER_OPTIONS` of their own.
- The mesh motion functions take `max_quadrature_degree` (default None, no cap), applied with
  `cap_quadrature_degree` to the residual before `MeshMotion` differentiates it, so the matrix keeps
  it. UFL estimates degree 3 on the cells and 12 on the facets for the C0IP forms of P2 on P2
  tetrahedra, and `MeshMotion` integrates them in every residual (and its lifting), once per Newton
  iteration. The 3d solver caps them by `mesh_motion_max_quadrature_degree`, by default 8 as in
  `threeDfsi_c0ip`, and rejects None. On `fsi2/mesh_sec_coarse` a cap of 2 changes a bending mesh
  motion by 7e-6 (harmonic), 2.8e-3 (mixed) and 2.9e-5 (C0IP) of the largest displacement; a cap of 4
  by 0, 2.2e-6 and 4.5e-8.
- Each step prints its time and the linear iterations per Newton step (`getLinearSolveIterations`),
  since `KSPConvCheck` cannot share the KSP's post-solve hook; a failed linear solve stops the SNES
  with DIVERGED_LINEAR_SOLVE.
- The slow tests pass, serially on `mesh_sec_coarse` with dt = 0.004 (7 min): two steps from rest
  with the iterative and the direct (v, p) solve agree in t, drag and lift to rtol 1e-4, and a start
  from another run's checkpoint, with a restart, reproduces its run to 1e-6. Each step from rest takes
  2 Newton iterations. Iterative: 31.0 and 27.5 FGMRES iterations per solve, 42.8 s and 24.9 s per
  step; direct (MUMPS LU): 39.6 s and 19.7 s. The first step includes the K_m Cholesky and the
  preconditioner setup.

## threeDfsi_simplified: MUMPS on the solid velocity (2026-10-28)

- Ports `MUMPS_ON_SOLID` and `requires_v_sub_IS` from `fsi2_simplified` (commit 1dea744): the v block
  of the Schur fieldsplit is split again, multiplicatively, into the v dofs on solid cells (with the
  interface), solved by MUMPS LU, and the fluid dofs, by one GAMG V-cycle with the translations as
  near-nullspace (block size of `is_v_f`). The fluid GAMG keeps the 3d smoothing of
  `DEFAULT_V_OPTIONS` (three SOR steps at omega = 0.4), not the two of the 2d solver. The inner ISes
  are set in the Jacobian callback after the first assembly, as in 2d. `DEFAULT_ITERATIVE_SOLVER_OPTIONS`
  now also has `ksp_atol = 1e-8`, as in 2d.
- Two steps from rest on `mesh_sec_coarse` (dt = 0.004, 4 ranks): 7.2 FGMRES iterations per solve
  against 18.5 with `DEFAULT_ITERATIVE_SOLVER_OPTIONS`, 6.6 s against 7.0 s for the second step; the
  QoIs agree to the linear tolerance (drag to 5e-6 relative).
- The BoomerAMG V-cycle on \hat S in `threeDfsi_simplified` (`DEFAULT_P_OPTIONS`, so in both
  `DEFAULT_ITERATIVE_SOLVER_OPTIONS` and `MUMPS_ON_SOLID`) uses `pc_hypre_boomeramg_strong_threshold
  = 0.5`, hypre's recommendation in 3d (its default 0.25 suits 2d). In the same two steps on
  `mesh_sec_coarse` with 4 ranks: 7.8 FGMRES iterations per solve with `MUMPS_ON_SOLID` (7.2 before)
  and 18.8 with the default iterative options (18.5 before), at the same step times; drag agrees to
  5e-6 relative.
- Medium mesh, 28 ranks, 10 steps (dt = 0.004) from the `threeDfsi_mixed_medium` checkpoint at t = 10,
  with the beam oscillating, quadrature caps of 6 on the residual and the mesh motion (2026-10-29):

  | | `MUMPS_ON_SOLID` | `DEFAULT_ITERATIVE_SOLVER_OPTIONS` |
  |---|---|---|
  | s/step, after the first | 18.3 | 25.4 |
  | first step (s) | 25.9 | 33.2 |
  | FGMRES iterations per solve | 9.6 | 23.2 |
  | s per Newton iteration, after the first step | 2.4 | 3.4 |

  Both take 74 Newton iterations (7 or 8 per step) and 11 Jacobian assemblies, the same as
  `threeDfsi_mixed` over these steps, and end with the same QoIs to 7 digits (lift to 4e-7
  relative). `MUMPS_ON_SOLID` is 28% faster per step: 2.4 times fewer FGMRES iterations, each
  about 1.8 times as expensive. It matches `threeDfsi_mixed` on 28 ranks over the same steps
  (18.3 s/step with K_m by MatMatSolve, above), which had the quadrature cap of 8. On 24 ranks
  (the user's runs) `MUMPS_ON_SOLID` took 19.5 s/step over 10 steps (9.5 FGMRES per solve) and the
  default options 25.3 s over the second of 2 steps (24.2).
- Eisenstat-Walker (`snes_ksp_ew`, version 2) on top of `MUMPS_ON_SOLID`, medium mesh, 28 ranks,
  8 steps from the `threeDfsi_mixed_medium` checkpoint at t = 10, quadrature caps of 6 (2026-10-29):

  | | fixed rtol 1e-5 | EW, rtol0 1e-4, rtolmax 1e-3 | EW, rtol0 1e-2, rtolmax 0.1 |
  |---|---|---|---|
  | Newton iterations | 60 | 62 | 19, 20, then `DIVERGED_MAX_IT` at t = 10.012 |
  | FGMRES iterations per solve | 9.6 | 5.9 | 1.9 |
  | s/step, after the first | 18.6 | 17.3 | 48.7 |
  | s per Newton iteration, after the first step | 2.45 | 2.24 | 2.43 |

  The preconditioner is strong enough that 1 or 2 FGMRES iterations reach rtol 1e-2, and the first
  Newton iterations contract as with rtol 1e-5 (30 -> 4 -> 0.5). Below a residual of about 0.5, a
  loose solve leaves Newton contracting by only about 0.5 per iteration, which EW reads as slow
  convergence and answers with the loosest tolerance, rtolmax; the slow contraction also makes
  wrapped_jacobian reassemble J at almost every iteration. PETSc's defaults (rtol0 0.3, rtolmax 0.9)
  are looser still and were not run to the end. With rtol0 1e-4 and rtolmax 1e-3, EW saves 7% per
  step at about the same Newton count, with the QoIs equal to 7e-7 relative. The gain is bounded:
  an FGMRES iteration costs about 0.06 s, against about 1.9 s for the rest of a Newton iteration
  (residual assembly, mesh motion, Jacobian), so even no FGMRES iterations at all would save only
  about a quarter of a step. EW is not enabled in the solver options.
- Fixed `ksp_rtol` on `MUMPS_ON_SOLID`, same set-up as the Eisenstat-Walker comparison above
  (2026-10-29):

  | `ksp_rtol` | Newton iterations | FGMRES iterations per solve | s/step, after the first |
  |---|---|---|---|
  | 1e-5 | 60 | 9.6 | 18.6 |
  | 1e-4 | 60 | 7.3 | 17.6 |
  | 1e-3 | 66 | 5.2 | 18.2 |
  | 1e-2 | 86 | 3.5 | 25.0 |

  The QoIs agree with rtol 1e-5 to 3e-7 relative for 1e-4 and 2e-6 for 1e-2. rtol 1e-4 keeps the
  Newton count and saves 5%, about as much as EW with rtolmax 1e-3 (17.3 s, within the scatter of
  about 1 s/step that one Newton iteration more or less gives). EW cannot do better here: the
  Jacobian is that of (v, p) only, without the mesh motion and the update of u_s, so the linear
  residual does not predict the nonlinear one, and a loose solve costs Newton progress that EW reads
  as slow convergence, to which it answers with a looser solve. A loose first solve of each time
  step alone would cost no Newton progress (rtol 1e-2 there gave the same residual as 1e-5), but
  EW cannot give it: PETSc caps rtol0 at rtolmax too, and EW with rtol0 1e-2 and rtolmax 1e-4 ran
  identically to a fixed rtol of 1e-4. It would save about 4 FGMRES iterations, 0.25 s, per step.
- `ksp_rtol` set to 1e-4 in `DEFAULT_ITERATIVE_SOLVER_OPTIONS` and `MUMPS_ON_SOLID`, after the same
  8 steps with the default iterative options (2026-10-29): 60 Newton iterations with both 1e-5 and
  1e-4, 25.5 and 20.8 FGMRES iterations per solve, 26.9 and 24.4 s/step after the first (9% less),
  QoIs equal to the 7 digits written.
- A loose first linear solve of each time step, by a hook in a scratch copy of
  threeDfsi_simplified that sets the KSP rtol in wrapped_jacobian (first_solve_rtol at Newton
  iteration 0, ksp_rtol after), on `MUMPS_ON_SOLID` with rtol 1e-4, same 8 steps (2026-10-29):

  | first solve | Newton iterations | FGMRES iterations per solve | s/step, after the first |
  |---|---|---|---|
  | rtol 1e-4 | 60 | 7.3 | 17.6 |
  | rtol 1e-2 | 60 | 6.7 | 17.0 |
  | rtol 1e-1 | 62 | 6.2 | 17.5 |

  With 1e-2 the first solve takes 2 FGMRES iterations instead of 5 or 6, and the following residuals
  are within a few percent of those with 1e-4; 3% less per step, about the scatter between runs.
  With 1e-1, 6 of the 8 steps took a Newton iteration more. Not added to the solver.
- Cost of a Newton iteration of threeDfsi_simplified, from timers and PETSc log stages in a scratch
  copy (medium mesh, 28 ranks, rtol 1e-4, steps 2 to 8 of the 8 from t = 10, 53 Newton iterations,
  7 Jacobian assemblies; wall time per Newton iteration, 2026-10-29):

  | | `MUMPS_ON_SOLID` | `DEFAULT_ITERATIVE_SOLVER_OPTIONS` |
  |---|---|---|
  | mesh motion (C0IP) | 1.49 s (65%) | 1.48 s (46%) |
  | linear solve, with PC setup | 0.52 s (22%) | 1.45 s (45%) |
  | residual assembly | 0.22 s (10%) | 0.22 s (7%) |
  | Jacobian assembly | 0.05 s (2%) | 0.05 s (2%) |
  | rest | 0.03 s | 0.02 s |
  | total | 2.30 s | 3.23 s |

  In the mesh motion, the MUMPS solve (MatMatSolve) takes 0.06 s per call on most ranks. The rest is
  the assembly of its residual, the full C0IP form with its interior-facet integrals, in every call,
  although the problem is linear and its matrix is assembled once: about 1.4 s on the slowest rank,
  while the others wait for it, in the ghost exchange (VecScatterEnd, 0 to 40 s per rank over the 53
  calls) or in the MUMPS solve (up to 21 s on some ranks against 3 s on most). In the linear solve
  with `MUMPS_ON_SOLID`, the SOR smoothing of GAMG on the fluid velocity takes 0.29 s per Newton
  iteration (MatSOR, median rank), the MUMPS triangular solves on the solid 0.05 s, and the PC setup
  (MUMPS factorization and GAMG setup, at each Jacobian assembly) about 0.05 s.
- The mesh motion residual by a product with its matrix (`MeshMotion._residual`), instead of an
  assembly of the form in every solve, on `MUMPS_ON_SOLID`, same 8 steps as the breakdown above
  (2026-10-29): 0.065 s instead of 1.49 s per mesh motion solve, so 0.88 s instead of 2.30 s per
  Newton iteration and 6.6 s instead of 17.4 s per step after the first (15.0 s instead of 24.5 s
  for the first), 62 s instead of 146 s for the 8 steps. Newton and FGMRES iterations are the same,
  and the QoIs equal to the 7 digits written. The product takes 0.01 s per call and the MUMPS solve
  0.05 s, and the ranks no longer wait for each other (VecScatterEnd at most 0.16 s over the 53
  calls, from 40 s). A Newton iteration is now 59% linear solve, 25% residual assembly, 7% mesh
  motion and 6% Jacobian assembly.
- Preconditioners for the fluid velocity block of `MUMPS_ON_SOLID` (the `fieldsplit_v_0_fieldsplit_f_`
  options), medium mesh, 28 ranks, rtol 1e-4, mesh motion by the matrix-vector residual, 8 steps
  from the `threeDfsi_mixed_medium` checkpoint at t = 10 (2026-10-29):

  | fluid block | Newton iterations | FGMRES iterations per solve | s/step, after the first |
  |---|---|---|---|
  | GAMG, 3 SOR steps, omega 0.4 (current) | 60 | 7.3 | 6.70 |
  | GAMG, 2 SOR steps, omega 0.4 | 60 | 7.8 | 6.08 |
  | GAMG, 3 SOR steps, omega 0.7 | 60 | 7.3 | 6.65 |
  | GAMG, 3 SOR steps, omega 1.0 | 61 | 14.3 | 9.47 |
  | GAMG, 3 steps of l1-Jacobi | 60 | 9.5 | 6.42 |
  | SOR, omega 0.4, no multigrid | 60 | 14.3 | 5.75 |
  | block Jacobi with ILU(0), no multigrid | 60 | 13.3 | 5.57 |

  The QoIs agree with the current options to 7e-7 relative. Damped Jacobi on the element mass
  matrix of P2 on a tetrahedron converges only for omega < 0.46 (eigenvalues of D^{-1} M in
  [0.25, 4.35], best omega 0.44), and SOR is Jacobi between ranks, which fits omega 1.0 doubling
  the FGMRES iterations with the solid split off. The fluid block is dominated by the mass at this
  dt, so a preconditioner without a coarse grid takes twice the FGMRES iterations but saves the
  GAMG V-cycle: 17% less per step with block Jacobi and ILU. It weakens as the viscous term grows
  relative to the mass (finer meshes, larger dt), and block Jacobi with the number of ranks, so it is
  not a default; untested above 28 ranks.
- The same comparison on the fine mesh, 28 ranks, 8 steps from the `threeDfsi_c0ip_fine_t8`
  checkpoint at t = 8 (dt = 0.004, 2026-10-29), against the medium mesh above:

  | fluid block | Newton iterations | FGMRES per solve, medium -> fine | s/step, after the first | vs current |
  |---|---|---|---|---|
  | GAMG, 3 SOR steps, omega 0.4 (current) | 60 | 7.3 -> 7.1 | 17.17 | |
  | GAMG, 2 SOR steps | 60 | 7.8 -> 7.8 | 15.69 | -9% (medium -9%) |
  | SOR, omega 0.4, no multigrid | 59 | 14.3 -> 15.6 | 14.36 | -16% (medium -14%) |
  | block Jacobi with ILU(0), no multigrid | 59 | 13.3 -> 14.3 | 13.37 | -22% (medium -17%) |

  QoIs agree with the current options to 8e-7 relative. With 2.5 times the fluid cells, the
  iterations without multigrid grow by 8 to 9%, and the GAMG V-cycle they save costs more, so their
  gain grows. On the fine mesh, nu dt / h^2 (h the P2 node spacing) is 0.034 at the median cell
  and 0.55 at most, and U dt / h, with U = 2, is 0.74 and 3.0, so the fluid block is still
  dominated by the mass and by local couplings.
