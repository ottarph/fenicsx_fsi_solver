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
