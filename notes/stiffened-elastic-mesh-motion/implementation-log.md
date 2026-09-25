# Stiffened elastic mesh motion: implementation log

Plan: `docs/stiffened-elastic-mesh-motion-plan.md`. New entry point
`src/xfsi_solver/solvers/fsi2_stiffened_elastic_diffmesh.py`. Branch
`iterative-solver` (the fieldsplit implementation through `770ff36` is on it;
no branch transfer needed). Runs are on the local 64-core machine; there is
no separate remote checkout.

## Environment (inspected 2026-09-25)

- DOLFINx 0.11.0, Basix 0.11.0, UFL 2026.1.0, PETSc/petsc4py 3.25.5,
  scifem 0.23.0, Python 3.14.6, MPICH (`mpiexec`), adios2 2.12.1.
- Meshes in `data/meshes/fsi2` (gitignored, already present, none generated):
  `mesh_sec` (5 861 quadratic triangles, the production mesh),
  `mesh_sec_coarse` (492), `mesh_quad_ssq_sec` (2 734 quadratic quads),
  `mesh_coarse` (affine triangles). All have a 2.5 m channel (x in [0, 2.5],
  y in [0, 0.41]); the comparison paper's channel is 2.2 m. The geometry is
  kept as is.
- Existing reference output: `output/qoi/fsi2_biharm_qoi.txt` (biharmonic,
  `mesh_sec`, dt 0.0025, 0 to 15 s, legacy time loop, 24 ranks) and its VTX
  output `output/pv/fsi2_biharm_dm.bp`; `data/fsi2_reference.txt` (published
  FSI2 data, t in [10, 14.62]).

## Integration audit of the fieldsplit code

- `jacobian_forms`, `diagnostic_index_sets` and `PHYSICAL_MARKERS` operate on
  the supplied `FSIProblem`: `jacobian_forms` differentiates
  `problem.residual`, so the mesh derivative is in `J[0][0]` in both Jacobian
  modes; the markers are those of every FSI2 mesh. The lazy import of
  `jacobian_forms` in `FieldSplitSolver` rebuilt the preconditioner forms
  from scratch; they are now passed in by `create_nonlinear_problem`.
- Hard-coded prefixes: `fsi2_harmonic_diffmesh_` in `create_nonlinear_problem`
  and `fsi2_harmonic_diffmesh_P_vp_` in `AuxiliaryOperators` are now
  parameters (defaults unchanged); the elastic solver uses
  `fsi2_stiffened_elastic_diffmesh_`.
- The one-sided interface measure took the fluid side from the first facet
  only. `scifem.compute_interface_data` orders each facet by the lower cell
  marker first, so it was correct, but every facet is now checked
  (`interface_fluid_entities`), serial and two ranks.

## Step 1: mesh-extension operator (`a5ead85`)

- `xfsi_solver.fsi.mesh_extension`: `FluidReferenceGeometry`,
  `HarmonicMeshExtension` (the previous forms verbatim; harmonic QoIs of 20
  coarse steps are bitwise identical to `670c5bf` for direct and fieldsplit),
  `StiffenedElasticMeshExtension`.
- `j_star` = mean initial fluid-cell volume / parent-cell volume, owned cells
  only: `mesh_sec` 3.946e-4 (5 120 fluid cells), `mesh_sec_coarse` 4.822e-3.
  Weight range (chi 2.5): `mesh_sec` 0.0099 to 4 871, `mesh_sec_coarse` 0.011
  to 1 697, `mesh_quad_ssq_sec` 0.017 to 94 396. On `mesh_sec` 59 fluid cells
  are curved (j_0 varies up to 4 % within a cell); the quads vary up to 3.6x.
- Quadrature (smooth u, relative change of the assembled residual vs degree
  20): triangles, degree 4 / 6 / 8: 9e-7 / 8e-10 / 9e-12 (coarse), 3e-9 /
  8e-13 (`mesh_sec`); interface term 1e-13 at any degree (the interface cells
  are straight); quads, degree 6 / 10 / 14: 6e-5 / 9e-8 / 1e-10. Defaults: 6
  for triangles, 10 for quadrilaterals.
- A standalone `LinearProblem` extension needs a (zero) solid-cell term, or
  the Dirichlet solid DOFs are missing from the sparsity pattern and their
  unit diagonal is silently dropped (singular matrix).

## Step 2: driver hooks and accepted time (`383af5c`)

- Legacy loop: `while t < T`, `t += dt`, inflow at `t`, label `t`. Its first
  step from rest has zero inflow and returns the zero state, so the state
  labelled `t` is the correct state at `t` (checked: identical to the
  accepted-time trajectory to 1e-12), but the endpoint depends on
  accumulated round-off: the biharmonic reference file ends with a step
  labelled 15.0 because 6000 additions of 0.0025 fall short of 15.
- `time_semantics="accepted"`: N = (T - t0)/dt steps (error unless integer),
  labels `t0 + n dt`, the initial state written to QoI and VTX. Restart files
  record their semantics and metadata; mismatches are rejected.

## Step 3: solver, diagnostics (`085a4bf`) and fieldsplit (`61d170f`)

- Coarse startup (8 steps): full/direct, no_ale/direct and no_ale/fieldsplit
  agree to < 1e-7 in all fields; FGMRES 7-11 iterations; sampled J_ALE > 0.99.
- Elastic `A_ff` in `DisplacementPC` equals the alpha-scaled weighted
  stiffness to 1e-12 and is symmetric; the dropped mesh terms in `A_II` are
  < 1e-6 of the solid mass.
- Two-rank test runs need a common output directory (pytest's `tmp_path` is
  per process; a missing file on one rank deadlocks the other in a
  collective).

## Mesh validity of the developed motion (standalone prediction)

Before the full runs reach developed motion, 11 biharmonic states in
[12, 15] s (tip A_y from -0.081 to 0.084, A_x down to -0.028) were read from
`output/pv/fsi2_biharm_dm.bp`, their solid displacement imposed on all
solid-cell DOFs of `mesh_sec`, and the elastic extension solved
standalone. Worst sampled fluid J_ALE over the states:

| chi \ nu | 0.3 | 0.35 | 0.4 | 0.45 |
|---|---|---|---|---|
| 0 | -2.83 | | | |
| 0.75 | -0.55 | -0.58 | -0.60 | -0.62 |
| 1.0 | -0.09 | -0.11 | -0.14 | -0.16 |
| 1.25 | 0.036 | 0.065 | 0.12 | 0.18 |
| 1.5 | -0.22 | -0.20 | -0.14 | -0.006 |
| 2.0 | -0.96 | | | |
| 2.5 | -1.89 | | | -1.80 |
| 3.0 | -2.92 | | | |

With chi = 2.5 the inverted cells are the large, soft cells at the bottom
wall under the beam tip (w about 0.15), which take the whole compression of
the 0.19 m gap; with chi = 1 it is the fluid cell at the lower tip corner of
the beam (w about 14). `mesh_sec` has a 190x range of initial cell sizes,
i.e. a 5e5 weight contrast at chi = 2.5.

Production runs started on `mesh_sec`, dt 0.0025, T 15, no_ale:
chi 2.5 / nu 0.3 (fieldsplit and direct), and fieldsplit with chi 2.0 / 0.3,
3.0 / 0.3, 2.5 / 0.45, 1.25 / 0.45, 1.25 / 0.3.

## Comparison tolerances (fixed 2026-09-25 14:50, before any elastic run reached developed motion)

`xfsi_solver.scripts.fsi2_qoi_statistics`: per-period mean, amplitude and
frequency over the window [12, 14.6] s (the published data end at 14.62 s).
The validated resolution is the biharmonic run on the same mesh and time
step, which deviates from the published reference by (mean relative to the
amplitude, amplitude and frequency relative):

| | drag | lift | A_x | A_y |
|---|---|---|---|---|
| mean | 3.6 % | 0.5 % | 2.9 % | 0.03 % |
| amplitude | 3.7 % | 2.9 % | 2.2 % | 1.3 % |
| frequency | 0.13 % | 0.11 % | 0.13 % | 0.13 % |

Acceptance for an elastic run on `mesh_sec`, dt 0.0025:

- against the published reference: tip displacement A_y amplitude within
  5 %, A_x mean and amplitude within 10 %, drag and lift amplitude within
  10 %, drag mean within 10 % of its amplitude, all frequencies within 2 %;
- against the biharmonic run (same discretization except the mesh motion):
  A_y amplitude within 3 %, A_x mean and amplitude within 5 %, drag and lift
  amplitude within 5 %, frequencies within 1 %;
- direct vs fieldsplit from equivalent states over a developed interval:
  relative QoI differences at the level of the nonlinear tolerance (no
  larger than 1e-5 of the signal amplitude over 100 steps).

## Coarse-mesh full runs (`mesh_sec_coarse`, no_ale/direct, dt 0.0025, T 15)

| chi / nu | outcome | first failure | max abs A_y | min sampled fluid J |
|---|---|---|---|---|
| 2.5 / 0.3 | failed | t = 9.7425: Newton trial with J_min = -0.009 rejected (domain error); last accepted J_min 0.014 at A_y 0.049 | 0.050 | 0.014 |
| 2.0 / 0.3 | failed | t = 10.2875, same mechanism | 0.065 | 0.000 |
| 1.5 / 0.45 | completed | | 0.080 | 0.263 |
| 1.25 / 0.45 | completed | | 0.081 | 0.368 |
| 1.0 / 0.3 | completed | | 0.081 | 0.318 |

The failures are geometric (mesh extension), not algebraic: Newton and the
linear solver converge until the sampled Jacobian of an iterate becomes
negative. The harmonic solver fails on this mesh at 9.57 s. The completed
coarse runs have frequency 1.7 % and A_y amplitude 4-5 % below the
biharmonic `mesh_sec` run, and a 25 % (of the amplitude) lower mean drag:
coarse-mesh discretization error; this mesh is not a production target.

## Meshes

- `create_mesh(fine=False, coarse=False, quads=False,
  semi_structured_quad=False, second_order=True)` (run in a scratch
  directory, it writes `data/meshes/fsi2/mesh_sec.xdmf` relative to the
  working directory) gives 5 851 cells, not the 5 861 of the existing
  `mesh_sec` (sha256 of `.h5`/`.xdmf` differ; presumably another gmsh
  version). The existing file is used; `run.json` records its checksum.
  `semi_structured_quad` must be `False` also for triangles, or `_ssq` is
  added to the file name.
- `fine=True` with the same options: `data/meshes/fsi2/mesh_fine_sec.xdmf`,
  21 890 cells (new file), weight range 0.0087 to 6 530 at chi 2.5.
- Standalone prediction on the fine mesh (biharmonic states interpolated):
  worst fluid J -1.37 (chi 2.5 / 0.3), -0.68 (2.0 / 0.3), -0.26 (1.0 / 0.3),
  0.087 (1.25 / 0.45). Refinement does not remove the chi = 2.5 failure.

Further runs started: `mesh_sec` chi 1.25 / 0.45 direct, chi 1.0 / 0.3 and
1.5 / 0.45 fieldsplit, chi 1.25 / 0.45 fieldsplit at dt 0.00125 (4 ranks),
and `mesh_fine_sec` chi 1.25 / 0.45 fieldsplit (8 ranks).

## Preconditioner approximations at a developed coarse state

`fsi2_harmonic_diffmesh_benchmark --solver stiffened_elastic`, chi 1.25 /
nu 0.45, `mesh_sec_coarse`, 40 steps from the elastic state at t = 12
(`output/benchmarks/elastic_coarse_chi1.25_nu0.45_t12`), snes_atol 1e-9,
ksp_rtol 1e-6. All modes agree with full/direct to <= 1.1e-10 in the QoIs
and 4.5e-11 in the fields.

| mode | Newton/step | FGMRES/solve (max) | s/step |
|---|---|---|---|
| full/direct | 3.08 | 1 | 0.184 |
| no_ale/direct | 5.70 | 1 | 0.175 |
| full/fieldsplit(no_ale P) | 3.58 | 9.1 (12) | 0.291 |
| no_ale/fieldsplit | 5.70 | 8.4 (10) | 0.246 |
| (u lu) | 5.70 | 8.4 (10) | 0.247 |
| (vp lu) | 5.70 | 1 | 0.181 |
| (u+vp lu) | 5.70 | 1 | 0.191 |
| (u gamg) | 5.70 | 8.4 (10) | 0.711 |
| (u amg) | 5.70 | 8.4 (10) | 0.706 |

With exact `(v,p)` solves one FGMRES iteration suffices, with or without the
exact displacement block: the dropped mesh terms of the solid/interface
displacement block and `H + theta dt E_s` still make `P_vp` the exact
Schur complement to the Krylov tolerance with the stiffened elastic mesh
(the old observation holds here). All outer iterations come from the AMG
velocity and selfp pressure approximations. The iterative fluid-displacement
solves take 11.5 (BoomerAMG) and 23.9 (GAMG, rigid body near-nullspace) CG
iterations to 1e-8, 5 ms each against 0.25 ms for the Cholesky solve.

## Primary run: `mesh_sec`, chi 2.5 / nu 0.3, dt 0.0025 — failed

no_ale/direct (serial) failed at t = 7.83: the last accepted state (t =
7.8275, A = (-0.0032, -0.0354)) has sampled fluid J_min 0.019 and deformation
condition number 97; the first Newton trial of the next step has J_min
-0.0028 (condition number 662) and is rejected as a function domain error,
so SNES stops (KSP converged, reason 4; residual 37.9 before the rejected
step). Converged residuals of the last accepted step: mesh/alpha 2e-13,
kinematic 5e-16, momentum 1e-9, incompressibility 4e-14. The worst cells
at the checkpoints t = 7.0 and 7.5 (J 0.75, 0.69) are at the top channel
wall, x about 0.55, y about 0.40: the large soft far-field cells, as in the
standalone prediction (there at the bottom wall for downward deflection).
First failure: invalid geometry from the mesh extension; not nonlinear,
linear-solver or physical-response failure. The harmonic solver fails on
this mesh at 7.35 s.

Machine: 32 physical cores (64 threads). Runs were oversubscribed (up to 68
ranks) between 15:48 and 15:51; since then at most 32 ranks run at once, and
the long runs use MPI (24 ranks is the usual count for this problem). The
serial chi 1.25 / 0.45 direct run was replaced by an 8-rank run.

## Parameter study on `mesh_sec` (dt 0.0025, no_ale, T 15)

| chi / nu | solver | outcome | failure time | max abs A_y | min fluid J |
|---|---|---|---|---|---|
| 2.5 / 0.3 | direct, fieldsplit | failed (both at the same state) | 7.83 | 0.035 | 0.019 |
| 3.0 / 0.3 | fieldsplit | failed | 7.81 | 0.035 | 0.001 |
| 2.0 / 0.3 | fieldsplit | failed | 8.33 | 0.052 | 0.009 |
| 2.5 / 0.45 | fieldsplit | failed | 8.325 | 0.051 | 0.003 |
| 1.25 / 0.45 | direct, 8 ranks | completed to 15.0 (6000 steps, 49 min) | | 0.084 | 0.154 |

All failures are geometric (the rejected Newton trial has J_min < 0, KSP
converged). chi 2.5 direct and fieldsplit agree to 9e-9 in A_y and 2.6e-5 in
lift over [0, 7.8275].

chi 1.25 / nu 0.45, direct (8 ranks): no rejected iterates, 24 017 Newton
iterations (4.0 per step), sampled min fluid J 0.154, max deformation
condition number 16, min solid J 0.952. Statistics over [12, 14.6]
(deviation from the published reference; mean relative to the amplitude):

| | drag | lift | A_x | A_y |
|---|---|---|---|---|
| elastic chi 1.25 / 0.45 | 216.6 +- 78.84 [3.854] | -0.87 +- 242.8 [1.929] | -0.0152 +- 0.01281 [3.858] | 0.00135 +- 0.08244 [1.929] |
| mean | +3.2 % | -0.6 % | -2.1 % | +0.1 % |
| amplitude | +3.7 % | +2.2 % | +1.7 % | +1.0 % |
| frequency | -0.12 % | -0.08 % | -0.12 % | -0.10 % |
| biharmonic | +3.8 / +3.8 / -0.13 % | -0.5 / +2.9 / -0.11 % | -3.0 / +2.3 / -0.13 % | +0.03 / +1.3 / -0.13 % |

Against the biharmonic run: drag mean -0.5 %, amplitudes -0.15 % (drag),
-0.7 % (lift), -0.6 % (A_x), -0.4 % (A_y), frequencies +0.02 to +0.03 %.
All within the pre-registered tolerances.

Further parameter-study failures (fieldsplit, geometric): chi 1.0 / 0.3 at
t = 9.405, chi 1.5 / 0.45 at t = 9.9125, both at full amplitude (A_y 0.080,
0.0835). Every outcome on `mesh_sec` agrees with the sign of the
standalone prediction of the worst developed J (-0.09, -0.006 fail; 0.18
completes); chi = 1.25 / 0.45 lies in a narrow window between the far-field
wall failure (larger chi) and the beam-tip corner failure (smaller chi).

## Developed interval, `mesh_sec`, chi 1.25 / 0.45: solver modes

100 steps (to t = 10.75) from the serial fieldsplit state at t = 10.5
(`output/benchmarks/elastic_mesh_sec_chi1.25_nu0.45_t10.5`), serial,
snes_atol 1e-9, ksp_rtol 1e-6, machine loaded with other runs (timings are
only comparable within the table):

| mode | Newton/step | FGMRES/solve (max) | s/step | max rel err drag / lift / A_y | rel err u / v / p |
|---|---|---|---|---|---|
| full/direct | 3.14 | 1 | 2.49 | reference | |
| no_ale/direct | 5.53 | 1 | 2.45 | 8e-12 / 1e-11 / 2e-12 | 3e-12 / 3e-12 / 1e-11 |
| no_ale/fieldsplit | 5.53 | 11.4 (14) | 3.12 | 8e-12 / 1e-11 / 2e-12 | 3e-12 / 3e-12 / 1e-11 |
| full/fieldsplit(no_ale P) | 3.45 | 11.8 (17) | 3.33 | 3e-11 / 8e-11 / 3e-13 | 4e-13 / 8e-13 / 1e-10 |
| no_ale/fieldsplit(u+vp lu) | 5.53 | 1 | 2.38 | 8e-12 / 1e-11 / 2e-12 | 3e-12 / 3e-12 / 1e-11 |

At full amplitude on the production mesh the exact-subsolve variant still
needs one FGMRES iteration: the dropped displacement-block terms do not
affect the Schur complement approximation with the stiffened extension.

## Production validation, chi 1.25 / 0.45 (defaults since `4b8a08a`)

- no_ale/fieldsplit, 8 ranks, 0-15 s: completed, 6 000 steps, 75 min, 4.0
  Newton iterations per step, 10.6 FGMRES per solve (max 15), max true
  relative residual 1.0e-6, J_min 0.154, max condition number 16. Direct (8
  ranks, 49 min) and fieldsplit agree over [0, 15] to <= 6e-7 of the signal
  range. The serial fieldsplit run (same parameters) gives the same numbers.
- dt 0.00125, 4 ranks, 12 000 steps, 176 min: J_min 0.153; developed
  statistics within 0.9 % of dt 0.0025, frequencies +0.08 %.
- Continued to 20 s (8 ranks, from the t = 15 state): J_min 0.159, identical
  statistics on [15, 17.5] and [17.5, 20].
- chi 1.25 / nu 0.3 (serial fieldsplit) also completes, J_min 0.035
  (predicted 0.036).
- Opt-in long tests pass: 800 coarse steps (serial) and 100 steps from the
  developed 8-rank state at t = 12 (8 ranks). Default suite: 91 passed.
- Fine mesh (8 ranks): in progress, J_min 0.048 at t = 10.48.
