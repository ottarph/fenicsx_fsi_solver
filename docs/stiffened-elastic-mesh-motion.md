# Stiffened elastic mesh motion: usage, validation and limitations

Implementation of [stiffened-elastic-mesh-motion-plan.md](stiffened-elastic-mesh-motion-plan.md):
the monolithic shared-space FSI2 solver with the reference-configuration,
Jacobian-stiffened linear-elastic mesh extension of Shamanskiy and Simeon
(Comput. Mech. 67, 583-600, 2021). The detailed record of decisions and
measurements is `notes/stiffened-elastic-mesh-motion/implementation-log.md`.

## Formulation

The problem is that of `fsi2_harmonic_diffmesh` (shared P2 displacement and
velocity on the whole mesh, P1 pressure on the fluid submesh, FSI2
materials, boundary conditions, inflow ramp, theta = 1/2 + dt), with the
harmonic mesh terms replaced by

```text
R_m(u; phi) = alpha [ int_{Omega_f^0} sigma_m(u) : eps(phi) dX - int_{Gamma_fs^0} (sigma_m(u) n_f) . phi dS ]
sigma_m(u)  = w (2 mu_0 eps(u) + lambda_0 tr(eps(u)) I),   eps(u) = sym(grad_X u)
w           = (j_star / j_0)^chi,   j_0 = |det D_xi G_0| (pointwise, initial geometry)
```

- `mu_0, lambda_0` from `E_0 = 1` and `nu_m`; `alpha = 1e-9` is the separate
  mesh-equation scale of the shared displacement rows (as `alpha_u` in the
  harmonic solver).
- `j_star` is the mean initial fluid-cell volume divided by the parent-cell
  volume (owned cells, global reduction), so `w = 1` in an affine cell of
  mean size. It is the same for residual, Jacobian, interface flux and
  preconditioner, since all are derived from one UFL form.
- `j_0` is `abs(ufl.JacobianDeterminant(mesh))` of the never-moved mesh, also
  on curved quadratic cells and on the one-sided interface facets (fluid
  cell); `cell_volume` weighting is an experimental DG0 alternative.
- The mesh law is linear in `u` and non-incremental (always from the
  reference configuration), fully implicit (no theta weighting). The
  interface displacement is shared with the solid through the displacement
  DOFs; the flux term uses the fluid-side stress and normal.
- Quadrature degree 6 on triangles and 10 on quadrilaterals (doubling it
  changes the assembled residuals by 8e-10 and 9e-8).

The channel of the repository's FSI2 meshes is 2.5 m long (the comparison
paper uses 2.2 m); the geometry is unchanged.

## Usage

```bash
conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh \
  --mesh data/meshes/fsi2/mesh_sec.xdmf --T 15 --dt 0.0025 \
  --linear-solver fieldsplit --jacobian-mode no_ale \
  --mesh-stiffening-exponent 1.25 --mesh-poisson-ratio 0.45 \
  --checkpoint-every 0.5 --output-dir output/fsi2_stiffened_elastic/run
```

MPI (MPICH's launcher in the conda environment):

```bash
conda run --no-capture-output -n xfsi_solver mpiexec -n 4 python -m xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh \
  --mesh data/meshes/fsi2/mesh_sec.xdmf --T 15 --dt 0.00125 --linear-solver fieldsplit --jacobian-mode no_ale \
  --mesh-stiffening-exponent 1.25 --mesh-poisson-ratio 0.45 --output-dir output/fsi2_stiffened_elastic/run_np4
```

Other options: `--linear-solver direct` (MUMPS), `--jacobian-mode full`,
`--preconditioner-mode no_ale` (full Newton, no-ALE preconditioning matrix),
`--displacement-fluid {cholesky,amg,gamg}`, `--snes-atol/--snes-rtol/--ksp-rtol`,
`--mesh-equation-scale`, `--mesh-quadrature-degree`, `--mesh-weighting`,
`--output-every N` (VTX every N steps), `--restart STATE` (a state written by
this solver with the same mesh, dt, mesh law and number of ranks),
`--sample-resolution`. From Python:

```python
from xfsi_solver.solvers.fsi2_harmonic_diffmesh import SolverConfig
from xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh import MeshMotionConfig, solve

result = solve("data/meshes/fsi2/mesh_sec.xdmf", T=15.0, dt_val=0.0025, output_dir="output/run",
               config=SolverConfig(jacobian_mode="no_ale", linear_solver="fieldsplit"),
               mesh_config=MeshMotionConfig(mesh_stiffening_exponent=1.25, mesh_poisson_ratio=0.45))
```

Time steps use accepted-time semantics: step `n` imposes the inflow at, and
reports the state at, `t0 + n dt`, through exactly `T` (`T - t0` must be a
multiple of `dt`); the initial state is the first QoI row. The harmonic
solver keeps its legacy loop (same states, round-off dependent endpoint).

## Output and diagnostics

`OUTPUT_DIR/qoi.txt` (`t, drag, lift, A_x, A_y`), `uv.bp`, `p.bp` (VTX),
`checkpoints/state_t*.npz` (per rank, with metadata: solver, mesh sha256,
dt, mesh law, time semantics), `diagnostics.jsonl` (one record per accepted
step, and one for a failed step) and `run.json` (configuration, software
versions, ranks, mesh path/checksum/cells, DOFs, `j_star`, weight range,
reference-geometry check, outcome and totals, or the failure with the last
accepted record).

Each step record has the time and step size, Newton iterations, SNES and
KSP convergence reasons, FGMRES iterations and true relative residuals of
every linear solve, timings (Jacobian, auxiliary assembly, preconditioner
setup, Krylov, step), preconditioner work, the final per-field residuals
(mesh rows also divided by `alpha`; kinematic, momentum, incompressibility),
QoIs, and the sampled geometry: min/max of `J_ALE = det(I + grad_X u)` and
the maximal condition number of `I + grad_X u`, separately in fluid and
solid, and non-finite counts. The geometry is sampled on a lattice with 6
subintervals per parent-cell edge (28 points per triangle, 49 per
quadrilateral, vertices and edges included) of every cell: a sampled
validity check, not a proof of bijectivity. A Newton iterate failing it is
rejected as a SNES function domain error (the step fails with an error; no
coefficient is clipped and no invalid state is accepted), an accepted state
failing it raises `InvalidStateError`.

`python -m xfsi_solver.scripts.fsi2_qoi_statistics --window 12 14.6 QOI...`
tabulates mean, amplitude and frequency against the published FSI2 data.
`python -m xfsi_solver.scripts.fsi2_harmonic_diffmesh_benchmark --solver
stiffened_elastic ...` compares linear/nonlinear solver modes from a
restart state.

## Fieldsplit integration

The production hierarchy, index sets and options of
[shared-space-iterative-solver.md](shared-space-iterative-solver.md) are
reused unchanged (prefix `fsi2_stiffened_elastic_diffmesh_`). The
block-triangular displacement preconditioner extracts `A_ff` (fluid
interior) and `A_fI` from the preconditioning matrix, so it applies the
elastic operator without a separate assembly (tested equal to the
alpha-scaled weighted stiffness to 1e-12; symmetric; Cholesky, factored
once). `displacement_fluid="gamg"` is GAMG-CG with the rigid body modes of
the fluid-interior DOFs as near-nullspace (not an exact nullspace of the
Dirichlet-restricted block); `"amg"` is BoomerAMG-CG.

## Tests

- `tests/test_mesh_extension.py`: weights (unit at chi = 0, inverse-volume
  power on affine cells, pointwise variation on curved cells, ghosts,
  normalization on 1 and 2 ranks), rigid-motion invariance, the Lame law,
  modulus-scale invariance and history independence of standalone
  extensions (`LinearProblem`), fluid-sided interface measure, quadrature
  increase on curved triangles and quadrilaterals.
- `tests/test_fsi2_stiffened_elastic_diffmesh.py`: replaced mesh terms,
  finite-difference Jacobians (all blocks, the elastic volume and interface
  terms separately), the no-ALE contract, coupled startup with full/direct,
  no_ale/direct and no_ale/fieldsplit, boundary conditions, times, geometry,
  restart and metadata, quadrilateral Newton step, CLI, benchmark script, 2
  ranks.
- `tests/test_fsi2_stiffened_elastic_fieldsplit.py`: exact block
  factorization vs LU, elastic `A_ff` extraction, AMG/GAMG solves, hierarchy
  views, 2 ranks.
- `tests/test_fsi2_stiffened_elastic_long.py` (opt-in, `pytest -m long`):
  800 coarse steps direct vs fieldsplit; 100 steps from a developed state
  (`XFSI_ELASTIC_STATE`).

## Validation (2026-09-25)

Local workstation, 32 physical cores; DOLFINx 0.11.0, Basix 0.11.0, UFL
2026.1.0, PETSc/petsc4py 3.25.5, MPICH, Python 3.14.6. Production mesh
`data/meshes/fsi2/mesh_sec.xdmf` (5 861 quadratic triangles, 2.5 m channel,
sha256 of `.xdmf` + `.h5` `1b3ae40cfff7ba50...` as recorded in `run.json`;
50 540 DOFs: 23 898 u, 23 898 v, 2 744 p). `j_star` 3.946e-4. Large output
stays in `output/fsi2_stiffened_elastic/` (not in Git).

Tolerances were fixed before any run reached developed motion, from the
deviation of the validated biharmonic run (same mesh and dt) from the
published FSI2 data (implementation log, "Comparison tolerances").

### The paper's parameters fail on this mesh (not validated)

`chi = 2.5`, `nu_m = 0.3`, dt 0.0025, no_ale, direct and fieldsplit: the run
stops at **t = 7.83 s** (last accepted state 7.8275, A_y = -0.035, sampled
fluid J_min 0.019; the next Newton trial has J_min -0.003 and is rejected).
The Newton and Krylov solves converge until then, and direct and fieldsplit
agree to 9e-9 in A_y; the first failure is invalid geometry from the mesh
extension. The worst cells are the large far-field cells at the channel
wall above/below the beam tip: with `w = (j_star/j_0)^2.5` and a 190x range
of initial cell sizes (weight contrast 5e5) these cells are the softest and
absorb the whole compression of the 0.2 m gap between beam and wall.
Reproduce with:

```bash
conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh \
  --mesh data/meshes/fsi2/mesh_sec.xdmf --T 15 --dt 0.0025 --linear-solver direct --jacobian-mode no_ale \
  --mesh-stiffening-exponent 2.5 --mesh-poisson-ratio 0.3 --output-dir output/fsi2_stiffened_elastic/chi2.5
```

(`run.json` then holds `"status": "failed"`, the failure record and the last
accepted diagnostics). The harmonic solver fails on this mesh at 7.35 s.

### Parameter study (`mesh_sec`, dt 0.0025, T 15)

| chi / nu_m | outcome | predicted worst J | min sampled fluid J |
|---|---|---|---|
| 3.0 / 0.3 | failed at 7.81 s (wall) | -2.92 | 0.001 |
| 2.5 / 0.3 | failed at 7.83 s (wall) | -1.89 | 0.019 |
| 2.5 / 0.45 | failed at 8.33 s (wall) | -1.80 | 0.003 |
| 2.0 / 0.3 | failed at 8.33 s (wall) | -0.96 | 0.009 |
| 1.5 / 0.45 | failed at 9.91 s | -0.006 | 0.001 |
| 1.0 / 0.3 | failed at 9.41 s (beam-tip corner) | -0.09 | 0.002 |
| 1.25 / 0.3 | see below | 0.036 | 0.035 |
| **1.25 / 0.45** | **completed 0-15 s** | 0.18 | 0.154 |

"Predicted" is the worst sampled J of the standalone elastic extension of 11
developed biharmonic beam shapes (t in [12, 15]); its sign predicted every
outcome. Stiffening protects the beam-tip corner (it fails for small chi)
at the expense of the far field (large chi); on this mesh only a narrow
window near chi = 1.25 works. This is a parameter choice of the same
method; no weight clipping, blending, remeshing or change of the solid was
used. The coarse mesh behaves alike (chi 2.5 and 2.0 fail at 9.74 and
10.29 s; 1.0, 1.25 and 1.5 complete).

### Full FSI2 run with the defaults (chi 1.25, nu_m 0.45)

Production solver (no_ale Jacobian, FGMRES + fieldsplit, 8 ranks) from rest
to exactly T = 15 s, dt 0.0025: 6 000 steps, 4.0 Newton iterations per step,
10.6 FGMRES iterations per solve (max 15), every true relative linear
residual <= 1.0e-6, no rejected iterate, sampled fluid J >= 0.154, max
deformation condition number 16, solid J >= 0.952; 75 min wall time
(direct on 8 ranks: 49 min). Direct and fieldsplit trajectories agree over
[0, 15] s to 6e-7 (drag), 2e-7 (lift), 4e-7 (A_x) and 6e-8 (A_y) of the
signal range; 100 steps from the developed state at t = 12 agree within the
opt-in long test's 1e-5 (8 ranks); from t = 10.5 all modes, including
full/direct, agree to 1.3e-10 (serial).

Statistics over [12, 14.6] s, `mean +- amplitude [frequency]`, and
deviations (mean relative to the amplitude):

| run | drag | lift | A_x | A_y |
|---|---|---|---|---|
| published FSI2 | 214.2 +- 76.05 [3.859] | 0.61 +- 237.5 [1.931] | -0.01494 +- 0.01260 [3.863] | 0.00125 +- 0.08166 [1.931] |
| biharmonic, dt 0.0025 | 217.0 +- 78.96 [3.853] | -0.58 +- 244.5 [1.928] | -0.01531 +- 0.01289 [3.858] | 0.00128 +- 0.08274 [1.928] |
| elastic, dt 0.0025 | 216.6 +- 78.84 [3.854] | -0.87 +- 242.8 [1.929] | -0.01520 +- 0.01281 [3.858] | 0.00135 +- 0.08244 [1.929] |
| elastic, dt 0.00125 | 215.9 +- 78.53 [3.857] | -0.80 +- 243.7 [1.931] | -0.01521 +- 0.01280 [3.861] | 0.00134 +- 0.08245 [1.930] |

| elastic dt 0.0025 vs | drag mean / amp / freq | lift | A_x | A_y |
|---|---|---|---|---|
| published | +3.2 / +3.7 / -0.12 % | -0.6 / +2.2 / -0.08 % | -2.1 / +1.7 / -0.12 % | +0.1 / +1.0 / -0.10 % |
| biharmonic | -0.5 / -0.15 / +0.02 % | -0.1 / -0.7 / +0.03 % | +0.9 / -0.6 / +0.02 % | +0.1 / -0.4 / +0.03 % |
| dt 0.00125 | -0.8 / -0.4 / +0.08 % | +0.03 / +0.4 / +0.08 % | -0.1 / -0.1 / +0.08 % | -0.01 / +0.02 / +0.08 % |

All deviations are within the pre-registered tolerances (published: A_y
amplitude 5 %, other amplitudes and A_x mean 10 %, frequencies 2 %;
biharmonic: A_y amplitude 3 %, others 5 %, frequencies 1 %). The elastic
and biharmonic runs are closer to each other than either is to the
published data, i.e. the remaining deviation is the discretization of this
mesh and time step, not the mesh motion.

### Sustained motion to 20 s

The fieldsplit run was continued from its t = 15 state to T = 20 s (8 ranks,
2 000 steps): sampled fluid J >= 0.159, max condition number 15.6, FGMRES
<= 15 iterations, no rejected iterate. The statistics of [15, 17.5] and
[17.5, 20] are identical to four digits (A_y 0.001349 +- 0.08244 [1.929],
drag 216.4 +- 78.4 [3.860]) and equal those of [12, 14.6]: the mesh quality
does not degrade over periods (no accumulated distortion, as expected of the
non-incremental extension).

### Mesh-resolution sensitivity

`data/meshes/fsi2/mesh_fine_sec.xdmf` (21 890 cells, generated with
`fine=True`, see Meshes below), chi 1.25 / nu_m 0.45, no_ale/fieldsplit, 8
ranks, dt 0.0025: **completed 0-15 s** (6 000 steps, 5.7 h), no rejected
iterate, sampled fluid J >= 0.046 (predicted 0.087; the finer cells at the
beam-tip corner deform more), max condition number 58. Statistics over
[12, 14.6]: drag 217.5 +- 78.87 [3.852], lift -0.54 +- 242.2 [1.928], A_x
-0.01523 +- 0.01282 [3.857], A_y 0.00127 +- 0.0825 [1.928]. Against
`mesh_sec`: drag mean +1.1 % of the amplitude, all amplitudes within 0.25 %,
frequencies -0.05 %; against the published data: A_y amplitude +1.0 %, drag
+4.3 / +3.7 % (mean / amplitude), frequencies -0.16 %; against biharmonic
within 1 %. The developed motion is converged in the mesh to about 1 %. The
fieldsplit iterations grow with the refinement: 22.6 FGMRES iterations per
solve on average (10.6 on `mesh_sec`), at most 113 (true relative residual
still <= 1e-6).

## Meshes

`data/` is not in Git. The existing meshes were used; the fine mesh was
generated (new file) with

```bash
conda run --no-capture-output -n xfsi_solver python -c "from xfsi_solver.scripts.create_mesh_FSI2 import create_mesh; create_mesh(fine=True, coarse=False, quads=False, semi_structured_quad=False, second_order=True)"
```

run from the repository root (it writes `data/meshes/fsi2/mesh_fine_sec.xdmf`
relative to the working directory, so run it elsewhere to avoid overwriting
an existing file). The same call with `fine=False` writes `mesh_sec.xdmf`;
with the installed gmsh it produces 5 851 cells, not the 5 861 of the
existing `mesh_sec` used here (equivalent, not identical: run it in a
scratch directory). `semi_structured_quad=False` is required for triangles,
otherwise `_ssq` is added to the name. `python -m
xfsi_solver.scripts.create_test_meshes` generates the test meshes.

## Limitations

- The paper's chi = 2.5 / nu_m = 0.3 is **not validated** on this
  repository's meshes: it fails at 7.83 s on `mesh_sec` (and at 9.74 s on the
  coarse mesh; the standalone prediction fails also on the fine mesh). The
  validated chi = 1.25 / nu_m = 0.45 sits in a narrow window (J_min 0.154
  on `mesh_sec`, 0.046 on the fine mesh); nu_m = 0.3 completes on `mesh_sec`
  with a sampled J_min of only 0.035. The window depends on the
  mesh grading (initial cell-size range 190x on `mesh_sec`), so other meshes
  need the standalone check (implementation log) or a full run.
- The validity check samples 28 (49) points per cell; it is not a proof of
  bijectivity. Invalid Newton trials stop the run (line search `none`);
  `--snes-linesearch bt` backtracks from rejected trials but was not needed
  and is not validated.
- Performance: at 50 000 DOFs MUMPS is faster than the fieldsplit solver
  (8 ranks: 49 vs 75 min for 0-15 s), as for the harmonic solver. On the
  fine mesh the average FGMRES count doubles (22.6 per solve, maximum 113):
  the one-V-cycle velocity and selfp pressure approximations are not
  mesh-independent at developed motion. The
  iterative fluid-displacement solves (`amg`, `gamg`) are validated but 20x
  slower than the once-factored Cholesky solve in 2D.
- Restart states are per rank count and require the same solver, mesh
  checksum, dt and mesh law; harmonic checkpoints are rejected.
- The 2.5 m channel of the repository's meshes is kept (the paper uses
  2.2 m).
