# Implementation log: Newton contraction under mesh refinement

Does the linear contraction of Newton's method with the approximate Jacobian J_0 (without the fluid
ALE derivatives) break down when the mesh is refined? The study runs the FSI2 window of
`notes/newton-convergence/report.md` on a family of refined meshes, from states transferred from the
`mesh_sec` checkpoint at t = 16.0025 instead of spun up from rest on every mesh.

## 2026-11-02: Meshes

`create_mesh_FSI2.create_mesh` takes `refinements`: the linear mesh is split uniformly by gmsh that
many times before it is made second order, so the meshes are nested and the new nodes on the cylinder
lie on the circle (checked: |r - 0.05| < 1e-16 at all nodes of the cylinder facets). With the options
of `mesh_sec` (triangles, `semi_structured_quad=False`, second order), the current gmsh makes 5851
cells, not the 5861 of `mesh_sec`, so `mesh_sec` cannot be refined itself (its .msh is gone, and
`dolfinx.mesh.refine` crashes on its quadratic geometry with a heap corruption). The family is
therefore regenerated, in `data/meshes/fsi2/refinement/`:

| mesh | cells | nodes |
|---|---:|---:|
| `mesh_sec_ref0` | 5851 | 11929 |
| `mesh_sec_ref1` | 23404 | 47262 |
| `mesh_sec_ref2` | 93616 | 188140 |

## 2026-11-02: State transfer

`xfsi_solver.scripts.transfer_fsi2_checkpoint` reads the latest checkpoint of a run on one mesh and
writes its state, interpolated onto another mesh of the domain (`interpolate_nonmatching`, padding
1e-6 for the curved boundary), as a checkpoint that `initial_state_dir` reads: u and v between the
whole meshes, p and z between the fluid submeshes, z only if the source has it. Tested on polynomial
fields from `mesh_sec_coarse` to `mesh_sec` (exact to 1e-10 away from the curved cells, serial and on
3 ranks), without z, and by starting `fsi2_biharmonic_mixed` from a transferred solver state. The
state of `output/newton_convergence/init_mixed/checkpoints` (t = 16.0025, with z) was transferred to
the three meshes (`output/newton_refinement/states/ref*`).

## 2026-11-02: First settle attempt: a transfer transient

One period of `fsi2_biharmonic_mixed` with reuse (gamma 0.2), from the transferred states, on 4, 8
and 16 ranks (`output/newton_refinement/first_try/`):

- ref0 ran the period, with the iterations of `mesh_sec` (7.39 per step against 7.57, median rate
  0.028), and the QoIs of `mesh_sec` (tip displacement to 7e-7, lift to 0.9).
- ref1 stopped at t = 16.0375 (DIVERGED_MAX_IT) and ref2 at t = 16.0175 (DIVERGED_DTOL). A J_0 reused
  into the next step grew the residual in its second iteration (factors 2 to 170), and a fresh J_0
  contracted by 0.03 to 0.3, worse step by step.

The cause is the transfer, not yet the refinement. On `mesh_sec` the initial residual of every step
is 97 in the v block. On the transferred meshes it grows step by step: on ref0 to 980 after 20 steps,
decaying to 170 by the end of the period; on ref1 to 4900 within 13 steps. The interpolated state is
not a discrete solution on the new mesh, and the mismatch rings, hardly damped by theta = 0.5 + dt;
the finer the mesh, the larger the mismatch. A transfer from `mesh_sec` onto `mesh_sec` itself keeps
the initial residual at 97 (15 steps), so the transfer loses no state. In the first step, the
interpolated u also leaves a residual of about 1e3 in the mesh motion rows on ref1 and ref2, which
the first Newton step removes.

Plan: on every mesh, 20 steps of backward Euler (`absolute_theta_shift=0.5`) to damp the transient,
then a period with the usual theta, both reassembling J_0 in every iteration, and the measurements
from the end of that.

## 2026-11-02: Damped settling and the measurements

On every mesh, 20 steps of backward Euler (`absolute_theta_shift=0.5`), then 200 steps with the
usual theta, both with J_0 reassembled in every iteration, on 2, 4 and 10 ranks. Backward Euler
killed the transient: the initial residual stayed at 105 (ref0), 55 to 63 (ref1) and 30 to 33
(ref2), and after the switch back at 97, 50 and 28, halving with each refinement like the l2 norm
of a residual that integrates over smaller cells. ref0 settled at the initial residual of
`mesh_sec` (97).

The measurements, one period from t = 16.5525 on each mesh (`output/newton_refinement/`):

| run | h | h/2 | h/4 |
|---|---|---|---|
| J_0 predicted, later iterations (median) | 3.6e-3 | 4.2e-3 | 4.9e-3 |
| J_0 predicted, first iteration | 3.2e-3 | 3.2e-3 | 3.1e-3 |
| J_0 predicted, its/step (rtol 1e-11) | 4.75 | 4.97 | 5.07 |
| J_0 not predicted, later iterations | 5.5e-3 | 5.9e-3 | 6.5e-3 |
| J_0 not predicted, first iteration | 0.17 | 0.16 | 0.16 |
| full Jacobian predicted, rates 0→1, 1→2 | 4.7e-4, 4.5e-6 | 4.9e-4, 4.3e-6 | 4.9e-4, 3.5e-6 |
| J_0 predicted reused, its/step, assemblies | 6.72, 109 | 6.27, 110 | 6.23, 110 |

The J_0 rate grows by 8 to 17% per halving of h; the ratio of successive updates falls with the
prediction (1.46e-2, 1.29e-2, 1.18e-2) and grows without it (6.4e-3, 8.2e-3, 9.2e-3). On ref0 the
rates reproduce those of `mesh_sec` in the Newton convergence report. The full Jacobian reaches
round-off before the relative tolerance on ref1 and ref2, and most of its steps there stop on
PETSc's default `snes_stol` (1e-8) after 3 iterations. The ref2 measurements ran side by side on 3
ranks each; full_pred took 2.6 hours, slowed in its last hour by another user's job on about 30
cores. Written up in `report.md`, with the figures of `make_figures.py`.

## 2026-11-02: Time step sweep and the spectral radius of J_0^-1 N

`xfsi_solver.tools.spectral.dominant_eigenvalues` finds the eigenvalues of largest modulus of
J_0^-1 N with SLEPc (Krylov-Schur on a shell operator: N, then a MUMPS solve with J_0, factorized
once). `fsi2_biharmonic_fulljacobian_initialguess` writes them with `spectral_every` and
`spectral_path`, at the solution of a step: it reassembles J there, assembles N (the u-derivatives of
the fluid integrals of the momentum rows and of the continuity rows, with the Dirichlet rows and
columns zeroed) into a matrix of the layout of J, and takes J_0 = J - N. `spectral_left_out` keeps
only the momentum or only the continuity part in N. Checked on the settled state of h: the spectral
radius (3.8e-3 to 4.5e-3 over six steps) tracks the measured later-iteration rates of `j0_pred`
(4.6e-3 to 6.5e-3), and its dominant eigenvalues are a complex pair, which explains the alternating
measured ratios.

Runs over the first 0.12 s of the measured period from the settled states (`output/newton_dt/`),
on 10 ranks next to another user's job of 20 to 30 cores; h/4 at dt = 0.000625 over 0.04 s only:

| mesh | dt | rho (median) | measured later | measured first |
|---|---|---:|---:|---:|
| h | 0.005 | 7.0e-3 | 6.6e-3 | 9.6e-3 |
| h | 0.0025 | 4.6e-3 | 5.8e-3 | 5.1e-3 |
| h | 0.00125 | 3.9e-3 | 5.1e-3 | 2.6e-3 |
| h | 0.000625 | 3.2e-3 | 4.4e-3 | 1.3e-3 |
| h/2 | 0.0025 | 5.5e-3 | 6.7e-3 | 5.1e-3 |
| h/2 | 0.00125 | 4.8e-3 | 6.7e-3 | 2.6e-3 |
| h/4 | 0.0025 | 6.3e-3 | 7.7e-3 | 5.0e-3 |
| h/4 | 0.000625 | 4.7e-3 | 7.6e-3 | 8.0e-4 |

- In h at dt = 0.0025, rho grows by 1.20 and 1.14 per halving, as the measured rates do, so the
  growth is not an artefact of the l2 norm of the residual.
- In dt on h, rho falls only as about dt^0.36 (1.53, 1.17, 1.21 per halving), against the scaling
  argument of the first version of the report, which predicted rho ∝ dt. Only the first iteration
  with the prediction contracts in proportion to dt. At dt = 0.005 the dominant eigenvalue is real
  in every sample (another mode); below, a complex pair in 83 to 100% of the samples.
- Along dt ∝ h, over the first 0.04 s: rho = 4.79e-3, 4.85e-3, 4.74e-3, within 2% at every sample.
  The measured later-iteration rate grows along it (5.6e-3, 6.3e-3, 7.6e-3), but every step takes
  5.00 iterations, as the first iteration speeds up in proportion to dt.
- Split on h (scratch copy of the solver, then `spectral_left_out`): only the momentum part left out
  gives 1.1e-2 (dt = 0.0025) and 6.9e-3 (0.000625), only the continuity part 7.2e-3 and 4.3e-3,
  both 4.6e-3 and 3.2e-3. The parts partly cancel, and each falls with dt as slowly as both.

Report and artifact extended with a section on the time step and the spectral radius, three
figures (`rho_vs_dt`, `rho_over_window`, `rho_vs_h`), findings 3 and 7 to 10, and a revised
explanation (the added mass of the pressure, as a hypothesis).

## 2026-11-02: The dominant eigenvectors

`xfsi_solver.tools.spectral.dominant_eigenpairs` returns the eigenvectors with the eigenvalues, and
`fsi2_biharmonic_fulljacobian_initialguess` writes them with `spectral_vectors_dir`, one checkpoint
per sample with the real and imaginary parts of each as functions `<f>_re_<i>`, `<f>_im_<i>` and the
solution of the step. Tested on one and two ranks: unit norm, and on the solid u = dt theta v
exactly, as N has no solid rows.

Runs over the first 0.04 s from the settled states (`output/newton_modes/`), 4 eigenpairs every
0.01 s: h at dt = 0.005, 0.0025, 0.00125 and 0.000625, h/2 at 0.0025 and 0.00125, h/4 at 0.0025 and
0.000625, on 2, 2 and 4 ranks next to another user's job of 20 to 45 threads. The eigenvalues agree
with those of the time step sweep at the same times to 1e-4. `analyse_modes.py` scales each mode to
||du_I||_L2 = 1 on the interface (N acts only on u in the fluid, where J_0 extends u from the
interface by the mesh motion, so du_I sets the mode) and measures it.

- The pressure part is large: ||dp|| ~ 4e7 against ||dv_I|| ~ 800 = 1/(dt theta), about
  rho_f L dv / dt. In the 2-norm of the vector the pressure dominates, so the solid part of a mode
  converges loosely at the SLEPc tolerance of 1e-6: the kinematic relation u = dt theta v on the
  solid holds to 1e-5 in the complex pairs but only to 1e-3 to 2e-2 in some real modes.
- Two families. The complex pair of largest modulus is concentrated at the tip, 92% of |du_I|^2 within
  the last 10% of the beam and the tip at dt = 0.0025, with its peak at the lower tip corner. The real
  modes change the area of the beam (88 to 97% of int |du_I . n| is net area change), and their
  pressure is a channel-wide field, flat upstream (the inflow velocity is fixed) and falling linearly
  to the outflow, the inertia of the fluid column that the area change pushes out.
- On h, as dt halves from 0.005 to 0.000625, the tip pair falls by about dt^0.25 and localizes further
  (solid velocity within 0.01 of a corner: 26%, 56%, 80%, 93%), and the largest real volume mode by
  about dt^0.6. At dt = 0.005 a real volume mode dominates, which is the real dominant eigenvalue of
  the sweep; below, the tip pair does.
- The pressure of the tip pair, scaled by rho_f |dv_I| / dt, falls from 0.5 to 0.006 as dt halves from
  0.005 to 0.000625: its pressure does not scale like 1/dt, so the added-mass explanation of the
  first version of the time step section does not hold for the dominant mode.
- The state at the lower tip corner, a re-entrant corner of the fluid, grows with refinement: max |grad
  v| within 0.01 of it 243, 357 and 489, max |p| 797, 957 and 1165 on h, h/2 and h/4, while at the
  upper tip corner and the beam-cylinder junction they stay flat. A Stokes corner of 270 degrees has
  grad v ~ r^-0.455, 1.37 per halving.
- A test of the volume modes by the Poisson ratio of the solid (0.2, 0.3, 0.45, 0.49 against 0.4, on h
  at dt = 0.0025, 4 steps from the settled state) is inconclusive: changing the material kicks the
  beam out of its settled state, and the dominant modes of those runs sit at single interface nodes
  along the beam (median |du_I| 1e-4 to 1e-5 of the peak), not at the tip. At 0.49 the second step
  does not reach the tolerance in 8 iterations. The same protocol at 0.4 reproduces the main run.
- In h at dt = 0.0025, the tip pair grows (4.89e-3, 5.77e-3, 6.52e-3: 1.18 and 1.13 per halving, as
  the spectral radius) and sharpens at the lower tip corner; the volume mode does not (3.54e-3,
  3.37e-3, 3.47e-3), and its interface displacement is the same on the three meshes.
- Along dt ∝ h, the tip pair keeps its modulus (4.89e-3, 4.97e-3, 4.89e-3) while concentrating at the
  corner (56%, 86%, 97% of its solid velocity within 0.01 of a corner). h/4 at dt = 0.000625 took
  4047 s on 4 ranks.

Report and artifact extended with a section on the slow modes, three figures (`mode_fields`,
`mode_interface`, `mode_families`) and findings 11 to 13; the added-mass explanation of the time step
section is replaced, and the next steps now target the tip pair and the volume modes.

## 2026-11-03: The 30 dominant eigenvalues

Question: are there only a few slow modes, with a gap below them, so that a correction of those
few would speed up J_0 noticeably?

- Runs `output/newton_modes/modes30_ref<n>_dt<dt>/`, from the settled states, over the first 0.02 s
  with `spectral_nev=30` at t = 16.5625 and 16.5725: h at dt = 0.0025 and 0.005, with the
  eigenvectors (about 50 s each on 4 ranks), and h/4 at dt = 0.0025 without them (732 s on 4
  ranks). The first four eigenvalues agree with the 4-eigenvalue runs on h to 3e-4 relative for
  the tip pair and to 1% for the real modes, within the SLEPc tolerance of 1e-6 on the vector.
- On h at dt = 0.0025, relative to the largest: the tip pair (1), a volume mode peaking at the lower
  tip corner (0.74 to 0.77, area change 91%), a second volume mode peaking on the bottom 0.025
  before the tip (0.59 to 0.62, 86 to 88%), and a real mode at the upper tip corner (0.50 to 0.55,
  on about 4 interface nodes). Then, from 0.44, a dense band of complex pairs falling only to 0.30
  at k = 30, peaking on the bottom 0.04 to 0.06 before the tip and spread over 10 to 60 nodes.
- At dt = 0.005: the two volume modes (1, 0.84 to 0.89), the tip pair (0.81), the upper-corner mode
  (0.41 to 0.44), then the band from 0.33 to 0.36 down to 0.23.
- On h/4 at dt = 0.0025 (sorted by modulus only): the tip pair (5.73e-3, 6.25e-3), three real
  eigenvalues at 0.46 to 0.57 (the first 3.14e-3 and 3.56e-3, as on h), then the band from 0.38 to
  0.39 down to 0.33. The next eigenvalue after the tip pair does not change with h, so the gap below
  the tip pair widens with h (0.75 on h, 0.56 on h/4); the band rises by 1.1 to 1.2.
- Gain: without the tip pair the spectral radius would be about 3.4e-3 on both meshes; without all
  four leading modes, the top of the band, 2.0 to 2.4e-3. At the tolerance of 1e-11, 5 iterations
  per step (first 2.3 decades, later 2.2) drop to 4 only if the later iterations contract by
  1.3e-3, 4.6 times below the measured 5.8e-3, so neither correction saves an iteration.
- Driver scripts in the session scratchpad: `modes30_runs.py` (`mode_runs.py` with
  `WINDOW, SAMPLE, NEV = 0.02, 0.01, 30` and the vectors optional) and `modes30_chain.sh`.

Report extended with finding 14, a subsection "Below the dominant modes" with the figure
`spectrum_decay`, a caveat on the two samples, and the first next step now quantified and split
into a measurement in the monolithic solver and the low-rank form it would take in `fsi2_simplified`.

## 2026-11-03: The CFL number

Question: is there a CFL condition in the FSI system? Not for stability (implicit fluid and solid
with θ = 0.5 + dt, Newton to 1e-11, incompressible fluid, elliptic mesh motion solved in the step,
monolithic coupling), but the CFL numbers turn out to line up the tip pair.

- `cfl_numbers.py` reads the state of the first eigenvector sample of `modes_ref<n>_dt0.0025`
  (t = 16.5625). Largest flow CFL number |v| dt / h (|v| at cell midpoints, h the cell diameter, mesh
  velocity left out): 0.88, 1.87, 3.84 on h, h/2, h/4 (99th percentile 0.64, 1.26, 2.54; max |v|
  2.23). Solid c_p dt / h 11, 22, 44 and c_s dt / h 4.5, 9.0, 18 (c_p = 17.3, c_s = 7.07).
- The tip pair against the largest CFL number (scaled with dt from those values): on h 3.43e-3,
  4.16e-3, 4.89e-3, 5.45e-3 at 0.22, 0.44, 0.88, 1.76 (about CFL^0.22); at about 0.9 the three
  meshes give 4.89e-3, 4.97e-3, 4.89e-3, at about 1.8 h and h/2 give 5.45e-3 and 5.77e-3. The volume
  mode lines up in dt instead (3.54e-3, 3.37e-3, 3.47e-3 at 0.0025; 2.42e-3, 2.45e-3 at 0.00125),
  except h/4 at 0.000625 (2.41e-3 against 1.72e-3 on h, a volume mode in only 2 of 4 samples).
- The sweep's spectral radius at dt <= 0.0025 goes as dt^0.26 at fixed h and h^-0.23 at fixed dt,
  about (dt/h)^0.25, which explains the flat line along dt ∝ h (CFL 0.88 to 0.96).
- Not derived: the tip pair is about 100 times smaller than dt max |∇v| at the corner (0.61 on h).

Report extended with finding 15, a section "The CFL number" with the figure `mode_cfl`, a caveat,
and a next step to test the scaling on h/2 at dt = 0.005 and h/4 at dt = 0.00125.

## 2026-11-03: Rounded tip corners

Question: is the growth of the tip pair with h the singularity at the lower tip corner, or the CFL
number? A rounded corner tells them apart: the singularity goes, the CFL number stays.

- `create_mesh_FSI2.create_mesh(tip_fillet=r)` replaces the two corners of the free end of the beam
  by quarter circles of radius r (lines to the tangent points, `addCircleArc` about the centres,
  the unused corner and centre points removed before the cut). With r = 0 the mesh is unchanged
  (the h5 of ref0 identical to the stored one). With r = 0.005 and 0, 1 and 2 refinements: 5885,
  23540 and 94160 cells, each quarter circle 2, 4 and 8 P2 edges (5, 9 and 17 nodes), its nodes on the circle to 5e-17, the
  solid area smaller by 1.07e-5 = 2 (1 - π/4) r^2.
- `analyse_modes.py` reads the radius from the run name (`modes_ref<n>_fillet<r>_dt<dt>`), takes
  the arc length and the normal around the quarter circles, and the radial normal there for the
  fluxes. On the sharp mesh, s, the normals and the interface dofs are identical to before; on
  the rounded one the interface is 0.0043 = (4 - π) r shorter, and the divergence check (u =
  (x - X0, 0)) leaves the same junction flux as on the sharp mesh.
- Runs (session scratchpad `fillet/fillet_runs.py`, `fillet_chain.sh`): the `mesh_sec` state at
  t = 16.0025 transferred onto each mesh, damp (20 steps) and settle (200 steps) as before, then the
  4-eigenpair runs on h at dt = 0.005 to 0.000625, h/2 at 0.0025 and 0.00125, h/4 at 0.0025 and
  0.000625, and the 30 eigenvalues on h at 0.0025; on 2, 2 and 4 ranks next to another user's job
  of 22 cores. Settle took 267, 1077 and 2805 s.
- The settled states: initial residual 97 on h as with sharp corners; the tip shifted slightly in
  phase (A_y = -4.3e-3 against -7.4e-3 at t = 16.5525).
- At dt = 0.0025, tip pair 3.46e-3, 3.38e-3, 3.37e-3 (sharp 4.89e-3, 5.77e-3, 6.52e-3) and volume mode
  4.16e-3, 4.12e-3, 4.11e-3 (sharp 3.54e-3, 3.37e-3, 3.47e-3), the volume mode now dominant. On h
  the tip pair falls as about dt^0.33 (4.17e-3 to 2.12e-3 from 0.005 to 0.000625), the volume mode as
  about dt^0.55. h/2 at 0.00125: 2.70e-3 and 2.82e-3.
- Settle runs, later iterations (median over 199 steps, floor 1e-9): 4.31e-3, 4.47e-3, 4.51e-3,
  against 5.32e-3, 6.01e-3, 6.51e-3 sharp; 4.98, 4.65, 4.53 iterations per step against 5.16, 5.13,
  4.98.
- Corner state (settled, within 0.01 of the lower tip corner): max |p| 763, 786, 766 (sharp 797, 957,
  1165); max |∇v| 294, 362, 417 (sharp 243, 357, 489). Largest flow CFL number 0.81, 1.75, 3.60.
- h/4 at dt = 0.000625 (3859 s on 4 ranks): spectral radius 1.95e-3 to 2.84e-3 (median 2.31e-3, sharp
  4.89e-3), led by a nearly real pair (Im 2% to 3%) on about 8 interface nodes on the bottom of the
  beam at s ≈ 0.295 (x ≈ 0.544, 0.056 before the tip), no area change, pressure ~40 times below the
  tip pair's: the location of the dense band. The same mode is third on sharp h/4 at that dt in two
  samples (2.53e-3, 2.80e-3) and fourth on rounded h (1.2e-3 to 1.8e-3). Tip pair among the 4 dominant
  in one sample only (1.98e-3), the volume mode in none.
- 30 eigenvalues on h: the volume mode first, then the tip pair at 0.75 to 0.81, the second volume
  mode at 0.69 to 0.72 (2.69e-3, as sharp), the band from 0.50 (1.87e-3, as sharp).

Report extended with finding 16, a section "Rounded tip corners" with the figure `fillet`, a caveat,
and the first next step tied to it; finding 15 now notes that its CFL alignment is a coincidence of
exponents.
