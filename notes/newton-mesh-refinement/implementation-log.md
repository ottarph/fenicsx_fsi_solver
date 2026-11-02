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
