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
