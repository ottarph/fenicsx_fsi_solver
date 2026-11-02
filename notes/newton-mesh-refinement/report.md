# Newton contraction under mesh refinement

Does the linear contraction of Newton's method with the approximate Jacobian J_0 break down when
the mesh is refined? The FSI2 window of `notes/newton-convergence/report.md`, one period of the
oscillating beam, is solved on three nested meshes of mesh size h, h/2 and h/4, from the same state.
The history of the work is in `implementation-log.md`.

## Findings

1. **The contraction of J_0 does not break down.** With J_0 reassembled in every iteration, the later
   iterations contract the residual by 3.6e-3, 4.2e-3 and 4.9e-3 (median) on the three meshes with
   the displacement prediction, and by 5.5e-3, 5.9e-3 and 6.5e-3 without it: 8 to 17% more per
   halving of h, roughly h^(-0.1) to h^(-0.2). A step takes 4.75, 4.97 and 5.07 iterations to a
   relative tolerance of 1e-11.
2. **The first iteration does not depend on h.** It contracts by 3.2e-3 with the prediction and by
   0.16 to 0.17 without it on every mesh.
3. **The ratio of successive Newton updates tells the same story, if less clearly.** It estimates how
   fast the error contracts, and falls from 1.5e-2 to 1.2e-2 with the prediction, but grows from
   6.4e-3 to 9.2e-3 without it, like the residual ratio. Part of the growth of the residual ratio
   may be the l2 norm of the residual, which weights the fine-scale components more on a finer
   mesh; either way it is slow.
4. **The full Jacobian stays quadratic.** It takes 3 iterations per step on every mesh, contracting
   by 4.7e-4 to 4.9e-4 and then by 3.5e-6 to 4.5e-6, so the problem itself does not get harder for
   Newton's method on finer meshes.
5. **With reuse, the cost per step does not grow.** J_0 with the prediction and reuse (gamma 0.2)
   takes 6.72, 6.27 and 6.23 iterations per step and assembles J_0 in 109, 110 and 110 of the 199
   steps. The default absolute `snes_atol` asks for a little less on a finer mesh, whose residual is
   smaller (median ||R_0|| 96, 49 and 25), so the counts fall slightly.
6. **J_0 with reuse is fragile to a disturbed state.** A state interpolated onto a finer mesh rings,
   hardly damped by theta = 0.5 + dt, and J_0 with reuse diverged within 7 to 15 steps on h/2 and
   h/4 (see Setup). This is about the transfer, not about the refinement, but it shows that a reused
   J_0 copes badly with high-frequency transients.

## Setup

### Meshes

`create_mesh_FSI2.create_mesh(refinements=n)` splits the linear gmsh mesh uniformly n times before
making it second order, so the meshes are nested and the new nodes on the cylinder lie on the
circle. The current gmsh does not reproduce `mesh_sec` exactly (5851 cells against 5861), and
dolfinx cannot refine a quadratic mesh, so the whole family is regenerated:

| mesh | relative h | cells | nodes |
|---|---|---:|---:|
| `refinement/mesh_sec_ref0` | h | 5851 | 11929 |
| `refinement/mesh_sec_ref1` | h/2 | 23404 | 47262 |
| `refinement/mesh_sec_ref2` | h/4 | 93616 | 188140 |

### States

Every mesh starts from the state of the Newton convergence report, `fsi2_biharmonic_mixed` on
`mesh_sec` at t = 16.0025, interpolated onto it by `xfsi_solver.scripts.transfer_fsi2_checkpoint`
(u and v on the whole mesh, p and z on the fluid submesh). The interpolated state is not a discrete
solution on the new mesh, and with theta = 0.5 + dt the mismatch rings: on the h/2 and h/4 meshes,
the initial residual of a step grew to 5000 within 15 steps, against 97 on `mesh_sec`, and a reused
J_0 diverged. A transfer onto `mesh_sec` itself keeps it at 97, so the transfer loses no state. So
every mesh first takes 20 steps of backward Euler (theta = 1), which damp the transient, then one
period of 200 steps with the usual theta, both with J_0 reassembled in every iteration. The settled
state at t = 16.5525 starts the measurements. On h, the measured rates reproduce those of `mesh_sec`
in the Newton convergence report (3.2e-3, 5.0e-3 and 3.3e-3 for the first three iterations).

### Runs

Each run solves one period (200 steps of dt = 0.0025, t from 16.5525 to 17.0525) on each mesh:

| run | solver and options |
|---|---|
| J_0, predicted | `fsi2_simplified`, `gamma_reassemble=None`, `snes_rtol` 1e-11, `snes_atol` 1e-13 |
| J_0, not predicted | the same, `predict_displacements=False` |
| full Jacobian, predicted | `fsi2_biharmonic_fulljacobian_initialguess`, `snes_rtol` 1e-11, `snes_atol` 1e-13 |
| J_0, predicted, reused | `fsi2_simplified`, `gamma_reassemble=0.2`, default tolerances (`snes_atol` 1e-7) |

The tight relative tolerance keeps the iterations going well past the default absolute tolerance,
whose meaning changes with h: the l2 norm of the residual halves with each refinement (median
||R_0|| 96, 49 and 25), since each entry integrates over a smaller cell. Contraction ratios are
taken only between residuals above 1e-9 ||R_0||, clear of round-off. On h/2 and h/4, the residual of
the full Jacobian reaches round-off before 1e-11 ||R_0||, so most of its steps stop on PETSc's
default step tolerance (`snes_stol` 1e-8) after 3 iterations instead; `fsi2_simplified` sets
`snes_stol` to 0, and no J_0 run reached its limit of 14 iterations. The first step of every run is
left out of the figures; the tables are over all 200 steps. The runs on the three meshes ran on 2, 4
and 3 to 10 MPI ranks, partly side by side, so their times are not compared.

## Results

| run | mesh | its/step | rate 0→1 | 1→2 | later iterations (median, IQR) | update ratio |
|---|---|---:|---:|---:|---|---:|
| J_0, predicted | h | 4.75 | 3.2e-3 | 5.0e-3 | 3.6e-3 [2.1e-3, 6.2e-3] | 1.46e-2 |
| | h/2 | 4.97 | 3.2e-3 | 6.1e-3 | 4.2e-3 [2.4e-3, 7.1e-3] | 1.29e-2 |
| | h/4 | 5.07 | 3.1e-3 | 7.2e-3 | 4.9e-3 [2.9e-3, 8.2e-3] | 1.18e-2 |
| J_0, not predicted | h | 5.61 | 0.17 | 3.4e-3 | 5.5e-3 [2.6e-3, 9.1e-3] | 6.4e-3 |
| | h/2 | 5.92 | 0.16 | 4.0e-3 | 5.9e-3 [3.2e-3, 1.1e-2] | 8.2e-3 |
| | h/4 | 6.03 | 0.16 | 4.7e-3 | 6.5e-3 [3.7e-3, 1.2e-2] | 9.2e-3 |
| full Jacobian, predicted | h | 3.00 | 4.7e-4 | 4.5e-6 | | |
| | h/2 | 3.00 | 4.9e-4 | 4.3e-6 | | |
| | h/4 | 3.00 | 4.9e-4 | 3.5e-6 | | |

"Rate k→k+1" is the median of ||R_(k+1)|| / ||R_k|| over the steps, "later iterations" the median
over all iterations after the first, and "update ratio" the median of ||δx_(k+1)|| / ||δx_k|| over
the same iterations.

![Contraction against the mesh size, and iterations per step](plots/rates_vs_h.svg)

The contraction of the later iterations grows slowly with refinement, by a factor 1.17 per halving
with the prediction and 1.09 without it. The first iteration does not change. The iteration counts
follow: 0.3 to 0.4 more iterations per step from h to h/4 (4.75 to 5.07 with the prediction, 5.61
to 6.03 without it).

![Residual relative to the first, per iteration, on the three meshes](plots/residual_histories.svg)

Relative to the residual of the initial guess, the histories of the three meshes lie on top of each
other for J_0, and for the full Jacobian up to its second iteration. Its third iteration reaches the
round-off floor, which lies higher on the finer meshes.

![Contraction of each step over the period, with the tip displacement](plots/rate_over_period.svg)

Over the period, the contraction of J_0 follows the phase of the oscillation in the same way on every
mesh: on h, about 6e-3 while the tip moves fast and about 1.5e-3 around the turning points. Each
halving of h raises it by about 20% in the fast phases (6.3e-3, 7.5e-3, 8.8e-3) and by about 10%
around the turning points (1.5e-3, 1.6e-3, 1.8e-3), so the growth with h is largest where the solid
moves fast and the neglected ALE derivatives, through the mesh velocity, matter most. The jumps of a
curve are steps where an iteration more or fewer changes which ratios enter its median.

![Newton iterations and assemblies per step with reuse](plots/reuse_iterations.svg)

With reuse, the three meshes reassemble J_0 in the same phases and as often, and take the same
number of iterations per step up to the slightly looser absolute tolerance on the finer meshes.

## Why the rate barely depends on h

J_0 leaves out N, the derivatives of the fluid equations with respect to the mesh displacement, and
the asymptotic contraction is the spectral radius of J_0^(-1) N. The displacement enters the Newton
step through u = u_old + dt θ E(v on the interface), with E the biharmonic extension, which smooths,
and the fluid equations depend on u through at most first derivatives, while J_0 holds the mass term
ρ/dt and the second-order viscous term. N is therefore of lower order than J_0, and J_0^(-1) N stays
bounded as h → 0, at about dt times the flow and solid velocities. The measurements agree: the rate
grows only slowly, and most in the phases where the solid moves fast.

## Caveats

- One period, from one state, and three meshes; the slow growth may continue. At 17% per halving it
  would take four to five more halvings to double the rate, and since the first iteration does not
  change, that would add less than one iteration per step.
- The states are transferred and settled, not spun up from rest on each mesh. The settled states
  follow the same oscillation (the tip displacements of the three meshes lie on top of each other
  over the measured period), but each finer mesh would drift to its own periodic orbit over a
  longer run.
- One time step, dt = 0.0025. The argument above predicts that the rate scales with dt, which the
  sketch of this study also meant to measure; it is not done yet.
- The rates are in the l2 norm of the residual vector, whose weighting of fine and coarse components
  changes with h. The update ratio, the other estimate, grows without the prediction and falls with
  it.
- The meshes are regenerated by the current gmsh, so h is not `mesh_sec` itself, but its iterates
  match those of `mesh_sec` in the report to the digits shown.

## Next steps

- The time step sweep on h (dt from 0.005 to 0.000625) and the diagonal dt ∝ h.
- The spectral radius of J_0^(-1) N at frozen states on each mesh with SLEPc, which measures the
  asymptotic rate without tolerances or norms.

## Reproducing

```bash
python -c "from xfsi_solver.scripts.create_mesh_FSI2 import create_mesh; create_mesh(quads=False, semi_structured_quad=False, second_order=True, refinements=2, path='data/meshes/fsi2/refinement/mesh_sec_ref2.xdmf')"
```

makes a mesh of the family (0, 1 and 2 refinements), and

```bash
mpirun -n 4 python -m xfsi_solver.scripts.transfer_fsi2_checkpoint data/meshes/fsi2/mesh_sec.xdmf output/newton_convergence/init_mixed/checkpoints data/meshes/fsi2/refinement/mesh_sec_ref2.xdmf output/newton_refinement/states/ref2 0.0025
```

transfers the start state to it. The damping, settling and measurement runs call
`fsi2_biharmonic_mixed.solve`, `fsi2_simplified.solve` and
`fsi2_biharmonic_fulljacobian_initialguess.solve` with the options above, writing to
`output/newton_refinement/<run>_ref<n>/`. The figures are made by

```bash
python notes/newton-mesh-refinement/make_figures.py
```
