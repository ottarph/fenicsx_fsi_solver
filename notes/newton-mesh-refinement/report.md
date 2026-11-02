# Newton contraction under mesh refinement

Does the linear contraction of Newton's method with the approximate Jacobian J_0 break down when
the mesh is refined? The FSI2 window of `notes/newton-convergence/report.md`, one period of the
oscillating beam, is solved on three nested meshes of mesh size h, h/2 and h/4, from the same state.
A time step sweep and the spectral radius of the error iteration of J_0 then separate the effects of h
and dt (see Time step and spectral radius). The history of the work is in `implementation-log.md`.

## Findings

1. **The contraction of J_0 does not break down.** With J_0 reassembled in every iteration, the later
   iterations contract the residual by 3.6e-3, 4.2e-3 and 4.9e-3 (median) on the three meshes with
   the displacement prediction, and by 5.5e-3, 5.9e-3 and 6.5e-3 without it: 8 to 17% more per
   halving of h, roughly h^(-0.1) to h^(-0.2). A step takes 4.75, 4.97 and 5.07 iterations to a
   relative tolerance of 1e-11.
2. **The first iteration does not depend on h.** It contracts by 3.2e-3 with the prediction and by
   0.16 to 0.17 without it on every mesh.
3. **The growth with h is real, not an artefact of the norm.** The spectral radius of J_0^(-1) N,
   the asymptotic contraction without tolerances or norms, grows by 1.20 and 1.14 per halving of h
   (4.6e-3, 5.5e-3 and 6.3e-3 at dt = 0.0025), as the measured rates do. The ratio of successive
   Newton updates, which falls from 1.5e-2 to 1.2e-2 with the prediction (and grows from 6.4e-3 to
   9.2e-3 without it), is a poorer estimate over only 2 to 4 iterations.
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
7. **A smaller time step helps the contraction only a little.** On h, the spectral radius falls from
   7.0e-3 to 4.6e-3, 3.9e-3 and 3.2e-3 as dt halves from 0.005 to 0.000625, roughly dt^0.36, and the
   measured contraction of the later iterations about as slowly. Only the first iteration with the
   prediction contracts in proportion to dt (9.6e-3 to 1.3e-3), so a step takes about 5 iterations
   at every dt.
8. **Refined in h and dt together, the contraction stays the same.** Over the first 0.04 s of the
   window, the spectral radius is 4.79e-3 at (h, 0.0025), 4.85e-3 at (h/2, 0.00125) and 4.74e-3 at
   (h/4, 0.000625), within 2% of each other at every sample time: the growth with h and the decrease
   with dt cancel. Refined in h alone, it is 6.4e-3 on h/4, and in dt alone, 3.4e-3 at dt = 0.000625.
   The measured contraction of the later iterations still grows along this line (5.6e-3, 6.3e-3,
   7.6e-3, from 1.16 to 1.59 times the spectral radius), but the first iteration, in proportion to
   dt, makes up for it: every step takes 5.00 iterations on all three.
9. **The two parts of the fluid ALE derivatives must be left out together.** Leaving out only the
   derivatives of the momentum residual gives a spectral radius of 1.1e-2 at dt = 0.0025 on h, only
   those of the continuity residual 7.2e-3, and both, as J_0 does, 4.6e-3: the two parts partly
   cancel, as if the geometry were frozen consistently in all the fluid equations. Each part alone
   falls with dt as slowly as both together.
10. **The measured contraction lies above the spectral radius.** Over the 2 to 4 iterations of a step,
    the measured rate is 1.2 to 2 times the spectral radius at dt = 0.0025 and below (about equal at
    0.005), most near the turning point of the tip,
    where the spectral radius falls earlier than the measured rate. The error iteration is far from
    normal, and its dominant eigenvalues are a complex pair, which makes the measured ratios alternate
    from one iteration to the next.

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

## Time step and spectral radius

J_0 leaves out N, the derivatives of the fluid momentum and continuity residuals with respect to the
mesh displacement (the blocks F and D of eq. (1) of `docs/restricted-iterative-solver.md`), and
near the solution Newton's method with J_0 iterates the error as e_(k+1) = -J_0^(-1) N e_k. The
spectral radius of J_0^(-1) N is therefore the asymptotic contraction factor.
`fsi2_biharmonic_fulljacobian_initialguess` computes it with `spectral_every`, at the solution of a
step: it assembles J and N there, factorizes J_0 = J - N once with MUMPS, and finds the eigenvalues
of largest modulus by Krylov-Schur in SLEPc (`xfsi_solver.tools.spectral`). On the settled state
of h, it tracks the measured contraction of `fsi2_simplified`, at 0.6 to 0.8 times it over the first
steps.

The runs cover the first 0.12 s of the period of the refinement study (from the fast phase of the
tip to its turning point at t = 16.66), from the same settled states, on h at dt = 0.005, 0.0025,
0.00125 and 0.000625, on h/2 at dt = 0.0025 and 0.00125, and on h/4 at dt = 0.0025 and, over the
first 0.04 s only, at 0.000625. The spectral radius is sampled every 0.005 s, at the solution of the
full Jacobian with the prediction; the measured rates are those of J_0 with the prediction,
reassembled in every iteration, to a relative tolerance of 1e-11.

| mesh | dt | spectral radius (median, IQR) | measured, later iterations | measured, first iteration |
|---|---|---|---:|---:|
| h | 0.005 | 7.0e-3 [6.6e-3, 7.3e-3] | 6.6e-3 | 9.6e-3 |
| h | 0.0025 | 4.6e-3 [3.6e-3, 5.1e-3] | 5.8e-3 | 5.1e-3 |
| h | 0.00125 | 3.9e-3 [3.2e-3, 4.4e-3] | 5.1e-3 | 2.6e-3 |
| h | 0.000625 | 3.2e-3 [2.6e-3, 3.6e-3] | 4.4e-3 | 1.3e-3 |
| h/2 | 0.0025 | 5.5e-3 [4.6e-3, 6.1e-3] | 6.7e-3 | 5.1e-3 |
| h/2 | 0.00125 | 4.8e-3 [4.0e-3, 5.4e-3] | 6.7e-3 | 2.6e-3 |
| h/4 | 0.0025 | 6.3e-3 [5.3e-3, 7.0e-3] | 7.7e-3 | 5.0e-3 |
| h/4 | 0.000625 (first 0.04 s) | 4.7e-3 [4.4e-3, 5.2e-3] | 7.6e-3 | 8.0e-4 |

The measured rates are medians over the same window as the spectral radius, 0.12 s except on h/4 at
dt = 0.000625.

![Asymptotic and measured contraction against dt on h](plots/rho_vs_dt.svg)

On h, the spectral radius falls with dt by a factor 1.53, then 1.17 and 1.21 per halving, roughly
dt^0.36, and the measured contraction of the later iterations by about 1.15 per halving. The first
iteration with the prediction contracts in proportion to dt: the prediction u_old + dt v_old is
off by about dt^2 a / 2, against a residual of the initial guess of order dt. With only the
momentum or only the continuity part left out, the spectral radius is larger (1.1e-2 and 7.2e-3 at
dt = 0.0025) and falls with dt about as slowly (by 1.59 and 1.67 from dt = 0.0025 to 0.000625,
against 1.42 for both).

![Spectral radius over the window on h, for the four time steps](plots/rho_over_window.svg)

Over the window, the spectral radius at dt = 0.0025 and below keeps its shape from one time step to
the next, rising through the fast phase and falling toward the turning point, each curve about 1.2
times below the last. At dt = 0.005 it stays high through the turning point: there, the dominant
eigenvalue is real in every sample, while at the smaller time steps it is a complex pair in 83 to
100% of them, so at the largest time step another mode takes over. The measured rate falls toward
the turning point later than the spectral radius and stays up to twice as large: over only 2 to 4
iterations, the ratio of residual norms of a far from normal error iteration is set by more than its
spectral radius.

![Spectral radius under refinement in h, in dt, and in both](plots/rho_vs_h.svg)

Refined in h at dt = 0.0025, the spectral radius grows by 1.20 and 1.14 per halving; refined in dt
on h, it falls by 1.17 and 1.21; refined in both, over the first 0.04 s of the window where all three
are run, it stays at 4.79e-3, 4.85e-3 and 4.74e-3. The measured rate of the later iterations grows
along the same line (5.6e-3, 6.3e-3 and 7.6e-3), so the transient of the error iteration, not its
asymptotic rate, grows with refinement; the first iteration contracts by 3.1e-3, 1.6e-3 and 8.0e-4,
and the steps take 5.00 iterations on all three.

### Why the rate depends so little on h and dt

The displacement enters the Newton step through u = u_old + dt θ E(v on the interface), with E the
biharmonic extension, and the fluid equations depend on u through at most first derivatives. With
J_0 dominated by the mass term ρ/dt, this argument predicts a spectral radius bounded in h and
proportional to dt. It holds for h, where the growth is slow, but not for dt: the spectral radius
falls only as about dt^0.36, for both parts of N alike. A plausible reason is the pressure: in
incompressible flow, the pressure answers an acceleration of the interface with an added-mass
response that scales like 1/dt, which would cancel the factor dt of the displacement for the modes
that limit the convergence. This is not checked; the eigenvectors of the dominant eigenvalues would
show where these modes live.

## Caveats

- One period, from one state, and three meshes; the slow growth may continue. At 17% per halving it
  would take four to five more halvings to double the rate, and since the first iteration does not
  change, that would add less than one iteration per step. Refining dt with h, as one would to keep
  the accuracy in time, removes the growth altogether over the range measured.
- The time step sweep and the spectral radius cover the first 0.12 s of the period, from the fast
  phase to the turning point, and on h/4 at dt = 0.000625 only the first 0.04 s. The states for
  dt other than 0.0025 start from the settled states of dt = 0.0025, so the first steps of those
  runs follow a state of another time step.
- The states are transferred and settled, not spun up from rest on each mesh. The settled states
  follow the same oscillation (the tip displacements of the three meshes lie on top of each other
  over the measured period), but each finer mesh would drift to its own periodic orbit over a
  longer run.
- The spectral radius is the asymptotic rate at the solution of each step; the measured rates of the
  2 to 4 iterations of a step lie up to twice above it, as the error iteration is far from normal.
- The explanation of the weak dependence on dt by the added mass of the pressure is not checked.
- The meshes are regenerated by the current gmsh, so h is not `mesh_sec` itself, but its iterates
  match those of `mesh_sec` in the report to the digits shown.

## Next steps

- The eigenvectors of the dominant eigenvalues of J_0^(-1) N, to see where the slow modes live
  (interface, beam tip, wake, corners) and to check the added-mass explanation.
- The real mode that dominates at dt = 0.005, and whether it limits larger time steps further.

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
`output/newton_refinement/<run>_ref<n>/`. The runs of the time step sweep start from
`output/newton_refinement/settle_ref<n>/checkpoints` and write to `output/newton_dt/`; the spectra
come from `fsi2_biharmonic_fulljacobian_initialguess.solve` with `spectral_every` (every 0.005 s),
`spectral_path` and, for the parts of N, `spectral_left_out`. The figures are made by

```bash
python notes/newton-mesh-refinement/make_figures.py
```
