# Linear solvers for the C0IP mesh motion: implementation log

Goal: choose the linear solver for the mesh-motion block K_m of
`solvers/fsi2_biharmonic_c0ip.py` (C0IP biharmonic, P2), and estimate how it
scales to the 3d benchmark, where K_m is much larger. K_m does not couple the
displacement components and has the same matrix for each, so it is solved by
an additive fieldsplit over the components, with the preconditioner of the x
component reused for the others (`tools/petsc_reuse.py:ReusePC`).

## CG with GAMG in the FSI solver (2026-10-13)

`mesh_motion_direct_solver=False` solves each component by CG with GAMG
(agg, threshold 0.2), in the unpreconditioned residual norm.

- With CG atol 1e-7 = `snes_atol`, Newton stalled. Each CG solve left a
  mesh-motion residual of up to 1e-7 per component, so the Newton residual
  settled at 1.0–1.3e-7, just above `snes_atol`. Once the mesh-motion residual
  was below atol, CG took 0 iterations and returned a zero update, and the
  default SNES step-norm test (`snes_stol` 1e-8) declared
  `CONVERGED_SNORM_RELATIVE`. The same as `ecdb8e0` in the split condensed
  solver. Fixed with `snes_stol: 0.0` and CG atol 1e-9.
- Effect of the CG rtol on Newton, `mesh_sec`, 16 ranks, restart from t = 16,
  with `snes_stol` 0 and CG atol 1e-9:

  | mesh motion | γ = 0, 10 steps | γ = 0.5, 10 steps | γ = 0.5, 40 steps | Jacobian assemblies, 40 steps |
  |---|---|---|---|---|
  | Cholesky | 50 | 96 | 353 | 19 |
  | CG rtol 1e-5 | | 96 | 353 | |
  | CG rtol 1e-3 | 60 | 96 | 360 | 41 |
  | CG rtol 1e-2 | | | 597 | 41 |
  | CG rtol 1e-1 | | | 668 | 41 |
  | Cholesky, update × 0.9 | 120 | 120 | | |

  - With a fresh Jacobian in every iteration (γ = 0), rtol 1e-3 costs a Newton
    iteration per step, as expected of an inexact solve.
  - With Jacobian reuse (γ = 0.5), the looser solves change when the Jacobian
    is reassembled (about twice as often), which on short windows can give
    fewer Newton iterations; over 40 steps the totals are about equal. A
    damped exact update (× 0.9) is worse, so the inexact solve does not help
    by damping.
  - The QoIs agree with the direct solve to all printed digits (about 7) at
    every rtol.
  - rtol 1e-3 halves the CG work (37 against 73 iterations per solve, 0.37
    against 0.54 s per step), but Cholesky is faster still: 0.145 s per step.

## Mesh-motion study: `component_solvers/c0ip_mesh_motion.py` (2026-10-14)

Solves K_m alone, without the FSI simulation, on the fluid submesh of the
FSI2 or the 3d geometry, with the displacement of a checkpoint as Dirichlet
condition on the whole fluid boundary (zero on the walls and the cylinder,
the solid's displacement on the interface). The unknowns are those of K_m,
except that the interface dofs become identity rows. `--mesh` interpolates the
deformation to another mesh of the same geometry, to solve for the same
deformation at different resolutions; the tip displacement (A in 2d, B in 3d)
is printed from the checkpoint and the boundary condition, and matched
exactly in every run. Cases: `fsi2` (C0IP checkpoint at t = 16, `mesh_sec`)
and `fsi3d` (checkpoint at t = 10 of the medium run).

Solvers: `direct` (MUMPS LU on the whole vector system), `direct_component`
(MUMPS Cholesky on the x component, reused), `cg_gamg`, and
`cg_laplacian_squared` (below). It reports the setup time (PCSetUp and
PCSetUpOnBlocks, where the fieldsplit sets up the components'
preconditioners), the time of a second solve with the solver set up (the cost
per Newton iteration in the FSI run), iterations, memory, and with
`--compare` the error relative to an exact solve (Cholesky per component).

- 3d C0IP penalty: no parameters existed for tetrahedra.
  `tools/dg_optpar.py:get_dg_penalty_parameters_tetrahedron` carries the
  triangle parameters over, keeping the facet area over cell volume ratio of
  the trace inequality. Not from the literature and not tuned; CG converges
  and the deformed meshes are valid (smallest det(I + grad u) about 0.9), but
  coercivity is not proven.
- At the same relative residual, the error can be much larger, since the
  operator is of fourth order: CG with GAMG at rtol 1e-5 on `mesh_sec` has an
  error of 6.3e-3, about 600 times the residual.
- Assembly takes about 20 s on the 3d medium mesh (16 ranks), not included in
  the solve times.

### CG with GAMG on the biharmonic

| mesh | unknowns | ranks | CG iterations | error | s per solve | Cholesky s per solve |
|---|---|---|---|---|---|---|
| 2d coarse | 1.9k | 4 | 29–32 | 2.9e-4 | 0.005 | |
| 2d `mesh_sec` | 21k | 4 | 58–61 | 6.3e-3 | 0.058 | 0.002 |
| 2d fine | 78k | 4 | 101–110 | 3.4e-2 | 0.35 | |
| 3d coarse | 66k | 8 | 33–35 | 1.6e-4 | 0.18 | 0.017 |
| 3d medium | 385k | 16 | 72–76 | 1.1e-3 | 4.6 | 0.15 |

rtol 1e-5. The iterations roughly double per refinement, and the error at a
fixed rtol grows.

### Laplacian-squared preconditioner

`LaplacianSquaredPC` applies L^{-1} M L^{-1} per component in CG, with L the
P2 Laplacian and M the mass matrix, both with the Dirichlet identity rows,
and L^{-1} one AMG V-cycle. For the clamped boundary condition, the condition
number grows like h^{-1} (Braess and Peisker, IMA J. Numer. Anal. 6, 1986).

- One V-cycle is a fixed symmetric linear operator, so CG applies: with
  BoomerAMG, CG and FGMRES need about the same iterations (87 and 92 on the
  fine 2d mesh). An inner Krylov solve to a tolerance would need flexible CG
  or FGMRES.
- BoomerAMG V-cycles give the same iterations as exact (MUMPS) solves with L.
  GAMG V-cycles do not work: 1000 iterations without converging, also with
  FGMRES, so a GAMG V-cycle is a poor approximation of L^{-1} here, not an
  asymmetric one. BoomerAMG is the default.
- 2d, rtol 1e-5, 4 ranks (condition number from CG's estimate):

  | mesh | condition number | CG iterations | error |
  |---|---|---|---|
  | coarse | 75 | 33 | 4e-5 |
  | `mesh_sec` | 360 | 62 | 3e-4 |
  | fine | 820 | 88 | 1.8e-3 |

  The condition number grows like about h^{-1.2}.

### Comparison at the same error

| case | solver | rtol | iterations | setup | s per solve | error |
|---|---|---|---|---|---|---|
| 2d fine, 4 ranks | GAMG | 1e-6 | 159–182 | | 0.56 | 3.8e-3 |
| | Laplacian squared | 1e-5 | 87–89 | | 0.73 | 1.8e-3 |
| 3d medium, 16 ranks | GAMG | 1e-6 | 97–100 | 8.9 s | 6.1 | 7.2e-5 |
| | Laplacian squared | 1e-5 | 52 | 1.5 s | 4.4 | 3.8e-5 |
| | Laplacian squared | 1e-4 | 42 | 1.5 s | 3.5 | 2.0e-4 |
| | Cholesky, reused | | | 5.8 s | 0.15 | exact |
| 3d fine, 28 ranks | Laplacian squared | 1e-5 | 64 | 4.7 s | 16.0 | 8.8e-5 |
| | Laplacian squared | 1e-4 | 50 | 4.8 s | 12.4 | 5.4e-4 |
| | Cholesky, reused | | | 20.0 s | 0.55 | exact |

- In 2d, GAMG on the biharmonic is slightly faster. In 3d, the
  Laplacian-squared preconditioner reaches half the error in 70% of the time,
  with a sixth of the setup.
- The reused Cholesky factorization is about 30 times faster per solve than
  either, on the medium and the fine 3d mesh.
- 3d fine mesh: the deformation of the medium checkpoint, 951k unknowns (317k
  per component). Cholesky: 9.4 GB of MUMPS memory, 19.6 GB peak. The peaks of
  the CG runs (21.6 and 22.0 GB) include the Cholesky of the reference solve.
- Laplacian squared at rtol 1e-5: 38, 52 and 64 iterations on the coarse,
  medium and fine 3d meshes.

### Estimates beyond the fine mesh

From two meshes (medium on 16 ranks, fine on 28), so order of magnitude only.

- CG never overtakes the reused Cholesky per solve. From the medium to the
  fine mesh, both solve times grew by the same factor (CG 3.6, Cholesky 3.7),
  keeping the ratio at about 29. In theory, CG costs about N^{7/6} per
  component (N times iterations growing like N^{1/6}), and a solve with the
  factorization costs about its size, N^{4/3} in 3d, so CG gains only N^{1/6}:
  closing a factor 30 would take 30^6 ≈ 7·10^8 times the unknowns. The
  factorization (about N^2) is done once per run of about 17,500 solves
  (about 7 Newton iterations × 2500 steps), so it stays negligible.
- Cholesky per component reaches 100 GB of MUMPS memory at about 2–3 million
  unknowns per component. The memory grew like N^{1.07} from the medium to the
  fine mesh (3.56 to 9.40 GB for 2.47 times the unknowns), lower than the
  theoretical N^{4/3}, probably because of the memory per rank and the
  different numbers of ranks:

  | growth | per component | mesh motion | FSI system |
  |---|---|---|---|
  | measured, N^{1.07} | 2.9M | 8.7M | 27M |
  | theory, N^{4/3} | 1.9M | 5.6M | 18M |

  6 to 9 times the fine mesh (3.0M FSI unknowns), about one more uniform
  refinement, and for the factorization alone.
- So the reused Cholesky factorization is the choice up to about one
  refinement beyond the fine mesh. Beyond it, an iterative solver is needed
  because of memory, not speed; the Laplacian-squared preconditioner would
  then need about 75 iterations at rtol 1e-5.
