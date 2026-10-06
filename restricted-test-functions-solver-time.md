# Solver time of the restricted-test-function FSI2 solver

The biharmonic FSI2 solver with restricted test functions
(`src/xfsi_solver/solvers/fsi2_biharmonic_diffmesh_restricted.py`) is about 7%
slower per time step than the base biharmonic solver
(`src/xfsi_solver/solvers/fsi2_biharmonic_diffmesh.py`). The two solvers give
the same results. The slowdown comes from the MUMPS numeric factorisation, not
from the extra assembly in the restricted solver. It goes away when the
mesh-motion equations are scaled by `alpha_u = 1e-9` again, as they are in the
base solver. That scaling does not change the solution.

## The slowdown

Full FSI2 benchmark runs on `mesh_sec`, 24 MPI ranks, T = 15 s,
dt = 0.0025 (from the bottom of the solver logs in `output/logs/`):

| Solver | Elapsed time | Time per step |
|---|---|---|
| Base biharmonic (`fsi2_biharm_log.txt`) | 2476 s | 0.413 s |
| Restricted test functions (`fsi2_bh_dm_restr.txt`) | 2652 s | 0.442 s |

## Where the time goes

Both solvers were profiled with PETSc `-log_view` for the first 20 time steps
(t ≤ 0.05) on 4 ranks. They took the same number of Newton iterations: 57
residual and 37 Jacobian evaluations.

| PETSc event | Base | Restricted | Difference |
|---|---|---|---|
| `SNESFunctionEval` (includes `post_residual`) | 1.11 s | 1.13 s | +0.02 s |
| `SNESJacobianEval` (includes `post_jacobian`) | 3.66 s | 3.77 s | +0.11 s |
| `MatLUFactorNum` (MUMPS numeric factorisation) | 10.09 s | 11.66 s | **+1.57 s** |
| `MatLUFactorSym` | 1 call | 1 call | none |
| `SNESSolve` (total) | 16.04 s | 17.77 s | +1.73 s |

About 90% of the difference is in the numeric factorisation. The extra
residual and Jacobian assembly done in the restricted solver's SNES callbacks
costs about 3 ms per Jacobian evaluation. The symbolic factorisation runs only
once in both solvers, so zeroing the interface rows with `zeroRowsLocal` does
not force MUMPS to redo its analysis.

## Cause: more pivoting in MUMPS

The two Jacobians have exactly the same number of nonzeros (3,427,268) and the
same size (71,756). So the extra zero-weighted terms that set up the sparsity
pattern do not make the matrix larger. The difference is in how many pivots
MUMPS has to delay during the numeric factorisation:

| Solver | Delayed pivots per step | Mean elimination flops |
|---|---|---|
| Base | 8424 at the first step, falling to 807 by step 20 | 1.02e10 |
| Restricted | about 8270 at every step | 1.22e10 |

In the base solver, the number of delayed pivots falls quickly as the solution
develops. In the restricted solver it stays near its first-step value. The
factorisation then costs about 20% more flops, which matches the slower
`MatLUFactorNum`.

The two solvers scale the biharmonic mesh-motion equations differently:

- **Base solver:** every term in `A_I` that is tested with `dz` or with `du`
  on the fluid carries `alpha_u = 1e-9`.
- **Restricted solver:** the same terms are unscaled, i.e. α = 1. With
  restricted test functions α is not needed as a coupling parameter, so it was
  removed.

Rescaling the solid kinematic rows of the restricted solver by 1/ρₛ or by
dt/ρₛ made no difference: the delayed pivots and flops stayed the same. Only
the scaling of the mesh-motion equations matters.

## What changes when the scaling is reintroduced

The test used the restricted solver with `alpha_u = 1e-9` put back on the three
mesh-motion terms of `A_I`:

```python
alpha_u = dolfinx.fem.Constant(mesh, 1.0e-9)
residual  = alpha_u * ufl.inner(z, dz) * dx_fluid
residual -= alpha_u * ufl.inner(ufl.grad(u), ufl.grad(dz)) * dx_fluid
residual += alpha_u * ufl.inner(ufl.grad(z), ufl.grad(du)) * dx_fluid
```

Same 20 steps on 4 ranks:

| | Restricted, α = 1 | Restricted, α = 1e-9 | Base |
|---|---|---|---|
| Delayed pivots | about 8270 at every step | 8352 falling to 806 | 8424 falling to 807 |
| Mean elimination flops | 1.22e10 | 1.02e10 | 1.02e10 |
| `MatLUFactorNum` | 11.66 s | 10.15 s | 10.09 s |
| `SNESSolve` (total) | 17.77 s | 16.25 s | 16.04 s |

With α = 1e-9, the restricted solver's factorisation behaves exactly like the
base solver's. Its remaining overhead is the small cost of the extra assembly
in the callbacks.

### The solution does not change

With restricted test functions, α multiplies only the rows of the mesh-motion
equations: the `dz` rows and the fluid `du` rows. The interface rows hold the
solid kinematic equation instead, and that equation is not multiplied by α.
So α only rescales a set of equations and does not change what the discrete
system solves. The QoI file from the α = 1e-9 run was byte-identical to the
α = 1 run over the profiled steps. In the base solver, by contrast, α also
controls how strongly the interface is coupled.

### What the SNES residual norm measures

Scaling the mesh-motion rows by 1e-9 also scales their share of the residual
norm that SNES uses in its convergence test by 1e-9. With `snes_atol = 1e-7`,
the stopping test then effectively only checks the flow, pressure and solid
equations, and hardly checks the mesh motion. The base solver has always had
this property. Over the profiled steps, the Newton iterations and the results
were the same as with α = 1. If it ever matters, one option is a convergence
test that checks the mesh-motion block separately.

## Limitations

- The profile covers only the first 20 time steps (t ≤ 0.05) on 4 ranks. The
  α = 1e-9 version has not yet been rerun on the full 15 s benchmark on 24
  ranks. So the full-run speedup is expected, not measured.
- It is not clear why MUMPS pivots differently with α = 1. The measurements
  show the effect, not the mechanism.
