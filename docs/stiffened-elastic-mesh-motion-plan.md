# Implementation handoff for Claude Code Opus 5.5

Implement a new monolithic FSI2 solver using the reference-configuration, stiffened linear-elastic mesh extension studied by **Shamanskiy and Simeon**. Deliver a solver that completes the full FSI2 run without biharmonic mesh motion and works with the harmonic solver's shared-space fieldsplit infrastructure. Implement and validate this task on the remote checkout of the task-provided feature branch; this document is a plan, not evidence of a completed implementation.

Read `AGENTS.md` and `CLAUDE.md`. Use `conda run --no-capture-output -n xfsi_solver ...`. Keep the configured Git identity and make coherent implementation-and-test commits with the attribution required for Claude Code. Preserve unrelated remote changes. Do not push, merge, rebase, or amend as part of this task.

## 1. Inspect the remote branch and establish the integration contract

This plan was prepared against `iterative-solver` at `52b04e2`. At that revision:

- `src/xfsi_solver/solvers/fsi2_harmonic_diffmesh.py` has shared quadratic displacement and velocity on the parent fluid-solid mesh, linear pressure on the fluid submesh, and a MUMPS solve.
- `docs/shared-space-iterative-solver-plan.md` specifies the intended `u | (v,p)` outer Schur split and nested `v | p` split, but the implementation is not present locally.
- `src/xfsi_solver/solvers/fsi2_harmonic_lagrange.py` uses a different coupling architecture and is not the base for this task.
- The local environment has DOLFINx 0.11.0 and PETSc 3.25.5; inspect the remote environment instead of assuming these versions.

Inspect the remote harmonic solver and its actual helpers first. Reuse the fieldsplit implementation if it has arrived since this revision. Preserve its public configuration, distributed index sets, option-prefix conventions, full/no-ALE Jacobian modes, direct diagnostic mode, and solver lifecycle. Do not replace working remote code with this older local snapshot. If the remote branch still only contains the fieldsplit plan, make the missing shared infrastructure an explicit prerequisite and implement the minimum complete reusable path specified there; do not label direct-only code fieldsplit-compatible without exercising it.

Add the new entry point:

`src/xfsi_solver/solvers/fsi2_stiffened_elastic_diffmesh.py`

Keep `(u, v, p)` and the fluid pressure submesh/entity maps. No extra biharmonic field or interface multiplier is needed. Retain the existing harmonic and biharmonic entry points. Prefer a small reusable mesh-extension form helper and reuse existing solver-setup helpers. A limited extraction of common setup is acceptable if needed for reuse and covered by harmonic regression tests; avoid an unrelated solver-framework rewrite.

Preserve FSI2 material parameters, physical boundary conditions, inflow ramp, theta scheme, and physical momentum/divergence residuals. Do not change the benchmark to FSI3 or stiffen the physical solid to obtain convergence. The current mesh generator uses a 2.5 m channel; record this geometry explicitly rather than silently changing it to the 2.2 m channel in the comparison paper.

## 2. Implement the mesh constitutive law and reference weighting

Use the **non-incremental LE** method: solve for total mesh displacement from the original reference configuration at each nonlinear iterate. All derivatives below are with respect to reference physical coordinates \(X\). Do not update the mesh geometry array or accumulate increments on successively deformed meshes.

Define

\[
\varepsilon_m(u)=\operatorname{sym}(\nabla_Xu),\qquad
\sigma_m(u)=2\mu_m(X)\varepsilon_m(u)
+\lambda_m(X)\operatorname{tr}(\varepsilon_m(u))I,
\]

\[
\mu_m(X)=\frac{E_0w(X)}{2(1+\nu_m)},\qquad
\lambda_m(X)=\frac{E_0w(X)\nu_m}{(1+\nu_m)(1-2\nu_m)},
\qquad \nu_m=0.3,\quad \chi=2.5.
\]

The reference-based mesh-Jacobian stiffening is

\[
w(G_{0,K}(\xi))=
\left(\frac{j_\star}{j_{0,K}(\xi)}\right)^\chi,
\qquad j_{0,K}(\xi)=|\det D_\xi G_{0,K}(\xi)|,
\]

where \(G_{0,K}\) maps a standard parent element to the **initial** physical cell. \(j_\star\) is one fixed positive global normalization. Choose and document it reproducibly, for example the global mean initial fluid-cell volume divided by the parent-cell volume on a single-cell-type mesh. Compute reductions from owned fluid cells only; ranks with no fluid cells must participate correctly. The same normalization must be used by residual, Jacobian, interface flux, and preconditioner.

Implementation requirements:

- Prefer pointwise reference geometry weighting, using the installed UFL/DOLFINx geometry APIs (e.g. `abs(ufl.JacobianDeterminant(mesh))` on the unchanged parent mesh). Verify the terminal denotes the cell-to-reference-physical map and evaluates correctly on cell and one-sided facet integrals.
- The meshes can have quadratic curved geometry. Do not silently replace the pointwise determinant with cell volume: they are proportional for affine elements of a common parent type, but not generally for curved triangles or non-affine quadrilaterals. A DG0 volume approximation may be an explicitly named experimental option, not the claimed exact default.
- Do not use `det(I + grad(u))`, the physical ALE deformation ratio, in this weight. That would create a deformation-dependent nonlinear mesh law and different Jacobian/update requirements.
- Keep the usual physical integration measure. Multiplying the integrand by \(j_0^{-\chi}\) already produces the intended \(j_0^{1-\chi}\) parent-coordinate integration factor; do not apply the weight twice.
- Reference orientation signs are distinct from ALE inversion. Use the positive geometry volume factor for weighting, reject degenerate geometry, and evaluate ALE validity separately.
- The fractional power is non-polynomial. Specify sufficient quadrature and perform a quadrature-increase check on a curved test mesh, including the interface term.
- Expose named configuration for `mesh_stiffening_exponent`, `mesh_poisson_ratio`, and `mesh_equation_scale`. Validate finite values, a supported Poisson range, and nonnegative exponent. Do not silently clip weights or substitute a different extension when convergence fails.

Use \(E_0=1\) as a dimensionless mesh-modulus normalization and initially retain the existing \(\alpha_u=10^{-9}\) as a **separate mesh-equation scale**. Neither is a physical-solid parameter. A common modulus factor cancels in an isolated Dirichlet extension, but the repository's shared displacement equations require explicit scaling checks; do not assume arbitrary rescaling has no algebraic effect here. Log the global weight range and chosen normalization.

## 3. Replace both the volume term and the interface flux

The current `A_I` contains a harmonic volume term and a one-sided fluid-interface flux subtraction. Replace their combined contribution with

\[
R_m(u;\varphi_u)=\alpha_u\left[
\int_{\Omega_f^0}\sigma_m(u):\varepsilon_m(\varphi_u)\,dX
-\int_{\Gamma_{fs}^0}(\sigma_m(u)n_f)\cdot\varphi_u\,dS
\right].
\]

The normal, stress, and weight on the interface must all use the fluid-side trace. Replace the harmonic flux as well as the harmonic volume operator. Do not leave `grad(u) * normal` in the new elastic flux, double-count interface terms, or interpret the artificial mesh traction as physical solid loading.

Reuse the existing shared-field coupling and solid kinematic equation. The fluid extension inherits interface displacement through shared degrees of freedom. The displacement residual is not a standalone elasticity problem with independently prescribed interface data, so preserve the interface formulation rather than introducing a new partitioned update. Verify fluid-side entity selection for every interface facet and on each MPI rank; the local snapshot's first-facet orientation shortcut deserves an explicit check.

Keep the mesh equation fully implicit without theta weighting. Since its coefficient and reference geometry are fixed, its displacement derivative is the same linear elastic operator in every Newton step. Retain `dolfinx.fem.petsc.NonlinearProblem` for the coupled solve; use `dolfinx.fem.petsc.LinearProblem` for standalone extension tests. Import `dolfinx.fem.petsc` as required by repository conventions. Never use the legacy `NewtonSolverNonlinearProblem`.

## 4. Adapt the fieldsplit displacement operator without changing the split layout

For the existing approximate Jacobian, keep the block contract

\[
J_{\mathrm{noALE}}=
\begin{pmatrix}
A&C&0\\
E_s&H&G\\
0&B&0
\end{pmatrix},\qquad
S_{vp}=\begin{pmatrix}H-E_sA^{-1}C&G\\B&0\end{pmatrix}.
\]

Only the mesh contribution to \(A\) changes directly. Continue to retain the solid tangent \(E_s\), solid kinematic coupling \(C\), and full nonlinear physical residual. Omitting fluid ALE derivatives must not omit the mesh-extension derivative. Pressure and velocity auxiliary operators keep their physical definitions but must still be updated for the current geometry where the existing implementation requires it.

- Supply the elastic mesh operator through the same displacement-block interface used for harmonic extension. Do not leave a scalar Laplacian hard-coded in an auxiliary `A` while presenting it as the new elastic inverse.
- Preserve the solid/interface versus fluid-interior displacement partition, including all extension couplings from shared interface unknowns. Use the actual new \(A\) in Schur actions; do not substitute an assumed exact solid condensation.
- The fluid-interior inverse now acts on **coupled vector elasticity** with variable Lamé coefficients. Start with direct subsolves as diagnostics, then use elasticity-capable AMG with correct vector block size and boundary-compatible near-nullspace candidates. Distinguish AMG near-nullspace information from an exact nullspace: the Dirichlet-restricted operator need not have rigid modes in its kernel.
- Preserve the distinction between the complete displacement block, which includes interface flux and solid mass, and its symmetric fluid-interior auxiliary operator. Do not apply CG to the complete block merely because the volume elasticity form is symmetric.
- Feed the same \(w,\nu_m,\alpha_u\) into residual, Jacobian, and mesh preconditioner. Include any existing row normalization consistently in residual/correction actions. Report sensitivity to coefficient contrast and scaling.
- Reuse the distributed `u | (v,p)` and nested `v | p` index-set machinery. Verify the assembled auxiliary reduced operator and the intended `Amat`/`Pmat` selection with PETSc solver views.
- Reuse constant mesh data and elastic stiffness where possible, but preserve state-dependent assembly of other pieces of \(A\) and the momentum preconditioner. Do not cache the entire coupled matrix as constant.

Support the remote implementation's equivalents of `full/direct`, `no_ale/direct`, and `no_ale/fieldsplit`. Retain `full/fieldsplit` as a diagnostic if it already exists. No silent fallback from fieldsplit to monolithic LU; direct coarse-grid solves are allowed.

## 5. Add diagnostics that distinguish mesh failure from solver failure

Keep the existing QoI columns `t, drag, lift, A_x, A_y` and independent output paths/prefixes for the new solver. Add machine-readable diagnostics containing:

- Actual accepted time, time-step size, SNES/KSP convergence reasons and iteration counts, solver mode, and timings.
- Global sampled minimum of \(J_{\mathrm{ALE}}=\det(I+\nabla_Xu)\), separately in fluid and solid, plus a mesh distortion measure such as a sampled deformation-gradient condition number.
- Initial reference mesh validity, weight extrema, non-finite-value checks, and separately scaled residual norms for mesh, kinematic, momentum, and incompressibility equations.

Evaluate mesh quality inside cells at a documented sufficiently rich set of points, not only at vertices; P2 displacement and curved geometry can invert between vertices. Call this a sampled validity check, not a proof of global bijectivity. Increase sampling/quadrature during verification. Abort visibly on an invalid accepted state. If invalid Newton trials occur, use supported SNES globalization/domain-error handling, not coefficient clipping or accepting an invalid iterate.

Audit time labels and final-time coverage. The local snapshot starts at `t=0` and solves inside `while t < T`, so a final log line is not sufficient evidence that the physical endpoint was reached. Preserve the chosen time discretization, but ensure the new runner advances and reports actual accepted states through the requested endpoint. Make any necessary time-label/endpoint correction narrow, documented, and tested separately from changing the mesh law.

Close both VTX writers on success and failure. Use rank-safe directory/file handling. Expose diagnostics and final state in a testable way without breaking existing callers. Reuse checkpoint/restart if available; do not require a new checkpoint subsystem to finish this task.

## 6. Verify the operator before running the full benchmark

Add `tests/test_fsi2_stiffened_elastic_diffmesh.py` and focused mesh-extension tests in the repository's existing structure. Required checks:

1. **Weight definition:** exponent zero gives unit weight; on unequal affine cells the coefficient ratio matches inverse volume to power \(\chi\); pointwise weighting varies appropriately on a curved/non-affine cell; distributed normalization and ghost values agree between one and two ranks.
2. **Elastic form:** rigid translations and infinitesimal rotations produce zero mesh stress; a simple strain field matches the intended Lamé law. On a small problem with prescribed boundary displacement, changing only the overall modulus scale leaves the standalone solution unchanged.
3. **Jacobian and interface:** directional finite differences confirm the new mesh tangent, including its fluid-side interface term. Check facet orientation and shared interface handling in serial and MPI. Existing tests of the no-ALE block omissions must still pass and preserve the solid tangent.
4. **History independence:** a standalone extension driven through a repeatable boundary-displacement cycle returns to its initial state at zero boundary displacement. This tests the reference-based LE formulation independently of FSI dynamics.
5. **Block compatibility:** compare a small direct block-factorized correction against monolithic LU using the new \(A\). Exercise the production approximate fieldsplit too, checking disjoint/exhaustive owned field indices, converged true residuals, and agreement with direct solves at equivalent nonlinear tolerances.
6. **Coupled startup:** run at least six steps on `mesh_sec_coarse.xdmf`, verify finite QoIs, positive sampled ALE determinants, boundary conditions, actual times, and full residual convergence. Compare direct and fieldsplit with numerical error tolerances and absolute floors for near-zero outputs; file existence alone is insufficient.
7. **Regression:** run the affected harmonic tests after shared-helper edits. Exercise triangular and quadrilateral assembly; use one and two MPI ranks for focused numerical checks. Do not infer parallel correctness from a serial pass.

## 7. Demonstrate full FSI2 completion on the remote machine

Primary acceptance is a run from rest through **T = 15 s**, with **dt = 0.0025 s**, on the regular benchmark mesh, using **stiffened elasticity and the production no-ALE fieldsplit solver**. This must include developed oscillations beyond the harmonic solver's documented failure near 7 s. Startup smoke tests and standalone displacement tests do not establish completion. Extend a successful run to 20 s if resources permit to check sustained behavior.

Use `data/meshes/fsi2/mesh_sec.xdmf` as the initial production target unless the remote work has established a different benchmark mesh; record the choice. The deliberately coarse test mesh is not the sole production validation. Keep the physical configuration fixed. At minimum, compare direct versus fieldsplit on a nontrivial evolved interval from equivalent states; run full direct reference simulations where practical. Use equivalent time steps and tolerances. Compare tip-displacement mean/amplitude/frequency and drag/lift over a developed interval such as [12,15] s, with published FSI2 data and any available validated biharmonic results. Using biharmonic output as a reference is allowed; using biharmonic motion inside the new solver is not.

Report relative errors and refinement sensitivity rather than declaring success solely from the absence of a crash. Use a smaller time step and/or finer mesh on an additional run or developed-motion comparison to assess discretization sensitivity. Define comparison tolerances before inspecting the final result, based on existing validated resolution; do not assume the comparison paper's accuracy transfers unchanged to this mesh and time discretization.

The literature's success on its IGA mesh is evidence for the method, not a guarantee for this repository's geometry and meshes. If a run fails:

- Save the last valid diagnostics and identify first failure: invalid geometry, nonlinear convergence, linear/preconditioner failure, or incorrect physical response.
- Compare accurate direct and approximate block solves at the same valid state. Determine whether failure comes from mesh extension or iterative algebra before tuning.
- Investigate normalization, quadrature, fluid-side traction, geometry validity, residual scaling, and preconditioner coefficient consistency first.
- Then perform a small documented parameter study near \(\chi=2.5\) (e.g. 2, 2.5, 3), and assess time-step/mesh-resolution sensitivity. If necessary, tune mesh Poisson ratio in the paper's discussed 0.3-0.45 range. Keep defaults tied to observed results and record deviations.
- Do not switch to a Gaussian weight, harmonic/biharmonic blend, remeshing, incremental updated-geometry elasticity, altered solid material, or unreported weight clipping to call this implementation successful. A materially different strategy requires a separate decision.

Persist through diagnosis and the relevant reruns. If available resources or a demonstrated numerical limitation prevent the full run, report it as **not yet validated**, with exact failure time, configuration, and reproduction command. Do not claim the objective is achieved.

## 8. Remote execution and deliverables

`data/` is gitignored. Inspect existing files before generating anything; preserve valuable reference data and existing meshes. Generate missing test meshes with:

```bash
conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.scripts.create_test_meshes
```

Generate a missing regular triangular quadratic-geometry mesh with the existing `create_mesh_FSI2.create_mesh` using `fine=False`, `coarse=False`, `quads=False`, `semi_structured_quad=False`, `second_order=True`. Ensure the destination exists. The generator's default `main()` creates a quadrilateral mesh with a different filename, so it does not supply `mesh_sec.xdmf` automatically. Document an exact safe invocation in the final implementation instructions.

Implement a command-line runner for the new module that exposes mesh path, final time, time step, solver/Jacobian modes, mesh parameters, output directory, and output frequency. Reuse an existing common CLI if present. Include reproducible commands in the implementation README; the following is the intended interface, not a currently available command:

```bash
conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh \
  --mesh data/meshes/fsi2/mesh_sec.xdmf --T 15 --dt 0.0025 \
  --linear-solver fieldsplit --jacobian-mode no_ale \
  --mesh-stiffening-exponent 2.5 --mesh-poisson-ratio 0.3 \
  --output-dir output/fsi2_stiffened_elastic
```

Document an MPI invocation using the remote environment's matching MPI launcher. Register long benchmark tests as opt-in, keeping them out of default fast pytest runs. Record software versions, MPI ranks, mesh provenance/checksum, DOFs, parameters, timings, Krylov work, true residuals, sampled geometry minima, and QoI comparisons. Keep large output/mesh artifacts out of Git; commit a compact validation report and exact reproduction commands.

Suggested commits, each with associated checks:

1. Add the reference-weighted elastic mesh operator and operator tests.
2. Add the new monolithic solver/runner, matching interface flux, diagnostics, and coupled tests.
3. Integrate the elastic displacement preconditioner into the existing shared fieldsplit path and add serial/MPI comparisons.
4. Document measured full-run results and any validated tuning.

Adapt these boundaries to the actual remote infrastructure. Completion requires the new solver, production fieldsplit compatibility demonstrated numerically, meaningful short tests, a full-run validation report, and documented remaining limitations. Review `git log --oneline` before the final response and report the commit sequence. Do not describe planned or still-running simulations as passing.

## Source and scope of the adaptation

A. Shamanskiy and B. Simeon, *Mesh moving techniques in fluid-structure interaction: robustness, accumulated distortion and computational efficiency*, Computational Mechanics 67, 583-600 (2021), [DOI: 10.1007/s00466-020-01950-x](https://link.springer.com/article/10.1007/s00466-020-01950-x). Read sections 3.3, 3.5, and 5: non-incremental LE, mesh-Jacobian stiffening, and FSI2 evaluation. Their FSI2 experiment uses \(\chi=2.5\), \(\nu_a=0.3\), and a 15 s run with 0.0025 s steps; reference-based LE completes it without accumulated distortion. The benchmark evidence uses a partitioned IGA implementation. The shared-space monolithic interface treatment, normalization convention, fieldsplit integration, diagnostics, and tests above are implementation requirements for this repository, not claims that the paper implements those choices.

The harmonic fieldsplit contract is documented in [shared-space-iterative-solver-plan.md](shared-space-iterative-solver-plan.md). Treat actual validated remote code as authoritative for helper names and interfaces while retaining this task's mesh-law requirements.
