**Implementation handoff: shared-space ALE-FSI iterative solver**

Implement this plan on the existing `iterative-solver` branch, which was branched from `main`. Use Conda environment `xfsi_solver`, follow the repository agent instructions (`CLAUDE.md` for Claude Code, `AGENTS.md` for Codex), and preserve unrelated work. This document is the plan; the solver has not yet been implemented or benchmarked.

The target is `src/xfsi_solver/solvers/fsi2_harmonic_diffmesh.py`. Its displacement and velocity spaces are quadratic vector spaces on the whole fluid-solid mesh; pressure is linear on the fluid submesh. Continuity is enforced through shared interface degrees of freedom. Do not introduce separate fluid/solid solution fields or Lagrange multipliers. The Lagrange-multiplier solver is outside this task.

The deliverable is a configurable PETSc FGMRES/fieldsplit solver with fluid ALE derivatives omitted from the Jacobian, validated against the existing full-Jacobian MUMPS solve. Preserve the full nonlinear residual, material laws, boundary conditions, time discretization, mesh-extension coefficient, and shared-space coupling. Geometric multigrid and Vanka are not required.

1. **Establish the reference and separate solver setup from time stepping.**

   Run commands through `conda run --no-capture-output -n xfsi_solver ...` so progress from long test and benchmark runs is visible. Inspect the installed DOLFINx and PETSc APIs before implementation. Keep `dolfinx.fem.petsc.NonlinearProblem`; do not use the legacy `NewtonSolverNonlinearProblem`. Use `dolfinx.fem.petsc.LinearProblem` for standalone variational linear problems; PETSc sub-KSPs remain appropriate for algebraic preconditioner components.

   Introduce small, testable helpers for residual/Jacobian construction, solver configuration, field index sets, and auxiliary operators. Keep the existing `solve()` calls compatible. Suggested configuration axes are `jacobian_mode={full,no_ale}` and `linear_solver={direct,fieldsplit}`, with an explicit diagnostic configuration using direct subsolves. Keep `full/direct` as the initial default and reference.

   Capture a baseline using `tests/test_fsi2_harmonic_diffmesh.py` and `data/meshes/fsi2/mesh_sec_coarse.xdmf`. Record nonlinear convergence, solution fields, tip displacement, drag/lift, and timings. Reuse the existing output-directory fixture. Expose enough state or diagnostics to compare numerical solutions without relying on plots or log parsing.

2. **Construct the approximate Jacobian by separating physical contributions.**

   With rows and columns ordered as `(u, v, p)`, write

   \[
   J_{\mathrm{full}}=
   \begin{pmatrix}
   A&C&0\\
   E_s+E_{\mathrm{ALE}}&H&G\\
   D&B&0
   \end{pmatrix},
   \qquad
   J_{\mathrm{noALE}}=
   \begin{pmatrix}
   A&C&0\\
   E_s&H&G\\
   0&B&0
   \end{pmatrix}.
   \]

   Here `A` is the solid kinematic mass plus the fluid harmonic operator including the existing interface flux subtraction; `C` is the solid velocity contribution to the kinematic equation; `E_s` is the solid stress tangent including its theta weight; `H` is the velocity derivative of momentum; `G` and `B` are the pressure and divergence blocks. `D` and `E_ALE` contain the fluid displacement derivatives.

   For `no_ale`, set block `(2,0)` to zero and replace block `(1,0)` by the derivative of the solid momentum contribution alone. **Do not zero the whole `(1,0)` block.** Shared interface rows contain both fluid and solid terms, so select terms by integration domain/physical contribution, not by a row mask.

   Omit fluid displacement derivatives from inertia, convection, viscosity, pressure stress, incompressibility, and the explicit mesh velocity `(u-u_old)/dt`. Retain all velocity derivatives, solid tangents, the mesh equation, and the kinematic equation. Preserve the original entity maps for mixed-mesh assembly.

   Leave residual evaluation unchanged: evaluate current geometry and mesh velocity at every nonlinear iterate. This is an approximate Newton method for the same nonlinear equations, not explicit mesh time stepping. Avoid compiling omitted derivative forms in `no_ale` mode.

   First verify `no_ale/direct` against `full/direct`. Then support `full` Jacobian with a `no_ale` preconditioning matrix as a diagnostic that isolates preconditioner quality from nonlinear-linearization effects. The final requested mode must also use `no_ale` as the Jacobian, realizing the derivative-assembly savings. Do not promise quadratic nonlinear convergence for that mode.

3. **Implement the outer displacement-versus-momentum Schur split.**

   Use outer FGMRES with `PCFIELDSPLIT` configured as a full Schur factorization. Define two explicit splits: `u` and `vp=(v,p)`. Elimination of displacement in the approximate Jacobian gives

   \[
   S_{vp}=
   \begin{pmatrix}
   H_{\mathrm{eff}}&G\\
   B&0
   \end{pmatrix},
   \qquad H_{\mathrm{eff}}=H-E_sA^{-1}C.
   \]

   For a right-hand side `(r_u,r_v,r_p)`, the reduced momentum right-hand side is `r_v-E_s A^{-1}r_u`; displacement recovery is `A^{-1}(r_u-C delta_v)`. Retain these coupling actions in the preconditioner.

   Retain the existing AIJ layout unless an actual implementation constraint justifies changing it. Build distributed index sets from owned scalar degrees of freedom, including vector block sizes. Test that they are disjoint and exhaustive. Inner splits must use the extracted `vp` matrix's numbering, not the parent numbering. Do not assume global field-contiguous storage, include ghosts as owned entries, or identify fields by coordinates.

   Begin on the coarse mesh with direct `A` solves and an explicitly formed reduced Schur matrix with LU. Use PETSc's `schur_precondition=full` only as a diagnostic. Compare the resulting correction with a monolithic LU correction before introducing approximate subsolves.

4. **Supply an assembled reduced preconditioning operator for nested field splitting.**

   The production hierarchy is

   ```text
   FGMRES on J_noALE
   +-- Schur fieldsplit: u | (v,p)
       +-- u: solid/interface mass approximation + harmonic extension
       +-- (v,p): Schur fieldsplit on an assembled auxiliary operator
           +-- v: AMG on a coupled fluid-solid effective velocity operator
           +-- p: unsteady pressure Schur inverse approximation
   ```

   Assemble the auxiliary reduced operator

   \[
   P_{vp}=\begin{pmatrix}\widehat H_{\mathrm{eff}}&G\\B&0\end{pmatrix}.
   \]

   Register it as a user Schur preconditioning matrix for the outer split. Configure the inner `v|p` split to extract blocks from this assembled matrix. Do not assume that a symbolic `MatSchurComplement` supports the submatrix extraction needed by a nested fieldsplit. A `preonly` reduced KSP applying the nested PC is a suitable initial design; inspect `ksp_view` to verify the actual operators and hierarchy.

   Avoid accidentally selecting full-Jacobian blocks when testing the `no_ale` preconditioner with a full Newton operator. Inspect the installed PETSc `Amat`/`Pmat` extraction behavior and configure it explicitly where necessary.

5. **Approximate the displacement inverse without changing interface equations.**

   `A` is not simply an SPD vector Laplacian: it combines a solid mass term with a tiny fluid mesh coefficient and a one-sided interface flux subtraction. Do not apply CG to the complete block without establishing symmetry and definiteness.

   Build an auxiliary partition of displacement degrees of freedom: `I` contains all degrees of freedom incident to solid cells, including the shared interface; `f` contains the remaining fluid-interior degrees of freedom. Preserve exactly one copy of every interface unknown.

   Start with a block-triangular approximate inverse in this ordering: solve an approximation to the solid/interface block, then solve the fluid-interior harmonic problem using the resulting interface increment. A solid mass approximation and AMG on the fluid-interior Laplacian are the first candidates. Retain the `A_fI` extension coupling. Dropping `A_If` or fluid corrections to `A_II` is a preconditioning approximation only; the actual Jacobian retains them.

   With alpha equal to `1e-9`, use documented algebraic normalization/scaling where needed. Transform right-hand sides and corrections consistently. Do not change alpha or silently rescale only some terms of the physical residual.

   Do not assume the paper's identity `delta_u_s = theta*dt*delta_v_s` is an exact condensation of this implementation: its shared displacement test functions also receive mesh/interface terms. Use the actual algebraic `A` and `C` in Schur coupling actions. Any future reformulation of the interface equations is outside this task.

6. **Construct the effective velocity and pressure preconditioners.**

   Assemble `H_eff_hat` on the existing global velocity space. Include fluid mass and viscosity, solid mass, and the solid elastic contribution. In the ideal solid kinematic limit,

   \[
   H_{\mathrm{eff},s}\approx
   \frac{\rho_s}{\Delta t}M_s+\theta^2\Delta t\,K_s^{\mathrm{tan}}.
   \]

   This expression motivates the auxiliary operator; it does not justify dropping the solid stiffness or claiming exact condensation at shared interface rows. Assemble contributions from both domains onto the same interface velocity degrees of freedom. Use elasticity-aware AMG with appropriate near-nullspace information. If the nonlinear tangent is unsuitable for an SPD auxiliary operator, use and document a positive mass-plus-elasticity approximation while retaining the tangent in `J_noALE`. Add convection to the auxiliary velocity operator only if convergence measurements justify it.

   For pressure, begin with an unsteady Stokes inverse approximation on the fluid pressure space:

   \[
   \widehat S_p^{-1}\simeq
   \frac{\rho_f}{\Delta t}K_p^{-1}
   +\theta\mu_f M_p^{-1},\qquad \mu_f=\rho_f\nu_f.
   \]

   This is a sum of inverse actions, not the inverse of a sum. Implement it with a small PETSc Python/shell PC, using an inexpensive pressure mass inverse and AMG for the pressure Laplace operator. Use the ALE metric for auxiliary operators and account for the difference between current and midpoint Jacobian factors when needed.

   Document auxiliary boundary conditions. A pressure-correction starting approximation is homogeneous Dirichlet at the do-nothing outflow and homogeneous Neumann elsewhere; treating the moving interface this way omits structural impedance and requires validation. Do not attach a constant-pressure nullspace automatically: inspect the actual operator and boundary conditions. Preserve the pressure Schur sign implied by the residual's positive divergence and negative pressure-stress convention.

   Pressure preconditioning remains the main numerical uncertainty. If iterations deteriorate with convection, test a PCD extension; match the code's advecting coefficient `theta*v-(u-u_old)/dt`. If they deteriorate with structural coupling, compare against an accurate pressure Schur solve to isolate missing interface/solid response. Do not claim parameter robustness from the fluid-only approximation.

7. **Control updates, tolerances, and object lifecycle.**

   Start by rebuilding state-dependent auxiliary operators at each Jacobian assembly. Reuse constant mesh/mass data where safe. Introduce preconditioner lagging only after obtaining correct baseline results, with explicit rebuild criteria and counters.

   Prefer fixed linear inverse approximations inside Schur matrix-vector products. FGMRES accommodates variable preconditioner applications, not a changing system operator. Use named option prefixes and verify the complete solver hierarchy at runtime.

   Expose tolerances rather than embedding tuning throughout the code. An initial outer KSP relative tolerance of `1e-6` is reasonable for experiments; tighten it when comparison errors or nonlinear convergence require it. Preserve existing nonlinear stopping criteria initially. Record true linear residuals and nonlinear residuals by field so the tiny mesh-equation scale does not hide poor displacement corrections.

   Keep convergence failures visible; no silent fallback to monolithic LU. Close both output writers on success and failure and destroy only PETSc objects owned by the new helpers. Preserve the existing high-level DOLFINx object's ownership responsibilities.

8. **Validate correctness before performance claims.**

   Add tests alongside the existing shared-space solver test. Use a small, nonzero admissible state, with positive deformation Jacobian, to verify that fluid ALE derivatives are nontrivial and that the solid tangent survives. Check retained derivative contributions with directional finite differences. For omitted directions, compare against a deliberately frozen-fluid-geometry linearization rather than requiring agreement with the full residual derivative.

   Verify direct block factorization against monolithic LU on a small assembled system, including the displacement recovery and reduced right-hand side. Validate auxiliary index sets and solver actions in serial and on two MPI ranks. Do not assume a serial test establishes correct submesh or nested-field numbering.

   Compare these modes with equivalent residual tolerances: `full/direct`, `no_ale/direct`, `full/fieldsplit` with the no-ALE preconditioner, and `no_ale/fieldsplit`. Compare normed field errors, tip displacement, drag/lift, and full nonlinear residuals, using absolute floors for near-zero quantities. Shared traces should remain continuous by construction.

   Separate short CI tests from benchmark runs. Benchmark at least the coarse and default meshes, time steps `0.0025` and `0.00125`, and a developed-motion state or short evolved window in addition to startup. Reuse a validated checkpoint if available; otherwise document how the state is generated. Report any baseline failure or unavailable benchmark rather than claiming success from startup alone.

   Record nonlinear iterations, outer and inner Krylov work, Jacobian/auxiliary setup times, total time per step, and memory when available. Include setup costs in comparisons. Confirm that production mode does not explicitly form a dense global Schur matrix or use monolithic LU. Direct coarse-grid solves are acceptable. No particular speedup is an acceptance requirement, but poor scaling or runtime must be reported honestly and diagnosed.

9. **Deliver a reviewable implementation and reproducible results.**

   Completion requires the working no-ALE iterative mode, a retained reference mode, numerical and MPI checks, reproducible benchmark commands, and documentation of unresolved robustness limits. Do not stop at changing PETSc options or at the direct-Schur diagnostic stage.

   Suggested logical commits, each including its relevant tests, are: expose solver configuration and the no-ALE Jacobian; implement distributed splits and direct block validation; implement displacement/effective-velocity preconditioning; implement the pressure PC and benchmark reporting. Commit each step as it is completed, without waiting for a separate instruction. Keep changes coherent rather than creating arbitrary checkpoint commits. Review the actual diff before staging, since files may be edited concurrently by the user; do not commit changes you did not make.

   Maintain an implementation log at `notes/shared-space-iterative-solver/implementation-log.md` and update it as work proceeds (decisions, API findings, measured results, open problems), not only at the end.

   Use the configured Git identity and the commit attribution trailer required by the agent instructions file for the tool doing the work. Stay on `iterative-solver`; do not amend, squash, rebase, or force-push. Finish by reviewing `git log --oneline` and reporting the commits, validation results, measured performance, and remaining limitations.

**References for implementation decisions.** Failer and Richter's [A Parallel Newton Multigrid Framework for Monolithic Fluid-Structure Interactions](https://doi.org/10.1007/s10915-019-01113-y), especially equations (4)-(12), motivates omitting fluid ALE derivatives while preserving solid elastic feedback. Its exact solid condensation and geometric multigrid infrastructure must not be assumed to match this shared-test-space implementation. See also the [DOLFINx NonlinearProblem API](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.fem.petsc.html), [PETSc Schur preconditioning documentation](https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/), and [PETSc field index sets](https://petsc.org/release/manualpages/PC/PCFieldSplitSetIS/).
