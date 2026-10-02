**An iterative solver for FSI with restricted displacement test functions**

At a fixed time step and nonlinear iterate, the full Newton system has the following block structure. Prescribed degrees of freedom have been eliminated and their contributions included in the right-hand side:

\[
\underbrace{
\begin{pmatrix}
A_S & C & 0 & 0 & 0 \\
E_S+F_S & H & G & 0 & F_I \\
D_S & B & 0 & 0 & D_I \\
-K_{zS} & 0 & 0 & M_z & -K_{zI} \\
0 & 0 & 0 & K_{Iz} & 0
\end{pmatrix}}_{J_{\mathrm{full}}}
\begin{pmatrix}
\delta u_S\\ \delta v\\ \delta p\\ \delta z\\ \delta u_I
\end{pmatrix}
=
\begin{pmatrix}
r_S\\r_v\\r_p\\r_z\\r_I
\end{pmatrix},
\qquad r=-R(U).
\tag{1}
\]

This is the restricted-test-function **biharmonic** formulation implemented in `src/xfsi_solver/solvers/fsi2_biharmonic_diffmesh_restricted.py`. The ordering is chosen to expose the algebra; it is not the storage order in that file. The harmonic specialization is described near the end.

The displacement field is still a single continuous finite element function. We only partition its coefficients: \(u_S\) contains every solid-supported degree of freedom, including the shared interface, and \(u_I\) contains the remaining fluid-interior degrees of freedom. Velocity \(v\) remains a global, shared fluid-solid field. Pressure \(p\) and the auxiliary biharmonic field \(z\) live on the fluid submesh. There are no interface multipliers or duplicated interface coefficients.

The five block rows in (1) are, respectively, solid kinematics, global momentum, fluid incompressibility, the auxiliary mesh equation, and the fluid-interior displacement equation. Their blocks mean:

| Block | Meaning |
|---|---|
| \(A_S\) | Solid displacement mass, \(\rho_s M_S/\Delta t\) |
| \(C\) | Velocity contribution to solid kinematics |
| \(E_S\) | Solid stress derivative with respect to \(u_S\), including its factor \(\theta\) |
| \(F_S,F_I\) | Fluid momentum derivatives with respect to displacement; \(F_S\) acts through the interface |
| \(H\) | Momentum derivative with respect to the global velocity: fluid inertia, transport and viscosity, plus solid inertia |
| \(G\) | Fluid pressure contribution to momentum |
| \(B\) | ALE divergence acting on velocity |
| \(D_S,D_I\) | Displacement derivatives of ALE incompressibility |
| \(M_z\) | Fluid mass matrix for the auxiliary field |
| \(K_{zS},K_{zI},K_{Iz}\) | Gradient couplings in the mixed biharmonic extension |

The mesh signs follow the implemented forms: \((z,\eta)_f-(\nabla u,\nabla\eta)_f\) and \((\nabla z,\nabla\psi_I)_f\). Thus the last two rows state

\[
M_z\delta z-K_{zI}\delta u_I-K_{zS}\delta u_S=r_z,
\qquad K_{Iz}\delta z=r_I.
\tag{2}
\]

The first row contains only the solid kinematic equation because interface displacement test functions have no fluid contribution. This property refers to the final restricted residual and Jacobian, after the interface row replacement. It is what makes the reduction below exact.

**Neglecting fluid ALE derivatives.**

Set \(F_S,F_I,D_S,D_I\) to zero, while retaining \(E_S\):

\[
\underbrace{
\begin{pmatrix}
A_S & C & 0 & 0 & 0 \\
E_S & H & G & 0 & 0 \\
0 & B & 0 & 0 & 0 \\
-K_{zS} & 0 & 0 & M_z & -K_{zI} \\
0 & 0 & 0 & K_{Iz} & 0
\end{pmatrix}}_{J_0}
\begin{pmatrix}
\delta u_S\\ \delta v\\ \delta p\\ \delta z\\ \delta u_I
\end{pmatrix}
=
\begin{pmatrix}
r_S\\r_v\\r_p\\r_z\\r_I
\end{pmatrix}.
\tag{3}
\]

This discards all displacement derivatives of the fluid momentum and continuity equations, including derivatives of mesh velocity \((u-u^{old})/\Delta t\). It retains fluid velocity and pressure derivatives, solid elasticity, solid kinematics, and the mesh equations. At shared interface rows, retain the solid contribution and discard only the fluid shape derivative.

The right-hand side remains the **full nonlinear residual**, evaluated using the current displacement, geometry and mesh velocity. Geometry is updated between nonlinear iterations. Consequently, (3) defines an approximate Newton method for the original fully implicit discrete equations. The approximation is in the derivative, not in the residual or the time discretization. Quadratic nonlinear convergence is not guaranteed.

Test-function restriction alone does not make (1) triangular: \(F_I\) and \(D_I\) feed fluid-interior mesh increments back into the physical equations. Their omission is essential for the sequential solve proposed here.

**Condensing the solid displacement.**

Let \(R_S\) restrict the global velocity coefficients to the solid-supported coefficients. Because displacement and velocity use matching bases and compatible essential conditions, the solid kinematic equation gives

\[
A_S=\frac{\rho_s}{\Delta t}M_S,
\qquad C=-\theta\rho_s M_SR_S
=-\theta\Delta t\,A_SR_S.
\tag{4}
\]

For the right-hand side of the current linear solve, define

\[
g_S=A_S^{-1}r_S.
\]

The first row of (3) then yields

\[
\boxed{\delta u_S=\theta\Delta t\,R_S\delta v+g_S.}
\tag{5}
\]

Substituting (5) into momentum gives the reduced physical system

\[
\boxed{
\underbrace{
\begin{pmatrix}
H_c&G\\B&0
\end{pmatrix}}_{K_c}
\begin{pmatrix}\delta v\\\delta p\end{pmatrix}
=
\begin{pmatrix}r_v-E_Sg_S\\r_p\end{pmatrix},
\qquad H_c=H+\theta\Delta t\,E_SR_S.
}
\tag{6}
\]

If \(K_S^{\mathrm{tan}}\) denotes the solid elastic tangent without the theta factor, then \(E_S=\theta R_S^T K_S^{\mathrm{tan}}\), and

\[
H_c=H+\theta^2\Delta t\,R_S^T K_S^{\mathrm{tan}}R_S.
\tag{7}
\]

In particular, the solid part of the effective velocity operator contains

\[
\frac{\rho_s}{\Delta t}M_S+\theta^2\Delta t\,K_S^{\mathrm{tan}}.
\]

This is an exact condensation of (3), not an approximation to its elastic feedback. It produces a sparse operator that can be assembled directly. No fluid mesh solve is hidden inside \(H_c\). Fluid and solid velocities remain coupled through their shared interface coefficients.

For an actual Newton right-hand side, \(g_S\) is the negative nodal kinematic defect, provided essential conditions have already been accounted for. For an arbitrary residual passed to a preconditioner, computing \(g_S\) requires a mass-inverse action. Do not omit this term simply because it vanishes at a converged state.

**Grouping the physical and mesh unknowns.**

Define groups of increments

\[
q=\begin{pmatrix}\delta u_S\\\delta v\\\delta p\end{pmatrix},
\qquad
m=\begin{pmatrix}\delta z\\\delta u_I\end{pmatrix},
\qquad
r_q=\begin{pmatrix}r_S\\r_v\\r_p\end{pmatrix},
\qquad
r_m=\begin{pmatrix}r_z\\r_I\end{pmatrix}.
\]

Then (3) becomes

\[
\boxed{
\begin{pmatrix}
K_q&0\\L&K_m
\end{pmatrix}
\begin{pmatrix}q\\m\end{pmatrix}
=
\begin{pmatrix}r_q\\r_m\end{pmatrix},
}
\tag{8}
\]

where

\[
K_q=\begin{pmatrix}A_S&C&0\\E_S&H&G\\0&B&0\end{pmatrix},
\qquad
L=\begin{pmatrix}-K_{zS}&0&0\\0&0&0\end{pmatrix},
\qquad
K_m=\begin{pmatrix}M_z&-K_{zI}\\K_{Iz}&0\end{pmatrix}.
\tag{9}
\]

The solid-displacement condensation explains how to solve \(K_q\). The outer grouping explains when to solve \(K_m\): after the physical correction is known.

Equivalently, after eliminating \(\delta u_S\) altogether, define \(q_c=(\delta v,\delta p)^T\). The remaining system is still triangular:

\[
\begin{pmatrix}
K_c&0\\L_c&K_m
\end{pmatrix}
\begin{pmatrix}q_c\\m\end{pmatrix}
=
\begin{pmatrix}
r_v-E_Sg_S\\r_p\\r_z+K_{zS}g_S\\r_I
\end{pmatrix},
\qquad
L_c=\begin{pmatrix}
-\theta\Delta t\,K_{zS}R_S&0\\0&0
\end{pmatrix}.
\tag{10}
\]

Both forms describe the same correction. Equation (8) is convenient for grouping the original unknowns; equation (10) exposes the smaller physical solve directly.

**One forward block solve.**

For a given right-hand side, apply the following steps in order:

1. Compute the solid kinematic correction \(g_S=A_S^{-1}r_S\).
2. Form the reduced momentum right-hand side \(b_v=r_v-E_Sg_S\).
3. Solve \(K_c(\delta v,\delta p)^T=(b_v,r_p)^T\).
4. Recover \(\delta u_S=\theta\Delta t\,R_S\delta v+g_S\).
5. Form the mesh right-hand side \(r_m-Lq=(r_z+K_{zS}\delta u_S,r_I)^T\).
6. Solve the mesh problem for \((\delta z,\delta u_I)\), then combine \(\delta u_S\) and \(\delta u_I\) into the single shared displacement correction.

If all component solves are exact, these six steps solve (3) exactly. With inexpensive approximate component solves, the same sequence is a preconditioner. Retaining step 5 is essential: independent additive physical and mesh solves would discard the transfer of interface motion into the fluid mesh.

**Using fieldsplit efficiently.**

Use FGMRES for the complete linear system and an outer multiplicative fieldsplit in the order **physical \(q\), then mesh \(m\)**. A forward block Gauss-Seidel application has the action

\[
\widehat q=\widehat K_q^{-1}r_q,
\qquad
\widehat m=\widehat K_m^{-1}(r_m-L\widehat q).
\tag{11}
\]

Here inverse symbols mean solve actions; matrices need not be inverted explicitly. The triangular factorization follows from the formulation, while the hatted inverses express the practical approximations. The hierarchy is

```text
FGMRES
  multiplicative split: physical, then mesh
    physical
      eliminate solid displacement through its mass block
      solve the coupled velocity-pressure problem with a Schur split
      recover solid displacement
    mesh
      solve the extension problem with the recovered interface motion
```

In the physical solve, split \(K_c\) into global velocity and fluid pressure. Its pressure Schur complement is

\[
S_p=-BH_c^{-1}G.
\tag{12}
\]

An exact full block-factorization application to \((b_v,r_p)\) is

\[
a=H_c^{-1}b_v,\qquad
\delta p=S_p^{-1}(r_p-Ba),\qquad
\delta v=a-H_c^{-1}G\delta p.
\tag{13}
\]

These equations also prescribe the approximate algorithm: replace the velocity and pressure inverse actions by suitable preconditioners while preserving the coupling and signs. The zero pressure diagonal is not itself a useful pressure preconditioner.

For velocity, start with algebraic multigrid on an auxiliary global operator containing fluid mass and viscosity and solid mass and elasticity. Preserve the solid stiffness term in (7) and the shared interface velocity coefficients. The actual tangent also contains transport and may be nonsymmetric or indefinite; an SPD auxiliary approximation does not make the Newton operator SPD.

For pressure, an initial unsteady Stokes approximation is

\[
\widehat S_p^{-1}\simeq
\frac{\rho_f}{\Delta t}K_p^{-1}
+\theta\mu_f M_p^{-1},\qquad \mu_f=\rho_f\nu_f.
\tag{14}
\]

Here \(M_p\) and \(K_p\) are auxiliary pressure mass and Laplace operators on the fluid geometry. Equation (14) is a sum of inverse actions, not the inverse of a sum. Their boundary conditions, nullspaces and geometric weights must match the intended pressure approximation. This starting choice omits some convection and structural-response effects; establish its effectiveness by comparison with an accurate pressure Schur solve. A pressure convection-diffusion approximation is a possible later refinement.

FGMRES accommodates varying accuracy in preconditioner applications. Keep inverse approximations used inside a Schur matrix-vector product fixed and linear: flexibility of the outer preconditioner does not justify changing the linear operator being solved.

**Solving the mesh block.**

Let \(b_z=r_z+K_{zS}\delta u_S\). For the biharmonic extension, eliminating the auxiliary field from (2) gives

\[
\underbrace{K_{Iz}M_z^{-1}K_{zI}}_{S_m}\delta u_I
=r_I-K_{Iz}M_z^{-1}b_z,
\qquad
\delta z=M_z^{-1}(b_z+K_{zI}\delta u_I).
\tag{15}
\]

This suggests a mesh Schur split ordered as \(z\), then \(u_I\): the first diagonal block is an invertible mass matrix. Starting with \(u_I\) instead would put a zero block first. The reduced operator \(S_m\) is fourth order. An effective iterative approximation must account for the biharmonic boundary conditions; two independent Poisson solves are not automatically equivalent to (15).

A practical first version can use a reusable direct mesh factorization while the physical iterative solver is developed. The mesh matrices are constant for the fixed reference mesh and extension law; only their right-hand sides change. Reuse their setup across Newton iterations and time steps. A dedicated mixed-biharmonic or auxiliary-space preconditioner can replace this factorization later.

For **harmonic** extension, remove \(z\), set \(m=\delta u_I\), and replace the last two rows of (1) by

\[
A_{IS}\delta u_S+A_{II}\delta u_I=r_I.
\]

The physical condensation is unchanged. The final mesh solve is simply

\[
A_{II}\delta u_I=r_I-A_{IS}\delta u_S,
\]

with a vector-Laplacian AMG preconditioner as the natural starting point.

**How to establish confidence in the algorithm.**

First compare the six-step block solve using accurate subsolves with a monolithic direct solve of the same \(J_0\). This isolates algebraic errors in signs, right-hand sides and recovery. Then replace one inverse action at a time, measuring the true residual after each solve. Check that the physical correction is independent of the mesh-extension inverse when solving \(J_0\); the recovered fluid displacement should still respond to the interface motion.

At each nonlinear iteration, evaluate the full residual, assemble the current approximate Jacobian, solve for the correction, and update all fields consistently. A line search, if used, applies the same step length to the complete correction. Compare converged solutions against the full-Jacobian reference and assess performance during developed motion as well as startup.

The same triangular algorithm may also precondition the full Jacobian (1). In that case, FGMRES must resolve the omitted ALE feedback. With \(J_0\) as the system operator, that feedback is instead handled by the outer nonlinear iterations. Keep these two experiments distinct.

Mesh-equation scaling is an algebraic choice once their rows are separated from solid kinematics. Scale right-hand sides consistently and continue monitoring the unscaled mesh residual. A scaling that helps direct-solver pivoting does not necessarily help AMG or the Krylov stopping test.

The expected gain is structural: solid displacement has an exact inexpensive elimination, the effective velocity operator is sparse and directly assemblable, and mesh extension is a subsequent reusable solve. Robust pressure preconditioning and an efficient biharmonic inverse remain numerical tasks to validate rather than guaranteed consequences of the restriction.

**Background.** This reduction follows the combination of solid kinematic condensation and omitted fluid ALE derivatives in Failer and Richter, [A Parallel Newton Multigrid Framework for Monolithic Fluid-Structure Interactions](https://doi.org/10.1007/s10915-019-01113-y). Their geometric multigrid and Vanka smoother are one way to solve the reduced physical problem; the algebra above also supports the field-split and algebraic-multigrid strategy described here. PETSc provides the relevant [multiplicative and Schur field splits](https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/).
