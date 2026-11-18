# Handoff to Claude: compressible three-term Mooney–Rivlin

## Objective and scope

Implement the compressible three-term Mooney–Rivlin (MR3) material in this solver library,
with an opt-in integration in `fsi2_simplified` and meaningful material and FSI2 verification.
Keep SVK as the default. The material implementation must support 3D and 2D plane strain;
3D FSI integration and aneurysm production simulations are follow-up work.

**Explicit user requirement: define the first Piola–Kirchhoff stress by UFL differentiation
of the strain-energy density, `P = ufl.diff(W, F)`. Do not implement a prederived stress
formula in production code.** Let UFL also differentiate the residual to obtain its tangent.

Work in `~/code/xfsi_solver` on `three-term-mooney-rivlin`. This handoff branch starts at
`origin/restricted`, commit `d31cec241ae7342d5880fe5c320ee490dff98831` (fetched 2026-11-18).
This is the solver code repository, distinct from `~/fsi-library`, the private knowledge
library, and from the solver clone inside `aneurism-fsi/external/`.

At handoff creation the working tree already had unrelated changes in
`src/xfsi_solver/solvers/fsi2_simplified.py`, plus untracked `fsi2_w_oseen.py` and
`fsi2_w_stokes.py` in that directory. Preserve them and their staging. They are not part of
this handoff commit. Inspect overlapping edits before changing `fsi2_simplified.py` and
stage only this task's implementation changes.

## Material definition

The source is Yamamoto et al., *Impact of Wall Property and Flow Rate Assumptions on
Simulations of Flow-Induced Vibration of Intracranial Aneurysms*, International Journal
for Numerical Methods in Biomedical Engineering (2025), Equation (5) and Table 1.
The user's attached copy is named `J49 Kei Sensitivity.pdf`. The complete formulation
needed here is reproduced below; do not depend on an attachment being accessible.

Use energy per reference volume:

```text
F = I + grad(u)
J = det(F)
Bbar = J**(-2/3) * F * F.T
I1bar = tr(Bbar)
I2bar = 0.5 * (tr(Bbar)**2 - tr(Bbar * Bbar))

W = C10*(I1bar - 3) + C01*(I2bar - 3)
    + C11*(I1bar - 3)*(I2bar - 3) + (J - 1)**2/D1
P = dW/dF
```

All invariants here are those of a THREE-dimensional tensor. Require positive `J` for
physical admissibility. Do not silently clip `J` or change the energy to hide inverted
states. All `Cij` have stress units; `D1` has inverse-stress units. Use Pa internally.
The three terms are the distortional terms; the volume penalty is separate.

The initial moduli are:

```text
mu0 = 2*(C10 + C01)
K0 = 2/D1
```

`C11` does not affect the initial linear-elastic moduli but must be exercised at finite
strain in the tests. This law has no viscous damping, solid pressure unknown, or collagen
fiber directions. Preserve the existing two-term incompressible `Solid.IMR` functionality;
it is a different formulation and must not be repurposed as compressible MR3.

### UFL construction and stress convention

Mark the deformation gradient as `ufl.variable` before forming the invariants and energy,
and obtain the stress with `ufl.diff(W, F)`. A helper returning energy from a variable F,
plus a clearly named first-Piola material helper, is a reasonable design. Follow the
repository's conventions and avoid a broad unrelated material API refactor.

For 2D, embed the in-plane gradient as:

```text
F3 = [[F2[0,0], F2[0,1], 0],
      [F2[1,0], F2[1,1], 0],
      [0,        0,       1]]
```

Mark `F3` as a UFL variable, construct the 3D energy, differentiate with respect to it,
and return the upper-left 2x2 block of P for the 2D residual. The dependency on u must
remain available for subsequent residual differentiation. This is plane strain, NOT
plane stress: out-of-plane stress generally need not vanish. Do not simply compute 2D
invariants and subtract 3, or change the exponent to -1.

Existing `Solid.STVK` returns Cauchy stress. The residual currently converts it via
`P = J*sigma*inv(F).T`. MR3 already returns P and must enter as
`inner(P, grad(test_velocity))*dx_solid` without that extra conversion. If a Cauchy
output is needed, derive it as `sigma = P*F.T/J`.

## Parameters for implementation testing

The paper includes D1 in the energy and describes its connection to the bulk modulus,
but does not give a numerical D1 in the attached main text/table. Do not claim that the
provisional value below reproduces the paper's compressibility.

The user authorizes deriving a provisional bulk modulus from existing SVK parameters:

```text
K_test = lambda_SVK + 2*mu_SVK/3
D1_test = 2/K_test
```

Expose an explicit D1 override so a verified value can be used later. Resolve existing
`mu_s`, `nu_s`, and `lambda_s` consistently with the current parameter helper. Document
that those SVK parameters determine the fallback bulk modulus, while the MR3 Cij determine
its shear response. Validate basic numerical inputs, including a positive finite D1/K
and sensible nonnegative coefficients for the supported test presets; do not claim global
material stability from these basic checks.

For a controlled FSI2 comparison, initially match BOTH bulk and infinitesimal shear moduli:

| Quantity | Value |
| --- | --- |
| SVK mu | 500000 Pa |
| SVK nu | 0.4 |
| SVK lambda | 2000000 Pa |
| K_test | 2333333.333333333 Pa |
| D1_test | 8.571428571428571e-7 Pa^-1 |
| C10 + C01 for matched initial shear | 250000 Pa |

A concrete test preset is `C10=0`, `C01=250000 Pa`, `C11=1800000 Pa`, with D1 as above.
The C11 value is borrowed from the paper's medium case as a finite-strain test value;
this combined preset is an artificial benchmark material, not a tissue fit. Include a
C11=0 control as well, but do not use it as the only MR3 integration test.

For later optional comparison, the paper's fits converted to Pa are:

| Case | C10 | C01 | C11 |
| --- | --- | --- | --- |
| Soft | 0 | 2000 | 900000 |
| Medium | 0 | 20000 | 1800000 |
| Stiff | 0 | 200000 | 1800000 |

The medium tissue fit has mu0=40000 Pa, 12.5 times softer initially than FSI2. Combining
it with FSI2's K_test gives an initial Poisson ratio about 0.4915, so it is more nearly
incompressible and can be harder to solve. Keep this separate from the matched-modulus
test. A copied bulk modulus does not preserve Poisson ratio if the shear modulus changes.

## Integration points and solver constraints

Inspect the current versions of:

- `src/xfsi_solver/fsi/materials.py`: material helpers and stress conventions.
- `src/xfsi_solver/fsi/forms.py`: `FSIForms.A_E_solid` and `build_residual`.
- `src/xfsi_solver/solvers/fsi2_simplified.py`: material parameters and FSIForms construction.
- `src/xfsi_solver/tools/parameters.py`: existing Lame-parameter resolution.
- `tests/test_materials.py` and `tests/test_fsi2_simplified.py`: test conventions.

Add a small, explicit opt-in material selection and coefficients, with SVK defaults and
existing callers preserved. Thread that choice through the shared forms. The current
residual is scaled by dt: preserve the existing dt factors and theta weighting for both
current and previous solid stress. Preserve condensation through
`u_s_new = u_s_old + dt*((1-theta)*v_old + theta*v)` and automatic residual differentiation
with respect to velocity and fluid pressure. No additional solid pressure field is needed
for this compressible formulation.

Do not change fluid physics, mesh motion, time integration, or the incompressible solver
as part of this task. The mesh-motion derivative is already approximated in this solver;
material AD does not make the entire ALE Jacobian exact.

Start diagnostic runs with a fresh Jacobian each Newton iteration (`gamma_reassemble=None`)
and conservative stepping. The displacement/mesh update uses a KSP post-solve hook that
assumes a full Newton correction. Do not enable backtracking as a quick fix: each trial
velocity would need consistent displacement and mesh updates. Only change this machinery
if evidence makes it necessary, and explain any expansion in scope.

Monitor solid J and fluid mesh J separately. Recheck quadrature sensitivity because MR3
contains nonpolynomial terms. Large bulk penalties can lead to conditioning problems or
volumetric locking; finite K is a modeling choice, not exact incompressibility.

## Verification and acceptance

Use tests that independently check the material and its integration, not merely two
copies of the production expression:

1. Zero energy/stress at identity and zero stress under a rigid rotation, in 3D and
   plane strain. Check appropriate energy/stress transformation for a rotated deformed
   state if useful.
2. Directional finite differences of W against P, and of P or the assembled solid
   residual against its UFL tangent, at nontrivial admissible deformations. Include the
   condensed velocity dependency to catch a disconnected AD variable.
3. Infinitesimal response consistent with `mu0=2*(C10+C01)` and `K0=2/D1`; compare to
   matched SVK only in the small-strain limit.
4. Independent finite simple-shear check with nonzero C11: for amount gamma,
   `W=(C10+C01)*gamma**2+C11*gamma**4`,
   `P12=2*(C10+C01)*gamma+4*C11*gamma**3` for F12=gamma, J=1.
   Analytic reference expressions are appropriate in TESTS, not production stress code.
5. Pure 3D dilation: distortional energy is zero and Cauchy stress is
   `2*(J-1)/D1 * I`. Check D1 units/factors and the 2D result against an independently
   evaluated embedded 3D plane-strain state.
6. Existing SVK behavior remains unchanged. Add a short, reproducible nonzero-C11 FSI2
   run that exercises coupled stress, with meaningful convergence/finite-output checks.
   Check a suitable finite-strain state or loading interval so the third term actually
   matters; an almost undeformed startup alone is insufficient evidence.
7. Compare selected outputs under time-step and quadrature refinement at a practical
   cost. Report the physical interval, solver settings, iterations, J diagnostics, and
   limitations. A short smoke test is not evidence of a fully converged periodic cycle.

FSI2 with MR3 is a MODIFIED benchmark. Published SVK drag, lift, frequency, and displacement
are contextual comparisons, not exact targets for this different finite-strain material.
Do not tune coefficients to force agreement with SVK benchmark numbers.

Provide a runnable example or documented invocation with the matched-modulus preset and
D1 override. Report test commands and outcomes, any remaining convergence limitations,
and reviewable implementation commits. Do not launch a new agent or message another chat
just because this document is a handoff; it is intended for a user-started Claude session.

## Execution rules

Read `AGENTS.md`, `CLAUDE.md`, and the knowledge library index/inbox before implementation.
Use the `xfsi_solver` Conda environment, for example
`conda run --no-capture-output -n xfsi_solver pytest ...`.
Before tests or MPI runs, read `~/fsi-library/infrastructure/running-simulations.md`.

Tests marked `slow` require explicit user permission; this handoff does not authorize them.
Do not run `pytest -m slow`, `pytest -m ""`, or select a slow test indirectly without that
permission. Follow the repository's physical-core budget, timeout, MPI thread-count,
unique-output, and long-run detachment rules. Do not overwrite existing simulation outputs.
Commit implementation and its tests in logical units on this branch, preserving unrelated
work. Report results honestly and follow the repository procedure for recording findings.
