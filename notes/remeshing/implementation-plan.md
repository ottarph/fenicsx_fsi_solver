# Plan: remeshing for the FSI2 benchmark (harmonic mesh motion)

Status: **design/planning only, nothing in this document is implemented
yet.** See [`literature-review.md`](literature-review.md) for the background
that motivates the choices below.

## 1. The problem, precisely

`src/xfsi_solver/solvers/fsi2_harmonic.py` solves the FSI2 benchmark as a
**fixed-reference, pseudo-solid ALE** problem: `mesh.geometry.x` is loaded
once from `data/meshes/fsi2/mesh_sec.xdmf` and **never changes** for the rest
of the run. Everything that "moves" is carried by the displacement unknown
`u`, defined on a single `("CG", 2, (2,))` function space spanning *both* the
solid and fluid cells of one shared mesh:

- In the solid subdomain (`dx_solid`), `u` is the true structural
  displacement (St. Venant–Kirchhoff, `A_E`/`A_T` in `fsi2_harmonic.py`), a
  Lagrangian quantity always measured from the true, undeformed, t=0
  geometry.
- In the fluid subdomain (`dx_fluid`), `u` is the ALE mesh displacement,
  obtained from a harmonic (Laplace) extension PDE (`A_I`), i.e. it solves
  `Δu = 0` **on the original undeformed fluid domain**, with `u` at the
  fluid–solid interface pinned to whatever the structural displacement is
  there.
- Because both live on **one shared function space**, interface continuity
  (`u_fluid = u_solid` at the interface) and traction balance are automatic —
  there's no explicit interface-coupling term for `u`/`v` anywhere in the
  residual, which is what makes the current monolithic scheme comparatively
  simple.

As the beam's oscillation grows, the harmonic extension eventually produces
`det(I + ∇u) ≤ 0` for some fluid cell — an inverted/degenerate element — and
the Newton solve (`snes_error_if_not_converged=True`,
`ksp_error_if_not_converged=True`) raises rather than silently corrupting the
solution. There is currently **no mesh-quality signal, no retry, and no
regression test that reaches this regime** (the existing smoke test only runs
6 steps on a coarse mesh).

## 2. The key design fork, and the recommendation

Remeshing means picking a *new* reference configuration for the region whose
mesh has become invalid. The hard question is: **what happens to the shared
function space at the interface once the fluid part of the mesh is
regenerated?**

### Option A — whole-domain remesh, keep the shared-mesh architecture (recommended)

Regenerate **both** the fluid and solid regions together as one new,
conforming, single mesh (same architecture as today), built from the
*current* deformed boundary of the whole domain (channel walls are
unchanged; the cylinder/flag/interface boundary comes from `X + u`
evaluated at the current boundary DOFs). Concretely:

- Keep exactly one shared mesh and one shared `u`/`v` function space, so
  **none of the FSI coupling machinery changes** (`A_T`, `A_I`, `A_E`,
  `A_P`, the `scifem.compute_interface_data` one-sided measures, the shared
  Dirichlet BCs). This is the single biggest simplification available and is
  why it's the recommended path.
- Since the new mesh's geometry for the solid region is now the *currently
  deformed* shape rather than the true t=0 undeformed shape, `u` can no
  longer directly be "the displacement relative to the mesh" for the solid,
  or the St. Venant–Kirchhoff constitutive law (which is only valid measured
  from the true undeformed reference) would silently use the wrong strain
  after every remesh. This is fixed with a standard **updated-Lagrangian /
  multiplicative decomposition across remesh events**: introduce a
  per-remesh "accumulated deformation gradient" field `F_hat`
  (piecewise-constant per cell, e.g. `("DG", 0)` tensor space, or evaluated
  at quadrature points), initialized to the identity. At every remesh event
  `n → n+1`, update it as
  `F_hat_(n+1) = F(u_at_remesh) · F_hat_(n)` (chain rule composition of
  deformation gradients), then reset `u ← 0`, `u_old ← 0` for the *next*
  segment, and use `F_total = F(u) · F_hat` — not `F(u)` alone — inside
  `Solid.STVK` for stress evaluation. STVK is hyperelastic with no
  history/dissipation, so this composition is exact (no path-dependence
  issue): the true total deformation gradient relative to the real t=0
  configuration is just the product of the deformation gradients of each
  successive segment. `F_hat` is transferred across the remesh the same way
  as every other field (§5).
  - The fluid subdomain doesn't need this trick at all: since harmonic
    extension has no notion of accumulated history, simply resetting
    `u ← 0` there after a remesh (the new mesh's fluid region *is* the
    current physical fluid domain) is correct as-is.
- Cost: a full LU factorization (MUMPS) + a new gmsh generation every time
  this triggers. This is why the remesh trigger (§4) should be a genuine
  mesh-quality check, not a fixed short cadence — remeshing should be rare
  relative to time steps.

### Option B — independent fluid/solid meshes with explicit interface coupling (not recommended for the first version)

Give the fluid domain its own `dolfinx.mesh.Mesh` object, regenerated
independently, and keep the solid mesh permanently fixed (so it never needs
the `F_hat` bookkeeping above). This is closer to what `fsi2_harmonic_diffmesh.py`
already gestures at (`dolfinx.mesh.create_submesh` for the fluid pressure
space) — but note that `create_submesh` there is still a *view* into the
same parent mesh/topology, not a freely-regenerable independent mesh, and
critically `u`/`v` remain whole-mesh even in the `diffmesh` variants (only
pressure is split out). Making the fluid mesh genuinely independent means the
interface no longer gets continuity "for free" from a shared function space,
and traction balance (which today falls out automatically from shared test
functions) has to be re-derived and enforced explicitly — via node-matched
multi-point constraints (since we control both meshes' interface point sets
and can force them to coincide in position) or a full mortar/Nitsche
coupling. This is materially more work (a new interface formulation, not
just a remeshing routine) and is flagged here as a **later option** if
Option A's per-remesh cost turns out to dominate runtime, not as the first
implementation target.

**Recommendation: implement Option A first.** It reuses the existing
monolithic formulation almost unchanged and isolates all new work into
(1) a boundary-extraction + gmsh-regeneration routine, (2) a field-transfer
routine, (3) the small `F_hat` addition to the solid constitutive law, and
(4) a trigger/orchestration loop. Revisit Option B only if profiling shows
whole-domain remeshing is too expensive at the cadence the benchmark needs.

## 3. Existing building blocks already in the repo (don't reinvent)

- **DOF-coordinate extraction pattern**: the "measurement spot" probe used
  identically in all three monolithic FSI2 solvers
  (`fsi2_harmonic.py:288-292`) already shows the exact primitives needed:
  `U.tabulate_dof_coordinates()` + `np.isclose`/boolean masking. Combined
  with `dfx.fem.locate_dofs_topological(U, dim-1, facets)` (used pervasively
  for BCs, e.g. `fsi2_harmonic.py:172`) and
  `facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])`, this is
  already everything needed to pull ordered interface/boundary DOF
  coordinates — it just hasn't been assembled into a reusable "get current
  boundary points" helper yet.
- **`scifem.compute_interface_data`** (`fsi2_harmonic.py:62-75`) already
  gives paired fluid-side/solid-side interface entity data — reusable to
  order interface facets into a boundary walk for the gmsh spline.
- **Nonmatching interpolation is already used and working** in
  `component_solvers/navier_stokes_ALE_fsi2_mm.py:103-136` (and the `_bih`
  and `static_` siblings), transferring a precomputed structural boundary
  displacement time series onto the fluid mesh via exactly
  `create_interpolation_data(V_to, V_from, cells, padding=...)` +
  `interpolate_nonmatching`. This is a working template for the
  old-mesh → new-mesh field transfer step, not something to design from
  scratch.
- **gmsh OCC mesh construction + sizing fields + physical tagging** already
  exist in full for the *initial* FSI2 geometry in
  `src/xfsi_solver/scripts/create_mesh_FSI2.py` — remeshing needs the same
  structure (rectangle channel, circular obstacle, tagged physical groups,
  graded `gmsh.model.mesh.setSize`, `dolfinx.io.gmsh.model_to_mesh`), just
  with the flag/interface boundary built from spline points instead of
  straight lines + a circle arc.
- **Local Jacobian-based stiffening is already implemented** (just not used
  by the monolithic solver): `component_solvers/navier_stokes_ALE_fsi2_mm.py:157-158,278`
  uses `alpha = alpha_0 * cell_volume**(-2)` instead of the monolithic
  solver's constant `alpha_u = 1e-9`. Worth porting into `fsi2_harmonic.py`
  regardless of remeshing, as a Phase 0 robustness win.
- **`PHYSICAL_MARKERS` is currently duplicated verbatim** across
  `fsi2_harmonic.py`, `fsi2_harmonic_diffmesh.py`, `fsi2_biharmonic_diffmesh.py`,
  `create_mesh_FSI2.py`, and the `component_solvers` scripts. A remeshing
  routine that regenerates a mesh at runtime *must* reproduce these exact
  tag numbers — this is a good forcing function to finally factor
  `PHYSICAL_MARKERS` into one shared module (e.g.
  `xfsi_solver/fsi/markers.py`) that both the mesh generator and the solvers
  import, rather than adding yet another copy for the remeshing code path.

## 4. Remesh trigger: mesh-quality metric

Check, every time step (cheap relative to the Newton solve), the minimum
scaled cell quality over the fluid subdomain. The Jacobian determinant `J`
is already computed in `A_I`/`A_E`; expose it as a `dfx.fem.Expression`
evaluated into a `("DG", 0)` function over `dx_fluid`, then take
`comm.allreduce(local_min, op=MPI.MIN)`. Two thresholds:

- **Warn/tighten threshold** (e.g. `J_min < 0.3× median`): nothing drastic,
  but worth logging so the trigger threshold can be tuned empirically against
  real FSI2 runs.
- **Remesh threshold** (e.g. `J_min` within some safety margin of 0, or a
  standard scaled-Jacobian quality measure below ~0.1–0.2 — exact values need
  calibration against this benchmark's actual degradation curve, which we
  don't have data for yet since no long run has been logged): trigger a
  remesh **between time steps** (never mid-Newton-iteration) at the *next*
  step boundary.

Fall back to a fixed max-step-count cadence as a safety net (e.g. force a
remesh if none has happened in the last `N` steps even if quality looks
fine) purely to bound worst-case behavior if the quality metric is
mis-calibrated — but the quality-triggered path should be the one that
actually fires in practice, per the literature consensus in
`literature-review.md` §2.

## 5. Field transfer across a remesh event

Fields to carry from the old mesh to the new one: `u`, `u_old`, `v`,
`v_old`, `p`, plus the new `F_hat` tensor field from §2. All via
`create_interpolation_data` / `interpolate_nonmatching`
(`component_solvers/navier_stokes_ALE_fsi2_mm.py` pattern), with generous
`padding` since the new mesh's boundary is deliberately built to coincide
with the old mesh's *current* physical boundary and float error must not
cause "point not found" failures right at that shared boundary.

After transfer:
- Reset `u ← 0`, `u_old ← 0` everywhere (the new mesh *is* the current
  configuration, so zero displacement relative to it is correct for both
  the solid, now expressed via `F_hat`, and the fluid).
- `v`, `v_old`, `p` carry over as interpolated (physically continuous
  quantities, no reference-frame subtlety).
- Rebuild every `dfx.fem.Function`, `dfx.fem.functionspace`,
  `dfx.fem.dirichletbc`, and the `dfx.fem.petsc.NonlinearProblem` itself
  against the new mesh — the existing time loop in `fsi2_harmonic.py` treats
  all of this as one-time setup before the `while` loop, so this becomes the
  one part of that function that needs to become re-entrant / loop-able.

## 6. Boundary/mesh regeneration mechanics

1. Gather (rank 0, since `dolfinx.io.gmsh.model_to_mesh` only meshes on
   `gmsh_model_rank`) the ordered current physical coordinates of:
   channel wall corners (fixed, reuse the analytic rectangle from
   `create_mesh_FSI2.py`), the cylinder boundary (fixed, reuse the analytic
   circle), and the flag + interface boundary (**not** fixed — build from
   `tabulate_dof_coordinates()[interface/flag boundary dofs] + u[those
   dofs]`, ordered into a walk around the boundary using
   `scifem.compute_interface_data`/facet adjacency, the same way
   `create_mesh_FSI2.py` already classifies boundary curves).
2. Feed the deformed points into `gmsh.model.occ.addSpline`/`addBSpline`
   (instead of `addLine`/`addCircleArc`), reusing `create_mesh_FSI2.py`'s
   curve-loop/surface/physical-group/sizing-field logic unchanged. This
   argues for **refactoring `create_mesh_FSI2.py`'s body into a reusable
   function parameterized by "flag+interface boundary curve provider"**
   (analytic for the initial mesh, spline-from-points for a remesh) rather
   than writing a second, parallel mesh-construction routine.
3. `gmsh.model.mesh.generate(2)` + `gmsh.model.mesh.setOrder(2)` (matching
   the existing second-order default) + `dolfinx.io.gmsh.model_to_mesh(...)`
   exactly as `create_mesh_FSI2.py` already does.
4. Broadcast/distribute the resulting mesh the same way `model_to_mesh`
   already does for the initial load (it handles the rank-0-generates,
   all-ranks-receive pattern internally — confirm this still holds when
   called repeatedly mid-run, not just once at startup; this needs a small
   standalone test since it's a new usage pattern for the API).

## 7. Phased implementation plan

Each phase is independently testable and delivers value even if later
phases are deferred.

- **Phase 0 — diagnostics and cheap mitigation (no remeshing yet).**
  Add the mesh-quality metric from §4 as a logged quantity (not yet a
  trigger) to `fsi2_harmonic.py`. Add a **long-running regression test**
  (marked slow/opt-in, not part of the default fast test suite) that runs
  the FSI2 benchmark until it currently fails, to (a) actually observe and
  record the failure — right now it's attested only by the user, not
  reproduced anywhere in-repo — and (b) get real `J_min` decay data to
  calibrate the Phase-4 trigger threshold against. While here, port the
  `cell_volume**(-2)` local stiffening from `navier_stokes_ALE_fsi2_mm.py`
  into `fsi2_harmonic.py`'s `A_I` to see how much it alone pushes back the
  failure point — useful signal even independent of remeshing.
  Also do the `PHYSICAL_MARKERS` module refactor (§3) now, since every later
  phase depends on mesh generation and the solver agreeing on tag numbers.

- **Phase 1 — boundary point extraction utility.**
  A function `current_boundary_points(mesh, facet_tags, U, u) -> dict` that
  returns ordered, deformed physical coordinates for each named boundary
  piece (channel corners, cylinder, flag+interface), gathered to rank 0.
  Unit-testable in isolation (feed a known analytic `u`, check the returned
  points match the expected deformed shape) without touching gmsh or the
  solver at all.

- **Phase 2 — mesh regeneration routine.**
  Refactor `create_mesh_FSI2.py` per §6 into a function usable both for the
  initial mesh (analytic boundary) and for a remesh (spline boundary from
  Phase 1's output), sharing sizing-field/tagging logic. Test by regenerating
  the *undeformed* mesh through the new spline path and diffing basic
  properties (volumes, physical group tag sets, `PHYSICAL_MARKERS`
  membership) against the existing analytic-path mesh — this validates the
  refactor didn't silently change tagging before ever touching a deformed
  case.

- **Phase 3 — field transfer routine.**
  A function that, given an old mesh's `u, u_old, v, v_old, p, F_hat` and a
  new mesh, returns the same set of `Function`s on the new mesh via
  `create_interpolation_data`/`interpolate_nonmatching` (§5), applying the
  `u ← 0` reset. Unit test: interpolate a known smooth analytic field across
  two different meshes of the same domain and check interpolation error is
  small and roughly `O(h^2)`.

- **Phase 4 — `F_hat` / updated-Lagrangian solid constitutive change.**
  Add the accumulated-deformation-gradient field and thread `F_total =
  F(u)·F_hat` through `Solid.STVK` in `fsi/materials.py`. Validate in
  isolation: take a pure-solid benchmark (`component_solvers/solid_elasticity_fullmesh.py`
  or similar), run it once straight through, then run it again with an
  artificial mid-simulation "remesh" that regenerates an identical mesh,
  resets `u`, and updates `F_hat` — confirm the two runs agree to solver
  tolerance. This isolates correctness of the trickiest piece of the whole
  plan before it's entangled with gmsh/MPI/Newton-solve orchestration.

- **Phase 5 — orchestration in `fsi2_harmonic.py`.**
  Wire the trigger (§4) into the time loop: on trigger, call Phase 1 → 2 → 3
  → 4 in sequence between time steps, rebuild function spaces/BCs/
  `NonlinearProblem`, and continue. Handle and log the case where the first
  post-remesh Newton solve needs more iterations (interpolation introduces a
  small non-physical residual, similar in character to a restart).

- **Phase 6 — validation against the benchmark.**
  Extend `plot_qois.py`/the existing harmonic-vs-biharmonic comparison
  (`output/figures/y-disp.svg`) to include a "harmonic + remeshing" run for
  the *full* benchmark duration, and confirm the tip y-displacement curve
  tracks the FeatFlow reference values past the point where today's
  no-remeshing run currently fails.

- **Phase 7 — optional/stretch: Option B (independent meshes).**
  Only pursue if Phase 6 shows Option A's remeshing cost (gmsh generation +
  full LU refactorization) is too high at the cadence FSI2 actually needs.
  Would need its own design pass (interface coupling reformulation) — not
  expanded further here.

## 8. Open questions / risks to track during implementation

- **Trigger threshold calibration**: no real `J_min` decay data exists yet
  for this benchmark (Phase 0 produces it). Pick a threshold with
  a documented empirical basis instead of a arbitrary guess.
- **Remeshing frequency vs. cost**: a full gmsh regeneration + MUMPS
  refactorization is expensive; if the benchmark needs remeshing every few
  hundred steps this could dominate runtime. Worth profiling after Phase 5.
- **MPI behavior of repeated `model_to_mesh` calls mid-run**: the existing
  usage is "call once at startup." Confirm there's no leaked gmsh state
  across repeated init/generate/finalize cycles in one process
  (`create_test_meshes.py` already notes `gmsh.clear()` is needed between
  repeated in-process `create_mesh` calls — same caution applies here, more
  so since this now happens inside a long-running MPI job rather than a
  one-off script).
- **Second-order (curved) element boundary reconstruction**: confirmed
  workable in principle (P2 DOFs already include edge midpoints — see
  `literature-review.md` §3) but should be spot-checked visually (e.g. plot
  the spline-reconstructed boundary against the true FE boundary for a
  strongly bent flag) before trusting it inside the full pipeline.
- **Interpolation accuracy at the exact new interface boundary**: this is
  where `padding` in `create_interpolation_data` matters most (§5/§6) —
  points sitting exactly on a shared boundary are the easiest to lose to
  floating-point search failures; needs explicit test coverage in Phase 3,
  not just an average-case error check.
- **Quality of `F_hat` bookkeeping under many successive remeshes**: verify
  numerically that composing `F_hat` across several remesh events doesn't
  accumulate meaningful error (St. Venant–Kirchhoff is exact under
  composition analytically, but each remesh interpolation step introduces a
  small perturbation before the next composition — worth a multi-remesh
  version of the Phase 4 isolation test).
