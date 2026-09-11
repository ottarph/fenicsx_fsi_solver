# Plan: remeshing for the FSI2 benchmark (harmonic mesh motion)

Status: **Phases 1-4 (§7) are implemented**, in
`src/xfsi_solver/remeshing/` with tests in `tests/test_remeshing_*.py` —
a working, tested, standalone remeshing loop. Phase 5
(swapping in a real solve) and the deferred monolithic-FSI
-integration work are not started. See [`literature-review.md`](literature-review.md)
for the background that motivates the choices below, and §9 for a summary
of what was actually built, including two real design corrections made
during implementation that are worth reading before extending this code.

**Scope update (supersedes §2 below).** The prototype originally worked on
the fluid domain *in isolation*, extracted as a submesh; §2 records why that
narrowing was made and is kept for that rationale. It no longer describes
what the code does. Remeshing now regenerates the **full** FSI2 mesh — both
subdomains, all marked curves, with the fluid–solid interface coming back as
a conforming internal boundary — per a later request from the user. See
`implementation-log.md` §10 for what changed and what it cost. Interface
motion is still **prescribed** rather than solved, and the monolithic-FSI
integration question in §2 is still open; the full mesh removes a structural
obstacle to it but does not answer it.

Revision note: this version narrows the plan per review. The goal at the
time was **only** a standalone fluid-domain remeshing capability: a
fluid-only mesh whose fluid–solid interface boundary is deformed by a
**prescribed** function (not derived from solving the coupled problem),
remeshed as needed, with fields interpolated between the old and new mesh.
Everything about how this eventually plugs into the monolithic FSI solver —
where `U`/`V` are still whole-mesh, spanning both solid and fluid, even in
the `diffmesh` variants — is **explicitly out of scope for now** and not
designed here; see §2. (Earlier revisions of this document sketched a
whole-domain remesh with `F_hat` bookkeeping, then an interface-coupling
reformulation with candidate mechanisms — both are superseded; this revision
deliberately does not attempt to solve the interface-coupling question at
all.)

**Local dev data**: the untracked `data/` folder (gitignored, not part of
version control) has been copied into this worktree from the main checkout
so the meshes below are available for prototyping:
`data/meshes/fsi2/mesh.xdmf` (linear/P1 triangles),
`data/meshes/fsi2/mesh_sec.xdmf` (second-order/P2 triangles, curved edges),
plus the quad variants (deferred, see §7), `data/fsi2_boundary/*.npy` (the
precomputed boundary-displacement series the legacy `component_solvers` use),
and `data/fsi2_reference.txt` (FeatFlow reference values).

## 1. The problem, precisely

`src/xfsi_solver/solvers/fsi2_harmonic.py` (and its `_diffmesh` sibling)
solve the FSI2 benchmark as a **fixed-reference, pseudo-solid ALE**
problem: `mesh.geometry.x` is loaded once and **never changes** for the rest
of the run. Everything that "moves" is carried by the displacement unknown
`u`, defined on a `("CG", 2, (2,))` function space spanning *both* the solid
and fluid cells of one shared mesh:

- In the solid subdomain (`dx_solid`), `u` is the true structural
  displacement (St. Venant–Kirchhoff, `A_E`/`A_T`), a Lagrangian quantity
  always measured from the true, undeformed, t=0 geometry.
- In the fluid subdomain (`dx_fluid`), `u` is the ALE mesh displacement,
  obtained from a harmonic (Laplace) extension PDE (`A_I`), i.e. it solves
  `Δu = 0` **on the original undeformed fluid domain**, with `u` at the
  fluid–solid interface pinned to whatever the structural displacement is
  there.
- Because both live on **one shared function space**, interface continuity
  (`u_fluid = u_solid` at the interface) and traction balance are automatic —
  there's no explicit interface-coupling term for `u`/`v` anywhere in the
  residual.

As the beam's oscillation grows, the harmonic extension eventually produces
`det(I + ∇u) ≤ 0` for some fluid cell — an inverted/degenerate element — and
the Newton solve (`snes_error_if_not_converged=True`,
`ksp_error_if_not_converged=True`) raises rather than silently corrupting the
solution. There is currently **no mesh-quality signal, no retry, and no
regression test that reaches this regime**. This is the eventual motivation
for remeshing — but see §2 for what's actually in scope right now, which
stops well short of touching this solver.

## 2. Original scope: standalone fluid-domain remeshing, interface motion prescribed

**Superseded in part** — see the scope update at the top of this document.
The fluid-only restriction below has been lifted (remeshing now covers the
full mesh); the prescribed-interface-motion restriction and the deferral of
monolithic-FSI integration both still stand, and the reasoning for the
latter is the part of this section still worth reading.

Per review, the plan is narrowed to build and validate a **self-contained
fluid-domain remeshing capability**, decoupled from the FSI solver entirely:

- Take just the fluid region of the FSI2 mesh (extracted the same way
  `fsi2_harmonic_diffmesh.py` already does via
  `dolfinx.mesh.create_submesh(mesh, tdim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"]))`
  — reusing that extraction is convenient, but nothing else about that
  solver is touched in this scope).
- The fluid–solid interface — one of this fluid domain's boundaries — is
  deformed by a **prescribed function**: a function *we choose*, standing in
  for "whatever the structural displacement would have been," not the
  output of solving anything. This is precisely the role that
  `data/fsi2_boundary/*.npy` (a precomputed structural-boundary displacement
  time series) already plays in the existing, unrelated
  `component_solvers/navier_stokes_ALE_fsi2_mm.py` — see §3 — though for the
  earliest prototyping steps a simple closed-form analytic function is
  enough and avoids even needing that data file.
- Remesh this fluid domain as the prescribed interface motion grows, and
  interpolate fields between the old and new mesh.

**Explicitly out of scope for now, and not designed in this document**: how
any of this plugs back into the monolithic FSI solver. `U` and `V`
(displacement/velocity) are whole-mesh even in the `diffmesh` variants —
only pressure is split onto `fluid_mesh` today
(`fsi2_harmonic_diffmesh.py:122-124`) — so wiring a remeshable, independent
fluid domain into the real coupled solver will eventually require deciding
how `U`/`V` relate to a fluid mesh that can change out from under them, and
how interface continuity/traction balance (today automatic via shared DOFs)
gets re-established. That question needs real thought before it's designed,
not a placeholder list of mechanisms — it is intentionally left for later,
once the fluid-only remeshing mechanics below are actually validated and
there's a working sense of what a remesh event costs and requires in
practice.

## 3. Existing building blocks already in the repo (don't reinvent)

- **`dolfinx.mesh.create_submesh`** already extracts exactly the fluid-only
  region this plan needs (`fsi2_harmonic_diffmesh.py:52-57`,
  `cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"])`) — the convenient starting
  point for getting a standalone fluid mesh with a tagged interface boundary
  without writing new extraction code, independent of anything else that
  solver does.
- **A working precedent for "fluid domain + prescribed interface motion"
  already exists**: `component_solvers/navier_stokes_ALE_fsi2_mm.py` (and
  its `_bih`/`static_` siblings) already solve a standalone fluid-only ALE
  problem where the interface displacement is *not* the output of a coupled
  structural solve but a precomputed time series
  (`data/fsi2_boundary/msh_x.npy`/`msh_conn.npy`/`uh.npy`), brought in via
  exactly the nonmatching-interpolation machinery this plan needs
  (`create_interpolation_data(V_to, V_from, cells, padding=...)` +
  `interpolate_nonmatching`, `navier_stokes_ALE_fsi2_mm.py:103-136`). This is
  the closest existing analogue to "interface deformed by a prescribed
  function" in this codebase, and a natural reference (and later, reuse
  target — see §7 Phase 5) once the prototype needs something more realistic
  than a closed-form analytic function.
- **DOF-coordinate extraction pattern**: the "measurement spot" probe used
  identically in all monolithic FSI2 solvers (e.g.
  `fsi2_harmonic_diffmesh.py:297-301`) shows the exact primitives needed:
  `U.tabulate_dof_coordinates()` + `np.isclose`/boolean masking, combined
  with `dfx.fem.locate_dofs_topological`.
- **`scifem.compute_interface_data`** (`fsi2_harmonic_diffmesh.py:76-86`)
  already gives paired fluid-side/solid-side interface entity data — useful
  for identifying/tagging the interface boundary of the extracted fluid
  mesh.
- **Nonmatching interpolation** — see the `navier_stokes_ALE_fsi2_mm.py`
  bullet above; the same `create_interpolation_data`/`interpolate_nonmatching`
  pair is the field-transfer mechanism for old-mesh → new-mesh (§5).
- **gmsh OCC mesh construction + sizing fields + physical tagging** already
  exist in full for the *initial* FSI2 geometry in
  `src/xfsi_solver/scripts/create_mesh_FSI2.py` (analytic primitives,
  `PHYSICAL_MARKERS`-driven curve classification at
  `create_mesh_FSI2.py:146-175`, graded `gmsh.model.mesh.setSize`,
  `dolfinx.io.gmsh.model_to_mesh`) — the sizing/tagging logic is directly
  reusable for the fluid-only remesh (§6); the on-disk `mesh.xdmf`/
  `mesh_sec.xdmf` in the copied `data/` folder are this script's own output
  with `quads=False`, so they can be loaded directly rather than
  regenerated for early prototyping.
- **DOLFINx mesh → gmsh discrete-entity round trip** is not yet in this repo
  but is the practitioner-recommended mechanism for capturing deformed
  fluid-mesh geometry (per expert input, `literature-review.md` §5) —
  nothing to reuse from this codebase directly, but Dokken's published
  script is a working reference to adapt (2D triangles instead of 3D tets;
  see §8 for the node-ordering caveat quads/second order will eventually
  introduce).
- **Mesh-quality metric: reuse `pvmeshquality`, don't write a new one.**
  Per review, use [`ottarph/pvmeshquality`](https://github.com/ottarph/pvmeshquality)
  (`pvmeshquality.py`, `MeshQuality` class) rather than the
  `dfx.fem.Expression`-based Jacobian metric originally sketched here. It
  wraps a DOLFINx mesh as a `pyvista.UnstructuredGrid`
  (`dfx.plot.vtk_mesh`) and calls PyVista/VTK's `cell_quality` (the Verdict
  library) — supports `"scaled_jacobian"` (the default) and any other
  Verdict measure PyVista exposes, per cell, with the mesh optionally warped
  by a displacement `Function`/array first (`grid.warp_by_vector`). `pyvista`
  is already a conda dependency (`environment.yml`). Two things to account
  for when adopting it:
  - **It only accepts `("CG", 1, (d,))` vector spaces for the warping
    displacement** (`assert fspace.ufl_element().degree == 1` in
    `pvmeshquality.py:31`) — our `U`/`V` are `("CG", 2, (2,))`, so the
    trigger check needs a CG1 copy of the relevant displacement (e.g.
    `u_cg1.interpolate(u)`) before calling it, not `u` directly.
  - **Not yet updated for DOLFINx 0.11** (tested against 0.10.0 per its
    README) — the repo's `environment.yml` pins `fenics-dolfinx >= 0.11`, so
    before relying on it, vendor/adapt it and smoke-test the API surface it
    touches (`dfx.plot.vtk_mesh`, `dfx.mesh.basix.CellType`,
    `mesh.basix_cell()`, `fspace.ufl_element().degree`,
    `V.dofmap.list`) against 0.11 — plausibly a small/no-op change per the
    reviewer's own expectation, but unverified. Only supports the cell types
    it lists (`triangle`, `quadrilateral`, `tetrahedron`, ... — no direct
    support for curved/second-order cells; quality is computed from the
    underlying straight-sided/corner-node geometry, consistent with using
    it on `mesh.xdmf` first per §7).
- **`PHYSICAL_MARKERS` is currently duplicated verbatim** across every
  FSI2-related file. A remeshing routine that regenerates a mesh at runtime
  *must* reproduce these exact tag numbers — good forcing function to factor
  it into one shared module (e.g. `xfsi_solver/fsi/markers.py`).
- **Local Jacobian-based stiffening is already implemented** (just not used
  by the monolithic solvers): `component_solvers/navier_stokes_ALE_fsi2_mm.py:157-158,278`
  uses `alpha = alpha_0 * cell_volume**(-2)` instead of the diffmesh/plain
  solvers' constant `alpha_u = 1e-9`. Noted here for completeness — porting
  it into a monolithic solver's `A_I` is a change to that solver, which is
  out of scope per §2; revisit if/when the deferred monolithic-integration
  work (§7) is taken up.

## 4. Remesh trigger: mesh-quality metric

Use `pvmeshquality.MeshQuality` (§3) on the fluid mesh, warped by a CG1 copy
of the current fluid deformation field, checked every "step" of whatever
loop is driving the deformation (in Phase 1–4's prototype, one check per
increment of the prescribed interface function; in a real solve, cheap
relative to the Newton solve it would sit alongside). Two thresholds:

- **Warn/tighten threshold**: log-only, to gather real degradation data
  against which to tune the next threshold.
- **Remesh threshold**: trigger a remesh **between steps** (never
  mid-solve, once solves exist) once `scaled_jacobian` (or whichever measure
  Phase 1 settles on) drops below a value calibrated from that logged data —
  not an arbitrary guess.

Fall back to a fixed max-step-count cadence purely as a safety net in case
the quality metric is mis-calibrated, matching `literature-review.md` §2's
recommended quality-triggered-with-cadence-fallback pattern.

## 5. Field transfer across a remesh event

Because the current scope is a standalone fluid domain (§2) with no coupled
solid solve feeding it, there is no interface-coupling question to resolve
here at all: the interface motion for the *next* segment is simply whatever
the prescribed function says next, independent of anything transferred at
the remesh. What does need transferring is whatever fields the fluid
problem is carrying at the time: in the earliest phases (§7 Phase 1–4) that
is just the interpolated known/prescribed deformation field itself (no
velocity/pressure exist yet, since there's no solve); once a real
fluid-only solve is introduced (§7 Phase 5, e.g. extending
`navier_stokes_ALE_fsi2_mm.py`), it extends to `v`, `p`, and the ALE
displacement `u`. In every case the mechanism is the same:
`create_interpolation_data`/`interpolate_nonmatching`
(`navier_stokes_ALE_fsi2_mm.py` pattern), with generous `padding` since the
new fluid mesh's boundary is deliberately built to coincide with the old
mesh's *current* physical boundary. After transfer, reset the ALE
displacement `u ← 0` (`u_old ← 0` too, once history exists) since the new
mesh *is* the current configuration.

## 6. Mesh regeneration mechanics (as implemented — see §9)

**Mechanism, current version: DOLFINx discrete-mesh round trip through
gmsh, with boundary curves built directly from the mesh's own facet tags**
(not hand-built splines, and — as of this revision — not recovered by
asking gmsh to guess the boundary either; see below). Implemented in
`src/xfsi_solver/remeshing/discrete_mesh.py::regenerate_mesh`; the
module's own docstring is the authoritative, up-to-date reference for the
mechanics and the empirical findings behind each choice below.

The steps below are written for the fluid-only version this section was
first drafted against. The mechanism is unchanged for the full mesh, with
two additions: step 4 builds *one surface per `cell_tags` subdomain*, each
declaring only the curves that bound it (derived from facet-to-cell
adjacency, so the interface curve is shared by both), and step 5's sizing
field now has to state its gradient explicitly rather than inherit it from
CAD points. See `implementation-log.md` §10.

1. Gather the *deformed* geometry of the mesh: `X_deformed =
   mesh.geometry.x + u` for every geometry node, together with the
   existing cell connectivity (only the 3 corner nodes per cell/facet are
   used, regardless of mesh order — see the module docstring for why).
2. Feed the 2D cells into gmsh as a **discrete entity**
   (`gmsh.model.addDiscreteEntity` + `addNodes` + `addElementsByType`,
   following Dokken's pattern from `literature-review.md` §5).
3. **Build the 1D boundary curves directly from `facet_tags`**, per a
   suggestion from Jørgen Dokken (a DOLFINx core developer) given directly
   during this work: for each `PHYSICAL_MARKERS` value present, use
   `dolfinx.mesh.entities_to_geometry(mesh, 1, facets)` to map each tagged
   facet to its geometry node indices, split into connected components
   (a physical group can be geometrically disconnected — e.g.
   "channel_side" is the top *and* bottom walls under one tag), and build
   one gmsh discrete curve entity per component, with explicit discrete
   point entities (shared across curves that meet there) at its two
   endpoints. Tag each curve directly with the *known* marker value —
   no classification/guessing needed, since the DOLFINx meshtag already
   said which physical group each facet belongs to.
   - This **replaces** an earlier version of this step that instead ran
     `gmsh.model.mesh.classifySurfaces(angle)` to auto-detect the boundary
     from the discrete surface's own geometry, then classified the
     resulting curves by bounding box against the known FSI2 geometry
     constants. That version worked (see §9's now-historical
     `classifySurfaces`-angle-threshold finding) but was FSI2-specific
     (hardcoded geometry constants), fragile to the angle-threshold choice,
     and — found only once superseded — considerably more sensitive to the
     *source* mesh's own triangulation quality than the meshtag-based
     version turned out to be. Kept only as a documented fallback if a
     future geometry's facet tags aren't fine-grained enough to build
     curves from directly.
4. Build the 2D surface as a discrete entity too, this time declaring
   `boundary=<every curve tag from step 3>` (required — without it,
   `generate` silently produces an empty mesh rather than raising), then
   `gmsh.model.mesh.createGeometry(...)` to reparametrize curves and
   surfaces into genuine, remeshable CAD entities — every surface always,
   but every curve *except* `PINNED_BOUNDARIES` (currently just
   `solid_fluid_interface`): skipping a curve here carries its nodes
   forward exactly, unchanged, rather than letting step 6 reseed them from
   the sizing field. See `implementation-log.md` §11 for why the surfaces
   have to be included explicitly too, and the wrong turn taken before
   finding that out.
5. Re-apply the graded sizing-field logic, using the curve-tag groups
   already known exactly from step 3 (no re-derivation needed).
6. Discard the old triangulation, `gmsh.model.mesh.generate(2)` +
   `gmsh.model.mesh.setOrder(...)` fresh, then
   `dolfinx.io.gmsh.model_to_mesh(...)`.
7. Rebuild the mesh and everything defined on it (function spaces,
   whatever fields the current prototype stage is carrying — see §5) from
   this new mesh. The solid side comes back with it, tagged as before and
   conforming across the interface, but what a real coupled solve would
   have to carry across that rebuild is still the open question of §2.

The hard limitation carries over unchanged from the earlier version: this
only works while the deformed boundary is still a *simple*
(non-self-intersecting) curve. Feeding it an already-inverted mesh can
still hang rather than fail cleanly (§8, and see `loop.py`'s independent
inversion guard, §9).

*Fallback if facet tags aren't available/fine-grained enough for a future
geometry*: hand-built `gmsh.model.occ.addSpline`/`addBSpline` boundary
curves from ordered, boundary-only DOF coordinates, or fall back further to
`classifySurfaces`-based automatic recovery (step 3 above, superseded but
still a working reference implementation if needed).

## 7. Phased implementation plan

All phases below stay within the narrowed scope of §2: a standalone fluid
-only mesh, an interface boundary deformed by a function we prescribe, no
FSI solver involved. The earliest phases are additionally **solver-free**
altogether (no `NonlinearProblem`/`LinearProblem` at all — deformation comes
from interpolating a known/prescribed function, not from solving anything).
Work on **triangles only** first (`data/meshes/fsi2/mesh.xdmf`, linear; then
`data/meshes/fsi2/mesh_sec.xdmf`, second-order/curved); quads are explicitly
deferred. Nothing here touches `fsi2_harmonic_diffmesh.py` or any other FSI
solver file.

- **Phase 1 — deform-by-prescribed-function + quality-metric validation
  (no FE solve, no gmsh yet).**
  Load `data/meshes/fsi2/mesh.xdmf`, extract the fluid submesh via
  `dolfinx.mesh.create_submesh` (§2/§3), define a prescribed interface
  displacement function and a closed-form extension into the fluid interior
  (§6), interpolate it into a CG1 vector `Function`, and run it through
  `pvmeshquality.MeshQuality` — confirm quality drops as the prescribed
  deformation is scaled up, and that a deliberately degenerate case is
  correctly flagged. Repeat on `data/meshes/fsi2/mesh_sec.xdmf` (P2) to
  confirm the CG1-only restriction (§3) is handled correctly
  (project/interpolate down, don't pass the P2 function directly). First
  proof that the trigger metric (§4) behaves sensibly.

- **Phase 2a — validate the discrete-mesh round trip in isolation, on
  undeformed data first (no FE solve).**
  Round-trip the *undeformed* fluid submesh of `mesh.xdmf` through
  `addDiscreteEntity`/`addNodes`/`addElementsByType`/`addPhysicalGroup`
  (Dokken's pattern) → `classifySurfaces` → `createGeometry` → re-tag →
  `generate(2)`, and diff the result against the original (volumes,
  physical group tag sets, boundary curve count/composition). Then repeat
  with the mesh **deformed by the same prescribed function from Phase 1**
  (still no FE solve) and check the regenerated mesh is valid
  (`pvmeshquality` quality > 0 everywhere) and visually matches the intended
  deformed shape. Once this works on `mesh.xdmf`, repeat on `mesh_sec.xdmf`.
  This isolates and retires the single biggest technical uncertainty in the
  whole plan (does `classifySurfaces`/`createGeometry` behave well for this
  2D fluid domain at all). If it doesn't work cleanly, fall back to the
  spline-based approach (§6).

- **Phase 2 — mesh regeneration routine.**
  Refactor `create_mesh_FSI2.py`'s tagging/sizing logic into a function
  usable both for the initial mesh and for a remesh (discrete-mesh round
  trip from Phase 2a, fed with deformed coordinates), scoped to the fluid
  region only.

- **Phase 3 — field-transfer validation with known functions (no FE
  solve).**
  Interpolate a **known analytic** function (smooth, unrelated to the
  interface-deformation function — just something to check transfer
  accuracy) onto the old fluid mesh, transfer it to a Phase-2-regenerated
  new fluid mesh via `create_interpolation_data`/`interpolate_nonmatching`,
  and compare against interpolating the same analytic function directly
  onto the new mesh — check the error is small and roughly `O(h^2)`.

- **Phase 4 — standalone fluid-domain remeshing loop.**
  Combine Phases 1–3 into a working, still solver-free prototype: step the
  prescribed interface function through a sequence of increasingly large
  deformations (standing in for "time"), check quality every step (§4),
  remesh via §6 when triggered, and carry the interpolated field(s) forward
  across each remesh (§5) via Phase 3's transfer routine. This is the
  concrete deliverable review asked for: prescribed interface motion →
  repeated remeshing → fields carried forward, self-contained and outside
  the FSI solver.

- **Phase 5 (next increment, beyond the current ask) — real fluid-only
  physics.**
  Replace the prescribed/synthetic fields carried in Phase 4 with an actual
  standalone fluid ALE solve on the same fluid-only domain — natural to
  build by extending `component_solvers/navier_stokes_ALE_fsi2_mm.py` (§3),
  which already solves Navier-Stokes ALE with a prescribed/precomputed
  interface displacement and no solid coupling — so the remeshing loop
  carries real velocity/pressure/ALE-displacement fields, still with zero
  solid coupling. Only worth starting once Phase 4 is solid.

- **Deferred, not planned here — monolithic FSI integration.**
  How any of this connects back to the coupled solver (`U`/`V` splitting
  off the parent mesh, interface continuity/traction balance without shared
  DOFs — §2) is intentionally left for later thought, informed by what
  Phases 1–5 actually show about remeshing cost and mechanics in practice.
  Also deferred: quads/second-order production mesh support (element-type
  /node-ordering work, §8) and the biharmonic (`fsi2_biharmonic_diffmesh.py`)
  variant.

## 8. Open questions / risks to track during implementation

- **Monolithic FSI integration is deliberately out of scope and unsolved**
  (§2) — splitting `U`/`V` off the parent mesh the way `P` already is split,
  and re-establishing interface continuity/traction balance without the
  "free" shared-function-space coupling, is a real reformulation. Not a risk
  to the current phases (which don't depend on it), but the reason this plan
  does not yet claim to solve remeshing for the actual FSI2 benchmark —
  only for a standalone fluid domain with a prescribed interface motion.
- **`pvmeshquality`'s CG1-only restriction and DOLFINx 0.11 compatibility**
  (§3) — needs an explicit adaptation/smoke-test pass (start of Phase 1)
  before it can be trusted as the trigger signal.
- **Trigger threshold calibration**: no real quality-decay data exists yet
  for this fluid domain (Phase 1/4 produce it).
- **Remeshing frequency vs. cost**: a full gmsh regeneration is not free;
  worth timing once Phase 4's loop exists, even before any real solve is
  involved (Phase 5).
- **MPI behavior of repeated `model_to_mesh` calls mid-run**: existing usage
  is "call once at startup" — `create_test_meshes.py` already notes
  `gmsh.clear()` is needed between repeated in-process `create_mesh` calls;
  same caution applies here, more so inside a long-running MPI job.
- **Second-order (curved) element boundary reconstruction**: P2 DOFs already
  include edge midpoints (`literature-review.md` §3), should work in
  principle for `mesh_sec.xdmf`, but should be spot-checked visually before
  trusting it (Phase 2a covers this mesh explicitly).
- **gmsh element-type codes and DOLFINx↔gmsh node ordering for
  higher-order/quad cells**: deliberately deferred (§7, "quads/second-order
  production mesh support") — Dokken's discrete-entity example is for linear
  tetrahedra; DOLFINx/basix and gmsh use different local node orderings for
  higher-order and quad cells, a well-known footgun (silently wrong/inverted
  elements, not a hard error) when hand-rolling this kind of conversion. Do
  not attempt quads until Phases 1–5 are solid on triangles.
- **`classifySurfaces` angle-threshold tuning**: the FSI2 fluid domain mixes
  sharp corners (flag trailing edge, channel corners) with a smooth
  circular-arc obstacle boundary and the interface with the solid — check
  the threshold and resulting curve count/composition explicitly in
  Phase 2a rather than assuming a default works.
- **Interpolation accuracy at the exact new interface boundary**: where
  `padding` in `create_interpolation_data` matters most (§5/§6) — needs
  explicit test coverage in Phase 3, not just an average-case error check.

## 9. Implementation status (Phases 1-4 done)

Code lives in `src/xfsi_solver/remeshing/`, tests in
`tests/test_remeshing_*.py` (29 tests, ~6s total). What's there and how it
maps to the phases above:

- `fsi2_geometry.py`, `markers.py` — the FSI2 geometry constants and
  `PHYSICAL_MARKERS`, duplicated from (kept consistent with)
  `create_mesh_FSI2.py` (§3's flagged refactor still not done repo-wide).
- `domain.py` — `FsiDomain` (mesh + cell tags + facet tags) and the loader
  that reads all three from an XDMF file. Replaced `fluid_domain.py`, whose
  `create_submesh` + `transfer_meshtags_to_submesh` extraction existed only
  to produce the fluid-only domain (see the scope update at the top).
- `deformation.py` — Phase 1's prescribed deformation, **plus**
  `incremental_interface_deformation`, added during Phase 4 (see below) for
  chaining multiple deformation+remesh segments.
- `quality.py` — the vendored `pvmeshquality` (§3), used as-is.
- `discrete_mesh.py` — Phase 2/2a's `regenerate_mesh`: the discrete
  -mesh round trip through gmsh, with boundary curves built directly from
  `facet_tags` via `entities_to_geometry` (see the finding below — this
  replaced an earlier `classifySurfaces`-based version), one surface per
  `cell_tags` subdomain, a `SizingField` whose defaults reproduce the
  original FSI2 mesh's own resolution, and `PINNED_BOUNDARIES` (currently
  `solid_fluid_interface`), which carries a curve's nodes forward exactly
  rather than letting `SizingField` reseed them (`implementation-log.md`
  §11).
- `transfer.py` — Phase 3's `transfer_field`, with `DEFAULT_PADDING`
  calibrated to `1e-2` (see finding below).
- `dof_geometry.py` — a utility that turned out to be necessary and is not
  mentioned in the phase plan above: converting between a mesh's geometry
  -node order and a CG1 function space's dof order (see finding below).
- `loop.py` — Phase 4's `run_prescribed_deformation_loop`, combining all of
  the above; validated over 150 steps / 9 remesh events in ~4.5s, reaching
  roughly 10x the total bending amplitude a single un-remeshed segment can
  sustain before cells invert.

Empirical findings from actually building this, beyond what was
anticipated in §8 above:

- **(Historical, superseded below) `classifySurfaces`' angle threshold is
  not a minor tuning knob.** The first working version of `discrete_mesh.py`
  recovered boundary curves by asking gmsh's `classifySurfaces` to
  auto-detect them from the discrete surface's own geometry. 30° (a
  plausible-looking default) made it split a smoothly bent boundary into
  hundreds of spurious curves from ordinary mesh-resolution noise, and the
  pipeline could *hang* for minutes rather than erroring; 60° cleanly
  recovered the true 8-curve FSI2 boundary topology (4 straight channel
  walls, the obstacle arc, 3 flag edges) in both the undeformed and
  moderately-deformed case, in well under a second. This finding is kept
  here for the record — it directly informed the design decision below —
  but no longer describes the current implementation.
- **Building boundary curves directly from the mesh's own `facet_tags`,
  instead of asking gmsh to guess and then reclassifying by bounding box,
  is both simpler and more robust** — a suggestion from Jørgen Dokken (a
  DOLFINx core developer) given directly during this work, implemented via
  `dolfinx.mesh.entities_to_geometry` (§6 has the mechanics). Tags are now
  exact by construction rather than inferred, which also removed the
  FSI2-specific hardcoded-geometry-constants bounding-box heuristic
  entirely (more general, not just simpler). It turned out to bring a
  second, unanticipated benefit: because curve recovery no longer depends
  on the discrete surface's own triangulation quality (only on the facet
  tags, which are correct regardless of how distorted the interior cells
  have become), it tolerates a considerably worse-quality *source* mesh
  than the `classifySurfaces`-based version did before hitting the same
  "can hang past actual inversion" limitation (§8) — a geometry that made
  the old version struggle regenerates cleanly with this one. The
  fundamental limitation (needs a still-simple, non-self-intersecting
  boundary) is unchanged and still guarded against by `loop.py`'s
  independent inversion check, not by this mechanism itself.
- **A discrete curve needs explicit point entities at its endpoints, and
  the discrete surface needs an explicit `boundary=` reference to its
  curves** — neither is optional once you're building the topology by hand
  instead of letting `classifySurfaces` infer it. Omitting the former makes
  `createGeometry` fail outright ("has no begin or end point"); omitting
  the latter makes `generate` silently produce an *empty* mesh rather than
  raising ("only 0 nodes on the boundary") — worth calling out since that
  failure mode gives no indication of what's actually wrong.
- **`regenerate_mesh` can hang, not fail cleanly, on an
  already-inverted (self-intersecting) boundary.** This isn't a corner case
  to special-case around after the fact -- it's *why* the remesh trigger
  has to fire on early degradation. `loop.py` now runs a cheap, independent
  inversion check (signed corner-triangle area) *before* ever calling
  `regenerate_mesh`, raising `RuntimeError` instead. This also means
  `quality_threshold` needs a real margin, not just ">0": quality vs.
  amplitude was found to fall off a cliff over a handful of steps once it
  starts degrading, so a too-low threshold plus a not-small-enough step can
  let the *next* check land past actual inversion.
- **Mesh-geometry-node order and CG1 dof order are genuinely different**,
  confirmed by direct comparison on the FSI2 fluid mesh (same set of
  points, different order) -- not just a theoretical possibility. An early
  version of the Phase 1 code added a CG1 `Function`'s raw `.x.array`
  directly to `mesh.geometry.x`, silently scrambling the mesh (hundreds of
  spurious "corners" downstream in `classifySurfaces`, traced back to this).
  `dof_geometry.py`'s permutation utility (derived from matching per-cell
  local vertex order between the geometry dofmap and a same-degree CG1
  space's dofmap) exists specifically to stop this class of bug from
  recurring, and `discrete_mesh.py`/`loop.py` now consistently work in
  geometry-node order (talking to gmsh) or CG1 dof order (talking to
  pvmeshquality), never mixing the two without going through it.
- **`transfer.DEFAULT_PADDING` needed recalibrating from the value picked
  in the previous plan revision** (`1e-3`): with a coarser sizing
  (`size_far=0.06`), that value still left points near the far-field
  boundary silently un-interpolated (stuck at 0 -- a large, easy-to-miss
  error, not a small numerical one). `1e-2` resolved it. Confirms §8's
  general point but the specific number needed updating once tested against
  a real (not just fine) sizing configuration.
- **A field cannot be transferred across a remesh to recover a mesh node's
  true material reference position, no matter how it's implemented** -- a
  genuine mathematical dead end discovered while building Phase 4's loop,
  not a bug in the transfer machinery itself (see `loop.py`'s module
  docstring for the full explanation: transferring a mesh's own identity
  geometry via nonmatching interpolation always returns the query point
  itself, carrying zero information about where it "really" started out).
  This forced a design change: `loop.py` drives the deformation
  incrementally, decaying from the mesh's *current* interface position
  (needs no cross-mesh tracking) rather than a fixed original position
  (which would need it). This is very likely relevant to the deferred
  monolithic-integration work too (§2/§7): whatever eventually tracks
  accumulated structural state across a remesh event will need either a
  genuinely solved extension field (not a transferred identity-like one)
  or a reformulation that, like this loop's fix, avoids needing "true
  reference position" at all.
