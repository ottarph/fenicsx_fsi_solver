# Implementation log: fluid-domain remeshing prototype

This is a retrospective, roughly chronological account of what was actually
done on this remeshing effort and why, for whoever picks this up next
(including a future session with no memory of this one). It complements but
doesn't replace the other two documents in this directory:

- [`literature-review.md`](literature-review.md) — background research on
  how other ALE-FSI solvers handle remeshing, done before any design work.
- [`implementation-plan.md`](implementation-plan.md) — the forward-looking
  plan/spec, revised several times as scope narrowed; its §9 has a terser
  "what's implemented" summary. This log is the fuller story of *how* it
  got there, including the dead ends.

All commits referenced below are on branch `claude/fsi-remeshing-strategy-365516`.

## 1. Why this started, and how the plan narrowed

The task began as an open-ended one: the FSI2 benchmark solver
(`src/xfsi_solver/solvers/fsi2_harmonic.py` and its `_diffmesh` sibling)
fails as the flag's oscillation grows, because its harmonic mesh-motion
extension eventually produces inverted fluid elements. Nothing about
remeshing existed in the repo. The first phase of work was pure research and
planning, not code:

- Surveyed general remeshing strategies used in ALE-FSI solvers
  (`literature-review.md`), and the specific architecture of this repo's
  monolithic solver (a fixed-reference "pseudo-solid" ALE scheme, where
  `mesh.geometry.x` never moves and everything is tracked through a
  displacement unknown `u`).
- Drafted a first version of `implementation-plan.md` scoped to
  whole-domain remeshing (both fluid and solid regenerated together),
  carrying an "`F_hat`" accumulated-deformation-gradient trick so the
  solid's hyperelastic constitutive law would stay correct across a
  remesh event that changed its reference configuration.
- A FEniCSx expert pointed at a GitHub issue
  (`scientificcomputing/fenics-in-the-wild#6`, Jørgen Dokken) showing how to
  round-trip a DOLFINx mesh through gmsh as *discrete entities* rather than
  rebuilding CAD geometry from scratch. This reshaped the plan's core
  meshing mechanism (commit `4032a63`): feed the current mesh into gmsh
  directly, then let gmsh's own `classifySurfaces`/`createGeometry` recover
  proper, remeshable curves from it — replacing an originally-sketched
  "hand-build boundary splines from ordered points" approach, which was
  kept only as a documented fallback.
- Two rounds of review narrowed the scope further, each dropping real
  complexity rather than adding it:
  - **Commit `928307f`**: scope to the fluid domain only — never remesh the
    solid. This killed the entire `F_hat` mechanism (unneeded once the
    solid's reference configuration never changes) but surfaced a bigger
    problem: the monolithic solver's displacement/velocity fields (`U`/`V`)
    are whole-mesh even in the `diffmesh` variant (only pressure is split
    onto its own submesh today), so a remeshable fluid domain would
    eventually need `U`/`V` split the same way, with interface continuity
    re-established some other way than "shared DOFs."
  - **Commit `b5a7abd`**: go further and decouple from the FSI solver
    *entirely* for now. Instead of trying to design the interface-coupling
    reformulation in the abstract, build and validate a **standalone**
    fluid-only remeshing capability first, with the interface driven by a
    function chosen for testing purposes, not derived from an actual
    coupled solve. The interface-coupling question was deliberately left
    unsolved and unlisted — a placeholder list of "candidate mechanisms"
    would have been guessing, not planning.

This is the scope the actual implementation (§2 below) was built against:
Phases 1–4 of `implementation-plan.md` §7, entirely solver-free, entirely
confined to `src/xfsi_solver/remeshing/`, never touching any existing FSI
solver file.

## 2. Building Phase 1: a deformation to test with, and a quality metric

First real code, commit `08ccb49`. Two independent pieces:

**Extracting a fluid-only mesh.** `fluid_domain.py` reuses
`dolfinx.mesh.create_submesh` the same way `fsi2_harmonic_diffmesh.py`
already does for pressure — filter cells by the `ALE_fluid` tag — but adds
`dolfinx.mesh.transfer_meshtags_to_submesh` to carry the *facet* tags
(inflow/outflow/channel walls/obstacle/interface) onto the new submesh too,
which the existing solver code doesn't need (it keeps using the parent
mesh's tags via `entity_maps`) but a standalone fluid problem does.

**A deformation to drive testing with.** `deformation.py`'s
`prescribed_interface_deformation` needed to satisfy two things at once:
produce a recognisable cantilever-bending shape on the flag, and be
*exactly* zero on every other boundary piece, since `discrete_mesh.py`'s
later curve-classification logic (§3) relies on the channel walls and
obstacle never moving. The first attempt used a Gaussian decay envelope,
which is smooth but never actually reaches zero — numerically this left
inflow/outflow with values around `1e-321` and the channel walls around
`3e-5` (for amplitude 0.1), not exact zeros. Switched to a
compactly-supported bump (`exp(1 - 1/(1-r²))` for `r<1`, else exactly `0`)
so those pieces are genuinely, provably fixed rather than "small enough in
practice."

**Vendoring the quality metric.** Per review, `quality.py` is
`ottarph/pvmeshquality` copied in essentially unmodified (only reformatted
for line length / import order to satisfy this repo's ruff config) —
smoke-tested against this repo's pinned `dolfinx==0.11.0`/`pyvista==0.48.4`
and found to need no code changes at all, confirming the reviewer's own
expectation. One thing worth remembering about it: it only accepts CG1
vector spaces for the warping displacement (asserted in its own code), so
anywhere it's used with a higher-order field, that field has to be
projected/interpolated down first.

**Calibration, not guessing.** Rather than pick a quality threshold out of
thin air, swept the deformation amplitude and recorded when cells actually
invert (checked independently via signed triangle area, not via the
quality metric itself): zero inverted cells at amplitude 0.05, six by 0.08.
Also discovered, while doing this, that Verdict's `scaled_jacobian` for
triangles is implemented so it can *never* go negative — it saturates
towards but never crosses zero even for a fully inverted cell. That means a
remesh trigger has to be "quality below some small positive cutoff," never
"quality non-positive," and the Phase 1 tests assert the metric tracks
*real* degeneracy (via the independent signed-area check) rather than
trusting the saturation behaviour blindly.

**A bug that shaped everything downstream.** An early draft of this phase
computed the deformed geometry by adding a CG1 `Function`'s `.x.array`
directly onto `mesh.geometry.x`. This silently produced a scrambled mesh:
DOLFINx's CG1 dof ordering and its mesh-geometry node ordering are *not*
the same permutation (confirmed later by direct comparison — same set of
points, different order), so the addition mixed up which displacement
value landed on which vertex. The fix at the time was to stop going through
a `Function` at all and evaluate the deformation formula directly on
`mesh.geometry.x` — but the *general* version of this problem (needing to
move data between the two orderings correctly, not just avoid it) came back
in Phase 4 and got a proper, tested utility (`dof_geometry.py`, §5).

## 3. Building Phase 2/2a: the discrete-mesh round trip

Commit `7ba6ffc`. This was flagged from the start as the single biggest
technical uncertainty in the whole plan — whether gmsh's
`classifySurfaces`/`createGeometry` pipeline, designed for cleaning up STL
scans, would actually do something sensible on a flat 2D fluid mesh with a
hole in it (the FSI2 fluid domain wraps around the obstacle+flag). It was
prototyped interactively before writing any "real" module code, and the
prototype effectively *became* the real code once it worked, rather than
being thrown away — Phase 2a (validate) and Phase 2 (build the reusable
routine) collapsed into one piece of work, `discrete_mesh.py`.

**It worked far better than expected, once the angle threshold was right.**
The first attempt used a 30° `classifySurfaces` angle (a plausible-looking
default) and got two very different failure modes depending on whether the
input was deformed: on the *undeformed* mesh it worked instantly, cleanly
recovering exactly the right 8-curve topology (4 straight channel walls,
the obstacle arc, and the flag's 3 straight edges) with all curve endpoints
matching the known FSI2 geometry constants to high precision. But on a
*deformed* mesh, 30° caused `classifySurfaces` to detect hundreds of
spurious "corners" from ordinary mesh-resolution noise along the smoothly
bent boundary, and the pipeline effectively hung (a background run was left
running for ~29 minutes before it was noticed and killed — see §6). Raising
the angle to 60° fixed both: the deformed case recovers the same clean
8-curve topology, in well under a second, for every amplitude tested.

**A second, more severe hang, found only later.** Even at 60°, feeding the
mechanism a boundary that had already gone *past* actual cell inversion
(not just poor quality — genuinely self-intersecting) could still hang
rather than raise. This is documented prominently in `discrete_mesh.py`'s
module docstring as a hard limitation, not a bug to route around: it's the
reason the whole remeshing scheme only works if the trigger fires on early
degradation, never on outright inversion. Phase 4 (§5) had to build an
explicit guard against this, because the loop's amplitude-stepping
mechanics ran straight into it.

**Sidestepping node-ordering for higher-order cells.** The plan had flagged
DOLFINx-vs-gmsh node ordering for higher-order/quad cells as a real risk.
It turned out to have a clean solution for the "build the discrete input"
side specifically: feed gmsh only the 3 corner nodes of each cell,
regardless of whether the source mesh is first- or second-order
(`mesh.geometry.dofmaps[0][:, :3]` — DOLFINx always lists a cell's corner
vertices before any higher-order DOFs). The source mesh's curvature doesn't
need to survive this step at all, since the *new* mesh gets fresh geometry
from gmsh's own `setOrder(2)` afterwards — so there's no need to get the P2
node ordering right in the input at all. Validated on both `mesh.xdmf` (P1)
and `mesh_sec.xdmf` (P2).

**Re-tagging.** Since the four fixed boundary pieces and the obstacle never
move (by construction of the Phase 1 deformation), classifying gmsh's
recovered curves back into `PHYSICAL_MARKERS` names is a simple
bounding-box check (`_classify_curve` in `discrete_mesh.py`) with no
dependence on how much the interface has deformed.

**Sizing.** A graded background mesh-size field (gmsh `Distance` +
`Threshold` fields, fine near the obstacle/interface, coarser far away)
reproduces the spirit of `create_mesh_FSI2.py`'s own sizing logic without
needing to port that script's point-by-point `setSize` calls, which assume
analytic geometry that doesn't exist for a remeshed domain.

## 4. Building Phase 3: field transfer

Commit `5086470`. Mechanically the simplest phase — `transfer_field` is a
thin wrapper around `dolfinx.fem.create_interpolation_data` /
`Function.interpolate_nonmatching`, the same API this repo's legacy
`component_solvers/navier_stokes_ALE_fsi2_mm.py` already uses for a
different purpose (bringing in a precomputed boundary displacement time
series). The only real work was calibrating `padding`, and it needed
calibrating twice:

- With the default-looking `1e-6`, about a dozen dofs near the obstacle
  boundary came back completely wrong (not slightly off — stuck at exactly
  `0`, since `interpolate_nonmatching` silently leaves unfound points at
  their `Function`'s default value rather than raising). `1e-3` fixed it
  for that sizing.
- Testing against a *coarser* sizing (`size_far=0.06`, closer to what the
  real loop would use) than the one used to find `1e-3` showed it wasn't
  enough — points near the coarser far-field boundary were still missed.
  `1e-2` was needed. The lesson generalized into the module docstring: this
  isn't a "small safety margin" constant, it has to be checked against
  whatever the coarsest cell size actually in play is, not assumed correct
  once and left alone.

Tests check the exact comparison the plan called for — interpolate a known
analytic function onto the old mesh, transfer it, and compare against
interpolating that same function directly onto the new mesh — for both a
CG2 scalar and a CG1 vector field, and there's a dedicated regression test
that reproduces the too-small-padding failure mode and checks
`DEFAULT_PADDING` avoids it.

## 5. Building Phase 4: the combined loop, and two real design corrections

Commit `e885cf2`. This is where the first three phases got wired together
into `run_prescribed_deformation_loop`, and where two things that looked
fine in isolation turned out to be broken once chained together across
*multiple* remesh events.

### 5.1 The reference-position trap

The first working design tracked, alongside the mesh, a "reference
coordinates" field — each current mesh node's true original (t=0) material
position — reasoning that `prescribed_interface_deformation` needs the
*material* position to evaluate its shape function correctly, and after a
remesh the new mesh's own geometry is the *current* (already bent) position,
not the reference one. The plan was to carry this reference field forward
across each remesh via `transfer_field`, exactly like any other field.

It produced badly wrong results, and the debugging path is worth recording
because the size of the error was initially confusing: a residual (recomputed
material position vs. what it should have been) of up to `0.0498` for a
`0.05`-amplitude deformation — i.e. off by essentially the *entire*
deformation, not a small numerical error. Padding (§4) had no effect at all
on it, which ruled out the "points near a boundary get missed" explanation
that fixed everything else. The actual cause, once isolated by checking the
transfer against ground truth (does the transferred field still equal the
identity at points that map exactly onto themselves?), was simpler and more
fundamental than a bug: **a field that is literally the identity — which
"the old mesh's own reference coordinates, before any deformation" is,
trivially — transfers via nonmatching interpolation to exactly the query
point itself, regardless of which source cell answers the query.** It
carries no information that could ever recover a *different* value. This
isn't fixable by better padding, a finer mesh, or smaller steps between
remesh events (the residual was found to scale almost exactly linearly with
the deformation, i.e. it doesn't shrink relative to the problem); it's a
basic property of what point-value interpolation can and can't do.
Recovering true material reference position after a remesh would need
actually *solving* an extension problem (this is, not coincidentally, close
to what the discarded `F_hat` mechanism from §1 was for on the solid side) —
out of scope for a phase explicitly meant to stay solver-free.

The fix was a genuine redesign, not a patch: `deformation.py` gained
`incremental_interface_deformation`, which decays with distance from the
mesh's *current* interface (read directly from that mesh's own
`facet_tags`) instead of distance from a fixed original position. Since the
deformation is purely vertical (dx=0 everywhere by construction), the
`x`-coordinate-based part of the shape function was already correctly
invariant across remeshes for free; only the spatial envelope needed this
change. The loop now tracks displacement purely relative to the *current*
mesh, resetting to zero at every remesh (mirroring the "reset the ALE
displacement" step in `implementation-plan.md` §5) — no cross-mesh
reference tracking at all. `transfer_field` is still exercised in the loop,
now legitimately: it carries a passive demonstration scalar field across
each remesh event, which is the "interpolating between the new and old
mesh" deliverable that was actually asked for, without conflating it with
the (impossible, this way) reference-tracking problem.

### 5.2 The inversion-hang, hit for real this time

Even after fixing §5.1, an early test run of the full loop hung — not the
first remesh event, but the *second* one, triggered on the very next check
after the first remesh succeeded. Tracing it with per-step logging (initial
attempts at this used file logging with explicit flushes rather than piped
stdout, since a genuinely hung subprocess leaves nothing to read from a
pipe either way) showed the actual cause: quality collapsed from a
comfortable ~0.16 to essentially zero (already-inverted) within a single
`0.01`-amplitude step. This is exactly the "hangs on an already-inverted
boundary" limitation documented in `discrete_mesh.py` (§3) — but here it
was reached by accident, via a threshold/step-size combination that looked
reasonable but wasn't, rather than by deliberately feeding in bad geometry.

Two changes came out of this, both now load-bearing rather than
defensive-programming decoration:

- `loop.py` runs a cheap, independent check (`_has_inverted_cells`, signed
  corner-triangle area — the same style of ground-truth check used
  throughout to validate the quality metric) *before* ever calling
  `regenerate_fluid_mesh`, raising a clear `RuntimeError` naming the
  amplitude and quality if the geometry has already inverted. This turns a
  multi-minute silent hang into an immediate, actionable failure.
- The default `quality_threshold` was raised (`0.15` → `0.35`) to build in
  a real margin, informed by directly observing how steep the
  quality-vs-amplitude curve gets once it starts degrading (roughly
  `0.49 → 0.33 → 0.16 → 0.01 → inverted` across five `0.01`-amplitude
  steps on this mesh) — "just above zero" was never going to be safe with a
  fixed step size, regardless of how the trigger is implemented.

### 5.3 Formalizing the ordering utility from §2

The geometry-node-vs-CG1-dof-ordering mismatch that caused a silent bug
back in Phase 1 came back in Phase 4, now unavoidably: the loop needs to
move data between "gmsh/geometry order" (talking to `discrete_mesh.py`) and
"CG1 dof order" (talking to `quality.py`) every single step. `dof_geometry.py`
formalizes the fix that was ad hoc in Phase 1: derive the permutation
between the two orderings once, from matching per-cell *local* vertex
order between the mesh's geometry dofmap and a same-degree CG1 space's
dofmap (verified empirically to be consistent, unlike the *global*
numbering, which isn't), with a docstring that explains the original bug
so it doesn't get silently reintroduced by someone reaching for the more
"obvious" direct-array approach again.

### 5.4 Where it landed

The validated end state: stepping the incremental deformation through 150
steps of `0.005` (a notional total bending amplitude of `0.75`, against a
single un-remeshed segment's limit of somewhere between `0.05` and `0.08`)
completes in about 4.5 seconds, with 9 remesh events and quality never
dropping below roughly `0.31` at any point — the "sawtooth" pattern of
degrade-then-reset that's exactly what a working remesh-on-demand scheme
should produce.

## 6. Housekeeping along the way

- **Local dev data.** `data/` (meshes, precomputed boundary data, FeatFlow
  reference values) is gitignored and wasn't present in this worktree by
  default; it was copied in from the main checkout early on (commit
  `928307f`'s description) so the actual FSI2 meshes could be used for
  prototyping and tests, rather than synthetic ones. It stays out of
  version control.
- **A genuinely hung process, found by the user.** The very first
  `classifySurfaces`-angle experiment (§3, the 30°-on-deformed-geometry
  case) was run in the background when its foreground timeout was hit, and
  then not explicitly cleaned up once the 60° fix was found and work moved
  on. It sat spinning at ~100% CPU for about 29 minutes before the user
  noticed it and asked about it mid-session; it was identified via `ps` and
  killed. Worth remembering for future sessions: a `timeout`-wrapped
  command that gets moved to the background on expiry doesn't go away on
  its own, and gmsh/PETSc hangs in this codebase are a real, reproducible
  failure mode (§3, §5.2), not a hypothetical one — check for stragglers
  after any exploratory run that might have hit one.
- **Branch handoff.** The user's main checkout worktree (on local branch
  `remeshing`) was left exactly at the commit this branch started from
  (`92c9e79`), so bringing this work into it for inspection in VS Code is a
  clean fast-forward (`git merge --ff-only claude/fsi-remeshing-strategy-365516`),
  not a merge — no conflicts possible. Recorded here since it's a natural
  question for whoever looks at this next: no, the two branches never
  diverged.

## 7. What's implemented, and what's explicitly not

**Implemented and tested** (`src/xfsi_solver/remeshing/`, 23 tests across
`tests/test_remeshing_*.py`, all passing in a few seconds):
`fsi2_geometry.py`, `markers.py`, `fluid_domain.py`, `deformation.py`,
`quality.py`, `discrete_mesh.py`, `transfer.py`, `dof_geometry.py`,
`loop.py` — Phases 1–4 of `implementation-plan.md` §7, in full.

**Not started**, and not pretended to be solved:

- **Phase 5** (swapping the loop's synthetic carried field for a real
  standalone fluid-only ALE solve, e.g. extending
  `component_solvers/navier_stokes_ALE_fsi2_mm.py`) — explicitly the next
  increment, not attempted here.
- **Monolithic FSI integration** (§2 of `implementation-plan.md`): how any
  of this connects back to the coupled solver, where `U`/`V` are still
  whole-mesh. This was deliberately deferred before implementation started,
  and nothing learned while building Phases 1–4 removes the need for real
  thought there — if anything, §5.1's finding (transferring an "identity"
  -like reference field across a remesh is mathematically dead) is relevant
  to it: whatever eventually tracks accumulated structural state across a
  remesh event in the real solver will need either a genuinely *solved*
  extension field, or a reformulation that (like this loop's fix) avoids
  needing "true reference position" transferred at all. That's a note for
  whoever picks up that work next, not a solution.
- Quads and the production sizing/second-order combination
  (`quads=True, second_order=True`) — deliberately deferred per the plan;
  only `mesh.xdmf` (P1) and `mesh_sec.xdmf` (P2) triangles were used.
