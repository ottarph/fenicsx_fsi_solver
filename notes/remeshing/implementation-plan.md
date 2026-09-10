# Plan: remeshing for the FSI2 benchmark (harmonic mesh motion)

Status: **design/planning only, nothing in this document is implemented
yet.** See [`literature-review.md`](literature-review.md) for the background
that motivates the choices below.

Revision note: this version incorporates review comments (fluid-only
remeshing scope, building from the `diffmesh` solver variants, using
`pvmeshquality` for the quality metric, and keeping the earliest
prototyping steps free of any FE solve). The previous "Option A: remesh the
whole domain, single shared mesh" recommendation is **withdrawn** — see §2.

**Local dev data**: the untracked `data/` folder (gitignored, not part of
version control) has been copied into this worktree from the main checkout
so the meshes below are available for prototyping:
`data/meshes/fsi2/mesh.xdmf` (linear/P1 triangles),
`data/meshes/fsi2/mesh_sec.xdmf` (second-order/P2 triangles, curved edges),
plus the quad variants (deferred, see §7), `data/fsi2_boundary/*.npy` (the
precomputed boundary-displacement series the legacy `component_solvers` use),
and `data/fsi2_reference.txt` (FeatFlow reference values).

## 1. The problem, precisely

`src/xfsi_solver/solvers/fsi2_harmonic.py` (and its `_diffmesh` sibling, see
§2) solve the FSI2 benchmark as a **fixed-reference, pseudo-solid ALE**
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
regression test that reaches this regime**.

## 2. Scope decision: fluid-only remeshing, built from the `diffmesh` variants

Per review: remeshing is scoped to **the fluid domain only** — the solid
mesh is never regenerated. This is also what the FSI-remeshing literature
generally does (`literature-review.md` §2/§4: the solid's Lagrangian
material description is path-independent and doesn't accumulate the kind of
distortion the ALE mesh-motion PDE does), and it removes an entire class of
complexity the previous version of this plan carried: since the solid is
never remeshed, its St. Venant–Kirchhoff strain stays measured from the true
t=0 reference forever, and **no accumulated-deformation-gradient bookkeeping
is needed** (the previous plan's `F_hat` mechanism is dropped entirely).

This is why the plan now builds from `fsi2_harmonic_diffmesh.py` /
`fsi2_biharmonic_diffmesh.py` rather than the plain `fsi2_harmonic.py`: they
already carve the fluid region out as its own `dolfinx.mesh.Mesh` object via
`dolfinx.mesh.create_submesh` —

```python
fluid_mesh, fluid_cell_map, fluid_vertex_map, _ = dfx.mesh.create_submesh(
    mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"]))
entity_maps = [fluid_cell_map]
...
P = dfx.fem.functionspace(fluid_mesh, ("CG", 1))          # fsi2_harmonic_diffmesh.py:52-57,124
...
problem = dfx.fem.petsc.NonlinearProblem(
    residual_blocked, [u, v, p], bcs=bcs,
    entity_maps=entity_maps, ...)                          # fsi2_harmonic_diffmesh.py:273-276
```

— and already prove out DOLFINx's mixed-mesh assembly (`entity_maps`) needed
to couple a field on a submesh with fields on the parent mesh. **But** note
precisely what's split today: only pressure `P` (and, in the biharmonic
sibling, the auxiliary field `Z`) lives on `fluid_mesh`. Displacement `U` and
velocity `V` remain on the whole parent `mesh`
(`fsi2_harmonic_diffmesh.py:122-123`), exactly as in the non-`diffmesh`
solver. **This is the real work fluid-only remeshing requires**: `U`/`V`
have to be split the same way `P` already is, onto an independently
regenerable `fluid_mesh` (displacement/velocity) and a permanently-fixed
`solid_mesh` (structural displacement/velocity), with interface continuity —
today automatic via shared DOFs — replaced by an **explicit interface
coupling condition**, since after the split the two meshes' interface DOFs
are no longer the same DOFs.

This interface-coupling reformulation is flagged as its **own dedicated
design step** (§7, "interface coupling") rather than resolved here: it needs
focused attention independent of, and probably after, the solver-free
prototyping work below, which is deliberately scoped to not depend on how it
gets resolved. Candidate mechanisms to evaluate when that step is reached
(not a decision yet):
- A node-matched multi-point constraint (MPC) tying fluid-side interface
  DOFs to solid-side interface DOFs whenever a remesh is built so that the
  two node sets coincide in position by construction. `dolfinx_mpc` is the
  standard tool for this in the DOLFINx ecosystem but is **not currently a
  dependency** of this repo — would need to be added and evaluated.
- A Dirichlet "copy-down" coupling (prescribe the fluid problem's interface
  BC each step from the solid solution, partitioned-scheme style) — simpler
  to implement, but changes the coupling from monolithic-exact to a
  segregated approximation, which needs care to not silently reduce
  accuracy/stability relative to today's exact monolithic coupling.
- A weak (Nitsche/mortar) interface coupling — most general, most work.

## 3. Existing building blocks already in the repo (don't reinvent)

- **`dolfinx.mesh.create_submesh` + `entity_maps` mixed-mesh assembly** is
  already working in this repo for pressure
  (`fsi2_harmonic_diffmesh.py:52-57,124,273-276`, `fsi2_biharmonic_diffmesh.py`
  similarly) — the direct precedent and starting point for splitting `U`/`V`
  the same way (§2).
- **DOF-coordinate extraction pattern**: the "measurement spot" probe used
  identically in all monolithic FSI2 solvers (e.g.
  `fsi2_harmonic_diffmesh.py:297-301`) shows the exact primitives needed:
  `U.tabulate_dof_coordinates()` + `np.isclose`/boolean masking, combined
  with `dfx.fem.locate_dofs_topological`.
- **`scifem.compute_interface_data`** (`fsi2_harmonic_diffmesh.py:76-86`)
  already gives paired fluid-side/solid-side interface entity data.
- **Nonmatching interpolation is already used and working** in
  `component_solvers/navier_stokes_ALE_fsi2_mm.py:103-136` (and the `_bih`
  and `static_` siblings) via `create_interpolation_data(V_to, V_from, cells,
  padding=...)` + `interpolate_nonmatching`. Working template for the
  old-mesh → new-mesh field transfer step.
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
  solvers' constant `alpha_u = 1e-9`. Worth porting as a Phase 0 robustness
  win independent of remeshing.

## 4. Remesh trigger: mesh-quality metric

Use `pvmeshquality.MeshQuality` (§3) on the **fluid submesh** (`fluid_mesh`
in the `diffmesh` solvers), warped by a CG1 copy of the fluid-region ALE
displacement, evaluated every time step (cheap relative to the Newton
solve). Two thresholds:

- **Warn/tighten threshold**: log-only, to gather real degradation data
  against which to tune the next threshold (Phase 0, §7).
- **Remesh threshold**: trigger a remesh **between time steps** (never
  mid-Newton-iteration) once `scaled_jacobian` (or whichever measure Phase 0
  settles on) drops below a value calibrated from that logged data — not an
  arbitrary guess.

Fall back to a fixed max-step-count cadence purely as a safety net in case
the quality metric is mis-calibrated, matching `literature-review.md` §2's
recommended quality-triggered-with-cadence-fallback pattern.

## 5. Field transfer across a remesh event

Because remeshing is fluid-only (§2), the set of fields needing transfer at
a remesh event is smaller than previously planned, and **no accumulated
deformation-gradient bookkeeping is needed** (nothing changes about the
solid, which isn't touched): transfer the fluid-region contents of `u`,
`u_old`, `v`, `v_old` and all of `p`, `p_old` if present, from the old fluid
mesh to the new one, via `create_interpolation_data`/`interpolate_nonmatching`
(`component_solvers/navier_stokes_ALE_fsi2_mm.py` pattern), with generous
`padding` since the new fluid mesh's boundary is deliberately built to
coincide with the old mesh's *current* physical boundary. After transfer,
reset the fluid-region `u ← 0`, `u_old ← 0` (the new fluid mesh *is* the
current fluid configuration). What happens to `v`/`p` at the interface, and
how the (still to be designed, §2) interface coupling re-establishes itself
immediately after a remesh, is part of the interface-coupling design step,
not decided here.

## 6. Fluid-mesh regeneration mechanics

**Recommended mechanism (per expert input — see `literature-review.md` §5):
DOLFINx discrete-mesh round trip through gmsh, not hand-built splines.**
Scoped now to the fluid region only:

1. Gather (rank 0, since a fresh `gmsh.model.mesh.generate` call only runs on
   `gmsh_model_rank`) the *deformed* geometry of the fluid submesh:
   `X_deformed = fluid_mesh.geometry.x + u|_fluid` for every node of every
   fluid cell, together with the existing cell connectivity
   (`dolfinx.mesh.entities_to_geometry`).
2. Feed that into gmsh as a **discrete entity** (`gmsh.model.addDiscreteEntity`
   + `gmsh.model.mesh.addNodes` + `gmsh.model.mesh.addElementsByType`,
   following Dokken's pattern from `literature-review.md` §5), reproducing
   the current, possibly near-degenerate, fluid triangulation inside gmsh.
3. `gmsh.model.mesh.classifySurfaces(angle)` to recover sharp-feature curves
   from the discrete boundary (channel corners, the cylinder/flag boundary,
   the interface with the solid — these already coincide with
   `PHYSICAL_MARKERS` boundary-piece transitions), then
   `gmsh.model.mesh.createGeometry()` to reparametrize them into genuine,
   remeshable CAD curves.
4. Re-tag the recovered curves using (plausibly, to be checked) the same
   classification logic `create_mesh_FSI2.py:146-175` already has, and
   re-apply the same graded sizing-field logic.
5. Discard the old triangulation, `gmsh.model.mesh.generate(2)` +
   `gmsh.model.mesh.setOrder(2)` (once quads/second-order are back in scope,
   §7) fresh, then `dolfinx.io.gmsh.model_to_mesh(...)`.
6. Rebuild `fluid_mesh` and everything downstream of it
   (`fluid_cell_map`/`entity_maps`, `P`, the fluid-side split of `U`/`V`
   once §2's interface-coupling step lands) from this new mesh; the solid
   side of the problem is completely untouched.

*Fallback*, unchanged from the previous revision: hand-built
`gmsh.model.occ.addSpline`/`addBSpline` boundary curves from ordered,
ONLY-boundary DOF coordinates, if `classifySurfaces`/`createGeometry`
doesn't behave well for this domain — see §7 Phase 2a, which exists
specifically to check this early.

## 7. Phased implementation plan

Per review, the earliest phases are deliberately **solver-free**: deform the
grid by interpolating a *known analytic function*, and validate field
transfer by interpolating *known analytic functions* between meshes — no
`NonlinearProblem`/`LinearProblem` solve anywhere until the interface
-coupling and orchestration phases. Work on **triangles only** first
(`data/meshes/fsi2/mesh.xdmf`, linear; then `data/meshes/fsi2/mesh_sec.xdmf`,
second-order/curved); quads are explicitly deferred.

- **Phase 0 — diagnostics and cheap mitigation (no remeshing yet).**
  Vendor/adapt `pvmeshquality` for DOLFINx 0.11 (§3) and add it as a logged
  (not yet triggering) quantity to `fsi2_harmonic_diffmesh.py`. Add a
  long-running regression test (slow/opt-in) that runs FSI2 until it
  currently fails, to observe and record the failure and get real
  `scaled_jacobian` decay data to calibrate §4's threshold against. Port the
  `cell_volume**(-2)` local stiffening (§3) into the diffmesh solver's `A_I`
  as a cheap robustness win, independent of remeshing. Do the
  `PHYSICAL_MARKERS` module refactor now.

- **Phase 1 — deform-by-known-function + quality-metric validation
  (no FE solve, no gmsh yet).**
  Load `data/meshes/fsi2/mesh.xdmf`, extract the fluid submesh the same way
  `fsi2_harmonic_diffmesh.py` does, interpolate a **known analytic**
  displacement function (not a PDE solution) into a CG1 vector `Function` on
  it, and run it through `pvmeshquality.MeshQuality` — confirm quality drops
  as the analytic deformation is scaled up, and that a deliberately
  degenerate deformation is correctly flagged. Repeat on
  `data/meshes/fsi2/mesh_sec.xdmf` (P2) to confirm the CG1-only restriction
  (§3) is handled correctly (project/interpolate down, don't pass the P2
  function directly). This is the first proof that the trigger metric (§4)
  behaves sensibly, entirely decoupled from the solver.

- **Phase 2a — validate the discrete-mesh round trip in isolation, on
  undeformed data first (no FE solve).**
  Round-trip the *undeformed* fluid submesh of `mesh.xdmf` through
  `addDiscreteEntity`/`addNodes`/`addElementsByType`/`addPhysicalGroup`
  (Dokken's pattern) → `classifySurfaces` → `createGeometry` → re-tag →
  `generate(2)`, and diff the result against the original (volumes,
  physical group tag sets, boundary curve count/composition). Then repeat
  with the mesh **deformed by the same known analytic function from Phase 1**
  (still no FE solve — the deformation is an interpolated function, not a
  PDE solution) and check the regenerated mesh is valid
  (`pvmeshquality` quality > 0 everywhere) and visually matches the intended
  deformed shape. Once this works on `mesh.xdmf`, repeat on `mesh_sec.xdmf`.
  This isolates and retires the single biggest technical uncertainty in the
  whole plan (does `classifySurfaces`/`createGeometry` behave well for this
  2D fluid domain at all) before it's entangled with the solver. If it
  doesn't work cleanly, fall back to the spline-based approach (§6).

- **Phase 2 — mesh regeneration routine.**
  Refactor `create_mesh_FSI2.py`'s tagging/sizing logic into a function
  usable both for the initial mesh and for a remesh (discrete-mesh round
  trip from Phase 2a, fed with deformed coordinates), scoped to the fluid
  region only.

- **Phase 3 — field-transfer validation with known functions (no FE
  solve).**
  Interpolate a **known analytic** function (smooth, not a solver output)
  onto the old fluid mesh, transfer it to a Phase-2-regenerated new fluid
  mesh via `create_interpolation_data`/`interpolate_nonmatching`, and
  compare against interpolating the same analytic function directly onto
  the new mesh — check the error is small and roughly `O(h^2)`. This is
  exactly the "interpolate known functions between the meshes" check from
  review, and exercises the same machinery Phase 5's real field transfer
  will use, without needing a working solver yet.

- **Interface coupling — dedicated design step (see §2).**
  Only after Phases 0–3 above give confidence in the mesh-regeneration and
  field-transfer mechanics: work out how `U`/`V` split onto independent
  fluid/solid meshes and re-couple at the interface. This is the part of
  the plan most likely to need its own follow-up design note once the
  building blocks above are validated — not expanded further here.

- **Phase 4 — orchestration in `fsi2_harmonic_diffmesh.py`.**
  Wire the trigger (§4) into the time loop: on trigger, regenerate the
  fluid mesh (§6), transfer fields (§5), re-couple at the interface (per the
  design step above), rebuild the fluid-side function spaces/BCs/
  `NonlinearProblem`, and continue.

- **Phase 5 — validation against the benchmark.**
  Extend `plot_qois.py`/the existing harmonic-vs-biharmonic comparison to
  include a "harmonic + remeshing" run for the *full* benchmark duration,
  confirming the tip y-displacement curve tracks the FeatFlow reference
  values (`data/fsi2_reference.txt`) past the point where today's
  no-remeshing run currently fails.

- **Phase 6 — quads and second-order, and biharmonic.**
  Once the above is solid on linear/second-order triangles, extend to the
  production `quads=True, second_order=True` mesh (bringing in the
  element-type/node-ordering work flagged in §8), and to
  `fsi2_biharmonic_diffmesh.py`.

## 8. Open questions / risks to track during implementation

- **Interface-coupling reformulation is unsolved and is now the largest risk
  in the plan** (§2) — splitting `U`/`V` off the parent mesh the way `P`
  already is split, and re-establishing interface continuity/traction
  balance without the "free" shared-function-space coupling, is a real
  reformulation, not a remeshing detail. Flagged as its own step rather than
  estimated here.
- **`pvmeshquality`'s CG1-only restriction and DOLFINx 0.11 compatibility**
  (§3) — needs an explicit adaptation/smoke-test pass (Phase 0) before it
  can be trusted as the trigger signal.
- **Trigger threshold calibration**: no real quality-decay data exists yet
  for this benchmark (Phase 0 produces it).
- **Remeshing frequency vs. cost**: a full gmsh regeneration + MUMPS
  refactorization is expensive; profile after Phase 4.
- **MPI behavior of repeated `model_to_mesh` calls mid-run**: existing usage
  is "call once at startup" — `create_test_meshes.py` already notes
  `gmsh.clear()` is needed between repeated in-process `create_mesh` calls;
  same caution applies here, more so inside a long-running MPI job.
- **Second-order (curved) element boundary reconstruction**: P2 DOFs already
  include edge midpoints (`literature-review.md` §3), should work in
  principle for `mesh_sec.xdmf`, but should be spot-checked visually before
  trusting it (Phase 2a covers this mesh explicitly).
- **gmsh element-type codes and DOLFINx↔gmsh node ordering for
  higher-order/quad cells**: deliberately deferred (§7 Phase 6) — Dokken's
  discrete-entity example is for linear tetrahedra; DOLFINx/basix and gmsh
  use different local node orderings for higher-order and quad cells, a
  well-known footgun (silently wrong/inverted elements, not a hard error)
  when hand-rolling this kind of conversion. Do not attempt quads until
  Phases 0–5 are solid on triangles.
- **`classifySurfaces` angle-threshold tuning**: the FSI2 fluid domain mixes
  sharp corners (flag trailing edge, channel corners) with a smooth
  circular-arc obstacle boundary and the interface with the solid — check
  the threshold and resulting curve count/composition explicitly in
  Phase 2a rather than assuming a default works.
- **Interpolation accuracy at the exact new interface boundary**: where
  `padding` in `create_interpolation_data` matters most (§5/§6) — needs
  explicit test coverage in Phase 3, not just an average-case error check.
