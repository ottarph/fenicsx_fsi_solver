# Remeshing in ALE-FSI solvers: literature review

Working notes gathered while planning remeshing support for the FSI2 benchmark
solver. Goal: survey how other solvers handle mesh degradation under large
boundary deformation, before designing an approach for this codebase (see
[`implementation-plan.md`](implementation-plan.md)).

## 1. Why harmonic (and biharmonic) mesh motion eventually fails

An ALE fluid solver never moves the mesh with the true material velocity;
instead an auxiliary "mesh motion" PDE extends the known interface
displacement into the fluid domain. Harmonic (Laplace) extension is the
cheapest choice, but its solutions near reentrant/sharp corners behave like
`r^(π/ω)` for corner angle `ω`; for `ω > π` (a reentrant corner, which is
exactly the shape you get once a flag/beam bends far enough that the fluid
domain develops a locally non-convex pocket next to it) the gradient blows up
and bijectivity (`det(I + ∇u) > 0` everywhere) is lost — elements invert.
Biharmonic extension has better regularity at corners and generally tolerates
larger deformation, but it too eventually collapses cells once the interface
displacement grows enough — this is exactly reproduced by Shamanskiy & Simeon
(2020) using a gravity-driven deformation test built on the *same FSI2
geometry, material parameters, and beam* as our benchmark, explicitly to push
biharmonic extension to degeneracy. That paper also compares harmonic,
biharmonic, incremental linear elasticity, and a nonlinear-elasticity /
continuation scheme ("TINE") that explicitly enforces bijectivity by using a
`ln J` term in the mesh-material's strain energy (so the mesh PDE itself
cannot produce an inverted element, only an infinitely stiff one). Their
conclusion: elasticity-based and nonlinear mesh motion techniques are
markedly more robust than harmonic/biharmonic and can be combined with
Jacobian-based local stiffening (making elements near already-small volumes
artificially stiffer) for extra headroom — but *none* of these techniques
make remeshing unnecessary in general; they only push the point of failure
further out. [Shamanskiy & Simeon 2020, arXiv:2006.14051](https://arxiv.org/pdf/2006.14051)
(published version: [Computational Mechanics](https://link.springer.com/article/10.1007/s00466-020-01950-x)).

Practical implication for this repo: before touching remeshing at all, our
own `alpha_u = 1e-9` constant-stiffness Laplace extension in
`fsi2_harmonic.py` is already known to be weaker than what this codebase
itself uses elsewhere (`component_solvers/navier_stokes_ALE_fsi2_mm.py` uses
`alpha = alpha_0 * cell_volume**(-2)` local stiffening). Swapping in local
stiffening, and/or defaulting to the already-implemented biharmonic monolithic
variant, is a cheap way to delay (not eliminate) the failure and is a
reasonable Phase 0 step independent of remeshing itself.

## 2. General remeshing strategies used in ALE-FSI / large-deformation solvers

1. **Pure mesh motion (r-adaptivity)** — no connectivity change, only node
   repositioning via an auxiliary PDE (harmonic, biharmonic, linear/nonlinear
   elasticity) or algebraic technique (spring analogy, radial basis function
   interpolation). Cheapest, always tried first; eventually insufficient for
   genuinely large deformation because the *topology* of a good mesh for the
   deformed shape can differ from the topology of a good mesh for the
   reference shape (this is exactly the situation once the flag bends enough
   to want finer/differently-graded elements on the concave side).

2. **Global/full remeshing** — periodically discard the current mesh and
   generate an entirely new one (usually via a general-purpose unstructured
   mesher: gmsh, Netgen, MMG, BAMG, TetGen) from the *current deformed
   boundary*. All solution fields are then transferred (interpolated) from
   the old mesh onto the new one, and the simulation continues. This is what
   general fluid/FSI codes with "ALE + remeshing" fall back to once mesh
   motion alone can't cope, and it's what the user's question is pointing at.
   Practical characteristics reported across the literature and DOLFINx/gmsh
   forum threads:
   - The reference/computational domain is *redefined* at each remesh event;
     "at each remeshing step, the reference domain is updated accordingly and
     all fields are projected onto the new mesh."
   - Remeshing is triggered either at a **fixed cadence** (every N steps —
     simple, but wastes cost when deformation is small, and can still miss a
     sudden distortion spike between checks) or by a **mesh-quality metric**
     crossing a threshold (checked every step — more expensive to check but
     only remeshes when actually needed; this is the generally recommended
     approach since remeshing itself, not the quality check, is the expensive
     part).
   - Field transfer is almost always simple nodal/FE interpolation of smooth
     fields (velocity, pressure, displacement), not a conservative L2
     projection — FSI literature treats the interpolation error as a second
     -order-in-mesh-size perturbation that's acceptable given remeshing
     itself is already an approximation of the true evolving domain.
   - A **graded sizing strategy** (fine near the interface/structure,
     coarsening with distance) should be reproduced at every remesh, not just
     at t=0, or mesh quality/cost will drift over the run.
   [General remeshing-in-ALE-FSI search results](https://www.google.com/search?q=remeshing+ALE+fluid-structure+interaction) (see search summary),
   [Saksono et al. 2007, "An adaptive remeshing strategy for flows with moving
   boundaries and fluid–structure interaction", IJNME](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1971).

3. **Local remeshing (mesh-connectivity change)** — instead of regenerating
   the whole domain, apply local operations (edge flips/swaps, local
   refinement/coarsening, node insertion/deletion) only where quality has
   degraded. Cheaper than global remeshing and avoids a full field-transfer
   step for the untouched part of the mesh, but implementation is
   substantially harder (needs a mesh library that exposes these local
   operators, e.g. MMG) and less commonly available inside a Python/FEniCSx
   workflow. [Barral et al., "Large displacement body-fitted FSI simulations
   using a mesh-connectivity-change moving mesh strategy"](https://pages.saclay.inria.fr/frederic.alauzet/proceedings/Barral_Large%20displacement%20body-fitted%20FSI%20simulations%20using%20a%20mesh-connectivity-change%20moving%20mesh%20strategy.pdf).

4. **"Extended ALE" / partial remeshing near the structure only** — remesh
   just the region adjacent to the moving boundary while leaving the far
   field mesh untouched, reducing remeshing cost and the volume over which
   fields must be transferred. Relevant to FSI2 specifically since the
   channel far-field mesh never needs to change — only the region around the
   cylinder/flag does.

Given this repo's dependency set (`gmsh`, no MMG/local-remesher, no meshio),
**global remeshing via gmsh's OCC kernel**, triggered by a mesh-quality
metric with a fixed-cadence fallback, and using simple nodal interpolation
for field transfer, is the approach that fits both the literature consensus
and what's actually available/idiomatic in this codebase. This is the
approach developed in the implementation plan.

## 3. The DOLFINx-specific mechanics

- **Boundary point extraction**: `FunctionSpace.tabulate_dof_coordinates()`
  combined with `dolfinx.fem.locate_dofs_topological(V, dim, facets)` gives
  physical (reference-configuration) coordinates for a given facet set. This
  codebase already uses both functions elsewhere (see
  `implementation-plan.md` §"existing building blocks"). Adding the current
  displacement `u` at those DOF coordinates gives the *current* (deformed)
  physical position — exactly the "boundary geometry from the FE dof points"
  the task description points at. Because the FSI2 mesh uses second-order
  (P2/curved) elements, this DOF set already includes edge-midpoint nodes, so
  a spline/curve built by connecting these deformed points in order
  reasonably approximates the true curved deformed boundary without any
  extra geometric reconstruction.

- **Mesh (re)generation**: gmsh's OCC kernel builds curves from an explicit
  point sequence via `gmsh.model.occ.addSpline(pointTags)` (or `addBSpline`);
  `addCurveLoop` + `addPlaneSurface` turn a closed sequence of such curves
  into a meshable surface; `gmsh.model.mesh.setSize` on point entities
  reproduces graded sizing. This repo's own
  `src/xfsi_solver/scripts/create_mesh_FSI2.py` already implements exactly
  this pattern for the *initial* geometry (straight lines + circle arcs
  instead of splines, since the initial boundary is analytic) — a remeshing
  routine is structurally the same construction, with the deforming part
  of the boundary supplied as digitized spline points instead of analytic
  primitives, and reusing the same physical-group tagging and sizing-field
  logic so the regenerated mesh stays consistent with the original one.
  Conversion back to a dolfinx mesh goes through
  `dolfinx.io.gmsh.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank,
  gdim=gdim)`, same as the existing script — note this runs the actual gmsh
  meshing algorithm **only on `gmsh_model_rank`** and then distributes the
  result, so a remeshing call made mid-run under MPI needs the deformed
  boundary point cloud gathered onto that rank first (it currently isn't
  gathered anywhere in the repo; every existing boundary-point extraction is
  rank-local as of writing).

- **Field transfer**: `dolfinx.fem.create_interpolation_data(V_to, V_from,
  cells, padding=...)` builds a `PointOwnershipData` once per pair of
  (destination cells, function spaces); `Function.interpolate_nonmatching(
  u0, cells, interpolation_data)` then evaluates it. `padding` widens the
  bounding-box search so points that sit exactly on a shared boundary (e.g.
  reused interface node positions) are reliably found in a source cell
  despite floating point error — this matters a lot here since a remeshed
  boundary is deliberately built to coincide with the old boundary's current
  physical position. This exact pattern (nonmatching interpolation between a
  1D boundary-displacement mesh and the fluid mesh, with `padding=1e-6`
  /`1e-8`) is **already implemented and working** in this repo's legacy
  `component_solvers/navier_stokes_ALE_fsi2_mm.py` (see plan for file/line
  references) — it is a good template to copy from rather than a new
  integration to prove out.

## 4. Sources consulted

- Shamanskiy, A. & Simeon, B. (2020). *Mesh deformation techniques in
  fluid-structure interaction: robustness, accumulated distortion and
  computational efficiency.* [arXiv:2006.14051](https://arxiv.org/pdf/2006.14051) /
  [Computational Mechanics (2020)](https://link.springer.com/article/10.1007/s00466-020-01950-x).
  Directly benchmarks harmonic vs. biharmonic vs. elasticity-based vs.
  nonlinear-elasticity mesh motion on Turek–Hron-style FSI geometry.
- Saksono, P. et al. (2007). *An adaptive remeshing strategy for flows with
  moving boundaries and fluid–structure interaction.* [IJNME](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1971).
- Barral, N. et al. *Large displacement body-fitted FSI simulations using a
  mesh-connectivity-change moving mesh strategy.* [PDF](https://pages.saclay.inria.fr/frederic.alauzet/proceedings/Barral_Large%20displacement%20body-fitted%20FSI%20simulations%20using%20a%20mesh-connectivity-change%20moving%20mesh%20strategy.pdf).
- FEniCS Discourse: [Remeshing (using gmsh?)](https://fenicsproject.discourse.group/t/remeshing-using-gmsh/713),
  [Interpolation on non-matching meshes](https://fenicsproject.discourse.group/t/interpolation-on-non-matching-meshes/8586),
  [Cells argument of create_interpolation_data and interpolate_nonmatching](https://fenicsproject.discourse.group/t/cells-argument-of-create-interpolation-data-and-interpolate-nonmatching/14742).
- DOLFINx API docs: `dolfinx.fem.create_interpolation_data`,
  `dolfinx.fem.Function.interpolate_nonmatching` (`main` branch docs, current
  as of this writing).
