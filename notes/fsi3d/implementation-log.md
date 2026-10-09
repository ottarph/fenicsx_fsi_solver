# 3d FSI benchmark of Failer & Richter: implementation log

Goal: the 3d benchmark of Failer & Richter (J. Sci. Comput. 82:28, 2020,
Sect. 5.1.2, Table 1, Fig. 1 and Fig. 3), solved with the formulation and the
solver of `solvers/fsi2_biharmonic_diffmesh_restricted_split_condensed.py`.
The paper is in `references/` (not committed).

## Benchmark (2026-10-09)

- Channel 0 < x < 2.8, 0 < y < 0.41, 0 < z < 0.41, with the cylinder
  (x - 0.5)^2 + (y - 0.2)^2 < 0.05^2 cut out over its full height. Elastic
  beam 0.5 < x < 0.9, 0.19 < y < 0.21, 0.1 < z < 0.3, without the cylinder,
  clamped where it touches the cylinder.
- rho_f = rho_s = 1000, nu_f = 1e-3, mu_s = 2e6, lambda_s = 8e6 (Table 1;
  the 2d FSI2 values of the existing solvers differ).
- Inflow v_x = U_bar * 36 y (H - y) z (H - z) / H^4, U_bar = 1.75, H = 0.41,
  ramped by 1/2 - 1/2 cos(pi t) for t < 1 (the 2d FSI2 ramp is over t < 2).
- QoIs: displacement of B = (0.9, 0.2, 0.3) and drag and lift on the cylinder
  and the beam; reference values as mean ± amplitude over t in [9, 10] in
  Fig. 3, for k = 0.004, 0.002 and 0.001 on three mesh levels.
- The paper uses hexahedral Q2 elements for all unknowns with LPS
  stabilization (63,826 dofs on level 1). Here: tetrahedra with P2 u, v, z and
  P1 p (Taylor-Hood), as in 2d.

## Mesh: `scripts/create_mesh_FSI3D.py`

- gmsh OCC: cut the cylinder out of the channel and the beam, fragment them so
  that the interface is conforming. The cell and facet tags are those of the
  2d FSI2 mesh, so the solver's `PHYSICAL_MARKERS` are unchanged; the four
  walls are `channel_side`.
- B lies in the middle of the beam edge x = 0.9, z = 0.3, which is not a mesh
  node in general. Fragmenting the point B into the geometry gives a vertex
  there, so `find_point_dof` finds it as in 2d.
- OCC pads bounding boxes by 1e-7, so the wall test uses `atol=1e-6`.
- Size field: Threshold on the distance to the beam and the cylinder,
  `resolution_close` on them up to `resolution_far` at a distance of 0.2.
- Second order geometry with gmsh's high-order optimization. The coarse mesh
  (`data/meshes/fsi3d/mesh_sec_coarse.xdmf`, 0.02 / 0.08) has 15,142 tets
  (1,624 in the beam, about one cell through its thickness) and 3,386
  vertices. Checked: positive det J at degree-6 quadrature points in all
  cells, beam volume 0.0014013 and fluid volume 0.46606 as computed by hand,
  obstacle area 0.12478.

## Solver: `solvers/threeDfsi_iterative.py`

A copy of the 2d solver with the 3d changes only: function spaces with 3
components, the 3d inflow and parameters, 6 rigid body modes (3 translations,
3 rotations) for the GAMG near-nullspace, block size 3, the point B and a
QoI header with B_x, B_y, B_z (`QOI_HEADER_3D`). `drag_lift_forms` now takes
the drag along -e_x and the lift along e_y in any dimension, unchanged in 2d.

### Quadrature degree

The first runs took 42 s per step on 4 ranks, 17 s on 8 ranks, with
2 Newton iterations per step and a single Jacobian assembly for the whole run.
`-log_view` (3 steps, 8 ranks, iterative K_c): 50 s of 71 s in 9 residual
evaluations, 7 s in one Jacobian assembly, 3 s in all linear solves. UFL
estimates degree 19 for the nonlinear ALE and St. Venant-Kirchhoff terms of
the momentum and continuity residuals (25 in the v-v Jacobian block), while the
u and z blocks stay at 5 to 7. Quadrature points per tetrahedron: degree 8:
45, degree 13: 146, degree 16: 729, degree 19: 1000, degree 25: 2197.

`tools/quadrature.py`: `cap_quadrature_degree(form, max_degree)` sets the
quadrature degree to `max_degree` in the integrals whose estimate, computed
with `ufl.algorithms.compute_form_data` as in
https://jsdokken.com/FEniCS-workshop/src/unified_form_language/ufl_forms.html,
is higher, and leaves the others alone. The solver caps the residual blocks
before `derivative_block`, so the Jacobian blocks inherit the cap (checked: no
block above 8), plus `residual_post_u`, A_0 and the drag/lift forms.
`max_quadrature_degree=8` by default.

With the cap, 8 ranks, 5 steps of dt = 0.004:

| K_c solve | first step | later steps |
|---|---|---|
| MUMPS LU | 15 s | 1.2 s |
| FGMRES, block upper Schur, GAMG + Cahouet-Chabard | 12 s | 1.5 s |

The first step includes the MUMPS factorization of K_m (and of K_c in the
direct solve). The iterative K_c solve takes about 22 FGMRES iterations in the
first Newton iteration of a step and about 90 to 110 in the second. On this
coarse mesh the direct K_c solve is as fast; the iterative one should pay off
on finer meshes. The QoIs over these 5 steps agree with the uncapped run to all 6
printed digits, but the deformation is still tiny (|u_B| ~ 1e-6), so this
says little about the effect of the cap at large deformations.

### Teardown error (fixed)

`CahouetChabard.destroy()` clashed with petsc4py's `destroy(pc)` hook for
python PC contexts, which raised an ignored `TypeError` when the PC was
deallocated (in 2d too). Harmless, since the explicit cleanup had run by then.
Fixed by renaming the explicit cleanup to `destroy_solvers()`, so petsc4py
finds no hook; `tests/test_cahouet_chabard.py` destroys a python PC with the
context.
