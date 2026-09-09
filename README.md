# FEniCSx FSI Solver
This repository contains a ``FEniCSx``-implementaion of a fully monolithic arbitrary Lagrangian-Eulerian 
fluid-structure interaction solver verified on the FSI2 benchmark by Turek and Hron, 2006.

The solver is compatible with both triangular and quadrilateral meshes and different choices of elements, 
for instance ``P2-P2-P1``, ``Q2-Q2-Q1``, ``Q2-Q2-DG1``, or ``Q2-Q2-DPC1``.

The code is licensed under an MIT-license found in ``LICENSE``. Certain functionalities are modifications 
of code authored by Jørgen S. Dokken, licensing and copyright information for this code is given in the 
relevant files.

## Formulation

The monolithic FSI equations and their one-step-theta time discretization are based on the exposition in:

> T. Wick, "Fluid-structure interactions using different mesh motion techniques," *Computers & Structures*,
> vol. 89, no. 13-14, pp. 1456-1467, 2011. https://doi.org/10.1016/j.compstruc.2011.02.019

As in that paper, the weak-form residual is assembled from terms grouped into four categories: time
derivative terms, pressure terms, terms that must be treated fully implicitly, and the remaining
(theta-weighted) terms. In code, these are the ``A_T(...)``, ``A_P(...)``, ``A_I(...)``, and ``A_E(...)``
functions defined in each monolithic solver (e.g. ``src/xfsi_solver/solvers/fsi2_harmonic.py``).

The time-discretization of monolithic fluid-structure interaction problems is complicated by, amongst other things, stability issues and discretization of terms like

$$
(J(u) \partial_t v, \phi)_{L^2} \tag{1}
$$
where 
$$
\mathrm{F}(u) = \mathrm{Id} + \nabla u, J(u) = \mathrm{det}(\mathrm{F}(u)).
$$

The value of $J(u)$ when approximating (1) can be chosen in different ways. Following the paper by Wick, we approximate (1) by the midpoint treatment

$$
(J(\bar u) (v^n - v^{n-1}) / \mathrm d t, \phi)_{L^2}, \quad \bar{u} = \frac{u^n + u^{n-1}}{2},
$$
and handle the other similar terms fully implicitly. Furthermore we use the shifted Crank-Nicolson method for stability and second order accuracy. A comprehensive treatment of time discretization for monolithic fluid-structure interaction is found in:

> T. Richter, *Fluid-structure Interactions: Models, Analysis and Finite Elements*, Lecture Notes in
> Computational Science and Engineering, vol. 118, Springer, Cham, 2017, Section 5.1.2.
> https://doi.org/10.1007/978-3-319-63970-3

## Installation

The non-Python dependencies (``fenics-dolfinx``, ``mpich``, ``gmsh``, ``adios2``, ``scifem``, ...) are
conda-only and are managed through ``environment.yml``. Create the ``xfsi_solver`` conda environment and
install this package into it in editable mode:

```bash
conda env create -n xfsi_solver -f environment.yml
conda run -n xfsi_solver pip install -e ".[dev]"
```

## Running the tests

``data/`` is gitignored, so the coarse meshes the test suite runs on don't ship with the repo and must be
generated locally first:

```bash
conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.scripts.create_test_meshes
```

Then run the tests:

```bash
conda run --no-capture-output -n xfsi_solver pytest
```

Note the ``--no-capture-output`` flag: without it, ``conda run`` buffers the whole test run's output and
only prints it once every test has finished, instead of showing progress (including which test is
currently running) as it happens.

Set ``XFSI_KEEP_TEST_OUTPUT=1`` to keep each test's solver output under ``output/test/`` for inspection
(e.g. in ParaView/PyVista) instead of discarding it after the run.

## Running a solver

Each solver in ``src/xfsi_solver/solvers/`` and ``src/xfsi_solver/component_solvers/`` exposes a
``solve(...)`` function and a ``main()`` that calls it with example arguments. Run one directly as a
module, from the repository root:

```bash
conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.solvers.fsi2_harmonic
```

Some solvers expect mesh/reference data to already exist under ``data/`` (generated via the scripts in
``src/xfsi_solver/scripts/``) before they can run.

## Reference data

``plot_qois.py`` compares solver output against the published FSI2 benchmark reference values, which
aren't included in this repository. To generate ``data/fsi2_reference.txt``:

1. Download ``ref_fsi2.point`` from the FeatFlow FSI benchmark reference page:
   https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_reference.html,
   and place it at the repository root.
2. Run:
   ```bash
   conda run --no-capture-output -n xfsi_solver python -m xfsi_solver.scripts.prepare_FSI2_reference_values
   ```

## Solver naming

Solver filenames are built from a base name (the benchmark or physics being solved, e.g. ``fsi2``,
``dfg_2d_3``, ``navier_stokes``, ``solid_elasticity``) plus suffixes describing how that base problem is
set up:

- ``static``: a single steady-state solve, instead of a time-stepping loop.
- ``ALE``: an arbitrary Lagrangian-Eulerian (moving-mesh) formulation.
- ``mm``: the ALE mesh-motion equation is solved monolithically together with the flow at every time
  step, instead of being precomputed once for the whole time series and read off a lookup table (as the
  plain ``_ALE_fsi2`` variant does).
- ``mm_bih``: like ``mm``, but the mesh motion uses the biharmonic (fourth-order) extension equation from
  ``biharm.py`` instead of the plain harmonic (Laplace) extension used by ``mm``.
- ``fullmesh``: interface/traction measures between the fluid and solid subdomains are built directly on
  the full parent mesh (using dolfinx's ``entity_maps``), instead of on submesh-transferred facet tags.
- ``diffmesh``: different fields of the same coupled problem are defined on *different* meshes, instead
  of all sharing one mesh. For example, in ``fsi2_harmonic.py`` the pressure function space is defined on
  the whole mesh (like displacement and velocity), whereas in ``fsi2_harmonic_diffmesh.py`` pressure is
  defined only on a submesh of the fluid domain — since pressure is only physically meaningful there.
  ``diffmesh`` variants require dolfinx's mixed-mesh assembly (``entity_maps``) to combine forms whose
  fields live on different meshes.
