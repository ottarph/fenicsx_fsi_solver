
This repository contains a ``FEniCSx``-implementaion of a fully monolithic arbitrary Lagrangian-Eulerian 
fluid-structure interaction solver verified on the FSI2 benchmark by Turek and Hron, 2006.

The solver is compatible with both triangular and quadrilateral meshes and different choices of elements, 
for instance ``P2-P2-P1``, ``Q2-Q2-Q1``, ``Q2-Q2-DG1``, or ``Q2-Q2-DPC1``.

The code is licensed under an MIT-license found in ``LICENSE``. Certain functionalities are modifications 
of code authored by Jørgen S. Dokken, licensing and copyright information for this code is given in the 
relevant files.

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
