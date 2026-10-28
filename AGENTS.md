# Environment

Choose the Conda environment `xfsi_solver` at the start of each task.
Then run commands using the selected environment, for example:

```bash
conda run -n xfsi_solver pytest
```

## DOLFINx main: the `fenicsx-dev` env and `.venv`

The `xfsi_solver` env has the DOLFINx release from conda-forge (0.11). Use the DOLFINx main branch only when
a task needs a feature that is not in the release yet, such as the edge weights of the graph partitioners in
the `mwe_partition_*.py` scripts. For that, the repository's `.venv` (git-ignored) is a virtualenv on the
conda env `fenicsx-dev`, which has FEniCSx built from source with
https://github.com/ottarph/conda-dolfinx-dev-env. The commits that were built are listed in
`$CONDA_PREFIX/share/fenicsx-build-info.txt` of `fenicsx-dev`.

Run the tests and scripts with the Python of `.venv`, and MPI runs with the `mpirun` of `fenicsx-dev`.
Activating `.venv` alone puts the system's `/usr/bin/mpirun` (another MPICH) on the path, not the env's:

```bash
.venv/bin/python -m pytest
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n fenicsx-dev mpirun -n 4 .venv/bin/python -u <script>
```

To build `fenicsx-dev` from scratch, in a checkout of conda-dolfinx-dev-env (takes about 15 minutes; run it
detached, and leave cores free with `--jobs`):

```bash
conda env remove -n fenicsx-dev -y
conda env create -f environment.yml
conda activate fenicsx-dev
# Before the build, so that DOLFINx is built against the final libadios2:
conda install -c conda-forge "h5py=*=mpi_mpich*" "adios2=*=mpi_mpich*" tqdm ruff
./build-fenicsx.sh --clean --jobs 24 main
# scifem compiles against the DOLFINx just built
python -m pip install --no-deps --no-build-isolation scifem io4dolfinx fenicsx-compat
python -m pip check
```

Then recreate `.venv` (a rebuilt env invalidates the old one), with xfsi_solver installed editable and
without dependencies, which all come from `fenicsx-dev`:

```bash
rm -rf .venv
~/miniforge3/envs/fenicsx-dev/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps -e .
```

- Never `conda install` FEniCSx packages, or packages that depend on them (`fenics-*`, `scifem`,
  `io4dolfinx`, `fenicsx-compat`), into `fenicsx-dev`: conda would replace the source build with the
  conda-forge release. Install these with pip and `--no-deps`.
- Do not install packages into `.venv`; add them to `fenicsx-dev`.
- Keep the code in `src/` working on both the release and main. When the API differs, prefer a form that
  works on both, for example keyword arguments, as for `dolfinx.mesh.transfer_meshtags_to_submesh`, whose
  `vertex_to_parent` and `cell_to_parent` are in a different order on main.

# Tests

- Run the tests marked `slow` only with the user's explicit permission. These are the 3D solver runs, which take
  minutes each. This covers `pytest -m slow`, `pytest -m ""`, and selecting a slow test by name with `-k`. A plain
  `pytest` leaves them out (`addopts = "-m 'not slow'"` in `pyproject.toml`). When a change needs them, for example
  for the restart and initial-state paths of the 3D solvers, ask first.

# Code
Use FEniCSx to solve fluid-structure interaction problems using the finite element method in monolithic arbitrary Lagrangian-Eulerian formulation. Use the class ``dolfinx.fem.petsc.NonlinearProblem`` to solve the nonlinear problems with Newton's method. For any linear problems, use the class ``dolfinx.fem.petsc.LinearProblem``. Do NOT use ``dolfinx.fem.petsc.NewtonSolverNonlinearProblem``, as this is for the old API.

## Conventions
- Import dolfinx without an alias, and import `dolfinx.fem.petsc` explicitly right below it. Always refer to dolfinx
  with its full name (`dolfinx.fem.X`, `dolfinx.fem.petsc.X`), never as `dfx` or another alias: editors resolve
  hover hints for `dolfinx.fem.petsc` only through the full name.
  ```python
  import dolfinx
  import dolfinx.fem.petsc
  ```
- Format with `ruff format` (line length 120, set in `pyproject.toml`) before committing.

## Git workflow

- Work on the task-provided feature branch. Never commit directly to the default branch.
- For substantial changes, create multiple logical, reviewable commits instead of one final monolithic commit.
- Each commit should represent one coherent change and should pass the relevant tests when practical.
- Keep implementation and its associated tests in the same commit.
- Do not create arbitrary checkpoint or "WIP" commits.
- Do not commit unrelated or pre-existing changes.
- Use concise imperative commit messages, for example:
  - Add database migration for notification preferences
  - Implement notification preference service
  - Expose preference API endpoints
  - Add integration tests and documentation
- Before finishing, review the commit sequence with `git log --oneline` and report it in the final summary.
- Do not amend, squash, rebase, or force-push unless explicitly requested.


### Commit attribution

- Create commits using the Git identity already configured by the environment.
- Do not change `user.name` or `user.email`.
- Add `Co-authored-by: Codex <noreply@openai.com>` to every commit substantially produced by Codex.


# Running MPI simulations

- Before starting, check the number of physical cores, counting cores, not
  hardware threads (`lscpu` on Linux). On Apple silicon, count only the
  performance cores (`sysctl -n hw.perflevel0.physicalcpu`), since ranks on the
  slower efficiency cores hold back the others. Also check the
  current load (`uptime`), and MPI runs already going
  (`ps -eo pid,etime,pcpu,args | grep "[p]ython"`).
- Do not oversubscribe. The total number of MPI ranks across all concurrent
  runs must leave two of these cores free, or one core free if the machine has
  six or fewer of them. Ask the user rather than share cores.
- Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` for every MPI run.
- Give each run its own output paths for qoi files, logs and VTX output.
- Detach runs longer than a few minutes from the session (`setsid nohup ... &`
  on Linux, `nohup ... &` on macOS), send all output to a log file, use
  `python -u`, and record the PID.
- Wrap short test runs in `timeout`.
- Treat a run as hung when its log has had no new time step for 1 minute while
  its ranks use 100% CPU; this is usually an MPI deadlock. Mesh loading and form
  compilation before the first time step can take longer. Stop only the hung
  run, by its recorded PID or process group, and report it.
- Never use `pkill -f <pattern>` or `killall python`.
- Report the state of every run: finished, failed, stopped or still running,
  and how far it got.