# Environment

The dependencies come from the conda environment `fenicsx-env`, built from `environment.yml`, plus the packages
in `requirements-pip.txt`, which are installed into it with `pip install --no-deps`. This package is installed
editable in a virtualenv `.venv/` in the checkout, made from `fenicsx-env`'s Python with
`--system-site-packages`. Run every command through the conda environment with the venv's Python, for example:

```bash
conda run --no-capture-output -n fenicsx-env .venv/bin/python -m pytest
```

- Without `--no-capture-output`, `conda run` buffers the whole child process's stdout/stderr and only prints it once
  the process exits, which hides progress on long-running commands (e.g. the test suite) until they finish.
- Bare `python` or `pytest` resolve to the conda environment, which does not have this package installed. Always
  use `.venv/bin/python`.
- Each checkout and each git worktree has its own `.venv/`. If it is missing, or built on another environment,
  create it (the README has the commands):
  ```bash
  ~/miniforge3/envs/fenicsx-env/bin/python -m venv --clear --system-site-packages .venv
  .venv/bin/pip install --no-deps -e .
  readlink -f .venv/bin/python  # must be fenicsx-env's Python
  ```
  Always pass `--clear`: over an existing `.venv/`, `venv` without it rewrites `pyvenv.cfg` but keeps the old
  `bin/python` link, so the venv silently keeps running the interpreter it was first built on.
- Never install anything else into `fenicsx-env` with pip, and always use `--no-deps`: h5py must stay conda's MPI
  build (`h5py.get_config().mpi` is `True`). Ask the user before changing `fenicsx-env`; other projects use it.

# Tests

- Run the tests marked `slow` only with the user's explicit permission. These are the 3D solver runs, which take
  minutes each. This covers `pytest -m slow`, `pytest -m ""`, and selecting a slow test by name with `-k`. A plain
  `pytest` leaves them out (`addopts = "-m 'not slow'"` in `pyproject.toml`). When a change needs them, for example
  for the restart and initial-state paths of the 3D solvers, ask first.
- Pass this rule on in the brief of any session or subagent that may run tests.
- `pytest-xdist` is installed in `fenicsx-env`. `pytest -n 8` runs the tests in 8 serial worker processes
  (`-n auto` uses one per CPU, `--dist loadfile` keeps each file on one worker). It speeds up the whole suite, not
  a single test: every worker is one rank. Tests that launch their own `mpiexec` still work but briefly use extra
  cores, so leave cores free and follow the core-count rules under "Running MPI simulations". Tests must write
  only under `tmp_path`, never to a fixed path, or workers can collide.

# Knowledge library

What is known about this code (results, decisions, runs, pitfalls) is kept in a separate, private library at
`~/fsi-library`, not in this repository. Look things up there first: start at its `index.md`, and search its
`inbox/` as well. The design logs that used to be in `notes/` are summarised there, and the originals can be read
from git history (`git show 32bd021:notes/<path>`). Two user-level skills hold the procedures: `fsi-lookup` for
looking things up, and `fsi-record-finding` for writing an inbox entry.

- A `notes/` folder may be used sparingly on a feature branch, as a working log for a long task. It must never be
  merged into `restricted` or `main`: before the branch is merged, move its conclusions to the library's inbox (cite
  the log by commit if needed) and delete the folder in a commit on the branch.
- When you learn something a future session would need (a result with its numbers, a decision and why, an approach
  that failed, a pitfall, a run worth finding again), commit the code under test, then write one file per finding
  to `~/fsi-library/inbox/` as described in `~/fsi-library/inbox/README.md` (use the `fsi-record-finding` skill).
- Don't edit the library's pages and don't commit there; a librarian merges the inbox.
- Code comments and docstrings in this repository must stay self-contained, since this repository is public and the
  library is not. Don't cite the library from them.

# Code
Use FEniCSx to solve fluid-structure interaction problems using the finite element method in monolithic arbitrary Lagrangian-Eulerian formulation. 

## Conventions
- Use the class ``dolfinx.fem.petsc.NonlinearProblem`` to solve the nonlinear problems with Newton's method. For any linear problems, use the class ``dolfinx.fem.petsc.LinearProblem``. Do NOT use ``dolfinx.fem.petsc.NewtonSolverNonlinearProblem``, as this is for the old API.
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
- Add `Co-authored-by: Claude <noreply@anthropic.com>` to every commit substantially produced by Claude Code.


# Running MPI simulations

## Before starting
- Check how many physical cores the machine has, counting cores, not hardware
  threads: `lscpu` on Linux. On Apple silicon, count only the performance cores
  (`sysctl -n hw.perflevel0.physicalcpu`): ranks on the slower efficiency cores
  hold back the other ranks, so efficiency cores are not used for MPI. Check the
  current load (`uptime`) and the MPI runs already going, including other
  users' (`ps -eo pid,etime,pcpu,args | grep "[p]ython"`).
- Do not oversubscribe. The total number of MPI ranks across all runs,
  including runs already going, must leave two physical cores free, or one core
  free if the machine has six or fewer such cores. If that leaves too few
  cores for the runs you want, ask the user rather than share cores.
- Always set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` for MPI runs, so
  each rank uses one core.
- Give every run its own output paths (qoi, logs, VTX `.bp`) so runs do not
  overwrite existing results or each other. Check the free disk space before
  long runs that write VTX output.

## Launching runs longer than a few minutes
- Session background tasks are stopped after about 30 minutes, so do not use
  them for long runs. Detach the run from the session instead and write all
  output to a log file:
  ```bash
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 setsid nohup \
      conda run --no-capture-output -n fenicsx-env \
      mpirun -n <ranks> .venv/bin/python -u <script> > output/logs/<run>.txt 2>&1 < /dev/null &
  echo $! > output/logs/<run>.pid
  ```
  `setsid` is not available on macOS; use `nohup ... &` there.
- Use `python -u` (or `PYTHONUNBUFFERED=1`) so progress reaches the log as it
  happens.
- Record the PID (or process group) of every run you start, so it can be
  checked or stopped without touching anything else.
- Short tests (a few time steps) may run in the foreground, but always wrap them
  in `timeout`, e.g. `timeout 300 mpirun -n 4 ...`.

## Watching runs
- Watch the logs with the Monitor tool, filtering for progress, completion and
  all failure signatures (`Traceback`, `ERROR`, `Killed`, `did not converge`,
  `DIVERGED`, the final `Elapsed time` line). Re-arm the monitor when it
  expires.
- Treat a run as hung when its log has had no new time step for 1 minute while
  its ranks still use 100% CPU. This is the usual sign of an MPI deadlock, e.g.
  a collective call (`norm()`, `allreduce`, assembly) reached by only some
  ranks. Mesh loading and form compilation before the first time step can take
  longer, so do not apply this threshold before the first time step is logged.
- On a hang, check the log and `ps` first, then stop only that run, by its
  recorded PID or process group (`kill -- -<pgid>`). Report it to the user; do
  not silently restart it.
- Never use `pkill -f <pattern>` or `killall python`: the pattern can match
  your own shell's command line or other people's runs.

## Reporting
- Report the state of every run: finished, failed, stopped or still running,
  and how far it got. Never present partial results as complete.
