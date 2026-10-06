# Environment

Choose the Conda environment `xfsi_solver` at the start of each task.
Then run commands using the selected environment, for example:

```bash
conda run --no-capture-output -n xfsi_solver pytest
```

Without `--no-capture-output`, `conda run` buffers the whole child process's
stdout/stderr and only prints it once the process exits, which hides
progress on long-running commands (e.g. the test suite) until they finish.

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
      conda run --no-capture-output -n xfsi_solver \
      mpirun -n <ranks> python -u <script> > output/logs/<run>.txt 2>&1 < /dev/null &
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
