# Environment

Choose the Conda environment `xfsi_solver` at the start of each task.
Then run commands using the selected environment, for example:

```bash
conda run -n xfsi_solver pytest
```

# Tests

- Run the tests marked `slow` only with the user's explicit permission. These are the 3D solver runs, which take
  minutes each. This covers `pytest -m slow`, `pytest -m ""`, and selecting a slow test by name with `-k`. A plain
  `pytest` leaves them out (`addopts = "-m 'not slow'"` in `pyproject.toml`). When a change needs them, for example
  for the restart and initial-state paths of the 3D solvers, ask first.

# Knowledge library

What is known about this code (results, decisions, runs, pitfalls) is kept in a separate, private library at
`~/fsi-library`, not in this repository. Look things up there first: start at its `index.md`, and search its
`inbox/` as well. The design logs that used to be in `notes/` are summarised there, and the originals can be read
from git history (`git show 32bd021:notes/<path>`).

- A `notes/` folder may be used sparingly on a feature branch, as a working log for a long task. It must never be
  merged into `restricted` or `main`: before the branch is merged, move its conclusions to the library's inbox (cite
  the log by commit if needed) and delete the folder in a commit on the branch.
- When you learn something a future session would need (a result with its numbers, a decision and why, an approach
  that failed, a pitfall, a run worth finding again), commit the code under test, then write one file per finding
  to `~/fsi-library/inbox/` as described in `~/fsi-library/inbox/README.md`.
- Don't edit the library's pages and don't commit there; a librarian merges the inbox.
- Code comments and docstrings in this repository must stay self-contained, since this repository is public and the
  library is not. Don't cite the library from them.

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