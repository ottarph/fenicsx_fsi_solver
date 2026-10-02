# Environment

Choose the Conda environment `xfsi_solver` at the start of each task.
Then run commands using the selected environment, for example:

```bash
conda run -n xfsi_solver pytest
```

# Code
Use FEniCSx to solve fluid-structure interaction problems using the finite element method in monolithic arbitrary Lagrangian-Eulerian formulation. Use the class ``dolfinx.fem.petsc.NonlinearProblem`` to solve the nonlinear problems with Newton's method. For any linear problems, use the class ``dolfinx.fem.petsc.LinearProblem``. Do NOT use ``dolfinx.fem.petsc.NewtonSolverNonlinearProblem``, as this is for the old API.

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