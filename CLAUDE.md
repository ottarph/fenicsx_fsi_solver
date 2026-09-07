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
- Add `Co-authored-by: Claude <noreply@anthropic.com>` to every commit substantially produced by Claude Code.
