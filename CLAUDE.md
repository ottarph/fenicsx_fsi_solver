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
- The `scaled_jacobian` mesh-quality measure (e.g. via PyVista/Verdict's `cell_quality`, as used in `src/xfsi_solver/remeshing/quality.py`) is implemented so that it is **non-negative for triangle cells regardless of orientation**, due to how Verdict computes it. This means it can never be used, by itself, to detect an inverted/degenerate triangle — it will not go negative or hit zero the way it does for other cell types. It is fine for tracking relative mesh quality (e.g. a remeshing trigger threshold), but any check for actual cell inversion on a triangle mesh must use an independent method, such as signed cell area (or `det(F)` in a solve context), never `scaled_jacobian` alone.
- Import `dolfinx.fem.petsc` explicitly right below `import dolfinx as dfx`, and reference it as `dfx.fem.petsc.X` (not a separate alias like `dfpetsc`):
  ```python
  import dolfinx as dfx
  import dolfinx.fem.petsc  # noqa: F401

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
