import functools
import importlib
import inspect
import os
import pkgutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Packages whose modules' solve() may take snes_converged_reason_on.
SOLVER_PACKAGES = ["xfsi_solver.solvers", "xfsi_solver.component_benchmarks"]


def _without_converged_reason(solve):
    """``solve`` with ``snes_converged_reason_on=False`` unless a test passes it."""

    @functools.wraps(solve)  # keeps __module__, from which restart_helpers runs solve in mpiexec
    def wrapped(*args, **kwargs):
        kwargs.setdefault("snes_converged_reason_on", False)
        return solve(*args, **kwargs)

    return wrapped


def pytest_configure(config):
    """Turn off the SNES converged reason of every solver in the tests.

    Printing it makes PETSc keep a stdout viewer for the communicator of each mesh until
    PetscFinalize, in a table of fixed size. The test suite creates a few hundred meshes in one
    process, which overflows the table in the optimized PETSc build, and the process crashes in
    a later test or at exit. This runs before the test modules are imported, so they import the
    wrapped solve(). The solves in mpiexec subprocesses keep the default, since those processes
    are short.
    """
    for package_name in SOLVER_PACKAGES:
        package = importlib.import_module(package_name)
        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package_name}.{info.name}")
            solve = getattr(module, "solve", None)
            if solve is not None and "snes_converged_reason_on" in inspect.signature(solve).parameters:
                module.solve = _without_converged_reason(solve)


def pytest_runtest_logstart(nodeid, location):
    """Print, flushed immediately, so a hanging test is visible as it runs."""
    print(f"\n>>> RUNNING {nodeid}", flush=True)


def pytest_runtest_logreport(report):
    if report.when == "call":
        elapsed = getattr(report, "duration", None)
        elapsed_str = f" ({elapsed:.1f}s)" if elapsed is not None else ""
        print(f">>> {report.outcome.upper()} {report.nodeid}{elapsed_str}", flush=True)


@pytest.fixture
def output_dirs(tmp_path):
    """pv/qoi/figures directories for a test's solve() outputs.

    By default outputs go to a fresh pytest tmp_path and are discarded like
    any other test artifact. Set XFSI_KEEP_TEST_OUTPUT=1 to instead write
    into output/test/{pv,qoi,figures} so the results can be inspected
    (e.g. in ParaView/pyvista) after the test run.
    """
    if os.environ.get("XFSI_KEEP_TEST_OUTPUT"):
        base = ROOT / "output" / "test"
    else:
        base = tmp_path

    dirs = {}
    for name in ("pv", "qoi", "figures"):
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d
    return dirs
