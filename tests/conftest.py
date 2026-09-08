import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


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
