import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The root-level driver scripts (dfg_2d_3.py, fsi2_harmonic.py, ...) are not
# part of the installable xfsi_solver package, so make them importable by
# module name for the tests.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
