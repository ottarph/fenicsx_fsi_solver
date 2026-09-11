import numpy as np
import pytest

from xfsi_solver.remeshing.discrete_mesh import SizingField
from xfsi_solver.remeshing.domain import load_fsi2_domain
from xfsi_solver.remeshing.loop import run_prescribed_deformation_loop

# Coarse-ish on purpose to keep the tests fast; not the production sizing.
TEST_SIZING = SizingField(size_near=0.02, size_far=0.06, size_outflow=0.08)


@pytest.fixture
def initial_domain():
    return load_fsi2_domain("data/meshes/fsi2/mesh.xdmf")


def test_loop_runs_without_remeshing_when_deformation_stays_small(initial_domain):
    result = run_prescribed_deformation_loop(
        initial_domain, step_amplitudes=[0.005] * 5, quality_threshold=0.35, sizing=TEST_SIZING
    )
    assert result.n_remesh_events == 0
    assert result.domain is initial_domain
    assert len(result.steps) == 5
    # Quality should be monotonically decreasing while nothing remeshes.
    qualities = [s.min_quality for s in result.steps]
    assert all(a >= b for a, b in zip(qualities, qualities[1:], strict=False))


def test_loop_triggers_remesh_and_quality_recovers(initial_domain):
    """Calibrated against data/meshes/fsi2/mesh.xdmf (see
    notes/remeshing/implementation-plan.md): quality first drops below 0.35
    around the 11th step of 0.005.
    """
    result = run_prescribed_deformation_loop(
        initial_domain, step_amplitudes=[0.005] * 12, quality_threshold=0.35, sizing=TEST_SIZING
    )
    assert result.n_remesh_events >= 1
    remesh_index = next(i for i, s in enumerate(result.steps) if s.remeshed)
    # Quality right after the remesh (next step's check) should be clearly
    # better than the value that triggered it -- the whole point of
    # remeshing being that it "resets" mesh quality.
    if remesh_index + 1 < len(result.steps):
        assert result.steps[remesh_index + 1].min_quality > result.steps[remesh_index].min_quality
    assert result.domain is not initial_domain
    assert result.domain.mesh.topology.index_map(2).size_local > 0


def test_loop_extends_deformation_beyond_single_segment_limit(initial_domain):
    """The whole point of remeshing: a single, un-remeshed segment starts
    inverting cells somewhere between total amplitude 0.05 and 0.08 (see
    test_remeshing_quality.py's DEGENERATE_AMPLITUDE). Stepping well past
    that (0.125 total, in small increments, with remeshing -- calibrated to
    trigger roughly every 10-11 steps, see notes/remeshing/implementation
    -plan.md) must complete without error and with multiple remesh events
    -- not just one.
    """
    result = run_prescribed_deformation_loop(
        initial_domain, step_amplitudes=[0.005] * 25, quality_threshold=0.35, sizing=TEST_SIZING
    )
    assert result.n_remesh_events >= 2
    assert all(s.min_quality > 0 for s in result.steps)


def test_loop_carries_field_across_remesh_events(initial_domain):
    """The "interpolate known functions between the meshes" deliverable
    (notes/remeshing/implementation-plan.md §7 Phase 4): the carried field
    should survive remeshing as a sensible, non-degenerate CG1 function --
    not empty, not all-zero, not NaN.
    """
    result = run_prescribed_deformation_loop(
        initial_domain, step_amplitudes=[0.005] * 12, quality_threshold=0.35, sizing=TEST_SIZING
    )
    assert result.n_remesh_events >= 1
    values = result.carried_field.x.array
    assert values.shape == result.carried_field.function_space.tabulate_dof_coordinates()[:, 0].shape
    assert np.all(np.isfinite(values))
    assert np.any(values != 0.0)


def test_loop_raises_instead_of_hanging_when_step_is_too_coarse(initial_domain):
    """A single step straight to a clearly-inverted amplitude (see
    test_remeshing_quality.py: 0.1 already has inverted cells) must raise a
    clear, fast error -- not hang trying to remesh a self-intersecting
    boundary (see discrete_mesh.py's docstring)."""
    with pytest.raises(RuntimeError, match="inverted cells"):
        run_prescribed_deformation_loop(
            initial_domain, step_amplitudes=[0.2], quality_threshold=0.35, sizing=TEST_SIZING
        )
