import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from xfsi_solver.scripts.plot_qois import plot_qoi  # noqa: E402


@pytest.fixture
def qoi_files(tmp_path):
    t = np.linspace(0.0, 2.0, 21)
    run = tmp_path / "run.txt"
    np.savetxt(run, np.column_stack([t, 2 * t, 3 * t, 4 * t, 5 * t]), delimiter="\t", header="t\tdrag\tlift\tA_x\tA_y")
    # The reference has its columns in another order than the run
    reference = tmp_path / "reference.txt"
    np.savetxt(reference, np.column_stack([t, -t, -2 * t]), delimiter="\t", header="t\tA_y\tdrag")
    return run, reference


def _plotted_lines(monkeypatch):
    """Record the legend entries and the (x, y) of the lines of every figure plot_qoi saves."""
    saved = []

    def savefig(path, *args, **kwargs):
        legend = plt.gca().get_legend()
        entries = [text.get_text() for text in legend.get_texts()] if legend is not None else []
        saved.append((entries, [(line.get_xdata(), line.get_ydata()) for line in plt.gca().get_lines()]))

    monkeypatch.setattr(plt, "savefig", savefig)
    return saved


def test_plot_qoi_picks_the_columns_by_name(qoi_files, monkeypatch):
    run, reference = qoi_files
    saved = _plotted_lines(monkeypatch)
    plot_qoi("A_y", run, run.parent / "a_y.pdf", reference_file=reference)
    ((entries, lines),) = saved
    (t, a_y), (t_ref, a_y_ref) = lines
    assert entries == ["reference"]
    np.testing.assert_allclose(a_y, 5 * t)
    np.testing.assert_allclose(a_y_ref, -t_ref)


def test_plot_qoi_restricts_to_the_time_interval(qoi_files, monkeypatch):
    run, _ = qoi_files
    saved = _plotted_lines(monkeypatch)
    plot_qoi("drag", run, run.parent / "drag.pdf", label="mine", time_interval=(0.5, 1.0))
    ((entries, ((t, drag),)),) = saved
    assert entries == ["mine"]
    np.testing.assert_allclose(t, np.linspace(0.5, 1.0, 6))
    np.testing.assert_allclose(drag, 2 * t)


def test_plot_qoi_without_labels_has_no_legend(qoi_files, monkeypatch):
    run, _ = qoi_files
    saved = _plotted_lines(monkeypatch)
    plot_qoi("drag", run, run.parent / "drag.pdf")
    ((entries, _),) = saved
    assert entries == []


def test_plot_qoi_writes_the_figure_in_a_new_folder(qoi_files):
    run, reference = qoi_files
    for suffix in (".pdf", ".svg"):
        path = run.parent / "figures" / f"drag{suffix}"
        plot_qoi("drag", run, path, reference_file=reference, ylabel="Drag")
        assert path.is_file()


def test_plot_qoi_skips_a_missing_qoi_file_and_reference(qoi_files):
    run, _ = qoi_files
    plot_qoi("drag", run.parent / "missing.txt", run.parent / "none.pdf")
    assert not (run.parent / "none.pdf").exists()
    plot_qoi("drag", run, run.parent / "drag.pdf", reference_file=run.parent / "missing.txt")
    assert (run.parent / "drag.pdf").is_file()


def test_plot_qoi_rejects_a_qoi_not_in_the_header(qoi_files):
    run, reference = qoi_files
    with pytest.raises(ValueError, match="No column 'lift'"):
        plot_qoi("lift", run, run.parent / "lift.pdf", reference_file=reference)
