from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from xfsi_solver.scripts.plot_qois import (  # noqa: E402
    FIGURE_DIR,
    FSI2_QOI_LABELS,
    INCOMPRESSIBLE_QOI_LABELS,
    LAGRANGE_QOI_LABELS,
    QOI_LABELS_3D,
    plot_qoi,
    plot_run_qois,
)
from xfsi_solver.tools.qoi import (  # noqa: E402
    INCOMPRESSIBLE_QOI_HEADER,
    LAGRANGE_QOI_HEADER,
    QOI_HEADER,
    QOI_HEADER_3D,
)


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


@pytest.mark.parametrize(
    ("labels", "header"),
    [
        (FSI2_QOI_LABELS, QOI_HEADER),
        (LAGRANGE_QOI_LABELS, LAGRANGE_QOI_HEADER),
        (INCOMPRESSIBLE_QOI_LABELS, INCOMPRESSIBLE_QOI_HEADER),
        (QOI_LABELS_3D, QOI_HEADER_3D),
    ],
    ids=["fsi2", "lagrange", "incompressible", "3d"],
)
def test_qoi_labels_are_the_columns_of_the_qoi_header(labels, header):
    assert ["t", *labels] == header.split("\t")


def _saved_figures(monkeypatch):
    """Record the path and the x-limits of every figure plot_qoi saves, without writing it."""
    saved = []
    monkeypatch.setattr(plt, "savefig", lambda path, *args, **kwargs: saved.append((path, plt.gca().get_xlim())))
    return saved


def test_plot_run_qois_writes_full_and_closeup_figures(qoi_files, tmp_path):
    run, reference = qoi_files
    figure_dir = tmp_path / "figures"
    plot_run_qois(run, 2.0, reference_file=reference, closeup_interval=(0.5, 1.0), figure_dir=figure_dir)
    expected = {f"{name}_{qoi}.pdf" for name in ("full", "closeup") for qoi in FSI2_QOI_LABELS}
    assert {path.name for path in figure_dir.iterdir()} == expected


def test_plot_run_qois_intervals(qoi_files, tmp_path, monkeypatch):
    run, _ = qoi_files
    saved = _saved_figures(monkeypatch)
    # The full figure reaches the end of the closeup interval when that is after T
    plot_run_qois(run, 1.0, reference_file=None, closeup_interval=(0.5, 3.0), figure_dir=tmp_path / "figures")
    intervals = {path.name: xlim for path, xlim in saved}
    assert intervals["full_drag.pdf"] == (0.0, 3.0)
    assert intervals["closeup_drag.pdf"] == (0.5, 3.0)
    saved.clear()
    # Without a closeup interval there is only the full figure, up to T
    plot_run_qois(run, 1.0, reference_file=None, closeup_interval=None, figure_dir=tmp_path / "figures")
    assert {path.name: xlim for path, xlim in saved} == {f"full_{qoi}.pdf": (0.0, 1.0) for qoi in FSI2_QOI_LABELS}


def test_plot_run_qois_skips_the_reference_for_qois_not_in_it(qoi_files, tmp_path, monkeypatch):
    run, reference = qoi_files  # the reference has only A_y and drag
    saved = []

    def savefig(path, *args, **kwargs):
        legend = plt.gca().get_legend()
        saved.append((Path(path).name, [text.get_text() for text in legend.get_texts()] if legend else []))

    monkeypatch.setattr(plt, "savefig", savefig)
    plot_run_qois(run, 2.0, reference_file=reference, reference_label="ref", closeup_interval=None, figure_dir=tmp_path)
    assert dict(saved) == {"full_drag.pdf": ["ref"], "full_lift.pdf": [], "full_A_x.pdf": [], "full_A_y.pdf": ["ref"]}


def test_plot_run_qois_figure_dir_defaults_to_the_folder_of_the_run(qoi_files, tmp_path, monkeypatch):
    run, _ = qoi_files
    monkeypatch.chdir(tmp_path)  # FIGURE_DIR is relative, and the folder is created
    saved = _saved_figures(monkeypatch)
    plot_run_qois(run, 2.0, reference_file=None)
    assert {path.parent for path, _ in saved} == {FIGURE_DIR / "run"}
