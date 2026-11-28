"""Ticket #122: the Compare table's F1 is the run's decision, and it says which.

The summary table used to divide the counters inline under an unlabelled "F1"
header, next to pages that meant something else by the same word. It now reads
``Score.from_decision`` under the localisation criterion, and its header names
that criterion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core.detection_run import ImageResult, ModelInfo, RunResult, RunStatus  # noqa: E402
from polyx.core.detection_run.operating_point import Criterion, Score  # noqa: E402
from polyx.gui.detector.pages.compare import ComparePage  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402
from tests.match_fixtures import misclassified_result, misclassified_run  # noqa: E402


def test_compare_summary_names_its_criterion(qapp, tmp_path, selected_language):
    selected_language("en")
    run = misclassified_run(tmp_path)
    result = run.results[0][0]
    state = DetectorState()
    state.model_slots[0].alias = run.models[0].alias
    state.model_slots[0].path = run.models[0].path
    state.params = run.params
    state.results = run.results
    page = ComparePage(state)
    page.refresh()

    localisation = Score.from_decision([result], Criterion.LOCALISATION, run.params.conf)
    assert page.tbl_summary.horizontalHeaderItem(4).text() == "True positives (localisation)"
    assert page.tbl_summary.horizontalHeaderItem(5).text() == "False positives (localisation)"
    assert page.tbl_summary.horizontalHeaderItem(6).text() == "False negatives (localisation)"
    assert page.tbl_summary.horizontalHeaderItem(7).text() == "F1 (localisation)"
    assert page.tbl_summary.item(0, 4).text() == str(localisation.tp)
    assert page.tbl_summary.item(0, 5).text() == str(localisation.fp)
    assert page.tbl_summary.item(0, 6).text() == str(localisation.fn)
    assert page.tbl_summary.item(0, 7).text() == f"{localisation.f1:.3f}"


def test_compare_shows_the_run_s_dense_models_not_the_current_slots(
    qapp, tmp_path, selected_language
):
    """One row per dense model that ran, labelled from the run record."""
    selected_language("en")
    first = tmp_path / "photos" / "a.png"
    second = tmp_path / "photos" / "b.png"
    state = DetectorState()
    # The Models tab after the run: two non-adjacent slots with other names.
    state.model_slots[0].alias = "slot_a"
    state.model_slots[0].path = tmp_path / "a.pt"
    state.model_slots[2].alias = "slot_c"
    state.model_slots[2].path = tmp_path / "c.pt"
    run = RunResult(
        run_dir=tmp_path / "run",
        models=[ModelInfo(alias="run_a"), ModelInfo(alias="run_c")],
        results={
            0: [misclassified_result(first, model_idx=0)],
            1: [misclassified_result(second, model_idx=1)],
        },
    )
    state.run_result = run
    state.results = run.results
    state.params = run.params
    page = ComparePage(state)
    page.refresh()

    assert page.tbl_summary.rowCount() == 2
    assert [page.tbl_summary.item(row, 0).text() for row in range(2)] == ["run_a", "run_c"]
    assert page.tbl_per_image.columnCount() == 3
    assert [
        page.tbl_per_image.horizontalHeaderItem(column).text()
        for column in range(page.tbl_per_image.columnCount())
    ] == ["Image", "run_a", "run_c"]
    assert page.tbl_per_image.rowCount() == 2
    assert [page.tbl_per_image.item(row, 0).text() for row in range(2)] == ["a.png", "b.png"]
    assert [page.tbl_per_image.item(row, 1).text() for row in range(2)] == ["2", "0"]
    assert [page.tbl_per_image.item(row, 2).text() for row in range(2)] == ["0", "2"]


def test_a_partial_run_without_a_record_renders_from_the_snapshot(
    qapp, tmp_path, selected_language
):
    """A run that streamed results and then died still renders labelled rows."""
    selected_language("en")
    first = tmp_path / "photos" / "a.png"
    state = DetectorState()
    # The Models tab after the failure no longer matches the models that ran.
    state.model_slots[0].alias = "renamed"
    state.model_slots[0].path = tmp_path / "renamed.pt"
    state.model_snapshot = [
        ModelInfo(alias="run_a", alias_generated=False),
        ModelInfo(alias="run_b", alias_generated=False),
    ]
    state.params.conf = 0.5
    state.results = {
        0: [ImageResult(image_path=first, model_idx=0)],
        1: [],
    }

    page = ComparePage(state)
    page.refresh()

    assert page.tbl_summary.rowCount() == 2
    assert [page.tbl_summary.item(row, 0).text() for row in range(2)] == ["run_a", "run_b"]
    assert [
        page.tbl_per_image.horizontalHeaderItem(column).text()
        for column in range(page.tbl_per_image.columnCount())
    ] == ["Image", "run_a", "run_b"]


def test_compare_rerenders_a_failed_partial_run(qapp, tmp_path, selected_language):
    """A failed run repaints the rows received so far under the snapshot labels."""
    selected_language("en")
    first = tmp_path / "photos" / "a.png"
    state = DetectorState()
    # The Models tab after the failure no longer matches the models that ran.
    state.model_slots[0].alias = "renamed"
    state.model_slots[0].path = tmp_path / "renamed.pt"
    page = ComparePage(state)

    state.model_snapshot = [
        ModelInfo(alias="run_a", alias_generated=False),
        ModelInfo(alias="run_b", alias_generated=False),
    ]
    state.results = {
        0: [misclassified_result(first, model_idx=0)],
        1: [],
    }
    state.run_failed.emit()

    assert page.tbl_summary.rowCount() == 2
    assert [page.tbl_summary.item(row, 0).text() for row in range(2)] == ["run_a", "run_b"]
    assert [
        page.tbl_per_image.horizontalHeaderItem(column).text()
        for column in range(page.tbl_per_image.columnCount())
    ] == ["Image", "run_a", "run_b"]
    assert page.tbl_per_image.rowCount() == 1
    assert page.tbl_per_image.item(0, 0).text() == "a.png"
    assert page.tbl_per_image.item(0, 1).text() == "2"
    assert page.tbl_per_image.item(0, 2).text() == "0"


def test_compare_rerenders_an_aborted_partial_run(qapp, tmp_path, selected_language):
    """An aborted run repaints the rows it received under the run's labels."""
    selected_language("en")
    first = tmp_path / "photos" / "a.png"
    state = DetectorState()
    state.model_slots[0].alias = "renamed"
    state.model_slots[0].path = tmp_path / "renamed.pt"
    page = ComparePage(state)

    run = RunResult(
        run_dir=tmp_path / "run",
        models=[ModelInfo(alias="run_a"), ModelInfo(alias="run_b")],
        results={
            0: [misclassified_result(first, model_idx=0)],
            1: [misclassified_result(first, model_idx=1)],
        },
        status=RunStatus.ABORTED,
    )
    state.run_result = run
    state.results = run.results
    state.params = run.params
    state.run_aborted.emit()

    assert page.tbl_summary.rowCount() == 2
    assert [page.tbl_summary.item(row, 0).text() for row in range(2)] == ["run_a", "run_b"]
    assert [
        page.tbl_per_image.horizontalHeaderItem(column).text()
        for column in range(page.tbl_per_image.columnCount())
    ] == ["Image", "run_a", "run_b"]
    assert page.tbl_per_image.rowCount() == 1
    assert page.tbl_per_image.item(0, 0).text() == "a.png"
