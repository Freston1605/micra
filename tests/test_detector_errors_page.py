"""Ticket #120: the Errors page lists the match the run decided.

The page used to re-match every image with box IoU at ``params.iou_tp``. A run
that matched by mask IoU then showed a different set of errors -- a false
positive and a false negative the run had never counted -- because the page
had its own rule. It now reads ``ImageResult.match``.
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

from polyx.core.detect.protocol import Instance  # noqa: E402
from polyx.core.detection_run import ImageResult, ModelInfo, RunResult  # noqa: E402
from polyx.core.detection_run.result import RunParams  # noqa: E402
from polyx.gui.detector.pages.errors import ErrorsPage  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402
from tests.match_fixtures import matched_result  # noqa: E402


def _page(image: Path) -> tuple[ErrorsPage, DetectorState]:
    state = DetectorState()
    state.model_slots[0].alias = "m0"
    state.params = RunParams()
    state.results = {0: [matched_result(image)]}
    page = ErrorsPage(state)
    page.refresh()
    return page, state


def test_errors_page_rows_equal_the_match_the_run_decided(qapp, tmp_path):
    page, state = _page(tmp_path / "plate.png")
    match = state.results[0][0].match

    assert page.table.rowCount() == (
        len(match.fp_idx) + len(match.fn_idx) + len(match.miscls_pairs)
    )
    assert page.table.rowCount() == 1
    # The one error is the stray prediction; the mask-matched true positive is
    # not an error, and no false negative is invented.
    assert page.table.item(0, 1).text() == "plate.png"
    assert page.table.item(0, 4).text() == "PP"
    assert page.table.item(0, 5).text() == "0.40"


def test_errors_page_rows_use_the_display_name_not_the_alias(qapp, tmp_path, selected_language):
    """#124: the row speaks prose while the folder keeps the stable alias."""
    from PySide6.QtCore import Qt

    selected_language("en")
    state = DetectorState()
    state.model_slots[0].alias = "model_1"
    state.model_slots[0].alias_generated = True
    state.results = {0: [matched_result(tmp_path / "plate.png")]}
    page = ErrorsPage(state)
    page.refresh()

    assert page.table.item(0, 0).text() == "Model 1"
    assert page.table.item(0, 0).data(Qt.UserRole) == "model_1"


def test_errors_page_labels_rows_from_the_run_not_the_current_slots(
    qapp, tmp_path, selected_language
):
    """Dense results keep the run's aliases after the Models tab changes."""
    from PySide6.QtCore import Qt

    selected_language("en")
    first = tmp_path / "photos" / "a.png"
    second = tmp_path / "photos" / "b.png"
    state = DetectorState()
    state.model_slots[0].alias = "slot_a"
    state.model_slots[0].path = tmp_path / "a.pt"
    state.model_slots[1].alias = "slot_b"
    state.model_slots[1].path = tmp_path / "b.pt"
    run = RunResult(
        run_dir=tmp_path / "run",
        models=[ModelInfo(alias="run_a"), ModelInfo(alias="run_c")],
        results={
            0: [matched_result(first, model_idx=0)],
            1: [matched_result(second, model_idx=1)],
        },
    )
    state.run_result = run
    state.results = run.results
    page = ErrorsPage(state)
    page.refresh()

    assert page.table.rowCount() == 2
    assert [page.table.item(row, 0).text() for row in range(2)] == ["run_a", "run_c"]
    assert [page.table.item(row, 0).data(Qt.UserRole) for row in range(2)] == [
        "run_a",
        "run_c",
    ]


def test_errors_model_filter_lists_the_run_models_in_dense_order(qapp, tmp_path, selected_language):
    """A run from slots 0 and 2 offers its two models, not the slot placeholders."""
    selected_language("en")
    first = tmp_path / "photos" / "a.png"
    second = tmp_path / "photos" / "b.png"
    state = DetectorState()
    state.model_slots[0].alias = "model_1"
    state.model_slots[0].alias_generated = True
    state.model_slots[0].path = tmp_path / "a.pt"
    state.model_slots[2].alias = "model_3"
    state.model_slots[2].alias_generated = True
    state.model_slots[2].path = tmp_path / "b.pt"
    page = ErrorsPage(state)

    run = RunResult(
        run_dir=tmp_path / "run",
        models=[
            ModelInfo(alias="model_1", alias_generated=True),
            ModelInfo(alias="model_3", alias_generated=True),
        ],
        results={
            0: [matched_result(first, model_idx=0)],
            1: [matched_result(second, model_idx=1)],
        },
    )
    state.run_result = run
    state.results = run.results
    state.run_finished.emit()

    assert [page.combo_model.itemText(i) for i in range(page.combo_model.count())] == [
        "All",
        "Model 1",
        "Model 3",
    ]
    assert [page.combo_model.itemData(i) for i in range(page.combo_model.count())] == [-1, 0, 1]

    page.combo_model.setCurrentIndex(2)
    assert page.table.rowCount() == 1
    assert page.table.item(0, 0).text() == "Model 3"
    assert page.table.item(0, 1).text() == "b.png"


def test_errors_model_filter_lists_the_active_models_before_a_run(
    qapp, tmp_path, selected_language
):
    """Before a run the filter offers the models loaded on the Models tab."""
    selected_language("en")
    state = DetectorState()
    state.model_slots[0].alias = "slot_a"
    state.model_slots[0].alias_generated = False
    state.model_slots[0].path = tmp_path / "a.pt"
    state.model_slots[2].alias = "slot_c"
    state.model_slots[2].alias_generated = False
    state.model_slots[2].path = tmp_path / "c.pt"
    page = ErrorsPage(state)

    assert [page.combo_model.itemText(i) for i in range(page.combo_model.count())] == [
        "All",
        "slot_a",
        "slot_c",
    ]


def test_errors_model_filter_keeps_the_run_models_and_resets_to_all_after_a_models_change(
    qapp, tmp_path, selected_language
):
    """A post-run rename cannot relabel the filter or the rows of the run."""
    selected_language("en")
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    state = DetectorState()
    state.model_slots[0].alias = "slot_a"
    state.model_slots[0].path = tmp_path / "a.pt"
    state.model_slots[1].alias = "slot_b"
    state.model_slots[1].path = tmp_path / "b.pt"
    run = RunResult(
        run_dir=tmp_path / "run",
        models=[ModelInfo(alias="run_a"), ModelInfo(alias="run_b")],
        results={
            0: [matched_result(first, model_idx=0)],
            1: [matched_result(second, model_idx=1)],
        },
    )
    state.run_result = run
    state.results = run.results
    page = ErrorsPage(state)
    page.combo_model.setCurrentIndex(2)
    assert page.combo_model.currentData() == 1

    state.model_slots[0].alias = "renamed_a"
    state.model_slots[1].alias = "renamed_b"
    state.model_slots[2].alias = "late_model"
    state.model_slots[2].alias_generated = False
    state.model_slots[2].path = tmp_path / "late.pt"
    state.models_changed.emit()

    assert [page.combo_model.itemText(i) for i in range(page.combo_model.count())] == [
        "All",
        "run_a",
        "run_b",
    ]
    assert [page.combo_model.itemData(i) for i in range(page.combo_model.count())] == [-1, 0, 1]
    assert page.combo_model.currentData() == -1
    assert page.table.rowCount() == 2
    assert [page.table.item(row, 0).text() for row in range(2)] == ["run_a", "run_b"]


def test_errors_model_filter_falls_back_to_all_when_the_selected_model_did_not_run(
    qapp, tmp_path, selected_language
):
    """A later run replaces the offered entries and the filter returns to All."""
    selected_language("en")
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    state = DetectorState()
    state.model_slots[0].alias = "slot_a"
    state.model_slots[0].path = tmp_path / "a.pt"
    state.model_slots[1].alias = "slot_b"
    state.model_slots[1].path = tmp_path / "b.pt"
    state.run_result = RunResult(
        run_dir=tmp_path / "run",
        models=[ModelInfo(alias="run_a"), ModelInfo(alias="run_b")],
        results={
            0: [matched_result(first, model_idx=0)],
            1: [matched_result(second, model_idx=1)],
        },
    )
    state.results = state.run_result.results
    page = ErrorsPage(state)
    page.combo_model.setCurrentIndex(2)

    solo = RunResult(
        run_dir=tmp_path / "run2",
        models=[ModelInfo(alias="solo")],
        results={0: [matched_result(first, model_idx=0)]},
    )
    state.run_result = solo
    state.results = solo.results
    state.run_finished.emit()

    assert [page.combo_model.itemText(i) for i in range(page.combo_model.count())] == [
        "All",
        "solo",
    ]
    assert page.combo_model.currentData() == -1
    assert page.table.rowCount() == 1
    assert page.table.item(0, 0).text() == "solo"


def test_errors_model_filter_rebuilds_when_a_run_aborts(qapp, tmp_path, selected_language):
    """An aborted run keeps offering the models it started with."""
    selected_language("en")
    first = tmp_path / "a.png"
    state = DetectorState()
    state.model_slots[0].alias = "slot_a"
    state.model_slots[0].path = tmp_path / "a.pt"
    state.model_slots[2].alias = "slot_c"
    state.model_slots[2].path = tmp_path / "c.pt"
    page = ErrorsPage(state)

    state.run_result = RunResult(
        run_dir=tmp_path / "run",
        models=[
            ModelInfo(alias="model_1", alias_generated=True),
            ModelInfo(alias="model_3", alias_generated=True),
        ],
        results={0: [matched_result(first, model_idx=0)], 1: []},
    )
    state.results = state.run_result.results
    state.run_aborted.emit()

    assert [page.combo_model.itemText(i) for i in range(page.combo_model.count())] == [
        "All",
        "Model 1",
        "Model 3",
    ]
    page.combo_model.setCurrentIndex(2)
    assert page.table.rowCount() == 0


def test_errors_model_filter_follows_a_failed_run(qapp, tmp_path, selected_language):
    """A failed attempt with no record lists its snapshot models, densely keyed."""
    selected_language("en")
    first = tmp_path / "photos" / "a.png"
    second = tmp_path / "photos" / "b.png"
    state = DetectorState()
    state.model_slots[0].alias = "slot_a"
    state.model_slots[0].path = tmp_path / "a.pt"
    state.model_slots[2].alias = "slot_c"
    state.model_slots[2].path = tmp_path / "c.pt"
    page = ErrorsPage(state)

    state.model_snapshot = [
        ModelInfo(alias="run_a", alias_generated=False),
        ModelInfo(alias="run_c", alias_generated=False),
    ]
    state.results = {
        0: [matched_result(first, model_idx=0)],
        1: [matched_result(second, model_idx=1)],
    }
    state.run_failed.emit()

    assert [page.combo_model.itemText(i) for i in range(page.combo_model.count())] == [
        "All",
        "run_a",
        "run_c",
    ]
    assert [page.combo_model.itemData(i) for i in range(page.combo_model.count())] == [-1, 0, 1]

    page.combo_model.setCurrentIndex(2)
    assert page.combo_model.currentData() == 1
    assert page.table.rowCount() == 1
    assert page.table.item(0, 0).text() == "run_c"
    assert page.table.item(0, 1).text() == "b.png"


def test_errors_model_filter_lists_the_record_models_with_empty_results(
    qapp, tmp_path, selected_language
):
    """The record, not the current slots, decides which models are offered."""
    selected_language("en")
    state = DetectorState()
    state.model_slots[0].alias = "slot_a"
    state.model_slots[0].path = tmp_path / "a.pt"
    state.run_result = RunResult(
        run_dir=tmp_path / "run",
        models=[ModelInfo(alias="run_a"), ModelInfo(alias="run_b")],
        results={},
    )
    state.results = {}
    page = ErrorsPage(state)

    assert [page.combo_model.itemText(i) for i in range(page.combo_model.count())] == [
        "All",
        "run_a",
        "run_b",
    ]
    assert [page.combo_model.itemData(i) for i in range(page.combo_model.count())] == [-1, 0, 1]


def test_errors_page_says_when_counts_have_no_match_identity(qapp, tmp_path, selected_language):
    """A folder that kept the counts but not the pairs must not read as clean."""
    from tests.match_fixtures import matched_run, saved_run

    selected_language("en")
    run = saved_run(matched_run(tmp_path))
    state = DetectorState()
    state.model_slots[0].alias = run.models[0].alias
    state.params = run.params
    state.results = run.results
    page = ErrorsPage(state)
    page.refresh()

    assert page.table.rowCount() == 0
    assert "cannot be listed" in page.lbl_identity.text()
    assert "1 image" in page.lbl_identity.text()
    assert "does not record which box was paired with which" in page.lbl_identity.text()


def test_errors_page_has_no_identity_caption_when_every_match_is_known(qapp, tmp_path):
    page, _state = _page(tmp_path / "plate.png")

    assert page.lbl_identity.text() == ""


def test_errors_page_lists_an_image_with_no_match_as_no_errors(qapp, tmp_path, selected_language):
    """No match means no row: the page does not invent a re-match.

    The image still carries a count, so the page explains why the list is
    empty instead of presenting it as a clean image.
    """
    selected_language("en")
    state = DetectorState()
    state.model_slots[0].alias = "m0"
    prediction = Instance(class_id=0, class_name="PET", conf=0.9, x1=0, y1=0, x2=10, y2=10)
    gt = Instance(class_id=0, class_name="PET", conf=1.0, x1=0, y1=0, x2=10, y2=10)
    state.results = {
        0: [
            ImageResult(
                image_path=tmp_path / "plate.png",
                model_idx=0,
                predictions=[prediction],
                gt=[gt],
                has_gt=True,
                tp=1,
            )
        ]
    }
    page = ErrorsPage(state)
    page.refresh()

    assert page.table.rowCount() == 0
    assert "cannot be listed" in page.lbl_identity.text()
