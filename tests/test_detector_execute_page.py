"""Ticket #253: the Run page consumes the dense model index as its preview slot.

The runner emits the engine's dense active-model index with each result. The
page already builds its previews in that same active order, so the signal
index is used directly: no lookup through the mutable model slots.
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

from polyx.core.detection_run import ImageResult  # noqa: E402
from polyx.gui.detector.pages.execute import RunPage  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402
from tests.conftest import write_image  # noqa: E402


def test_run_page_previews_the_result_under_its_dense_index(qapp, tmp_path):
    plate = write_image(tmp_path / "photos" / "plate.png", width=64, height=64)
    state = DetectorState()
    state.model_slots[0].alias = "model_a"
    state.model_slots[0].path = tmp_path / "a.pt"
    state.model_slots[2].alias = "model_c"
    state.model_slots[2].path = tmp_path / "c.pt"
    page = RunPage(state)
    result = ImageResult(image_path=plate, model_idx=1, annotated_path=plate)

    # Dense index 1 is the second model that ran, which sits in slot 2.
    page._on_image_done(1, result)

    assert not page.previews[1].preview.pixmap().isNull()
    assert page.previews[0].preview.pixmap().isNull()
    assert page.previews[2].preview.pixmap().isNull()


def test_run_page_emits_run_failed_once(qapp, monkeypatch):
    """A failure closes the running state and refreshes the pages exactly once."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "critical", lambda *_args: None)
    state = DetectorState()
    state.set_running(True)
    page = RunPage(state)
    emitted: list[bool] = []
    state.run_failed.connect(lambda: emitted.append(True))

    page._on_failed("boom")

    assert emitted == [True]
    assert state.is_running() is False
    assert page.btn_start.isEnabled() is True
    assert page.btn_stop.isEnabled() is False
