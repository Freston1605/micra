"""The shipped training engine refuses a dataset its loader cannot read.

``core.training.train`` is the seam shared by the GUI Trainer and the
``polyx-train`` CLI. Both hand the YAML to Ultralytics, whose label lookup is
``images/`` -> ``labels/`` (or beside a plain-folder image). The dataset audit
now reports a codec-only legacy location; the engine must stop before the
framework starts, not train those images as background.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import ultralytics

from polyx.core.errors import CodedFailure
from polyx.core.training import TrainingConfig, train
from tests.conftest import split_dataset

PET_BOX = "0 0.5 0.5 0.10 0.10"


def test_train_refuses_a_label_location_ultralytics_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = split_dataset(
        tmp_path / "dataset",
        {
            "train": {"train_0": f"{PET_BOX}\n"},
            "val": {"val_0": f"{PET_BOX}\n"},
        },
        data_yaml=True,
    )
    for split in ("train", "val"):
        image = root / "images" / split / f"{split}_0.png"
        native = root / "labels" / split / f"{image.stem}.txt"
        (image.parent / f"{image.stem}.txt").write_text(
            native.read_text(encoding="utf-8"), encoding="utf-8"
        )
        native.unlink()

    def _must_not_start(*_args, **_kwargs):
        pytest.fail("Ultralytics must not start on a dataset the audit rejects")

    monkeypatch.setattr(ultralytics, "YOLO", _must_not_start)
    cfg = TrainingConfig(weights="unused.pt", data_yaml=str(root / "data.yaml"))

    with pytest.raises(CodedFailure) as excinfo:
        train(cfg)

    assert excinfo.value.code == "train.dataset_audit_failed"
    assert "label_location_not_native" in excinfo.value.parameters["problems"]
