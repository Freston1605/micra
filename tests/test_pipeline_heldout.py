"""Held-out test evaluation driven only through its headless command.

A finished execution is trained for one epoch with the suite's Ultralytics
double, then a separate action scores its frozen ``best.pt`` on the test
cohort. The validation and test folds hold different images and different
classes, so a run that touched validation would not produce the figures the
tests assert. Every assertion reads a file the command wrote, the paths the
predictor was asked for, or the execution tree before and after.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.conftest import split_dataset, write_image
from tests.test_pipeline_local_tracer import _FakeNetwork, _FakeResult
from tests.test_pipeline_yolo_seg import _FakeSegResult

NAMES = ("PET", "PP", "LDPE")
SQUARE = [10.0, 8.0, 30.0, 8.0, 30.0, 28.0, 10.0, 28.0]


# ── Ultralytics doubles that remember what they were asked to score ────────


class _RecordingYOLO:
    """Ultralytics stand-in shared by training and held-out evaluation.

    It records every source list and every prediction argument, so a test can
    prove which cohort was read instead of inferring it from a score.
    """

    result_class: type = _FakeResult
    prediction_sets: list[list[list[dict]]] = []
    instances: list[_RecordingYOLO] = []
    sources: list[list[str]] = []
    calls: list[dict] = []

    def __init__(self, weights: object) -> None:
        self.weights = str(weights)
        self.names = dict(enumerate(NAMES))
        self.model = _FakeNetwork()
        type(self).instances.append(self)

    def predict(self, source: object = None, **kwargs: object) -> list:
        paths = [str(path) for path in (source or [])]  # type: ignore[union-attr]
        type(self).sources.append(paths)
        type(self).calls.append(dict(kwargs))
        sets = type(self).prediction_sets
        per_image = sets.pop(0) if sets else []
        return [
            type(self).result_class(
                [item for item in per_image if int(item["image_id"]) == image_id]
            )
            for image_id in range(len(paths))
        ]


class _RecordingSegYOLO(_RecordingYOLO):
    """The same recorder, returning Ultralytics results that carry masks.

    Segmentation predictions arrive one list per image, the shape the
    adapter's ``coco_segmentation_predictions_from_results`` consumes.
    """

    result_class = _FakeSegResult

    def predict(self, source: object = None, **kwargs: object) -> list:
        paths = [str(path) for path in (source or [])]  # type: ignore[union-attr]
        type(self).sources.append(paths)
        type(self).calls.append(dict(kwargs))
        sets = type(self).prediction_sets
        per_image = sets.pop(0) if sets else [[] for _path in paths]
        return [type(self).result_class(detections) for detections in per_image]


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    cls: type[_RecordingYOLO],
    prediction_sets: list[list[list[dict]]],
) -> None:
    """Replace the Ultralytics wrapper and reset its recorded observations."""
    import ultralytics

    cls.prediction_sets = list(prediction_sets)
    cls.instances = []
    cls.sources = []
    cls.calls = []
    monkeypatch.setattr(ultralytics, "YOLO", cls)


# ── Cohorts and predictions ────────────────────────────────────────────────


def _build_detection_cohorts(root: Path) -> Path:
    """Train, validation and test images that cannot be mistaken for each other.

    Validation keeps the class distribution the audit requires. The test fold
    carries the same classes at different coordinates, so a test prediction
    set is perfect on test and wrong on validation: that is what makes "only
    test was scored" observable.
    """
    labels = {
        "train": {
            "plate_t1": "0 0.375 0.375 0.25 0.25\n1 0.75 0.375 0.25 0.25\n",
            "plate_t2": "2 0.375 0.75 0.25 0.25\n",
        },
        "val": {
            "plate_v1": "0 0.375 0.375 0.25 0.25\n1 0.75 0.375 0.25 0.25\n",
            "plate_v2": "2 0.375 0.75 0.25 0.25\n",
        },
        "test": {
            "plate_e1": "0 0.25 0.25 0.25 0.25\n1 0.75 0.25 0.25 0.25\n",
            "plate_e2": "2 0.25 0.75 0.25 0.25\n",
        },
    }
    root = split_dataset(root, labels, names=NAMES, data_yaml=True)
    for image in root.rglob("*.png"):
        write_image(image, width=64, height=48)
    return root / "data.yaml"


def _perfect_test_predictions() -> list[dict]:
    """Boxes exactly on the test labels, in test-cohort order."""
    return [
        {"image_id": 0, "category_id": 0, "bbox": [8.0, 6.0, 16.0, 12.0], "score": 0.9},
        {"image_id": 0, "category_id": 1, "bbox": [40.0, 6.0, 16.0, 12.0], "score": 0.8},
        {"image_id": 1, "category_id": 2, "bbox": [8.0, 30.0, 16.0, 12.0], "score": 0.85},
    ]


def _square(offset: float = 0.0) -> list[float]:
    return [value + (offset if index % 2 == 0 else 0.0) for index, value in enumerate(SQUARE)]


def _seg_detection(class_index: int, offset: float) -> dict:
    polygon = _square(offset)
    return {
        "class_id": class_index,
        "polygon": polygon,
        "xyxy": [polygon[0], polygon[1], polygon[4], polygon[5]],
        "score": 0.9,
    }


def _perfect_seg_test_predictions() -> list[list[dict]]:
    """Masks exactly on the test polygons, one list per test image."""
    return [
        [_seg_detection(0, 0.0), _seg_detection(1, 32.0)],
        [_seg_detection(2, 0.0), _seg_detection(0, 32.0)],
    ]


def _build_seg_export(root: Path) -> Path:
    """A CVAT-shaped COCO export whose test fold differs from validation."""
    images = [
        (1, "train/plate_t1.png", "train"),
        (2, "train/plate_t2.png", "train"),
        (3, "val/plate_v1.png", "val"),
        (4, "val/plate_v2.png", "val"),
        (5, "test/plate_e1.png", "test"),
        (6, "test/plate_e2.png", "test"),
    ]
    for _image_id, file_name, _split in images:
        write_image(root / "images" / file_name, width=64, height=48)
    planned = [
        (1, 1, 0.0),
        (1, 2, 32.0),
        (2, 3, 0.0),
        (2, 1, 32.0),
        (3, 1, 0.0),
        (3, 2, 32.0),
        (4, 3, 0.0),
        (4, 2, 32.0),
        (5, 1, 0.0),
        (5, 2, 32.0),
        (6, 3, 0.0),
        (6, 1, 32.0),
    ]
    annotations = [
        {
            "id": index,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [_square(offset)],
            "bbox": [10.0 + offset, 8.0, 20.0, 20.0],
            "area": 400.0,
            "iscrowd": 0,
        }
        for index, (image_id, category_id, offset) in enumerate(planned, start=1)
    ]
    export = {
        "info": {"description": "CVAT export of three distinguishable folds", "domain": "real"},
        "images": [
            {
                "id": image_id,
                "file_name": file_name,
                "width": 64,
                "height": 48,
                "split": split,
            }
            for image_id, file_name, split in images
        ],
        "categories": [{"id": index, "name": name} for index, name in enumerate(NAMES, start=1)],
        "annotations": annotations,
    }
    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / "instances_default.json").write_text(
        json.dumps(export, indent=2) + "\n", encoding="utf-8"
    )
    return root


# ── Command seams ──────────────────────────────────────────────────────────


def _pipeline(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


def _heldout(*arguments: str) -> int:
    from polyx.pipeline.heldout import main

    return main(list(arguments))


def _train_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A finished detection execution with an empty validation score."""
    data_yaml = _build_detection_cohorts(tmp_path / "dataset")
    _install_fakes(monkeypatch, _RecordingYOLO, [[], []])
    output = tmp_path / "executions" / "detection"
    code = _pipeline(
        f"data_yaml={data_yaml}",
        f"output_dir={output}",
        "protocol.epochs=1",
    )
    assert code == 0
    return data_yaml, output


def _finished_segmentation_execution(tmp_path: Path, store: Path) -> Path:
    """The artifacts a completed YOLO-seg execution leaves, without a fit.

    Segmentation scoring is the subject here, not the training loop (covered
    by ``test_pipeline_yolo_seg``): the record and the checkpoint are written
    through the pipeline's own serialisers so the action reads a real layout.
    """
    from polyx.pipeline.artifacts import write_json, write_yaml
    from polyx.pipeline.config import config_to_dict, load_config, protocol_to_dict

    output = tmp_path / "executions" / "segmentation"
    (output / "checkpoints").mkdir(parents=True)
    config = load_config(
        [
            f"coco_dataset={store}",
            f"output_dir={output}",
            f"scratch_dir={tmp_path / 'scratch'}",
            "family=yolo_segmentation",
        ]
    )
    write_yaml(config_to_dict(config), output / "config.yaml")
    write_yaml(protocol_to_dict(config), output / "resolved_protocol.yaml")
    best = output / "checkpoints" / "best.pt"
    best.write_bytes(b"frozen-checkpoint")
    write_json(
        {
            "status": "completed",
            "protocol": protocol_to_dict(config),
            "dataset_fingerprint": "segmentation-execution-fingerprint",
            "family": asdict(config.family),
            "backend": asdict(config.backend),
            "selected_epoch": 0,
            "selection": {"metric": "with_class_score", "value": 0.0},
            "checkpoint": str(best),
            "epochs_completed": 1,
        },
        output / "completion.json",
    )
    return output


def _import_seg_store(tmp_path: Path) -> Path:
    export = _build_seg_export(tmp_path / "export")
    dataset_root = tmp_path / "user_data" / "segmentation_datasets"
    code = _pipeline(
        f"coco_import={export}",
        f"dataset_root={dataset_root}",
        f"output_dir={tmp_path / 'executions' / 'import'}",
        "dataset_name=mask_batch",
    )
    assert code == 0
    return dataset_root / "mask_batch" / "v1"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(root: Path) -> dict[str, str]:
    """Every file under ``root`` and its bytes, to prove nothing was touched."""
    from polyx.core.provenance import sha256_file

    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ── Tests ──────────────────────────────────────────────────────────────────


def test_the_frozen_checkpoint_is_scored_on_the_test_cohort_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One result artifact names the split, cohort, digest and operating point."""
    data_yaml, execution = _train_detection(tmp_path, monkeypatch)
    before = _snapshot(execution)
    _install_fakes(monkeypatch, _RecordingYOLO, [_perfect_test_predictions()])
    results = tmp_path / "results" / "detection"

    code = _heldout(f"--execution={execution}", f"--output={results}")

    assert code == 0, capsys.readouterr().err
    artifact = _read(results / "heldout_test.json")
    assert artifact["split"] == "test"
    assert artifact["task"] == "detect"
    assert artifact["primary"]["metric"] == "with_class_score"
    assert artifact["primary"]["value"] == pytest.approx(1.0)
    assert "held-out test split 'test'" in artifact["primary"]["criterion"]
    assert artifact["confidence_threshold"] == 0.25
    assert artifact["confidence_source"] == "protocol"
    assert artifact["iou_threshold"] == 0.5
    assert artifact["overlap"] == "box IoU"

    # The predictor saw exactly the test images, in cohort order.
    test_images = sorted((data_yaml.parent / "images" / "test").glob("*.png"))
    assert _RecordingYOLO.sources == [[str(path) for path in test_images]]
    # One Ultralytics load: no Lightning run was needed to score.
    assert len(_RecordingYOLO.instances) == 1

    from polyx.core.provenance import sha256_file

    checkpoint = execution / "checkpoints" / "best.pt"
    assert artifact["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert artifact["checkpoint"]["path"] == str(checkpoint)

    # The cohort identity covers the test files and the declaration, no fold.
    assert set(artifact["test_cohort"]["hashes"]) == {
        "data.yaml",
        "images/test/plate_e1.png",
        "images/test/plate_e2.png",
        "labels/test/plate_e1.txt",
        "labels/test/plate_e2.txt",
    }
    assert artifact["test_cohort"]["images"] == 2
    assert len(artifact["test_cohort"]["fingerprint"]) == 64

    # Code, metrics, selection, training state and completion are untouched.
    assert _snapshot(execution) == before
    metrics = _read(execution / "metrics.json")
    assert metrics["selection"]["value"] == 0.0
    assert metrics["selection"]["value"] != artifact["primary"]["value"]

    report = (results / "heldout_test.md").read_text(encoding="utf-8")
    assert "Held-out test evaluation" in report
    assert str(checkpoint) in report
    assert artifact["test_cohort"]["fingerprint"] in report


def test_a_fixed_threshold_is_applied_recorded_and_never_tuned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one requested threshold decides the operating point, by name."""
    _, execution = _train_detection(tmp_path, monkeypatch)
    uncertain = [
        {"image_id": 0, "category_id": 0, "bbox": [16.0, 12.0, 16.0, 12.0], "score": 0.5},
        {"image_id": 0, "category_id": 1, "bbox": [40.0, 12.0, 16.0, 12.0], "score": 0.5},
        {"image_id": 1, "category_id": 2, "bbox": [16.0, 30.0, 16.0, 12.0], "score": 0.5},
    ]
    _install_fakes(monkeypatch, _RecordingYOLO, [uncertain])
    results = tmp_path / "results" / "threshold"

    code = _heldout(f"--execution={execution}", f"--output={results}", "--confidence=0.6")

    assert code == 0, capsys.readouterr().err
    artifact = _read(results / "heldout_test.json")
    assert artifact["confidence_threshold"] == 0.6
    assert artifact["confidence_source"] == "cli"
    # Every candidate fell below the requested threshold: nothing was widened.
    assert artifact["primary"]["value"] == 0.0
    assert artifact["evaluation"]["selection"]["confidence_threshold"] == 0.6
    assert not any("recommend" in key.lower() for key in _all_keys(artifact))

    # A second result in the same folder is refused; --force replaces it.
    _install_fakes(monkeypatch, _RecordingYOLO, [uncertain])
    assert _heldout(f"--execution={execution}", f"--output={results}") == 1
    _install_fakes(monkeypatch, _RecordingYOLO, [uncertain])
    assert _heldout(f"--execution={execution}", f"--output={results}", "--force") == 0


def test_the_result_is_never_written_inside_the_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """The execution tree is read-only for this action, path included."""
    _, execution = _train_detection(tmp_path, monkeypatch)
    before = _snapshot(execution)
    _install_fakes(monkeypatch, _RecordingYOLO, [_perfect_test_predictions()])
    selected_language("en")

    code = _heldout(f"--execution={execution}", f"--output={execution / 'heldout'}")

    assert code == 1
    assert "outside" in capsys.readouterr().err.lower()
    assert _RecordingYOLO.instances == []
    assert _snapshot(execution) == before
    assert not (execution / "heldout").exists()


def test_a_dataset_without_a_test_split_is_not_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No test fold means no score: validation is never substituted."""
    from tests.conftest import build_tiny_dataset

    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    _install_fakes(monkeypatch, _RecordingYOLO, [[], []])
    execution = tmp_path / "executions" / "validation-only"
    assert _pipeline(f"data_yaml={data_yaml}", f"output_dir={execution}", "protocol.epochs=1") == 0
    before = _snapshot(execution)
    _install_fakes(monkeypatch, _RecordingYOLO, [_perfect_test_predictions()])
    results = tmp_path / "results" / "not-evaluated"

    code = _heldout(f"--execution={execution}", f"--output={results}")

    captured = capsys.readouterr()
    assert code != 0
    assert "NOT EVALUATED" in captured.err
    assert "test" in captured.err.lower()
    # Neither the model nor any validation image was touched.
    assert _RecordingYOLO.instances == []
    assert _RecordingYOLO.sources == []
    assert not results.exists()
    assert _snapshot(execution) == before


def test_an_unfinished_execution_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """Without the completion record there is no frozen checkpoint to score."""
    _, execution = _train_detection(tmp_path, monkeypatch)
    (execution / "completion.json").unlink()
    before = _snapshot(execution)
    _install_fakes(monkeypatch, _RecordingYOLO, [_perfect_test_predictions()])
    selected_language("en")
    results = tmp_path / "results" / "unfinished"

    code = _heldout(f"--execution={execution}", f"--output={results}")

    assert code == 1
    assert "finished" in capsys.readouterr().err.lower()
    assert _RecordingYOLO.instances == []
    assert not results.exists()
    assert _snapshot(execution) == before


def test_a_non_ultralytics_family_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mask R-CNN belongs to its own ticket; the action says so by name."""
    from polyx.pipeline.artifacts import write_json

    execution = tmp_path / "executions" / "mask-rcnn"
    (execution / "checkpoints").mkdir(parents=True)
    best = execution / "checkpoints" / "best.pt"
    best.write_bytes(b"frozen-checkpoint")
    write_json(
        {
            "status": "completed",
            "protocol": {"split": "val", "confidence": 0.25, "image_size": 1280},
            "family": {
                "name": "mask_rcnn",
                "framework": "torchvision",
                "weights": "maskrcnn_resnet50_fpn_v2",
                "task": "segment",
                "run_name": "mask_rcnn",
            },
            "selected_epoch": 0,
            "selection": {"metric": "with_class_score", "value": 0.5},
            "checkpoint": str(best),
            "epochs_completed": 1,
        },
        execution / "completion.json",
    )
    _install_fakes(monkeypatch, _RecordingYOLO, [])
    results = tmp_path / "results" / "mask-rcnn"

    code = _heldout(f"--execution={execution}", f"--output={results}")

    message = capsys.readouterr().err
    assert code == 1
    assert "Ultralytics" in message
    assert "mask_rcnn" in message
    assert _RecordingYOLO.instances == []
    assert not results.exists()


def test_yolo_segmentation_reports_the_same_result_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Masks are scored by the same evaluator and the same artifact shape."""
    store = _import_seg_store(tmp_path)
    execution = _finished_segmentation_execution(tmp_path, store)
    before = _snapshot(execution)
    _install_fakes(monkeypatch, _RecordingSegYOLO, [_perfect_seg_test_predictions()])
    results = tmp_path / "results" / "segmentation"

    code = _heldout(f"--execution={execution}", f"--output={results}")

    assert code == 0, capsys.readouterr().err
    artifact = _read(results / "heldout_test.json")
    assert artifact["split"] == "test"
    assert artifact["task"] == "segment"
    assert artifact["overlap"] == "mask IoU"
    assert artifact["primary"]["metric"] == "with_class_score"
    assert artifact["primary"]["value"] == pytest.approx(1.0)
    assert "held-out test split 'test'" in artifact["primary"]["criterion"]
    assert "mask" in artifact["evaluation"]["figures"]["mask_iou"]["criterion"].lower()

    # Only the test store images were predicted.
    test_images = sorted((store / "images" / "test").glob("*.png"))
    assert _RecordingSegYOLO.sources == [[str(path) for path in test_images]]
    assert set(artifact["test_cohort"]["hashes"]) == {
        "dataset.json",
        "images/test/plate_e1.png",
        "images/test/plate_e2.png",
    }
    assert _snapshot(execution) == before


def _all_keys(value: object) -> list[str]:
    """Every mapping key anywhere in one artifact."""
    if isinstance(value, dict):
        return [str(key) for key in value] + [
            nested for item in value.values() for nested in _all_keys(item)
        ]
    if isinstance(value, list):
        return [nested for item in value for nested in _all_keys(item)]
    return []
