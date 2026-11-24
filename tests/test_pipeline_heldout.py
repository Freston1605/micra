"""Held-out test evaluation driven only through its headless command.

A finished execution is trained for one epoch with the suite's Ultralytics
double, then a separate action scores its frozen ``best.pt`` on the test
cohort. The validation and test folds hold different images and different
classes, so a run that touched validation would not produce the figures the
tests assert. Every assertion reads a file the command wrote, the paths the
predictor was asked for, or the execution tree before and after.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import split_dataset, write_image
from tests.test_pipeline_local_tracer import _FakeNetwork, _FakeResult
from tests.test_pipeline_yolo_seg import _FakeSegResult, _square

NAMES = ("PET", "PP", "LDPE")


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


# ── torchvision doubles that stand in for a frozen Mask R-CNN ──────────────


def _mask_outputs(detections: list[dict], image: object) -> dict:
    """One image's torchvision inference dict, shaped like Mask R-CNN's."""
    import torch

    from polyx.pipeline.evaluation import rasterize_segmentation

    height, width = int(image.shape[-2]), int(image.shape[-1])  # type: ignore[attr-defined]
    boxes = torch.tensor([item["xyxy"] for item in detections], dtype=torch.float32).reshape(-1, 4)
    labels = torch.tensor([item["class_id"] + 1 for item in detections], dtype=torch.int64)
    scores = torch.tensor([item["score"] for item in detections], dtype=torch.float32)
    masks = [
        torch.from_numpy(
            rasterize_segmentation([item["polygon"]], width, height).astype(np.float32)
        )
        for item in detections
    ]
    stacked = torch.stack(masks).unsqueeze(1) if masks else torch.zeros((0, 1, height, width))
    return {"boxes": boxes, "labels": labels, "scores": scores, "masks": stacked}


class _HeldOutMaskDetector:
    """Torchvision factory stand-in for the held-out inference path.

    The factory hands back a stock-shaped detector: torchvision sizes its own
    heads for COCO's 91 classes, and rebuilding for the execution's dataset has
    to replace them. The stand-in records the call arguments and how many
    images each forward pass received, so a test can prove the cohort and the
    one-image-at-a-time policy without reading private state.
    """

    prediction_sets: list[list[dict]] = []
    instances: list[object] = []
    factory_calls: list[dict] = []
    inference_batch_sizes: list[int] = []

    @classmethod
    def build(cls) -> object:
        import torch

        class _Detector(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Stock torchvision heads: a module tree with COCO's 91-class
                # output layers, replaced for the dataset when rebuilt.
                self.roi_heads = torch.nn.Module()
                self.roi_heads.box_predictor = torch.nn.Module()
                self.roi_heads.box_predictor.cls_score = torch.nn.Linear(8, 91)
                self.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(8, 364)
                self.roi_heads.mask_predictor = torch.nn.Module()
                self.roi_heads.mask_predictor.conv5_mask = torch.nn.Conv2d(8, 256, 3)
                self.roi_heads.mask_predictor.mask_fcn_logits = torch.nn.Conv2d(256, 91, 1)
                _HeldOutMaskDetector.instances.append(self)

            def forward(self, images: list, targets: list | None = None) -> object:
                _HeldOutMaskDetector.inference_batch_sizes.append(len(images))
                return [
                    _mask_outputs(_HeldOutMaskDetector.prediction_sets.pop(0), image)
                    for image in images
                ]

        return _Detector()


def _install_mask_fakes(monkeypatch: pytest.MonkeyPatch, prediction_sets: list[list[dict]]) -> None:
    """Replace only the torchvision factory; the conversion and evaluator run."""
    import torchvision.models.detection

    _HeldOutMaskDetector.prediction_sets = list(prediction_sets)
    _HeldOutMaskDetector.instances = []
    _HeldOutMaskDetector.factory_calls = []
    _HeldOutMaskDetector.inference_batch_sizes = []

    def factory(**kwargs: object) -> object:
        _HeldOutMaskDetector.factory_calls.append(kwargs)
        return _HeldOutMaskDetector.build()

    monkeypatch.setattr(torchvision.models.detection, "maskrcnn_resnet50_fpn_v2", factory)


def _write_mask_checkpoint(path: Path, class_count: int) -> None:
    """The payload ``_MaskModule.export_native`` writes, with real head shapes.

    The heads are the adapter's own: ``class_count + 1`` rows because
    torchvision reserves label 0 for background. A rebuild that loads this
    into a stock 91-class detector is the size mismatch the held-out path must
    not hit.
    """
    import torch
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    classes = class_count + 1
    state: dict[str, torch.Tensor] = {}
    for prefix, module in (
        ("roi_heads.box_predictor.", FastRCNNPredictor(8, classes)),
        ("roi_heads.mask_predictor.", MaskRCNNPredictor(8, 256, classes)),
    ):
        state.update({f"{prefix}{name}": value for name, value in module.state_dict().items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "epoch": 0}, path)


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


def _finished_mask_execution(tmp_path: Path, store: Path) -> Path:
    """The artifacts a completed Mask R-CNN execution leaves, without a fit.

    Held-out scoring is the subject here, not the training loop (covered by
    ``test_pipeline_mask_rcnn``): the record, the declaration and the native
    checkpoint are written through the pipeline's own serialisers and the
    family's own export shape, so the action reads a real layout.
    """
    from polyx.pipeline.artifacts import write_json, write_yaml
    from polyx.pipeline.config import config_to_dict, load_config, protocol_to_dict

    output = tmp_path / "executions" / "mask_rcnn"
    (output / "checkpoints").mkdir(parents=True)
    config = load_config(
        [
            f"coco_dataset={store}",
            f"output_dir={output}",
            f"scratch_dir={tmp_path / 'scratch'}",
            "family=mask_rcnn",
        ]
    )
    write_yaml(config_to_dict(config), output / "config.yaml")
    write_yaml(protocol_to_dict(config), output / "resolved_protocol.yaml")
    best = output / "checkpoints" / "best.pt"
    _write_mask_checkpoint(best, len(NAMES))
    write_json(
        {
            "status": "completed",
            "protocol": protocol_to_dict(config),
            "dataset_fingerprint": "mask-execution-fingerprint",
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """One result artifact names the split, cohort, digest and operating point."""
    selected_language("en")
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
    # The cohort is the execution's own declaration, and both fingerprints
    # travel: the dataset the execution scored on and the fold that was read.
    assert artifact["test_cohort"]["source"] == "declared"
    assert (
        artifact["test_cohort"]["execution_fingerprint"]
        == (artifact["execution_dataset_fingerprint"])
    )
    assert artifact["test_cohort"]["classes"] == list(NAMES)

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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """No test fold means no score: validation is never substituted."""
    from tests.conftest import build_tiny_dataset

    selected_language("en")
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


def test_a_family_without_an_inference_path_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A framework the action cannot predict from is named, not guessed at."""
    from polyx.pipeline.artifacts import write_json

    execution = tmp_path / "executions" / "mmdetection"
    (execution / "checkpoints").mkdir(parents=True)
    best = execution / "checkpoints" / "best.pt"
    best.write_bytes(b"frozen-checkpoint")
    write_json(
        {
            "status": "completed",
            "protocol": {"split": "val", "confidence": 0.25, "image_size": 1280},
            "family": {
                "name": "mmdet_mask",
                "framework": "mmdetection",
                "weights": "some_factory",
                "task": "segment",
                "run_name": "mmdet_mask",
            },
            "selected_epoch": 0,
            "selection": {"metric": "with_class_score", "value": 0.5},
            "checkpoint": str(best),
            "epochs_completed": 1,
        },
        execution / "completion.json",
    )
    _install_fakes(monkeypatch, _RecordingYOLO, [])
    results = tmp_path / "results" / "mmdetection"

    code = _heldout(f"--execution={execution}", f"--output={results}")

    message = capsys.readouterr().err
    assert code == 1
    assert "mmdet_mask" in message
    assert "mmdetection" in message
    assert "Mask R-CNN" in message
    assert _RecordingYOLO.instances == []
    assert not results.exists()


def test_yolo_segmentation_reports_the_same_result_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """Masks are scored by the same evaluator and the same artifact shape."""
    selected_language("en")
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


def test_a_frozen_mask_rcnn_checkpoint_is_scored_on_the_test_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """The family's own inference path scores the frozen export, once."""
    selected_language("en")
    store = _import_seg_store(tmp_path)
    execution = _finished_mask_execution(tmp_path, store)
    before = _snapshot(execution)
    _install_mask_fakes(monkeypatch, _perfect_seg_test_predictions())
    results = tmp_path / "results" / "mask-rcnn"

    code = _heldout(f"--execution={execution}", f"--output={results}")

    assert code == 0, capsys.readouterr().err
    artifact = _read(results / "heldout_test.json")
    assert artifact["split"] == "test"
    assert artifact["task"] == "segment"
    assert artifact["family"]["name"] == "mask_rcnn"
    assert artifact["family"]["framework"] == "torchvision"
    assert artifact["overlap"] == "mask IoU"
    assert artifact["primary"]["metric"] == "with_class_score"
    assert artifact["primary"]["value"] == pytest.approx(1.0)
    assert "held-out test split 'test'" in artifact["primary"]["criterion"]

    # The frozen export was rebuilt at the protocol's image size for the
    # dataset's classes, and read one image at a time, as validation does.
    # Two calls, one image each: the two test images and nothing else.
    call = _HeldOutMaskDetector.factory_calls[-1]
    assert call["weights"] is None
    assert call["min_size"] == call["max_size"] == 1280
    assert len(_HeldOutMaskDetector.inference_batch_sizes) == 2
    assert set(_HeldOutMaskDetector.inference_batch_sizes) == {1}

    from polyx.core.provenance import sha256_file

    checkpoint = execution / "checkpoints" / "best.pt"
    assert artifact["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert artifact["checkpoint"]["path"] == str(checkpoint)
    assert artifact["checkpoint"]["selected_epoch"] == 0
    assert set(artifact["test_cohort"]["hashes"]) == {
        "dataset.json",
        "images/test/plate_e1.png",
        "images/test/plate_e2.png",
    }
    assert artifact["test_cohort"]["images"] == 2
    assert artifact["confidence_threshold"] == 0.25
    assert artifact["confidence_source"] == "protocol"
    assert {path.name for path in results.iterdir()} == {
        "heldout_test.json",
        "heldout_test.md",
    }

    # The execution is untouched: no selection, study or completion artifact.
    assert _snapshot(execution) == before


def test_the_mask_rcnn_result_matches_the_ultralytics_result_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """Both families reach the same artifact contract through one evaluator."""
    selected_language("en")
    store = _import_seg_store(tmp_path)
    seg_execution = _finished_segmentation_execution(tmp_path, store)
    mask_execution = _finished_mask_execution(tmp_path, store)
    seg_results = tmp_path / "results" / "segmentation"
    mask_results = tmp_path / "results" / "mask-rcnn"

    _install_fakes(monkeypatch, _RecordingSegYOLO, [_perfect_seg_test_predictions()])
    assert _heldout(f"--execution={seg_execution}", f"--output={seg_results}") == 0
    _install_mask_fakes(monkeypatch, _perfect_seg_test_predictions())
    assert _heldout(f"--execution={mask_execution}", f"--output={mask_results}") == 0, (
        capsys.readouterr().err
    )

    seg = _read(seg_results / "heldout_test.json")
    mask = _read(mask_results / "heldout_test.json")

    # The same record, field for field, with the family as the only variable.
    assert set(mask) == set(seg)
    assert set(mask["family"]) == set(seg["family"])
    assert mask["family"]["framework"] == "torchvision"
    assert seg["family"]["framework"] == "ultralytics"
    assert mask["family"]["name"] == "mask_rcnn"
    assert seg["family"]["name"] == "yolo_segmentation"
    for key in (
        "action",
        "status",
        "split",
        "task",
        "overlap",
        "confidence_threshold",
        "confidence_source",
        "iou_threshold",
    ):
        assert mask[key] == seg[key], key
    assert set(mask["checkpoint"]) == set(seg["checkpoint"])
    assert set(mask["test_cohort"]) == set(seg["test_cohort"])
    assert mask["test_cohort"]["fingerprint"] == seg["test_cohort"]["fingerprint"]
    assert set(mask["primary"]) == set(seg["primary"])
    assert mask["primary"] == seg["primary"]

    # Same figures, same criteria and the same values: one scoring routine.
    assert set(mask["evaluation"]) == set(seg["evaluation"])
    assert set(mask["evaluation"]["figures"]) == set(seg["evaluation"]["figures"])
    assert set(mask["evaluation"]["per_class"]) == set(seg["evaluation"]["per_class"])
    for name, figure in seg["evaluation"]["figures"].items():
        assert mask["evaluation"]["figures"][name]["criterion"] == figure["criterion"], name
        assert mask["evaluation"]["figures"][name]["value"] == pytest.approx(figure["value"]), name
    for class_name, figures in seg["evaluation"]["per_class"].items():
        assert set(mask["evaluation"]["per_class"][class_name]) == set(figures)
        for name, figure in figures.items():
            assert mask["evaluation"]["per_class"][class_name][name]["value"] == pytest.approx(
                figure["value"]
            )

    # Both reports name the same facts in the same words.
    mask_report = (mask_results / "heldout_test.md").read_text(encoding="utf-8")
    seg_report = (seg_results / "heldout_test.md").read_text(encoding="utf-8")
    assert "# Held-out test evaluation" in seg_report
    assert "# Held-out test evaluation" in mask_report
    assert "Overlap rule: mask IoU >= 0.50" in seg_report
    assert "Overlap rule: mask IoU >= 0.50" in mask_report


def test_a_copied_execution_is_scored_from_its_own_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The recorded checkpoint path counts only inside the execution it names.

    A copied tree still carries the original's absolute ``checkpoint`` string;
    scoring the original's bytes there would report one execution's weights as
    another's result, so the copy's own ``checkpoints/best.pt`` is the file
    that is read.
    """
    _, execution = _train_detection(tmp_path, monkeypatch)
    copied = tmp_path / "executions" / "detection-copy"
    shutil.copytree(execution, copied)
    assert Path(_read(copied / "completion.json")["checkpoint"]) == (
        execution / "checkpoints" / "best.pt"
    )

    own_bytes = b"the copy's own frozen weights"
    (copied / "checkpoints" / "best.pt").write_bytes(own_bytes)

    _install_fakes(monkeypatch, _RecordingYOLO, [_perfect_test_predictions()])
    results = tmp_path / "results" / "copy"
    code = _heldout(f"--execution={copied}", f"--output={results}")

    assert code == 0, capsys.readouterr().err
    artifact = _read(results / "heldout_test.json")
    assert artifact["checkpoint"]["path"] == str(copied / "checkpoints" / "best.pt")
    assert artifact["checkpoint"]["sha256"] == hashlib.sha256(own_bytes).hexdigest()


def test_a_cohort_with_other_classes_is_refused_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """A held-out cohort is only this execution's if it names the same classes."""
    _, execution = _train_detection(tmp_path, monkeypatch)
    other = split_dataset(
        tmp_path / "other-cohort",
        {
            "test": {
                "plate_x1": "0 0.25 0.25 0.25 0.25\n",
                "plate_x2": "1 0.25 0.75 0.25 0.25\n",
            }
        },
        names=("PS", "PVC", "PA"),
        data_yaml=True,
    )
    results = tmp_path / "results" / "other-classes"
    _install_fakes(monkeypatch, _RecordingYOLO, [])
    selected_language("en")

    code = _heldout(
        f"--execution={execution}",
        f"--output={results}",
        f"--dataset={other / 'data.yaml'}",
    )

    error = capsys.readouterr().err
    assert code == 1
    assert "classes" in error
    assert "PS" in error and "PET" in error, "both class maps are named"
    assert _RecordingYOLO.instances == [], "a refused cohort must not reach the model"
    assert not results.exists()


def test_an_external_cohort_with_the_same_classes_is_recorded_as_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--dataset may name another location, but never another class map."""
    _, execution = _train_detection(tmp_path, monkeypatch)
    external = split_dataset(
        tmp_path / "external-cohort",
        {
            "test": {
                "plate_e1": "0 0.25 0.25 0.25 0.25\n1 0.75 0.25 0.25 0.25\n",
                "plate_e2": "2 0.25 0.75 0.25 0.25\n",
            }
        },
        names=NAMES,
        data_yaml=True,
    )
    results = tmp_path / "results" / "external"
    _install_fakes(monkeypatch, _RecordingYOLO, [[]])

    code = _heldout(
        f"--execution={execution}",
        f"--output={results}",
        f"--dataset={external / 'data.yaml'}",
    )

    assert code == 0, capsys.readouterr().err
    artifact = _read(results / "heldout_test.json")
    assert artifact["test_cohort"]["source"] == "external"
    # Both fingerprints travel: the execution's dataset and the cohort read.
    assert (
        artifact["test_cohort"]["execution_fingerprint"]
        == (artifact["execution_dataset_fingerprint"])
    )
    assert (
        artifact["test_cohort"]["fingerprint"] != artifact["test_cohort"]["execution_fingerprint"]
    )
    assert _RecordingYOLO.sources == [
        [str(path) for path in sorted((external / "images" / "test").glob("*.png"))]
    ]


def test_the_heldout_token_is_the_evaluators_own_split_constant() -> None:
    """One token names the fold: selection wording and the scored cohort agree."""
    from polyx.pipeline import heldout
    from polyx.pipeline.evaluation import TEST_SPLIT, SharedEvaluator

    assert heldout.TEST_SPLIT == TEST_SPLIT
    evaluator = SharedEvaluator(
        {"images": [], "annotations": [], "categories": []},
        TEST_SPLIT,
        task="detect",
        confidence_threshold=0.25,
    )
    assert "held-out test split 'test'" in evaluator.class_criterion


def test_the_report_and_the_not_evaluated_line_speak_the_active_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """Both surfaces are interface copy: labels and the honest refusal translate."""
    from tests.conftest import build_tiny_dataset

    data_yaml, execution = _train_detection(tmp_path, monkeypatch)
    _install_fakes(monkeypatch, _RecordingYOLO, [_perfect_test_predictions()])
    results = tmp_path / "results" / "language"
    selected_language("es")
    assert _heldout(f"--execution={execution}", f"--output={results}") == 0, capsys.readouterr().err
    report = (results / "heldout_test.md").read_text(encoding="utf-8")
    assert "# Evaluación de la cohorte de prueba retenida" in report
    assert "Huella de la cohorte de prueba" in report
    assert "- Held-out test evaluation" not in report

    # A dataset without a test fold: the refusal is translated too.
    validation_only = build_tiny_dataset(tmp_path / "validation-only")
    _install_fakes(monkeypatch, _RecordingYOLO, [[], []])
    plain = tmp_path / "executions" / "validation-only"
    assert (
        _pipeline(f"data_yaml={validation_only}", f"output_dir={plain}", "protocol.epochs=1") == 0
    )
    _install_fakes(monkeypatch, _RecordingYOLO, [])
    code = _heldout(f"--execution={plain}", f"--output={tmp_path / 'results' / 'none'}")

    error = capsys.readouterr().err
    assert code == 2
    assert "NO EVALUADO" in error
    assert "no test split" not in error
    assert data_yaml.is_file()


def _all_keys(value: object) -> list[str]:
    """Every mapping key anywhere in one artifact."""
    if isinstance(value, dict):
        return [str(key) for key in value] + [
            nested for item in value.values() for nested in _all_keys(item)
        ]
    if isinstance(value, list):
        return [nested for item in value for nested in _all_keys(item)]
    return []
