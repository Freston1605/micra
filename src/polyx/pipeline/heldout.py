"""Score a frozen checkpoint on the held-out test cohort.

This is the separate step the pipeline leaves open after model selection: a
finished execution's kept ``best.pt`` is scored once on the ``test`` fold under
one confidence threshold, and the figures are written to their own artifact.
It never trains, never re-selects and never writes into the execution.

The shared evaluator remains the one scorer, so the held-out with-class
operating-point F1 has the same definition as the selection figure. The
threshold is recorded and never tuned: this command applies the one it is
given (by default the protocol's own confidence) and reports no candidate.

Usage:
    python -m polyx.pipeline.heldout --execution <finished execution> \
        --output <result folder>
    python -m polyx.pipeline.heldout --execution <finished execution> \
        --output <result folder> --confidence 0.25 --device cpu

Only families with a native inference path are scored here: the Ultralytics
detection and segmentation families and the torchvision Mask R-CNN family.
Each one produces its predictions through its own frozen-weights path and
hands them to the same evaluator, so the result records the same shape.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from polyx.core.data import DatasetDeclaration, SplitKind
from polyx.core.errors import CodedFailure
from polyx.core.i18n import N_, _
from polyx.core.provenance import sha256_file
from polyx.core.quality.dataset_audit import FINDING_SAMPLE_LIMIT
from polyx.pipeline.artifacts import write_json
from polyx.pipeline.coco import (
    coco_predictions_from_results,
    coco_segmentation_predictions_from_results,
)
from polyx.pipeline.dataset import (
    BoxImage,
    ResolvedBoxDataset,
    coco_store_ground_truth,
    dataset_fingerprint,
    ordered_categories,
    read_coco_store,
    resolve_box_dataset,
    split_coco_images,
)
from polyx.pipeline.errors import PipelineError, report_failure
from polyx.pipeline.evaluation import (
    PREDICTION_FLOOR,
    TEST_SPLIT,
    SharedEvaluator,
    prediction_budget,
)

#: The families with a native inference path, by framework and the tasks each
#: one can score here. Mask R-CNN is the torchvision sibling; a framework
#: outside this table is refused by name.
SUPPORTED_FAMILIES: dict[str, tuple[str, ...]] = {
    "ultralytics": ("detect", "segment"),
    "torchvision": ("segment",),
}
#: Artifact names; machine tokens, never translated.
RESULT_NAME = "heldout_test.json"
REPORT_NAME = "heldout_test.md"


class HeldOutNotEvaluated(Exception):
    """The dataset declares no test fold, so there is nothing to score.

    It is not a failure of the execution: it is the honest answer. The
    command reports it on stderr and exits non-zero instead of falling back to
    the validation fold, which would turn the held-out cohort into a second
    validation set.

    The sentence is an English message id with its parameters, translated when
    the command prints it, so the reason reaches a person in their language
    while the journal keeps the invariant text.
    """

    def __init__(self, template: str, **parameters: object) -> None:
        self.template = template
        self.parameters = parameters
        super().__init__(template % parameters if parameters else template)

    def display(self) -> str:
        """The operator-facing sentence, translated in the active language."""
        translated = _(self.template)
        return translated % self.parameters if self.parameters else translated


def _read_completion(execution: Path) -> dict[str, Any]:
    """The completion record is the one proof that the checkpoint is frozen."""
    path = execution / "completion.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        record = None
    if not isinstance(record, dict) or record.get("status") != "completed":
        raise PipelineError(
            "heldout.execution_not_finished",
            N_(
                "held-out evaluation needs a finished training execution: "
                "%(path)s has no readable completion record"
            ),
            path=str(execution),
        )
    return record


def _required_record(record: dict[str, Any], field: str, execution: Path) -> Any:
    """One completion field this action cannot work without."""
    value = record.get(field)
    if not value:
        raise PipelineError(
            "heldout.execution_incomplete",
            N_("the execution at %(path)s is incomplete: its completion record names no %(field)s"),
            path=str(execution),
            field=field,
        )
    return value


def _selected_checkpoint(record: dict[str, Any], execution: Path) -> Path:
    """The kept checkpoint of *this* execution, by containment, then by layout.

    A completed execution records an absolute ``checkpoints/best.pt``. That
    reference is honoured only while it still points inside the execution: a
    copied tree carries the original's path, and scoring the original's bytes
    would report another execution's weights as this one's result. The copy's
    own canonical location is the fallback; no other file is substituted.
    """
    recorded = Path(str(record.get("checkpoint") or ""))
    if recorded.is_absolute():
        resolved = recorded.resolve()
        if resolved.is_file() and resolved.is_relative_to(execution):
            return resolved
    local = execution / "checkpoints" / "best.pt"
    if local.is_file():
        return local
    raise PipelineError(
        "heldout.checkpoint_missing",
        N_("the selected checkpoint of %(path)s is missing: %(checkpoint)s"),
        path=str(execution),
        checkpoint=str(recorded),
    )


def _declared_dataset(execution: Path, task: str) -> Path:
    """The dataset declaration the finished execution recorded.

    It is also the class map the held-out cohort is checked against, so an
    execution that records no source cannot be scored: naming another cohort
    with ``--dataset`` would leave nothing to verify it against.
    """
    kind = "YOLO box" if task == "detect" else "COCO mask"
    try:
        raw = yaml.safe_load((execution / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        raw = {}
    key = "data_yaml" if task == "detect" else "coco_dataset"
    value = str(raw.get(key) or "").strip() if isinstance(raw, dict) else ""
    if not value:
        raise PipelineError(
            "heldout.dataset_unrecorded",
            N_(
                "the execution at %(path)s records no %(kind)s dataset; the held-out "
                "cohort cannot be verified against the classes it was trained on"
            ),
            path=str(execution),
            kind=kind,
        )
    return Path(value).expanduser().resolve()


def _test_fold_images(resolved: ResolvedBoxDataset, dataset: Path) -> tuple[BoxImage, ...]:
    """Reviewed test images, refusing the fold if any image has no label file.

    A missing label file is not a reviewed negative: dropping the image would
    shrink the scored denominator silently, and scoring it as background would
    invent an annotation nobody made. Resolution has already hashed the files;
    this refusal happens before the cohort record or any prediction is built,
    so the cohort the result reports is the cohort it scored.
    """
    unlabelled = [image.path.name for image in resolved.unlabelled_images(TEST_SPLIT)]
    if unlabelled:
        sample = unlabelled[:FINDING_SAMPLE_LIMIT]
        raise PipelineError(
            "heldout.unlabelled_test_image",
            N_(
                "the test fold of %(path)s is not fully annotated: missing label "
                "files: %(count)s (first: %(first)s). An image whose label file is "
                "absent is refused rather than scored as background; write an empty "
                "label file for a verified negative or remove the image"
            ),
            path=str(dataset),
            count=len(unlabelled),
            first=", ".join(sample),
            images=sample,
        )
    return resolved.eligible_images(TEST_SPLIT)


def _box_cohort(dataset: Path) -> dict[str, Any]:
    """Resolve the box test fold and hash exactly the files it holds."""
    declaration = DatasetDeclaration.load(dataset)
    test = declaration.split(TEST_SPLIT)
    if test is None:
        raise HeldOutNotEvaluated(N_("no test split is declared in %(path)s"), path=str(dataset))
    if test.kind is not SplitKind.DIRECTORY or not test.exists:
        raise HeldOutNotEvaluated(
            N_("the test split of %(path)s is not an image directory"), path=str(dataset)
        )
    if not test.images():
        raise HeldOutNotEvaluated(
            N_("the test split of %(path)s holds no images"), path=str(dataset)
        )
    resolved = resolve_box_dataset(dataset, TEST_SPLIT)
    images = _test_fold_images(resolved, dataset)
    root = dataset.parent

    def key(path: Path) -> str:
        if path.is_relative_to(root):
            return path.relative_to(root).as_posix()
        return f"{TEST_SPLIT}/{path.name}"

    hashes = {"data.yaml": sha256_file(dataset)}
    for image in images:
        # _test_fold_images refused every ineligible image of this fold.
        assert image.label is not None
        hashes[key(image.path)] = image.sha256
        hashes[key(image.label)] = sha256_file(image.label)
    return {
        "kind": "yolo_boxes",
        "dataset": str(dataset),
        "split": TEST_SPLIT,
        "images": len(images),
        "hashes": hashes,
        "fingerprint": dataset_fingerprint(hashes),
        "paths": [image.path for image in images],
        "ground_truth": resolved.ground_truth(TEST_SPLIT),
        "classes": declaration.class_names(),
        "class_count": declaration.class_count,
    }


def _mask_cohort(dataset: Path) -> dict[str, Any]:
    """Resolve the COCO test fold and hash exactly the files it holds."""
    coco = read_coco_store(dataset)
    images = split_coco_images(coco, TEST_SPLIT)
    if not images:
        raise HeldOutNotEvaluated(N_("no test split is declared in %(path)s"), path=str(dataset))
    hashes = {"dataset.json": sha256_file(dataset / "dataset.json")}
    paths: list[Path] = []
    for image in images:
        relative = str(image["file_name"])
        path = dataset / relative
        try:
            hashes[relative] = sha256_file(path)
        except OSError as exc:
            raise PipelineError(
                "heldout.test_image_missing",
                N_("the test cohort of %(path)s is missing %(image)s"),
                path=str(dataset),
                image=relative,
            ) from exc
        paths.append(path)
    classes = [name for _category_id, name in ordered_categories(coco)]
    return {
        "kind": "coco_instances",
        "dataset": str(dataset),
        "split": TEST_SPLIT,
        "images": len(images),
        "hashes": hashes,
        "fingerprint": dataset_fingerprint(hashes),
        "paths": paths,
        "ground_truth": coco_store_ground_truth(coco, TEST_SPLIT),
        "classes": classes,
        "class_count": len(classes),
    }


def _cohort(dataset: Path, task: str) -> dict[str, Any]:
    """The test fold of the recorded dataset, in the evaluator's own order."""
    if not dataset.exists():
        raise PipelineError(
            "heldout.dataset_missing",
            N_("the test cohort declaration is missing: %(path)s"),
            path=str(dataset),
        )
    return _box_cohort(dataset) if task == "detect" else _mask_cohort(dataset)


def _dataset_classes(dataset: Path, task: str) -> list[str]:
    """The class names a dataset declaration holds, in the family's dense order."""
    try:
        if task == "detect":
            return DatasetDeclaration.load(dataset).class_names()
        return [name for _category_id, name in ordered_categories(read_coco_store(dataset))]
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        raise PipelineError(
            "heldout.classes_unreadable",
            N_("cannot read the classes of the execution's dataset %(path)s: %(reason)s"),
            path=str(dataset),
            reason=exc.display() if isinstance(exc, CodedFailure) else str(exc),
        ) from exc


def _require_same_classes(
    execution: Path, execution_classes: list[str], cohort_classes: list[str]
) -> None:
    """Refuse a cohort that does not describe the execution's own classes.

    A held-out figure only means something on the classes the model was
    trained to predict: matching class counts are not enough, because a
    cohort that renames or reorders them would score one polymer as another.
    """
    if execution_classes == cohort_classes:
        return
    raise PipelineError(
        "heldout.class_mismatch",
        N_(
            "the test cohort of %(path)s declares the classes %(cohort)s, but the "
            "execution was trained on %(execution)s; the held-out cohort must name "
            "the execution's own classes"
        ),
        path=str(execution),
        cohort=cohort_classes,
        execution=execution_classes,
    )


def _predictions(
    family: dict[str, Any],
    task: str,
    checkpoint: Path,
    paths: list[Path],
    *,
    image_size: int,
    class_count: int,
    device: str,
    batch: int,
) -> list[dict]:
    """Predict from the frozen native weights, never from a Lightning run.

    Each family keeps its own inference path; the cohort, the low prediction
    floor, the per-category budget and the COCO conversion are shared, so the
    evaluator sees the same candidates it would during validation.
    """
    if str(family.get("framework", "")) == "torchvision":
        from polyx.pipeline.adapters import mask_rcnn_predictions

        return mask_rcnn_predictions(
            checkpoint,
            paths,
            weights_name=str(family.get("weights") or ""),
            class_count=class_count,
            image_size=image_size,
            device=device,
        )
    return _ultralytics_predictions(
        task,
        checkpoint,
        paths,
        image_size=image_size,
        class_count=class_count,
        device=device,
        batch=batch,
    )


def _ultralytics_predictions(
    task: str,
    checkpoint: Path,
    paths: list[Path],
    *,
    image_size: int,
    class_count: int,
    device: str,
    batch: int,
) -> list[dict]:
    """Ultralytics owns letterboxing, NMS and polygon decoding here.

    The same low prediction floor and per-category budget the validation path
    uses are applied, so the operating point and the ranking tail come from
    the same candidates.
    """
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    raw = model.predict(
        source=[str(path) for path in paths],
        imgsz=image_size,
        batch=batch,
        conf=PREDICTION_FLOOR,
        max_det=prediction_budget(class_count),
        device=device,
        verbose=False,
    )
    results = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    image_ids = list(range(len(paths)))
    if task == "detect":
        return coco_predictions_from_results(results, image_ids=image_ids)
    return coco_segmentation_predictions_from_results(results, image_ids=image_ids)


def _confidence(protocol: dict[str, Any], requested: float | None) -> tuple[float, str]:
    """The one threshold this run applies and where it came from."""
    if requested is None:
        value = float(protocol.get("confidence", 0.25))
        source = "protocol"
    else:
        value = float(requested)
        source = "cli"
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise PipelineError(
            "heldout.confidence_range",
            N_("the confidence threshold must be within [0, 1], got %(value)s"),
            value=str(value),
        )
    return value, source


def _artifact(
    *,
    execution: Path,
    checkpoint: Path,
    completion: dict[str, Any],
    family: dict[str, Any],
    task: str,
    cohort: dict[str, Any],
    confidence: float,
    confidence_source: str,
    device: str,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """The separate held-out record, naming every fact a reader must verify."""
    primary = evaluation["figures"]["with_class_score"]
    return {
        "action": "held_out_test_evaluation",
        "status": "scored",
        "source_execution": str(execution),
        "family": {
            "name": str(family.get("name", "")),
            "framework": str(family.get("framework", "")),
            "task": task,
        },
        "split": TEST_SPLIT,
        "task": task,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "selected_epoch": completion.get("selected_epoch"),
        },
        "execution_dataset_fingerprint": completion.get("dataset_fingerprint"),
        "test_cohort": {
            key: value for key, value in cohort.items() if key not in ("paths", "ground_truth")
        },
        "confidence_threshold": confidence,
        "confidence_source": confidence_source,
        "iou_threshold": float(evaluation["selection"]["iou_threshold"]),
        "overlap": "mask IoU" if task == "segment" else "box IoU",
        "device": device,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "primary": {
            "metric": "with_class_score",
            "value": primary["value"],
            "criterion": primary["criterion"],
        },
        "evaluation": evaluation,
    }


def _markdown(artifact: dict[str, Any]) -> str:
    """A short report; the JSON beside it stays the machine contract.

    Every label is a translated message with named placeholders, so a
    translator can reorder a label and its value. The JSON keys and their
    machine tokens (the split, the task, the cohort source, the threshold
    source) never pass through gettext (ADR-0004).
    """
    cohort = artifact["test_cohort"]
    checkpoint = artifact["checkpoint"]
    lines = [
        f"# {_('Held-out test evaluation')}",
        "",
        _("- Source execution: %(path)s") % {"path": artifact["source_execution"]},
        _("- Family: %(name)s (%(framework)s)")
        % {"name": artifact["family"]["name"], "framework": artifact["family"]["framework"]},
        _("- Task: %(task)s") % {"task": artifact["task"]},
        _("- Split scored: %(split)s") % {"split": artifact["split"]},
        _("- Checkpoint: %(path)s") % {"path": checkpoint["path"]},
        _("- Checkpoint SHA-256: %(sha256)s") % {"sha256": checkpoint["sha256"]},
        _("- Selected epoch: %(epoch)s") % {"epoch": checkpoint["selected_epoch"]},
        _("- Test cohort: %(images)s images") % {"images": cohort["images"]},
        _("- Test cohort source: %(source)s") % {"source": cohort["source"]},
        _("- Test cohort fingerprint: %(fingerprint)s") % {"fingerprint": cohort["fingerprint"]},
        _("- Execution dataset fingerprint: %(fingerprint)s")
        % {"fingerprint": cohort["execution_fingerprint"]},
        _("- Confidence threshold: %(threshold).2f (%(source)s)")
        % {
            "threshold": artifact["confidence_threshold"],
            "source": artifact["confidence_source"],
        },
        _("- Overlap rule: %(overlap)s >= %(iou).2f")
        % {"overlap": artifact["overlap"], "iou": artifact["iou_threshold"]},
        _("- Device: %(device)s") % {"device": artifact["device"]},
        _("- Evaluated at: %(moment)s") % {"moment": artifact["evaluated_at"]},
        "",
        f"## {_('Figures')}",
        "",
        f"| {_('Figure')} | {_('Value')} | {_('Criterion')} |",
        "| --- | --- | --- |",
    ]
    for name, figure in artifact["evaluation"]["figures"].items():
        lines.append(f"| {name} | {figure['value']:.6f} | {figure['criterion']} |")
    if artifact["evaluation"].get("per_class"):
        lines.extend(
            [
                "",
                f"## {_('Per class')}",
                "",
                f"| {_('Class')} | {_('Figure')} | {_('Value')} | {_('Criterion')} |",
                "| --- | --- | --- | --- |",
            ]
        )
        for class_name, figures in artifact["evaluation"]["per_class"].items():
            for name, figure in figures.items():
                lines.append(
                    f"| {class_name} | {name} | {figure['value']:.6f} | {figure['criterion']} |"
                )
    lines.extend(
        [
            "",
            _(
                "The primary figure is the class-aware operating-point F1, the same "
                "definition the shared evaluator uses for model selection. It is "
                "reported here once, at the recorded threshold; this action does not "
                "tune or recommend a threshold."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_heldout(
    execution: Path,
    output: Path,
    *,
    dataset: Path | None = None,
    confidence: float | None = None,
    device: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Score one finished execution's frozen checkpoint on the test fold.

    The execution is read, never written. The result names the split, the
    cohort fingerprint, the checkpoint digest and the fixed threshold, and it
    is written under ``output``; an existing result there is kept unless
    ``force`` asks to replace it.
    """
    execution = Path(execution).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output == execution or output.is_relative_to(execution):
        raise PipelineError(
            "heldout.output_inside_execution",
            N_("write the held-out result outside %(path)s: the execution is read-only here"),
            path=str(execution),
        )
    completion = _read_completion(execution)
    family = _required_record(completion, "family", execution)
    protocol = _required_record(completion, "protocol", execution)
    if not isinstance(family, dict):
        family = {}
    if not isinstance(protocol, dict):
        protocol = {}
    framework = str(family.get("framework", ""))
    task = str(family.get("task", ""))
    if task not in SUPPORTED_FAMILIES.get(framework, ()):
        raise PipelineError(
            "heldout.family_unsupported",
            N_(
                "held-out evaluation covers the Ultralytics detection and "
                "segmentation families and the torchvision Mask R-CNN family; "
                "%(path)s was produced by %(family)s (%(framework)s)"
            ),
            path=str(execution),
            family=str(family.get("name", "unknown")),
            framework=framework or "unknown",
        )
    results = (output / RESULT_NAME, output / REPORT_NAME)
    if not force and any(path.exists() for path in results):
        raise PipelineError(
            "heldout.output_in_use",
            N_("%(path)s already holds a held-out result; pass --force to replace it"),
            path=str(output),
        )
    checkpoint = _selected_checkpoint(completion, execution)
    declared = _declared_dataset(execution, task)
    cohort_dataset = Path(dataset).expanduser().resolve() if dataset is not None else declared
    cohort = _cohort(cohort_dataset, task)
    _require_same_classes(execution, _dataset_classes(declared, task), cohort["classes"])
    cohort["source"] = "declared" if cohort_dataset == declared else "external"
    cohort["execution_fingerprint"] = completion.get("dataset_fingerprint")
    threshold, threshold_source = _confidence(protocol, confidence)
    backend = completion.get("backend")
    recorded_device = str(backend.get("device") or "") if isinstance(backend, dict) else ""
    inference_device = str(device or recorded_device or "cpu")
    image_size = int(protocol.get("image_size", 1280))
    requested_batch = int(protocol.get("batch", 8) or 1)
    batch = max(1, min(requested_batch, len(cohort["paths"])))
    predictions = _predictions(
        family,
        task,
        checkpoint,
        cohort["paths"],
        image_size=image_size,
        class_count=int(cohort["class_count"]),
        device=inference_device,
        batch=batch,
    )
    evaluator = SharedEvaluator(
        cohort["ground_truth"],
        TEST_SPLIT,
        task=task,
        confidence_threshold=threshold,
    )
    evaluation = evaluator.evaluate(predictions)
    artifact = _artifact(
        execution=execution,
        checkpoint=checkpoint,
        completion=completion,
        family=family,
        task=task,
        cohort=cohort,
        confidence=threshold,
        confidence_source=threshold_source,
        device=inference_device,
        evaluation=evaluation,
    )
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULT_NAME
    report_path = output / REPORT_NAME
    write_json(artifact, result_path)
    report_path.write_text(_markdown(artifact), encoding="utf-8")
    return {"artifact": artifact, "result": result_path, "report": report_path}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one held-out evaluation and report where its artifact landed."""
    parser = argparse.ArgumentParser(
        description=_("Score a frozen checkpoint on the held-out test cohort")
    )
    parser.add_argument(
        "--execution",
        required=True,
        type=Path,
        help=_("the finished training execution directory"),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=_("where to write heldout_test.json and heldout_test.md"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=_(
            "the test cohort: a YOLO data.yaml or a canonical COCO store; "
            "defaults to the execution's dataset"
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help=_("one fixed confidence threshold (default: the execution's protocol confidence)"),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=_("the inference device (default: the execution's backend device)"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=_("replace an existing held-out result in --output"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = evaluate_heldout(
            args.execution,
            args.output,
            dataset=args.dataset,
            confidence=args.confidence,
            device=args.device,
            force=args.force,
        )
    except HeldOutNotEvaluated as exc:
        print(_("NOT EVALUATED: %(reason)s") % {"reason": exc.display()}, file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - the command reports any failure as exit 1
        report_failure(exc, stage="held-out")
        return 1

    artifact = result["artifact"]
    primary = artifact["primary"]
    localisation = artifact["evaluation"]["figures"]["localisation_score"]
    print(f"RESULT: {result['result']}")
    print(f"SPLIT: {artifact['split']}")
    print(f"PRIMARY: {primary['metric']}={primary['value']:.6f}")
    print(f"CRITERION: {primary['criterion']}")
    print(f"LOCALISATION: {localisation['value']:.6f}")
    print(f"CHECKPOINT_SHA256: {artifact['checkpoint']['sha256']}")
    print(f"TEST_COHORT: {artifact['test_cohort']['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
