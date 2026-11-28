"""Deep audit of a YOLO dataset before training.

It goes beyond "does the file exist?": it analyzes the class distribution
between train and val, detects classes that are never validated (the classic
error that inflates the metrics), dominance by a few images,
degenerate/out-of-range boxes, and classes indistinguishable by size. Designed
so the lab chemist does not train blindly on a bad split.

Public function: ``audit_dataset(yaml_path) -> DatasetAudit``.
Pure Python (no Qt) → testable headless."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polyx.core.data import SPLIT_NAMES, DatasetDeclaration, SplitDeclaration, SplitKind
from polyx.core.errors import CodedFailure
from polyx.core.i18n import _
from polyx.core.labels import BBox, find_label_file, read_boxes

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"

#: The audit's artifact codes stay bare, one token per cause. A declaration
#: failure keeps its code where the audit already had one; the rest are named
#: here instead of leaking the seam's dotted prefix into saved findings.
_FINDING_CODES = {
    "dataset.data_yaml_missing": "data_yaml_missing",
    "dataset.data_yaml_unreadable": "data_yaml_unreadable",
    "dataset.declaration_not_mapping": "data_yaml_not_mapping",
    "dataset.names_invalid": "classes_invalid",
    "dataset.split_list_unreadable": "split_list_unreadable",
}


def _finding_code(code: str) -> str:
    """Map a declaration failure to the audit's artifact code; no unknown passes."""
    return _FINDING_CODES[code]


#: How many file names one finding names. The count is complete; the list is a
#: readable sample so a saved artifact does not carry a folder.
_FINDING_SAMPLE = 10


@dataclass
class Finding:
    level: str  # ok | warn | error
    title: str
    detail: str = ""
    #: Stable machine identity for saved artifacts. The prose is translated at
    #: display time; only the code and its parameters survive serialization.
    code: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetAudit:
    findings: list[Finding] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    names_by_id: dict[int, str] = field(default_factory=dict)  # class id -> name
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)  # split -> {class: n}
    box_area_median: dict[str, float] = field(
        default_factory=dict
    )  # class -> median normalised area
    ok: bool = True  # False if there is any Finding at error level

    def add(
        self,
        level: str,
        title: str,
        detail: str = "",
        *,
        code: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Record one finding with a stable machine code and its parameters.

        ``title`` and ``detail`` are translated for people reading the report;
        only ``code`` and ``parameters`` are written to scientific artifacts.
        """
        self.findings.append(Finding(level, title, detail, code=code, parameters=parameters or {}))
        if level == LEVEL_ERROR:
            self.ok = False


# ──────────────────────────────────────────────────────────────────
def _scan_split(images: list[Path]) -> dict:
    """Reads labels of a split. Returns per-class counts, box stats and issues.

    Labels are located and parsed by the label codec, so the split layout and
    the polygon dialect are understood in one place: a YOLO-seg polygon line
    is valid YOLO but not a box, and the codec's box reader counts it
    malformed instead of misreading its first four coordinates as a box. The
    count is returned so the audit can report it instead of dropping it.
    """

    #: Labels the codec found somewhere the pinned loader does not read. The
    #: shipped GUI/CLI trainer hands the YAML to Ultralytics, which resolves a
    #: label only at ``images/`` -> ``labels/`` (or beside a plain-folder image);
    #: every other codec location would train that image as background.
    mismatched: list[tuple[str, str]] = []

    def records() -> Iterable[tuple[Path, Sequence[BBox] | None, int]]:
        from ultralytics.data.utils import img2label_paths

        for img in images:
            label = find_label_file(img)
            native = Path(img2label_paths([str(img)])[0])
            if label != native and (label is not None or native.is_file()):
                mismatched.append((img.name, str(native)))
            if label is None:
                yield img, None, 0
            else:
                boxes, malformed = read_boxes(label)
                yield img, boxes, malformed

    scan = _scan_box_records(records())
    scan["label_location_mismatch"] = mismatched
    return scan


def _scan_box_records(records: Iterable[tuple[Path, Sequence[BBox] | None, int]]) -> dict:
    per_class: dict[int, int] = defaultdict(int)
    areas: dict[int, list[float]] = defaultdict(list)
    per_class_per_img: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )  # class -> {img: n}
    degenerate = out_range = duplicates = empty = no_label = malformed = 0
    #: A bounded sample of the images that carry no label file at all, so the
    #: refusal can name files a chemist can open instead of only a count.
    unlabelled: list[str] = []
    n_imgs = 0
    for img, boxes, malformed_lines in records:
        n_imgs += 1
        if boxes is None:
            no_label += 1
            if len(unlabelled) < _FINDING_SAMPLE:
                unlabelled.append(img.name)
            continue
        malformed += malformed_lines
        seen = set()
        for box in boxes:
            c = box.class_id
            per_class[c] += 1
            per_class_per_img[c][img.name] += 1
            areas[c].append(box.w * box.h)
            if box.w <= 0 or box.h <= 0 or box.w < 1e-4 or box.h < 1e-4:
                degenerate += 1
            if not (0 <= box.cx <= 1 and 0 <= box.cy <= 1 and 0 <= box.w <= 1 and 0 <= box.h <= 1):
                out_range += 1
            key = (c, round(box.cx, 5), round(box.cy, 5), round(box.w, 5), round(box.h, 5))
            if key in seen:
                duplicates += 1
            seen.add(key)
        if not boxes:
            empty += 1
    return dict(  # noqa: C408
        per_class=dict(per_class),
        areas=dict(areas),
        per_class_per_img={k: dict(v) for k, v in per_class_per_img.items()},
        degenerate=degenerate,
        out_range=out_range,
        duplicates=duplicates,
        empty=empty,
        no_label=no_label,
        no_label_images=unlabelled,
        label_location_mismatch=[],
        malformed=malformed,
        n_imgs=n_imgs,
    )


# ── Auditor: method object holding the shared context ────────────
class _Auditor:
    """Holds the mutable state shared by all the audit checks.

    Replaces the free functions that used to receive ``audit``/``tr``/``vl``/
    ``cls_name`` as arguments. Each check is now a method reading ``self.*``.
    """

    def __init__(self, yaml_path: Path | None = None):
        self.yaml_path = yaml_path or Path()
        self.audit = DatasetAudit()
        self.declaration: DatasetDeclaration | None = None
        self.scans: dict[str, dict[str, Any]] = {}
        self.tr: dict[str, Any] = {}
        self.vl: dict[str, Any] = {}
        self.source = "yolo"

    @classmethod
    def from_scans(cls, scans: dict[str, dict[str, Any]], class_names: list[str]) -> _Auditor:
        """An auditor over label scans built from another label source.

        The COCO-mask entry point builds the same per-split scan dictionaries a
        YOLO dataset produces, then hands them to the same check methods. It is
        an alternate *input* to the auditor, not a second validator.
        """
        auditor = cls()
        auditor.source = "coco"
        auditor.audit = DatasetAudit(class_names=list(class_names))
        auditor.scans = scans
        auditor.tr = scans["train"]
        auditor.vl = scans["val"]
        return auditor

    # ── orchestration ──
    def run(self) -> DatasetAudit:
        audit = self.audit
        try:
            self.declaration = DatasetDeclaration.load(self.yaml_path)
        except CodedFailure as exc:
            title = (
                _("data.yaml not found")
                if exc.code == "dataset.data_yaml_missing"
                else _("Could not read data.yaml")
            )
            audit.add(
                LEVEL_ERROR,
                title,
                exc.display(),
                code=_finding_code(exc.code),
                parameters=exc.parameters,
            )
            return audit

        audit.class_names = self.declaration.class_names()
        audit.names_by_id = dict(self.declaration.names)
        declared_nc = self.declaration.declared_nc
        if declared_nc is not None and declared_nc != self.declaration.class_count:
            audit.add(
                LEVEL_WARN,
                _("nc does not match the declared names"),
                _("The data.yaml declares nc: %(nc)s but names %(count)s classes.")
                % {"nc": declared_nc, "count": self.declaration.class_count},
                code="nc_mismatch",
                parameters={"nc": declared_nc, "count": self.declaration.class_count},
            )
        self._scan_all_splits()
        return self.run_checks()

    def run_checks(self) -> DatasetAudit:
        """The checks shared by YOLO scans and COCO-mask scans."""
        self._check_structure()
        self._check_class_coverage()
        self._check_imbalance()
        self._check_dominance()
        self._check_size_similarity()
        self._check_mechanical()
        self._check_label_files()
        self._check_polygons()
        self._check_val_size()
        self._finalize()
        return self.audit

    # ── helpers ──
    def _cls_name(self, cid: int) -> str:
        audit = self.audit
        name = audit.names_by_id.get(cid)
        if name is not None:
            return name
        return (
            audit.class_names[cid]
            if 0 <= cid < len(audit.class_names)
            else _("class %(id)s") % {"id": cid}
        )

    def _scan_all_splits(self) -> None:
        assert self.declaration is not None
        for split in SPLIT_NAMES:
            declared = self.declaration.split(split)
            try:
                images = list(declared.images()) if declared is not None else []
            except CodedFailure as exc:
                self.audit.add(
                    LEVEL_ERROR,
                    _("Could not read a split list"),
                    exc.display(),
                    code=_finding_code(exc.code),
                    parameters=exc.parameters,
                )
                images = []
            self._record_box_split(split, declared, _scan_split(images))
            if declared is not None and declared.kind is SplitKind.LIST:
                self.audit.add(
                    LEVEL_WARN,
                    _("%(split)s split is a text list") % {"split": split},
                    _(
                        "A .txt split list can be audited and previewed, but a "
                        "headless box execution needs image directories."
                    ),
                    code="split_list_not_trainable",
                    parameters={"split": split},
                )
        self._finish_box_scans()

    def _record_box_split(self, split: str, declared: SplitDeclaration | None, scan: dict) -> None:
        audit = self.audit
        exists = declared.exists if declared is not None else False
        path = declared.path if declared is not None else None
        if declared is None and split in ("train", "val"):
            audit.add(
                LEVEL_ERROR,
                _("No %(split)s split declared") % {"split": split},
                _(
                    "The data.yaml has no '%(split)s' key. Ultralytics needs it "
                    "to train and measure metrics."
                )
                % {"split": split},
                code="split_missing",
                parameters={"split": split},
            )
        elif declared is not None and not exists:
            audit.add(
                LEVEL_ERROR if split in ("train", "val") else LEVEL_WARN,
                _("%(split)s path does not exist") % {"split": split},
                _(
                    "The data.yaml points to “%(path)s”, which does not exist on this "
                    "computer. This usually happens when the project is moved to another "
                    "folder or PC: fix the paths in the data.yaml (or use relative paths)."
                )
                % {"path": path},
                code="split_path_missing",
                parameters={"split": split, "path": str(path)},
            )
        scan["path_exists"] = exists
        self.scans[split] = scan
        audit.per_class[split] = {self._cls_name(c): n for c, n in scan["per_class"].items()}

    def _finish_box_scans(self) -> None:
        self.tr = self.scans["train"]
        self.vl = self.scans["val"]
        for c, ar in self.tr["areas"].items():
            if ar:
                self.audit.box_area_median[self._cls_name(c)] = statistics.median(ar)

    def _check_structure(self) -> None:
        audit, tr, vl = self.audit, self.tr, self.vl
        if not audit.class_names:
            audit.add(
                LEVEL_ERROR,
                _("No classes defined"),
                _("The data.yaml has no 'names' key."),
                code="classes_missing",
            )
        if tr["n_imgs"] == 0 and tr.get("path_exists"):
            audit.add(
                LEVEL_ERROR,
                _("Empty train"),
                _("The train folder exists but contains no images."),
                code="train_empty",
            )
        if vl["n_imgs"] == 0 and vl.get("path_exists"):
            audit.add(
                LEVEL_ERROR,
                _("Empty val"),
                _("The val folder exists but contains no images → metrics cannot be measured."),
                code="val_empty",
            )
        if self.source == "coco" and tr["n_imgs"] > 0 and vl["n_imgs"] == 0:
            audit.add(
                LEVEL_WARN,
                _("No validation split in the COCO export"),
                _(
                    "Every mask was imported into train because no image carries a split "
                    "marker. Mark the validation images before training: class coverage and "
                    "metrics cannot be audited without them."
                ),
                code="coco_val_split_missing",
            )

    def _check_class_coverage(self) -> None:
        audit, tr, vl = self.audit, self.tr, self.vl
        train_classes = set(tr["per_class"].keys())
        val_classes = set(vl["per_class"].keys())

        missing_in_val = train_classes - val_classes
        if missing_in_val and vl["n_imgs"] > 0:
            names_2 = ", ".join(sorted(self._cls_name(c) for c in missing_in_val))
            audit.add(
                LEVEL_ERROR,
                _("Classes without validation: %(classes)s") % {"classes": names_2},
                _(
                    "These classes have examples in train but ZERO in val. mAP will not "
                    "measure their performance (it will give you a false sense of success). "
                    "Add images of these classes to the validation split."
                ),
                code="classes_missing_in_val",
                parameters={"classes": sorted(self._cls_name(c) for c in missing_in_val)},
            )
        if len(val_classes) == 1 and len(train_classes) > 1:
            audit.add(
                LEVEL_ERROR,
                _("Single-class validation (%(class)s)")
                % {"class": self._cls_name(next(iter(val_classes)))},
                _(
                    "The validation set contains a single class even though the model "
                    "learns several. Multiclass metrics are not valid."
                ),
                code="single_class_val",
                parameters={
                    "class": self._cls_name(next(iter(val_classes))),
                    "train_classes": sorted(self._cls_name(c) for c in train_classes),
                },
            )

    def _check_imbalance(self) -> None:
        audit, tr = self.audit, self.tr
        counts = tr["per_class"]
        if len(counts) >= 2:
            mx, mn = max(counts.values()), min(counts.values())
            if mn > 0 and mx / mn >= 10:
                top = self._cls_name(max(counts, key=counts.get))
                bot = self._cls_name(min(counts, key=counts.get))
                audit.add(
                    LEVEL_WARN,
                    _("Strong class imbalance in train"),
                    _(
                        "“%(top)s” has %(top_count)d boxes and “%(bottom)s” only "
                        "%(bottom_count)d (%(ratio).0f× more). Consider balancing or using "
                        "class weights."
                    )
                    % {
                        "top": top,
                        "top_count": mx,
                        "bottom": bot,
                        "bottom_count": mn,
                        "ratio": mx / mn,
                    },
                    code="class_imbalance",
                    parameters={"top": top, "bottom": bot, "ratio": mx / mn},
                )

    def _check_dominance(self) -> None:
        audit, tr = self.audit, self.tr
        for c, per_img in tr["per_class_per_img"].items():
            total = tr["per_class"].get(c, 0)
            if total >= 50 and per_img:
                top_img, top_n = max(per_img.items(), key=lambda kv: kv[1])
                if top_n / total >= 0.40:
                    audit.add(
                        LEVEL_WARN,
                        _("“%(class)s”: little diversity") % {"class": self._cls_name(c)},
                        _(
                            "%(percent).0f%% of its boxes come from a single image "
                            "(%(image)s). The model may overfit to that scene."
                        )
                        % {
                            "percent": top_n / total * 100,
                            "image": top_img,
                        },
                        code="class_dominated_by_image",
                        parameters={
                            "class": self._cls_name(c),
                            "image": top_img,
                            "share": top_n / total,
                        },
                    )

    def _check_size_similarity(self) -> None:
        audit = self.audit
        meds = audit.box_area_median
        cls_list = list(meds.items())
        for i in range(len(cls_list)):
            for j in range(i + 1, len(cls_list)):
                (na, aa), (nb, ab) = cls_list[i], cls_list[j]
                if aa > 0 and ab > 0:
                    ratio = max(aa, ab) / min(aa, ab)
                    if ratio < 1.25:
                        audit.add(
                            LEVEL_WARN,
                            _("“%(first)s” and “%(second)s” are almost identical in size")
                            % {"first": na, "second": nb},
                            _(
                                "Their boxes are the same size, so the model can only tell "
                                "them apart by colour/texture. If they also look alike, "
                                "expect confusion between the two."
                            ),
                            code="similar_class_sizes",
                            parameters={"first": na, "second": nb, "ratio": ratio},
                        )

    def _check_mechanical(self) -> None:
        audit, tr, vl = self.audit, self.tr, self.vl
        for split, sc in (("train", tr), ("val", vl)):
            if sc["degenerate"]:
                audit.add(
                    LEVEL_WARN,
                    _("Degenerate boxes in %(split)s: %(count)d")
                    % {"split": split, "count": sc["degenerate"]},
                    _("Boxes with width/height ~0. Check the annotation."),
                    code="degenerate_boxes",
                    parameters={"split": split, "count": sc["degenerate"]},
                )
            if sc["out_range"]:
                audit.add(
                    LEVEL_ERROR,
                    _("Coordinates outside [0,1] in %(split)s: %(count)d")
                    % {"split": split, "count": sc["out_range"]},
                    _("Badly normalised labels; YOLO will interpret them wrongly."),
                    code="out_of_range_boxes",
                    parameters={"split": split, "count": sc["out_range"]},
                )
            if sc["duplicates"]:
                audit.add(
                    LEVEL_WARN,
                    _("Duplicate boxes in %(split)s: %(count)d")
                    % {"split": split, "count": sc["duplicates"]},
                    _("The same box repeated within one file."),
                    code="duplicate_boxes",
                    parameters={"split": split, "count": sc["duplicates"]},
                )
        malformed = sum(scan["malformed"] for scan in self.scans.values())
        if malformed:
            audit.add(
                LEVEL_WARN,
                _("Label lines that are not detection boxes: %(count)d") % {"count": malformed},
                _(
                    "Those lines were skipped. A YOLO-seg polygon line is valid YOLO "
                    "but not a detection box, so the audit counts boxes only."
                ),
                code="non_box_label_lines",
                parameters={"count": malformed},
            )

    def _check_label_files(self) -> None:
        """Where each split's labels live, as the audit and the trainer see it.

        Every scanned split is checked, a declared test fold included: a
        missing label file is not a reviewed negative, and a label only the
        codec can find would be read as background by the native trainer.
        """
        audit = self.audit
        for split, sc in self.scans.items():
            count = int(sc["no_label"])
            if count:
                sample = list(sc["no_label_images"])
                audit.add(
                    LEVEL_ERROR,
                    _("Images with no label file in %(split)s: %(count)d")
                    % {"split": split, "count": count},
                    _(
                        "An absent label file is not a reviewed negative: without one, the "
                        "image would enter the ground truth as background without anyone "
                        "having verified it holds no particles. Write an empty label file "
                        "for a verified negative, or remove the image. Unlabelled images: "
                        "%(images)s (%(shown)d of %(count)d)."
                    )
                    % {"images": ", ".join(sample), "shown": len(sample), "count": count},
                    code="images_without_labels",
                    parameters={"split": split, "count": count, "images": sample},
                )
            mismatched = list(sc["label_location_mismatch"])
            if mismatched:
                sample = mismatched[:_FINDING_SAMPLE]
                audit.add(
                    LEVEL_ERROR,
                    _("Labels the training loader cannot find in %(split)s: %(count)d")
                    % {"split": split, "count": len(mismatched)},
                    _(
                        "The native trainer resolves each label from its image path; "
                        "these files exist only at a legacy location it does not read, "
                        "so it would train those images as background. Move each label "
                        "to the path the loader expects. Images: %(images)s "
                        "(%(shown)d of %(count)d). First expected path: %(expected)s."
                    )
                    % {
                        "images": ", ".join(name for name, _expected in sample),
                        "shown": len(sample),
                        "count": len(mismatched),
                        "expected": sample[0][1],
                    },
                    code="label_location_not_native",
                    parameters={
                        "split": split,
                        "count": len(mismatched),
                        "images": [name for name, _expected in sample],
                        "expected": [expected for _name, expected in sample],
                    },
                )

    def _check_polygons(self) -> None:
        """Degenerate or structurally invalid mask polygons reject a dataset.

        Only COCO-mask scans carry these counters; YOLO box scans keep them at
        zero, so the shared check runs harmlessly for both label sources.
        """
        audit = self.audit
        for split, scan in self.scans.items():
            invalid = int(scan.get("invalid_polygons", 0))
            degenerate = int(scan.get("degenerate_polygons", 0))
            unknown = int(scan.get("unknown_categories", 0))
            if invalid:
                audit.add(
                    LEVEL_ERROR,
                    _("Invalid mask polygons in %(split)s: %(count)d")
                    % {"split": split, "count": invalid},
                    _(
                        "A mask polygon needs at least three finite, in-bounds points and "
                        "an even coordinate count. RLE segmentation is not a polygon; "
                        "export polygons from the annotation tool."
                    ),
                    code="invalid_mask_polygons",
                    parameters={"split": split, "count": invalid},
                )
            if degenerate:
                audit.add(
                    LEVEL_ERROR,
                    _("Degenerate mask polygons in %(split)s: %(count)d")
                    % {"split": split, "count": degenerate},
                    _(
                        "These polygons enclose zero area (collinear or repeated points) "
                        "and cannot be used as segmentation ground truth."
                    ),
                    code="degenerate_mask_polygons",
                    parameters={"split": split, "count": degenerate},
                )
            if unknown:
                audit.add(
                    LEVEL_ERROR,
                    _("Annotations with an unknown category in %(split)s: %(count)d")
                    % {"split": split, "count": unknown},
                    _(
                        "Every mask must reference a category declared in the export's "
                        "categories list."
                    ),
                    code="unknown_mask_category",
                    parameters={"split": split, "count": unknown},
                )
        orphans = sum(int(scan.get("orphan_annotations", 0)) for scan in self.scans.values())
        if orphans:
            audit.add(
                LEVEL_ERROR,
                _("Annotations with an unknown image: %(count)d") % {"count": orphans},
                _(
                    "These masks reference an image that is not declared in the export's "
                    "images list."
                ),
                code="orphan_mask_image",
                parameters={"count": orphans},
            )

    def _check_val_size(self) -> None:
        audit, vl = self.audit, self.vl
        if 0 < vl["n_imgs"] < 5:
            audit.add(
                LEVEL_WARN,
                _("Very small validation set (%(count)d images)") % {"count": vl["n_imgs"]},
                _("With so few images the metrics are unstable. Aim for ≥ 10-20."),
                code="small_validation_set",
                parameters={"count": vl["n_imgs"]},
            )

    def _finalize(self) -> None:
        audit = self.audit
        if audit.ok and not any(f.level == LEVEL_WARN for f in audit.findings):
            audit.add(
                LEVEL_OK,
                _("No dataset problems detected"),
                _("Class distribution, validation and format all look correct."),
                code="no_problems",
            )


# ──────────────────────────────────────────────────────────────────
def audit_dataset(yaml_path: str | Path) -> DatasetAudit:
    return _Auditor(Path(yaml_path)).run()


def _declared_for_audit(split: str, path: Path | None) -> SplitDeclaration | None:
    """A minimal split fact for the audit-only path, which has no declaration."""
    if path is None:
        return None
    return SplitDeclaration(
        name=split,
        kind=SplitKind.DIRECTORY,
        path=path,
        exists=path.exists(),
        outside_root=False,
        _root=path,
        _yaml_dir=path,
    )


def audit_resolved_boxes(
    class_names: Sequence[str],
    split_paths: Mapping[str, Path | None],
    records: Mapping[str, Iterable[tuple[Path, Sequence[BBox] | None]]],
) -> DatasetAudit:
    """Run the same box audit on resolved images and labels without rediscovering files."""
    auditor = _Auditor()
    auditor.audit.class_names = list(class_names)
    for split, path in split_paths.items():
        scan = _scan_box_records((image, boxes, 0) for image, boxes in records[split])
        auditor._record_box_split(split, _declared_for_audit(split, path), scan)
    auditor._finish_box_scans()
    return auditor.run_checks()


# ── COCO mask polygons ─────────────────────────────────────────────────────
# The canonical segmentation ground truth is an external CVAT COCO export.
# The geometry rules below feed the same auditor as the YOLO box path: one
# validator, one DatasetAudit result, two label sources.

_MIN_POLYGON_AREA = 1e-6  # px²; a zero-area ring is not a mask

#: Split names a COCO export may carry as an image-level attribute. Without a
#: marker an image belongs to train, which the audit reports as a warning.
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Absolute shoelace area of one polygon, in pixel squared."""
    total = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def validate_polygon(
    segmentation: Any, width: int, height: int
) -> tuple[list[list[tuple[float, float]]], int, int]:
    """Validate a COCO ``segmentation`` field.

    Returns ``(polygons, invalid_count, degenerate_count)``. A dictionary
    (RLE) is invalid because the project has no RLE codec and the annotation
    tool is asked to export polygons; a flat coordinate list must hold at
    least three finite, in-bounds points and enclose a non-zero area.
    """
    if not isinstance(segmentation, (list, tuple)) or not segmentation:
        return [], 1, 0
    polygons: list[list[tuple[float, float]]] = []
    invalid = degenerate = 0
    for ring in segmentation:
        if not isinstance(ring, (list, tuple)):
            invalid += 1
            continue
        numbers = list(ring)
        if (
            len(numbers) < 6
            or len(numbers) % 2 != 0
            or any(not isinstance(value, (int, float)) for value in numbers)
            or any(not math.isfinite(float(value)) for value in numbers)
        ):
            invalid += 1
            continue
        points = [
            (float(numbers[index]), float(numbers[index + 1]))
            for index in range(0, len(numbers), 2)
        ]
        if any(x < 0 or x > width or y < 0 or y > height for x, y in points):
            invalid += 1
            continue
        if len(set(points)) < 3 or polygon_area(points) <= _MIN_POLYGON_AREA:
            degenerate += 1
            continue
        polygons.append(points)
    return polygons, invalid, degenerate


def coco_image_split(image: dict[str, Any]) -> str:
    """The normalised split marker of one COCO image, ``train`` by default.

    Public because the segmentation execution filters the canonical dataset by
    the same rule the import audit counted: one owner for the convention.
    """
    value: Any = image.get("split")
    attributes = image.get("attributes")
    if value is None and isinstance(attributes, dict):
        value = attributes.get("split")
    elif value is None and isinstance(attributes, list):
        for item in attributes:
            if isinstance(item, dict) and str(item.get("name", "")).lower() == "split":
                value = item.get("value")
                break
    if value is None or not str(value).strip():
        return "train"
    return _SPLIT_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())


def _empty_coco_scan() -> dict[str, Any]:
    """A scan with the keys the box checks read, plus polygon counters."""
    return {
        "per_class": defaultdict(int),
        "areas": defaultdict(list),
        "per_class_per_img": defaultdict(lambda: defaultdict(int)),
        "degenerate": 0,
        "out_range": 0,
        "duplicates": 0,
        "empty": 0,
        "no_label": 0,
        "no_label_images": [],
        "label_location_mismatch": [],
        "malformed": 0,
        "n_imgs": 0,
        "path_exists": False,
        "invalid_polygons": 0,
        "degenerate_polygons": 0,
        "unknown_categories": 0,
        "orphan_annotations": 0,
    }


def audit_coco_masks(coco: dict[str, Any]) -> DatasetAudit:
    """Audit the polygons of a COCO instance dataset with the shared checks.

    The split is read from an image-level ``split`` attribute (CVAT metadata
    or ``attributes``); an image without one belongs to train. COCO category
    ids are arbitrary (CVAT starts at 1), so they are mapped to the sorted
    positions the auditor indexes class names by.
    """
    images: dict[int, dict[str, Any]] = {}
    for item in coco.get("images", []):
        images[int(item["id"])] = item
    category_names = {int(item["id"]): str(item["name"]) for item in coco.get("categories", [])}
    ordered_ids = sorted(category_names)
    id_to_index = {category_id: index for index, category_id in enumerate(ordered_ids)}
    class_names = [category_names[category_id] for category_id in ordered_ids]

    scans: dict[str, dict[str, Any]] = {
        split: _empty_coco_scan() for split in ("train", "val", "test")
    }
    image_split: dict[int, str] = {}
    for image_id, item in images.items():
        split = coco_image_split(item)
        image_split[image_id] = split
        scan = scans.setdefault(split, _empty_coco_scan())
        scan["n_imgs"] += 1
        scan["path_exists"] = True

    for annotation in coco.get("annotations", []):
        try:
            image_id = int(annotation["image_id"])
        except (KeyError, TypeError, ValueError):
            scans["train"]["orphan_annotations"] += 1
            continue
        image = images.get(image_id)
        if image is None:
            # An annotation that belongs to no image cannot be audited per
            # split; it is an error of the export, not a mask to repair.
            scans["train"]["orphan_annotations"] += 1
            continue
        scan = scans[image_split[image_id]]
        try:
            class_id = id_to_index.get(int(annotation["category_id"]))
        except (KeyError, TypeError, ValueError):
            class_id = None
        if class_id is None:
            scan["unknown_categories"] += 1
            continue
        width, height = int(image["width"]), int(image["height"])
        polygons, invalid, degenerate = validate_polygon(
            annotation.get("segmentation"), width, height
        )
        scan["invalid_polygons"] += invalid
        scan["degenerate_polygons"] += degenerate
        image_area = max(width * height, 1)
        for points in polygons:
            area = polygon_area(points) / image_area
            scan["per_class"][class_id] += 1
            scan["areas"][class_id].append(area)
            scan["per_class_per_img"][class_id][str(image_id)] += 1
        if not polygons and not invalid and not degenerate:
            scan["empty"] += 1

    for scan in scans.values():
        scan["per_class"] = dict(scan["per_class"])
        scan["areas"] = dict(scan["areas"])
        scan["per_class_per_img"] = {
            key: dict(value) for key, value in scan["per_class_per_img"].items()
        }

    auditor = _Auditor.from_scans(scans, class_names)
    audit = auditor.audit
    for split, scan in scans.items():
        audit.per_class[split] = {
            auditor._cls_name(class_id): count for class_id, count in scan["per_class"].items()
        }
        for class_id, areas in scan["areas"].items():
            if areas:
                audit.box_area_median[auditor._cls_name(class_id)] = statistics.median(areas)
    return auditor.run_checks()
