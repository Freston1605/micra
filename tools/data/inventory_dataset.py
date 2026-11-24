"""Inventory a labelled YOLO dataset without changing it.

The dataset is not versioned, so what trained a model has to be recovered from
the files themselves. This tool writes three portable tables that describe the
dataset as it is: one row per image, one row per plate group, and one version
record. It never renames, moves or rewrites an image, a label or the split.

    .venv/bin/python tools/data/inventory_dataset.py --root data
    .venv/bin/python tools/data/inventory_dataset.py --root data \
        --manifest tools/data/split_manifest.py --output-dir data/inventory

Only names, dimensions, hashes and the presence of metadata are recorded.
EXIF capture dates are copied when a photo carries them because they help
group acquisition sessions; GPS coordinates never are — only whether a photo
contains them. Filename-derived categories are marked as such: a name can
report that a plate was declared spiked, never that a sample was independently
confirmed.

Exit status is 0 when the dataset is internally consistent, 1 when the
inventory found problems (it is still written, so the problems can be read
from it), and 2 for a usage or precondition error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a declared dependency
    yaml = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is a declared dependency
    Image = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_KEYS = ("train", "val", "test")
SCHEMA_VERSION = 1

#: Groups whose file names declare a spiked sediment plate. It is a naming
#: convention, not a laboratory record, so the inventory labels the claim.
SEDIMENT_PREFIXES = ("sedcalci_", "sed_marino_")
MIXTURE_PREFIXES = ("mezc_",)

COLUMNS_IMAGES = (
    "image_id",
    "image_path",
    "label_path",
    "split",
    "plate_group",
    "group_source",
    "width",
    "height",
    "has_exif",
    "has_gps_metadata",
    "exif_capture_date",
    "image_sha256",
    "label_sha256",
    "annotation_status",
    "boxes_pet",
    "boxes_pp",
    "boxes_ldpe",
    "boxes_total",
    "duplicate_group",
)

COLUMNS_PLATES = (
    "plate_group",
    "group_source",
    "sample_id",
    "sample_type",
    "spike_status",
    "provenance_source",
    "split",
    "image_count",
    "boxes_pet",
    "boxes_pp",
    "boxes_ldpe",
    "notes",
)


class InventoryError(RuntimeError):
    """A precondition failed; nothing was written."""


@dataclass
class _Image:
    """One image and its label, as found on disk."""

    path: Path
    label: Path
    split: str
    stem: str
    group: str
    group_source: str
    width: int = 0
    height: int = 0
    has_exif: bool = False
    has_gps: bool = False
    capture_date: str = ""
    image_sha256: str = ""
    label_sha256: str = ""
    status: str = "paired"
    counts: dict[int, int] = field(default_factory=dict)

    @property
    def total_boxes(self) -> int:
        return sum(self.counts.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(root: Path, declared_yaml: Path | None) -> tuple[dict[str, str], dict[str, Path]]:
    """Resolve the class names and the split directories a dataset declares."""
    if yaml is None:
        raise InventoryError("PyYAML is required to read the dataset declaration")
    if declared_yaml is None:
        candidates = [root / "dataset.yaml", root / "data.yaml"]
        yaml_path = next((path for path in candidates if path.is_file()), None)
        if yaml_path is None:
            raise InventoryError(f"no dataset.yaml or data.yaml in {root}")
    else:
        yaml_path = declared_yaml
        if not yaml_path.is_file():
            raise InventoryError(f"no dataset declaration at {yaml_path}")
    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = {str(key): str(value) for key, value in dict(parsed.get("names") or {}).items()}
    if not names:
        raise InventoryError("the dataset declaration names no classes")

    directories: dict[str, Path] = {}
    for split in SPLIT_KEYS:
        declared = parsed.get(split)
        if not declared:
            continue
        path = Path(str(declared))
        directories[split] = path if path.is_absolute() else root / path
    if not directories:
        raise InventoryError("the dataset declaration names no split")
    return names, directories, yaml_path


def _load_manifest(path: Path | None) -> dict[str, str]:
    """Map a stem to its plate group from an optional split manifest."""
    if path is None:
        return {}
    if not path.is_file():
        raise InventoryError(f"manifest not found: {path}")
    spec = importlib.util.spec_from_file_location("_inventory_manifest", path)
    if spec is None or spec.loader is None:
        raise InventoryError(f"cannot load manifest: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = getattr(module, "PLATE_GROUPS", None)
    if not isinstance(groups, dict):
        raise InventoryError(f"manifest declares no PLATE_GROUPS: {path}")
    return {str(stem): str(group) for group, stems in groups.items() for stem in list(stems)}


def _load_groups(path: Path | None) -> dict[str, str]:
    """Map an image stem to its plate group from an export's mapping.csv."""
    if path is None:
        return {}
    if not path.is_file():
        raise InventoryError(f"groups CSV not found: {path}")
    groups: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            group = (row.get("plate_group") or "").strip()
            stem = (row.get("image_id") or Path(row.get("new_image") or "").stem).strip()
            if group and stem:
                groups[stem] = group
    if not groups:
        raise InventoryError(f"groups CSV names no plate_group: {path}")
    return groups


def _image_metadata(path: Path) -> tuple[int, int, bool, bool, str]:
    """Return size and metadata presence; never decode pixels or copy GPS."""
    if Image is None:
        raise InventoryError("Pillow is required to read image headers")
    with Image.open(path) as image:
        width, height = image.size
        exif = image.getexif()
        has_exif = bool(exif)
        has_gps = 34853 in exif
        capture_date = ""
        if 34665 in exif:
            capture_date = str(exif.get_ifd(34665).get(36867) or "")
    return width, height, has_exif, has_gps, capture_date


def _label_counts(path: Path, class_ids: set[int]) -> dict[int, int]:
    """Count boxes per class, or fail closed on an unusable label line."""
    counts = dict.fromkeys(sorted(class_ids), 0)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except (IndexError, ValueError) as exc:
            raise InventoryError(f"{path}:{number}: unreadable label line") from exc
        if class_id not in class_ids:
            raise InventoryError(f"{path}:{number}: unknown class id {class_id}")
        if len(values) != 4 or not all(0.0 <= value <= 1.0 for value in values):
            raise InventoryError(f"{path}:{number}: not a normalised box")
        counts[class_id] += 1
    return counts


def _derive_sample_type(group: str) -> tuple[str, str, str]:
    """Sample type, spike status and provenance source from a plate group name."""
    lowered = group.lower()
    if lowered.startswith(MIXTURE_PREFIXES):
        return "laboratory_mixture", "not_declared", "filename"
    if lowered.startswith(SEDIMENT_PREFIXES):
        return "sediment", "declared_spiked", "filename"
    for polymer in ("pet", "pp", "ldpe"):
        if lowered.startswith(f"{polymer}_"):
            return "laboratory_pure_plate", "not_declared", "filename"
    return "unknown", "unknown", "unknown"


def _collect(
    root: Path, manifest: dict[str, str], declared_yaml: Path | None
) -> tuple[list[_Image], dict[str, str], list[str], Path]:
    """Walk every declared split; return the images, class names and problems."""
    names, directories, yaml_path = _read_yaml(root, declared_yaml)
    class_ids = {int(key) for key in names}
    images: list[_Image] = []
    problems: list[str] = []

    for split, directory in directories.items():
        if not directory.is_dir():
            raise InventoryError(
                f"split '{split}' declares {directory} but it is not a directory "
                "(a Windows export can leave the split as path text in file names)"
            )
        labels_dir = root / "labels" / split
        if not labels_dir.is_dir():
            labels_dir = directory.parent.parent / "labels" / split
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            group = manifest.get(path.stem)
            record = _Image(
                path=path,
                label=labels_dir / f"{path.stem}.txt",
                split=split,
                stem=path.stem,
                group=group or path.stem,
                group_source="manifest" if group else "filename",
            )
            record.image_sha256 = _sha256(path)
            record.width, record.height, record.has_exif, record.has_gps, record.capture_date = (
                _image_metadata(path)
            )
            if record.label.is_file():
                record.label_sha256 = _sha256(record.label)
                record.counts = _label_counts(record.label, class_ids)
            else:
                record.status = "missing_label"
                problems.append(f"missing label: {split}/{path.name}")
            images.append(record)

        if labels_dir.is_dir():
            stems = {image.label.stem for image in images if image.split == split}
            for label in sorted(labels_dir.glob("*.txt")):
                if label.name == "classes.txt" or label.stem in stems:
                    continue
                problems.append(f"unpaired label: {split}/{label.name}")

    return images, names, problems, yaml_path


def _duplicate_groups(images: list[_Image]) -> dict[str, str]:
    """Name byte-identical images; a cross-split copy is a fold leak signal."""
    by_hash: dict[str, list[_Image]] = {}
    for image in images:
        by_hash.setdefault(image.image_sha256, []).append(image)
    groups: dict[str, str] = {}
    for digest, copies in by_hash.items():
        if len(copies) < 2:
            continue
        name = digest[:12]
        for copy in copies:
            groups[copy.image_sha256] = name
    return groups


def _group_rows(
    images: list[_Image], names: dict[str, str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """One row per plate group, plus the groups that cross a split."""
    by_group: dict[str, list[_Image]] = {}
    for image in images:
        by_group.setdefault(image.group, []).append(image)

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for group, members in sorted(by_group.items()):
        splits = sorted({member.split for member in members})
        if len(splits) > 1:
            problems.append(f"plate group '{group}' crosses splits: {', '.join(splits)}")
        sample_type, spike_status, source = _derive_sample_type(group)
        rows.append(
            {
                "plate_group": group,
                "group_source": members[0].group_source,
                "sample_id": "",
                "sample_type": sample_type,
                "spike_status": spike_status,
                "provenance_source": source,
                "split": ",".join(splits),
                "image_count": len(members),
                "boxes_pet": sum(member.counts.get(0, 0) for member in members),
                "boxes_pp": sum(member.counts.get(1, 0) for member in members),
                "boxes_ldpe": sum(member.counts.get(2, 0) for member in members),
                "notes": "" if sample_type != "unknown" else "sample type not inferred",
            }
        )
    expected = {"PET", "PP", "LDPE"}
    if not expected.issubset(set(names.values())):
        problems.append(f"class names are not the study classes: {sorted(names.values())}")
    if {int(key) for key in names} != {0, 1, 2}:
        problems.append(f"class ids are not 0/1/2: {sorted(names)}")
    return rows, problems


def _fingerprint(images: list[_Image], root: Path, data_yaml: Path) -> str:
    """The pipeline's dataset fingerprint over the relative paths and bytes.

    The declaration enters under the fixed key ``data.yaml`` because that is
    what ``polyx.pipeline.dataset`` uses, whatever the file is called, so the
    inventory can be compared with a training execution's recorded digest.
    """
    from polyx.pipeline.dataset import dataset_fingerprint

    hashes = {str(image.path.relative_to(root)): image.image_sha256 for image in images}
    hashes.update(
        {
            str(image.label.relative_to(root)): image.label_sha256
            for image in images
            if image.status == "paired"
        }
    )
    hashes["data.yaml"] = _sha256(data_yaml)
    return dataset_fingerprint(hashes)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summary(images: list[_Image], plates: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts that make coverage and concentration readable."""
    splits: dict[str, Any] = {}
    for split in sorted({image.split for image in images}):
        members = [image for image in images if image.split == split]
        split_plates = [row for row in plates if split in str(row["split"]).split(",")]
        splits[split] = {
            "images": len(members),
            "plate_groups": len(split_plates),
            "sample_types": sorted({str(row["sample_type"]) for row in split_plates}),
            "boxes": {
                "PET": sum(image.counts.get(0, 0) for image in members),
                "PP": sum(image.counts.get(1, 0) for image in members),
                "LDPE": sum(image.counts.get(2, 0) for image in members),
            },
        }
    return splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write an inventory of a labelled YOLO dataset (read-only)"
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="dataset declaration (default: <root>/dataset.yaml, then data.yaml)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="split manifest module with PLATE_GROUPS (default: group by file stem)",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=None,
        help="mapping.csv from export_renamed_dataset.py (image_id -> plate_group)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the tables (default: <root>/inventory)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing inventory")
    args = parser.parse_args(argv)

    root = args.root
    output = args.output_dir or root / "inventory"
    targets = [output / "images.csv", output / "plates.csv", output / "dataset-version.json"]
    if not args.force and any(target.exists() for target in targets):
        print(f"ERROR: inventory already exists in {output} (use --force)", file=sys.stderr)
        return 2

    try:
        manifest = _load_manifest(args.manifest) | _load_groups(args.groups)
        images, names, problems, yaml_path = _collect(root, manifest, args.data_yaml)
        plates, plate_problems = _group_rows(images, names)
        problems.extend(plate_problems)
        duplicates = _duplicate_groups(images)
        cross_split = [
            image
            for image in images
            if duplicates.get(image.image_sha256)
            and len({other.split for other in images if other.image_sha256 == image.image_sha256})
            > 1
        ]
        if cross_split:
            problems.append(f"byte-identical images across splits: {len(cross_split)}")
        rows = [
            {
                "image_id": image.image_sha256[:16],
                "image_path": str(image.path.relative_to(root)),
                "label_path": str(image.label.relative_to(root)),
                "split": image.split,
                "plate_group": image.group,
                "group_source": image.group_source,
                "width": image.width,
                "height": image.height,
                "has_exif": str(image.has_exif).lower(),
                "has_gps_metadata": str(image.has_gps).lower(),
                "exif_capture_date": image.capture_date,
                "image_sha256": image.image_sha256,
                "label_sha256": image.label_sha256,
                "annotation_status": image.status,
                "boxes_pet": image.counts.get(0, 0),
                "boxes_pp": image.counts.get(1, 0),
                "boxes_ldpe": image.counts.get(2, 0),
                "boxes_total": image.total_boxes,
                "duplicate_group": duplicates.get(image.image_sha256, ""),
            }
            for image in images
        ]
        version = {
            "schema_version": SCHEMA_VERSION,
            "dataset": root.name,
            "class_names": {key: names[key] for key in sorted(names, key=int)},
            "fingerprint": _fingerprint(images, root, yaml_path),
            "splits": _summary(images, plates),
            "totals": {
                "images": len(images),
                "plate_groups": len(plates),
                "boxes": sum(image.total_boxes for image in images),
                "images_without_label": sum(image.status != "paired" for image in images),
                "images_with_gps_metadata": sum(image.has_gps for image in images),
                "duplicate_groups": len(set(duplicates.values())),
            },
            "problems": problems,
        }
    except InventoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "images.csv", COLUMNS_IMAGES, rows)
    _write_csv(output / "plates.csv", COLUMNS_PLATES, plates)
    version["files"] = {
        "images.csv": _sha256(output / "images.csv"),
        "plates.csv": _sha256(output / "plates.csv"),
    }
    _write_json(output / "dataset-version.json", version)

    print(f"INVENTORY: {output}")
    print(f"IMAGES: {len(images)}")
    print(f"PLATES: {len(plates)}")
    print(f"BOXES: {version['totals']['boxes']}")
    print(f"FINGERPRINT: {version['fingerprint']}")
    for split, summary in version["splits"].items():
        print(
            f"SPLIT: {split} images={summary['images']} groups={summary['plate_groups']} "
            f"boxes={summary['boxes']}"
        )
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
