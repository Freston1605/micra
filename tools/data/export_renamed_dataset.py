"""Write a copy of a labelled YOLO dataset with portable file names.

The dataset's provenance lives in its file names: a plate group, a declared
spike, a polymer. Those names also carry characters that a download, a ZIP or a
second filesystem can mangle. This tool writes a **new** copy in which every
image and its label are named from the image's SHA-256, and records the mapping
back to the original name in ``mapping.csv``.

Nothing in the source tree is renamed, moved or rewritten. The copy keeps the
same split, class map and bytes; only the paths change, so the dataset
fingerprint changes while the labels do not.

    .venv/bin/python tools/data/export_renamed_dataset.py --root data --out data_portable
    .venv/bin/python tools/data/export_renamed_dataset.py --root data \
        --out data_portable --manifest tools/data/split_manifest.py

Exit status is 0 when every image was exported and verified, 1 when the
dataset itself cannot be exported safely (a missing or unpaired label, a split
that is not a directory, a duplicate image), and 2 for a usage or precondition
error such as a missing declaration or an occupied output folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a declared dependency
    yaml = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_KEYS = ("train", "val", "test")
COLUMNS = (
    "image_id",
    "split",
    "original_image",
    "original_label",
    "new_image",
    "new_label",
    "plate_group",
    "group_source",
    "image_sha256",
    "label_sha256",
    "boxes",
)


class ExportError(RuntimeError):
    """A usage or precondition failure; nothing was written."""


class DatasetProblem(RuntimeError):
    """The dataset is not in a state that can be exported; nothing was written."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path | None) -> dict[str, str]:
    """Map a stem to its plate group from an optional split manifest."""
    if path is None:
        return {}
    if not path.is_file():
        raise ExportError(f"manifest not found: {path}")
    spec = importlib.util.spec_from_file_location("_export_manifest", path)
    if spec is None or spec.loader is None:
        raise ExportError(f"cannot load manifest: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = getattr(module, "PLATE_GROUPS", None)
    if not isinstance(groups, dict):
        raise ExportError(f"manifest declares no PLATE_GROUPS: {path}")
    return {str(stem): str(group) for group, stems in groups.items() for stem in list(stems)}


def _declaration(root: Path, declared: Path | None) -> tuple[Path, dict[str, Any]]:
    if yaml is None:
        raise ExportError("PyYAML is required to read the dataset declaration")
    if declared is None:
        candidates = [root / "dataset.yaml", root / "data.yaml"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise ExportError(f"no dataset.yaml or data.yaml in {root}")
    else:
        path = declared
        if not path.is_file():
            raise ExportError(f"no dataset declaration at {path}")
    parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not parsed.get("names"):
        raise ExportError("the dataset declaration names no classes")
    return path, parsed


def _plan(
    root: Path, declared: Path, parsed: dict[str, Any], manifest: dict[str, str]
) -> list[dict]:
    """List every (image, label) pair and give each its hash-derived name."""
    plan: list[dict] = []
    for split in SPLIT_KEYS:
        if not parsed.get(split):
            continue
        directory = Path(str(parsed[split]))
        directory = directory if directory.is_absolute() else root / directory
        if not directory.is_dir():
            raise DatasetProblem(
                f"split '{split}' declares {directory} but it is not a directory "
                "(a Windows export can leave the split as path text in file names)"
            )
        labels = root / "labels" / split
        if not labels.is_dir():
            labels = directory.parent.parent / "labels" / split
        stems = set()
        for image in sorted(directory.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
                continue
            stems.add(image.stem)
            label = labels / f"{image.stem}.txt"
            if not label.is_file():
                raise DatasetProblem(f"missing label for {split}/{image.name}")
            digest = _sha256(image)
            plan.append(
                {
                    "split": split,
                    "image": image,
                    "label": label,
                    "digest": digest,
                    "new_image": f"{digest[:16]}{image.suffix.lower()}",
                    "new_label": f"{digest[:16]}.txt",
                    "group": manifest.get(image.stem, image.stem),
                    "group_source": "manifest" if image.stem in manifest else "filename",
                    "boxes": sum(
                        1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip()
                    ),
                }
            )
        if labels.is_dir():
            for label in sorted(labels.glob("*.txt")):
                if label.name == "classes.txt" or label.stem in stems:
                    continue
                raise DatasetProblem(f"unpaired label {split}/{label.name}")

    if not plan:
        raise DatasetProblem("no images found in the declared splits")
    _verify_names(plan)
    return plan


def _verify_names(plan: list[dict]) -> None:
    """Refuse any two images that would share an exported name.

    The name is a 16-hex prefix of the image digest, so a shared name has three
    distinct causes and each one needs its own answer: a real prefix collision
    between different images, one image carrying two different annotation
    files, a duplicate image that sits in two folds, or a duplicate inside one
    fold. Collapsing them to one message would hide a collision as a duplicate.
    """
    by_name: dict[str, list[dict]] = {}
    for entry in plan:
        by_name.setdefault(entry["new_image"], []).append(entry)

    for name, copies in sorted(by_name.items()):
        if len(copies) < 2:
            continue
        originals = ", ".join(f"{entry['split']}/{entry['image'].name}" for entry in copies)
        digests = {entry["digest"] for entry in copies}
        if len(digests) > 1:
            raise DatasetProblem(
                f"image id prefix collision: {', '.join(sorted(digests))} all shorten to "
                f"{name} ({originals}); widen the prefix"
            )
        labels = {entry["label"].read_bytes() for entry in copies}
        if len(labels) > 1:
            raise DatasetProblem(
                f"the same image carries different labels: {originals}; "
                "one of the two annotations is wrong and neither can be exported"
            )
        splits = {entry["split"] for entry in copies}
        if len(splits) > 1:
            raise DatasetProblem(
                f"byte-identical image in two folds ({', '.join(sorted(splits))}): {originals}"
            )
        raise DatasetProblem(f"duplicate image in one fold: {originals}")


def _write_declaration(path: Path, parsed: dict[str, Any], splits: list[str]) -> None:
    """Copy the declaration, pointing each split at the new folder."""
    payload = {
        key: value
        for key, value in parsed.items()
        if key not in {*SPLIT_KEYS, "path"} and key != "names"
    }
    lines = ["# Portable export of a labelled MICRA dataset.", "#", "# Generated by"]
    lines.append("# tools/data/export_renamed_dataset.py; do not edit by hand.")
    lines.append("")
    lines.append("path: .")
    lines.extend(f"{split}: images/{split}" for split in splits)
    if payload:
        lines.append("")
        lines.append(yaml.safe_dump(payload, sort_keys=True).strip())
    lines.append("")
    lines.append("names:")
    for key, value in sorted(parsed["names"].items(), key=lambda item: int(item[0])):
        lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _counts(plan: list[dict], classes: dict[int, str]) -> dict[str, dict[str, int]]:
    """Boxes per split and class, read from the labels being copied."""
    totals: dict[str, dict[str, int]] = {}
    for entry in plan:
        split = totals.setdefault(entry["split"], dict.fromkeys(classes.values(), 0))
        for line in entry["label"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                split[classes[int(line.split()[0])]] += 1
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy a labelled YOLO dataset with hash-derived file names"
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="split manifest module with PLATE_GROUPS (default: group by file stem)",
    )
    parser.add_argument("--force", action="store_true", help="write into a non-empty folder")
    args = parser.parse_args(argv)

    root, out = args.root, args.out
    if out.resolve().is_relative_to(root.resolve()):
        print("ERROR: --out must be outside --root", file=sys.stderr)
        return 2
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: {out} is not empty (use --force)", file=sys.stderr)
        return 2

    try:
        declared, parsed = _declaration(root, args.data_yaml)
        plan = _plan(root, declared, parsed, _load_manifest(args.manifest))
        classes = {int(key): str(value) for key, value in dict(parsed["names"]).items()}
        expected = _counts(plan, classes)
    except DatasetProblem as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    splits = list(dict.fromkeys(entry["split"] for entry in plan))
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for entry in plan:
        image_target = out / "images" / entry["split"] / entry["new_image"]
        label_target = out / "labels" / entry["split"] / entry["new_label"]
        image_target.write_bytes(entry["image"].read_bytes())
        label_target.write_bytes(entry["label"].read_bytes())
        if _sha256(image_target) != entry["digest"]:
            raise ExportError(f"copy mismatch for {entry['image']}")
        if _sha256(label_target) != _sha256(entry["label"]):
            raise ExportError(f"copy mismatch for {entry['label']}")
        rows.append(
            {
                "image_id": entry["new_image"].removesuffix(entry["image"].suffix.lower()),
                "split": entry["split"],
                "original_image": entry["image"].name,
                "original_label": entry["label"].name,
                "new_image": str(image_target.relative_to(out)),
                "new_label": str(label_target.relative_to(out)),
                "plate_group": entry["group"],
                "group_source": entry["group_source"],
                "image_sha256": entry["digest"],
                "label_sha256": _sha256(label_target),
                "boxes": str(entry["boxes"]),
            }
        )

    _write_declaration(out / "dataset.yaml", parsed, splits)
    with open(out / "mapping.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["image_id"]))

    actual = _counts(
        [
            {"split": str(Path(row["new_image"]).parent.name), "label": out / row["new_label"]}
            for row in rows
        ],
        classes,
    )
    differing = {split: totals for split, totals in expected.items() if actual.get(split) != totals}
    if differing:
        print(f"ERROR: per-class counts differ after export: {differing}", file=sys.stderr)
        return 1

    print(f"EXPORT: {out}")
    print(f"IMAGES: {len(rows)}")
    print(f"MAPPING: {out / 'mapping.csv'}")
    for split, totals in expected.items():
        print(f"SPLIT: {split} boxes={totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
