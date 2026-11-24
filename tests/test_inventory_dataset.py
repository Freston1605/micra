"""Tests for tools/data/inventory_dataset.py.

The inventory is read-only: every assertion reads a table the script wrote, or
the files it was given, to prove the dataset itself was not touched.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.conftest import split_dataset, write_image

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "data" / "inventory_dataset.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _inventory():
    return _load_module("inventory_dataset", SCRIPT)


def _dataset(tmp_path: Path) -> Path:
    root = split_dataset(
        tmp_path / "data",
        {
            "train": {
                "plate_t1": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n",
                "plate_t2": "2 0.5 0.5 0.2 0.2\n",
            },
            "val": {"plate_v1": "2 0.5 0.5 0.2 0.2\n"},
        },
        names=("PET", "PP", "LDPE"),
        data_yaml=True,
    )
    for index, image in enumerate(sorted((root / "images").glob("*/*.png"))):
        # Distinct pixels per image: identical bytes would be a duplicate signal.
        write_image(image, width=64, height=48, value=10 + index)
    return root


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_inventory_records_every_image_without_touching_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _inventory()
    root = _dataset(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    assert module.main(["--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "IMAGES: 3" in out
    assert "PROBLEM:" not in out

    images = _read(root / "inventory" / "images.csv")
    assert len(images) == 3
    assert {row["annotation_status"] for row in images} == {"paired"}
    by_plate = {row["plate_group"]: row for row in images}
    assert by_plate["plate_t1"]["boxes_pet"] == "1"
    assert by_plate["plate_t1"]["boxes_pp"] == "1"
    assert by_plate["plate_t1"]["boxes_total"] == "2"
    assert by_plate["plate_v1"]["boxes_ldpe"] == "1"
    assert all(row["image_id"] == row["image_sha256"][:16] for row in images)
    assert all((root / row["image_path"]).is_file() for row in images)

    version = json.loads((root / "inventory" / "dataset-version.json").read_text())
    assert version["schema_version"] == 1
    assert version["class_names"] == {"0": "PET", "1": "PP", "2": "LDPE"}
    assert version["totals"]["images"] == 3
    assert version["problems"] == []
    assert len(version["files"]["images.csv"]) == 64
    assert len(version["files"]["plates.csv"]) == 64

    after = {path: path.read_bytes() for path in before}
    for path, content in after.items():
        assert path.read_bytes() == content, f"the inventory rewrote {path}"


def test_inventory_reports_a_missing_label_and_a_negative_is_not_invented(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _inventory()
    root = _dataset(tmp_path)
    (root / "labels" / "train" / "plate_t2.txt").unlink()

    assert module.main(["--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM: missing label: train/plate_t2.png" in out

    images = {row["plate_group"]: row for row in _read(root / "inventory" / "images.csv")}
    assert images["plate_t2"]["annotation_status"] == "missing_label"
    assert images["plate_t2"]["boxes_total"] == "0"
    version = json.loads((root / "inventory" / "dataset-version.json").read_text())
    assert version["totals"]["images_without_label"] == 1


def test_inventory_uses_the_manifest_and_flags_a_group_across_splits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _inventory()
    root = _dataset(tmp_path)
    manifest = tmp_path / "manifest.py"
    manifest.write_text(
        'PLATE_GROUPS = {"plate_a": ["plate_t1", "plate_v1"], "plate_b": ["plate_t2"]}\n',
        encoding="utf-8",
    )

    assert module.main(["--root", str(root), "--manifest", str(manifest)]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM: plate group 'plate_a' crosses splits: train, val" in out

    plates = _read(root / "inventory" / "plates.csv")
    assert len(plates) == 2
    assert {row["plate_group"]: row["image_count"] for row in plates} == {
        "plate_a": "2",
        "plate_b": "1",
    }


def test_inventory_reads_plate_groups_from_an_export_mapping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _inventory()
    root = _dataset(tmp_path)
    groups = tmp_path / "mapping.csv"
    stems = sorted(path.stem for path in (root / "images").glob("*/*.png"))
    rows = ["image_id,new_image,plate_group"]
    rows += [f"{stem},images/train/{stem}.png,plate_a" for stem in stems]
    groups.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert module.main(["--root", str(root), "--groups", str(groups)]) == 1
    assert "PROBLEM: plate group 'plate_a' crosses splits: train, val" in capsys.readouterr().out
    plates = _read(root / "inventory" / "plates.csv")
    assert {
        row["plate_group"]: row["group_source"] for row in plates if row["plate_group"] == "plate_a"
    } == {"plate_a": "manifest"}


def test_inventory_reports_a_layout_that_is_not_a_split_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The layout is a dataset problem: still inventory it, exit 1."""
    module = _inventory()
    root = tmp_path / "data"
    root.mkdir()
    (root / "dataset.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  0: PET\n  1: PP\n  2: LDPE\n",
        encoding="utf-8",
    )

    assert module.main(["--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM: split 'train' declares" in out
    version = json.loads((root / "inventory" / "dataset-version.json").read_text())
    assert version["totals"]["images"] == 0
    assert any("not a directory" in problem for problem in version["problems"])


def test_inventory_still_writes_a_healthy_split_beside_a_broken_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _inventory()
    root = _dataset(tmp_path)
    declaration = (root / "data.yaml").read_text(encoding="utf-8")
    (root / "data.yaml").write_text(
        declaration.replace("val: images/val", "val: images/val-missing"), encoding="utf-8"
    )

    assert module.main(["--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM: split 'val' declares" in out
    images = _read(root / "inventory" / "images.csv")
    assert {row["split"] for row in images} == {"train"}
    assert len(images) == 2


def test_inventory_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _inventory()
    root = _dataset(tmp_path)
    assert module.main(["--root", str(root)]) == 0
    capsys.readouterr()

    assert module.main(["--root", str(root)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert module.main(["--root", str(root), "--force"]) == 0
