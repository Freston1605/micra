"""Tests for tools/data/export_renamed_dataset.py.

The export is a copy: every assertion checks the new tree, the mapping back to
the original names, and that the source bytes were not touched.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from tests.conftest import split_dataset, write_image

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "data" / "export_renamed_dataset.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _export():
    return _load_module("export_renamed_dataset", SCRIPT)


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
        write_image(image, width=64, height=48, value=10 + index)
    return root


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_renames_by_hash_and_keeps_the_mapping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _export()
    root = _dataset(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    out = tmp_path / "portable"

    assert module.main(["--root", str(root), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "IMAGES: 3" in printed
    assert "SPLIT: train boxes={'PET': 1, 'PP': 1, 'LDPE': 1}" in printed

    images = sorted((out / "images").glob("*/*.png"))
    assert len(images) == 3
    assert all(len(image.stem) == 16 and image.stem.isalnum() for image in images)
    assert len({image.stem for image in images}) == 3
    for image in images:
        label = out / "labels" / image.parent.name / f"{image.stem}.txt"
        assert label.is_file()

    mapping = _read(out / "mapping.csv")
    assert len(mapping) == 3
    assert {row["original_image"] for row in mapping} == {
        "plate_t1.png",
        "plate_t2.png",
        "plate_v1.png",
    }
    assert {row["split"] for row in mapping} == {"train", "val"}
    for row in mapping:
        assert row["new_image"] == f"images/{row['split']}/{row['image_id']}.png"
        original_image = root / "images" / row["split"] / row["original_image"]
        original_label = root / "labels" / row["split"] / row["original_label"]
        assert (out / row["new_image"]).read_bytes() == original_image.read_bytes()
        assert (out / row["new_label"]).read_bytes() == original_label.read_bytes()

    declaration = (out / "dataset.yaml").read_text()
    assert "train: images/train" in declaration
    assert "val: images/val" in declaration
    assert "  2: LDPE" in declaration

    for path, content in before.items():
        assert path.read_bytes() == content, f"the export touched {path}"


def test_export_refuses_a_missing_label(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _export()
    root = _dataset(tmp_path)
    (root / "labels" / "train" / "plate_t2.txt").unlink()

    assert module.main(["--root", str(root), "--out", str(tmp_path / "portable")]) == 2
    assert "missing label for train/plate_t2.png" in capsys.readouterr().err
    assert not (tmp_path / "portable").exists()


def test_export_refuses_an_unpaired_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _export()
    root = _dataset(tmp_path)
    (root / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    assert module.main(["--root", str(root), "--out", str(tmp_path / "portable")]) == 2
    assert "unpaired label train/orphan.txt" in capsys.readouterr().err


def test_export_records_the_plate_group_from_the_manifest(tmp_path: Path) -> None:
    module = _export()
    root = _dataset(tmp_path)
    manifest = tmp_path / "manifest.py"
    manifest.write_text('PLATE_GROUPS = {"plate_a": ["plate_t1", "plate_t2"]}\n', encoding="utf-8")

    out = tmp_path / "portable"
    assert module.main(["--root", str(root), "--out", str(out), "--manifest", str(manifest)]) == 0
    groups = {row["original_image"]: row["plate_group"] for row in _read(out / "mapping.csv")}
    assert groups["plate_t1.png"] == "plate_a"
    assert groups["plate_t2.png"] == "plate_a"
    assert groups["plate_v1.png"] == "plate_v1"


def test_export_refuses_to_write_inside_the_source_or_over_a_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _export()
    root = _dataset(tmp_path)

    assert module.main(["--root", str(root), "--out", str(root / "copy")]) == 2
    assert "outside --root" in capsys.readouterr().err

    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file").write_text("x", encoding="utf-8")
    assert module.main(["--root", str(root), "--out", str(tmp_path / "taken")]) == 2
    assert "is not empty" in capsys.readouterr().err
