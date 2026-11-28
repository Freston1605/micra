"""The dataset audit reads its labels through the label codec.

A conventional dataset (``images/<split>`` beside ``labels/<split>``) must
report the class and box statistics the audit has always reported, and a
YOLO-seg polygon line must not be counted as a detection box: the polygon
dialect is valid YOLO but not a box, so the codec's box reader counts it
malformed instead of mangling its first four coordinates into a box. The
audit reports those lines as a warning rather than dropping them silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from polyx.core.quality.dataset_audit import LEVEL_ERROR, LEVEL_WARN, audit_dataset
from tests.conftest import split_dataset, write_image

PET_BOX = "0 0.5 0.5 0.10 0.10"
PP_BOX = "1 0.5 0.5 0.30 0.30"
POLYGON = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"


def make_dataset(
    root: Path,
    *,
    train_labels: list[str],
    val_labels: list[str] | None = None,
) -> Path:
    """A conventional YOLO dataset: ``images/<split>`` + ``labels/<split>``."""
    val = train_labels if val_labels is None else val_labels
    splits = {
        "train": {f"train_{i}": text for i, text in enumerate(train_labels)},
        "val": {f"val_{i}": text for i, text in enumerate(val)},
    }
    split_dataset(root, splits, suffix=".jpg", data_yaml=True)
    return root / "data.yaml"


def test_audit_finds_conventional_split_labels(tmp_path: Path):
    """``labels/<split>`` is the codec's conventional location; the audit uses it."""
    yaml_path = make_dataset(
        tmp_path / "dataset",
        train_labels=[f"{PET_BOX}\n{PP_BOX}\n", f"{PET_BOX}\n{PP_BOX}\n"],
        val_labels=[f"{PET_BOX}\n{PP_BOX}\n", f"{PET_BOX}\n{PP_BOX}\n"],
    )

    audit = audit_dataset(yaml_path)

    assert audit.per_class["train"] == {"PET": 2, "PP": 2}
    assert audit.per_class["val"] == {"PET": 2, "PP": 2}
    assert audit.box_area_median["PET"] == pytest.approx(0.01)
    assert audit.box_area_median["PP"] == pytest.approx(0.09)
    # Labels were found: the dataset is valid and nothing is unlabelled.
    assert audit.ok
    assert not [finding for finding in audit.findings if finding.code == "images_without_labels"]


def test_audit_does_not_count_a_polygon_line_as_a_box(tmp_path: Path):
    """A YOLO-seg polygon is not a detection box and must not pollute the stats."""
    yaml_path = make_dataset(
        tmp_path / "dataset",
        train_labels=[f"{PET_BOX}\n", f"{POLYGON}\n"],
        val_labels=[f"{PET_BOX}\n", f"{POLYGON}\n"],
    )

    audit = audit_dataset(yaml_path)

    # Only the one real box exists in each split; the polygon adds nothing.
    assert audit.per_class["train"] == {"PET": 1}
    assert audit.per_class["val"] == {"PET": 1}
    # If the polygon were misread as a box its 0.9 x 0.1 area would shift this.
    assert audit.box_area_median["PET"] == pytest.approx(0.01)
    # The one polygon per split is reported instead of silently discarded.
    malformed = [
        finding
        for finding in audit.findings
        if finding.level == LEVEL_WARN and "YOLO-seg" in finding.detail
    ]
    assert malformed, audit.findings
    assert "2" in malformed[0].title, malformed[0].title
    # The explanation names the dialect; it survives the locale.
    assert "YOLO-seg" in malformed[0].detail


def test_audit_warns_a_list_split_is_not_trainable_headlessly(tmp_path: Path):
    """The audit reads a .txt split, and says the headless path will refuse it."""
    yaml_path = make_dataset(
        tmp_path / "dataset",
        train_labels=[f"{PET_BOX}\n"],
        val_labels=[f"{PP_BOX}\n"],
    )
    root = yaml_path.parent
    (root / "train.txt").write_text("images/train/train_0.jpg\n", encoding="utf-8")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["train"] = "train.txt"
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    audit = audit_dataset(yaml_path)

    codes = {finding.code for finding in audit.findings}
    assert "split_list_not_trainable" in codes
    # The list still supplies the images: the warning is about the training seam.
    assert audit.per_class["train"] == {"PET": 1}


def test_audit_warns_when_nc_disagrees_with_the_declared_names(tmp_path: Path):
    yaml_path = make_dataset(
        tmp_path / "dataset",
        train_labels=[f"{PET_BOX}\n"],
        val_labels=[f"{PP_BOX}\n"],
    )
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["nc"] = 3
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    audit = audit_dataset(yaml_path)

    codes = {finding.code for finding in audit.findings}
    assert "nc_mismatch" in codes


def test_audit_accepts_nc_over_a_sparse_name_map(tmp_path: Path):
    """nc counts the ids the mapping spans, so {0, 2} with nc: 3 is consistent."""
    yaml_path = make_dataset(
        tmp_path / "dataset",
        train_labels=[f"{PET_BOX}\n"],
        val_labels=[f"{PET_BOX}\n"],
    )
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["names"] = {0: "PET", 2: "LDPE"}
    data["nc"] = 3
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    audit = audit_dataset(yaml_path)

    codes = {finding.code for finding in audit.findings}
    assert "nc_mismatch" not in codes


def test_audit_reports_a_non_mapping_declaration_with_a_bare_code(tmp_path: Path):
    path = tmp_path / "dataset" / "data.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("- one\n- two\n", encoding="utf-8")

    audit = audit_dataset(path)

    codes = {finding.code for finding in audit.findings}
    assert "data_yaml_not_mapping" in codes


def test_audit_resolves_names_by_id_for_sparse_class_ids(tmp_path: Path):
    """A sparse `names` map labels class 2 correctly, not `class 2` or `LDPE` by index."""
    yaml_path = make_dataset(
        tmp_path / "dataset",
        train_labels=["2 0.5 0.5 0.3 0.3\n"],
        val_labels=["2 0.5 0.5 0.3 0.3\n"],
    )
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["names"] = {0: "PET", 2: "LDPE"}
    data["nc"] = 3
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    audit = audit_dataset(yaml_path)

    assert audit.per_class["val"] == {"LDPE": 1}
    assert audit.per_class["train"] == {"LDPE": 1}
    assert "class 2" not in audit.per_class["val"]


def test_missing_declaration_is_one_finding_with_the_legacy_code(tmp_path: Path):
    """The declaration owns the failure; the audit maps it once, with no re-mint."""
    audit = audit_dataset(tmp_path / "absent" / "data.yaml")

    assert [finding.code for finding in audit.findings] == ["data_yaml_missing"]
    assert audit.findings[0].parameters["path"].endswith("data.yaml")


# ── Absent vs empty label files (#223) ───────────────────────────────────


def test_an_absent_label_file_is_an_error_in_train_and_val(tmp_path: Path):
    """No label file means nobody reviewed the image; it cannot be background."""
    root = split_dataset(
        tmp_path / "dataset",
        {
            "train": {"kept": f"{PET_BOX}\n", "unlabelled": None},
            "val": {"kept": f"{PET_BOX}\n", "unlabelled": None},
        },
        data_yaml=True,
    )

    audit = audit_dataset(root / "data.yaml")

    assert not audit.ok
    findings = [finding for finding in audit.findings if finding.code == "images_without_labels"]
    assert {finding.parameters["split"] for finding in findings} == {"train", "val"}
    assert all(finding.level == LEVEL_ERROR for finding in findings)
    # The refusal is actionable: it names the file, not only the count.
    assert all(finding.parameters["images"] == ["unlabelled.png"] for finding in findings)
    assert "unlabelled.png" in findings[0].detail


def test_an_absent_label_file_is_an_error_in_a_declared_test_fold(tmp_path: Path):
    """A test fold is audited too: held-out scoring must not read background."""
    root = split_dataset(
        tmp_path / "dataset",
        {
            "train": {"kept": f"{PET_BOX}\n"},
            "val": {"kept": f"{PET_BOX}\n"},
            "test": {"kept": f"{PET_BOX}\n", "unlabelled": None},
        },
        data_yaml=True,
    )

    audit = audit_dataset(root / "data.yaml")

    findings = [finding for finding in audit.findings if finding.code == "images_without_labels"]
    assert [finding.parameters["split"] for finding in findings] == ["test"]
    assert findings[0].level == LEVEL_ERROR
    assert findings[0].parameters["images"] == ["unlabelled.png"]


def test_an_empty_label_file_is_a_reviewed_negative(tmp_path: Path):
    """An empty file records a decision; an absent one records nothing."""
    root = split_dataset(
        tmp_path / "dataset",
        {
            "train": {"kept": f"{PET_BOX}\n", "cleared": ""},
            "val": {"kept": f"{PET_BOX}\n", "cleared": ""},
        },
        data_yaml=True,
    )

    audit = audit_dataset(root / "data.yaml")

    assert audit.ok, [finding for finding in audit.findings if finding.level == LEVEL_ERROR]
    assert not [finding for finding in audit.findings if finding.code == "images_without_labels"]


def test_a_split_layout_same_directory_label_is_present_for_the_audit(tmp_path: Path):
    """The audit sees a legacy split-layout label; the resolved dataset accepts it too."""
    root = split_dataset(
        tmp_path / "dataset",
        {
            "train": {"a": f"{PET_BOX}\n", "b": f"{PET_BOX}\n"},
            "val": {"a": f"{PET_BOX}\n", "b": f"{PET_BOX}\n"},
        },
        data_yaml=True,
    )
    for split in ("train", "val"):
        for image in (root / "images" / split).glob("*.png"):
            conventional = root / "labels" / split / f"{image.stem}.txt"
            conventional.unlink()
            write_image(image)  # the same bytes; only the label location moves
            (image.parent / f"{image.stem}.txt").write_text(f"{PET_BOX}\n", encoding="utf-8")

    audit = audit_dataset(root / "data.yaml")

    assert not [finding for finding in audit.findings if finding.code == "images_without_labels"]
    assert audit.per_class["train"] == {"PET": 2}
    assert audit.per_class["val"] == {"PET": 2}
