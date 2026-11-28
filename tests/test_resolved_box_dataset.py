"""Eligibility of a resolved box image: a label file must exist.

The ticket is about the difference between a reviewed negative and an image
nobody has reviewed. An explicit empty label file is the first: the image
stays in the evaluator's ground truth with zero annotations. A missing label
file anywhere the codec searches is the second: the audit refuses the dataset
and the evaluator must not silently number it as background.
"""

from __future__ import annotations

from pathlib import Path

from polyx.pipeline.dataset import resolve_box_dataset
from tests.conftest import PET_BOX, box_dataset, write_image


def test_ground_truth_contains_only_eligible_images(tmp_path: Path):
    """The evaluator's denominator is the images that have a label file."""
    dataset = resolve_box_dataset(
        box_dataset(
            tmp_path / "dataset",
            val={"val_boxed": f"{PET_BOX}\n", "val_unlabelled": None},
        ),
        "val",
    )

    ground_truth = dataset.ground_truth("val")

    assert [image["file_name"] for image in ground_truth["images"]] == ["val_boxed.png"]
    assert [image["id"] for image in ground_truth["images"]] == [0]
    assert [annotation["image_id"] for annotation in ground_truth["annotations"]] == [0]
    # The unlabelled image is still resolved and fingerprinted: the audit has
    # to see it and refuse the execution.
    assert [image.path.name for image in dataset.splits["val"]] == [
        "val_boxed.png",
        "val_unlabelled.png",
    ]


def test_an_explicitly_empty_label_stays_a_true_negative(tmp_path: Path):
    """A reviewed negative is an image with an empty label and no annotations."""
    dataset = resolve_box_dataset(
        box_dataset(
            tmp_path / "dataset",
            val={"val_boxed": f"{PET_BOX}\n", "val_cleared": ""},
        ),
        "val",
    )

    ground_truth = dataset.ground_truth("val")

    assert [image["file_name"] for image in ground_truth["images"]] == [
        "val_boxed.png",
        "val_cleared.png",
    ]
    cleared = next(
        image for image in ground_truth["images"] if image["file_name"] == "val_cleared.png"
    )
    assert cleared["id"] == 1
    assert not [
        annotation
        for annotation in ground_truth["annotations"]
        if annotation["image_id"] == cleared["id"]
    ]
    assert [image.path.name for image in dataset.eligible_images("val")] == [
        "val_boxed.png",
        "val_cleared.png",
    ]


def test_audit_refuses_an_image_without_a_label_file(tmp_path: Path):
    """The same dataset the evaluator filters is refused by the preflight audit."""
    dataset = resolve_box_dataset(
        box_dataset(
            tmp_path / "dataset",
            val={"val_boxed": f"{PET_BOX}\n", "val_unlabelled": None},
        ),
        "val",
    )

    audit = dataset.audit()

    assert not audit.ok
    findings = [finding for finding in audit.findings if finding.code == "images_without_labels"]
    assert [
        (finding.parameters["split"], finding.parameters["images"]) for finding in findings
    ] == [("val", ["val_unlabelled.png"])]


def test_a_fully_labelled_dataset_keeps_its_ground_truth(tmp_path: Path):
    """Filtering out absent labels must not change a dataset that has none."""
    dataset = resolve_box_dataset(
        box_dataset(
            tmp_path / "dataset",
            val={"val_boxed": f"{PET_BOX}\n", "val_cleared": ""},
        ),
        "val",
    )

    ground_truth = dataset.ground_truth("val")

    assert dataset.audit().ok
    assert [image["file_name"] for image in ground_truth["images"]] == [
        "val_boxed.png",
        "val_cleared.png",
    ]
    assert [annotation["image_id"] for annotation in ground_truth["annotations"]] == [0]


def test_a_plain_folder_keeps_a_same_directory_label_eligible(tmp_path: Path):
    """Without an ``images/`` ancestor, the codec reads the label beside the image.

    That plain-folder location is the one both the audit and the adapter's
    resolved boxes use, so the image is eligible.
    """
    root = tmp_path / "plain"
    for split in ("train", "val"):
        image = write_image(root / split / f"{split}_plate.png")
        (image.parent / f"{image.stem}.txt").write_text(f"{PET_BOX}\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: train\nval: val\nnames:\n  0: PET\n  1: PP\n",
        encoding="utf-8",
    )

    dataset = resolve_box_dataset(data_yaml, "val")

    assert len(dataset.splits["val"]) == 1
    assert all(image.is_eligible for image in dataset.splits["val"])
    assert dataset.audit().ok
    assert [image["file_name"] for image in dataset.ground_truth("val")["images"]] == [
        "val_plate.png"
    ]


def test_a_split_layout_same_directory_label_is_eligible(tmp_path: Path):
    """A warned legacy label beside a split image is annotated data, not background."""
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        image = write_image(root / "images" / split / f"{split}_plate.png")
        (image.parent / f"{image.stem}.txt").write_text(f"{PET_BOX}\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: PET\n  1: PP\n",
        encoding="utf-8",
    )

    dataset = resolve_box_dataset(data_yaml, "val")

    assert len(dataset.splits["val"]) == 1
    assert all(image.is_eligible for image in dataset.splits["val"])
    assert dataset.audit().ok
    assert [image["file_name"] for image in dataset.ground_truth("val")["images"]] == [
        "val_plate.png"
    ]


def test_a_plain_folder_nested_label_is_eligible(tmp_path: Path):
    """The codec's warned nested legacy location is resolved, not treated as absent."""
    root = tmp_path / "plain"
    for split in ("train", "val"):
        image = write_image(root / split / f"{split}_plate.png")
        labels = image.parent / "labels"
        labels.mkdir()
        (labels / f"{image.stem}.txt").write_text(f"{PET_BOX}\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: train\nval: val\nnames:\n  0: PET\n  1: PP\n",
        encoding="utf-8",
    )

    dataset = resolve_box_dataset(data_yaml, "val")

    assert len(dataset.splits["val"]) == 1
    assert all(image.is_eligible for image in dataset.splits["val"])
    assert dataset.audit().ok
    assert [image["file_name"] for image in dataset.ground_truth("val")["images"]] == [
        "val_plate.png"
    ]


def test_an_unlabelled_image_sorting_first_does_not_shift_ground_truth_ids(tmp_path: Path):
    """The eligible image is renumbered from zero, not left at its sorted slot."""
    dataset = resolve_box_dataset(
        box_dataset(
            tmp_path / "dataset",
            val={"val_a_unlabelled": None, "val_b_boxed": f"{PET_BOX}\n"},
        ),
        "val",
    )

    audit = dataset.audit()
    assert not audit.ok
    findings = [finding for finding in audit.findings if finding.code == "images_without_labels"]
    assert [
        (finding.parameters["split"], finding.parameters["images"]) for finding in findings
    ] == [("val", ["val_a_unlabelled.png"])]

    ground_truth = dataset.ground_truth("val")
    assert [image["file_name"] for image in ground_truth["images"]] == ["val_b_boxed.png"]
    assert [image["id"] for image in ground_truth["images"]] == [0]
    assert [annotation["image_id"] for annotation in ground_truth["annotations"]] == [0]


def test_validation_paths_are_the_eligible_cohort(tmp_path: Path):
    """Prediction ids and ground truth share one eligible, renumbered cohort."""
    from polyx.pipeline.adapters import PreparedDataset, _YoloDataModule
    from polyx.pipeline.config import Protocol

    yaml_path = box_dataset(
        tmp_path / "dataset",
        val={"val_boxed": f"{PET_BOX}\n", "val_cleared": ""},
    )
    dataset = resolve_box_dataset(yaml_path, "val")
    prepared = PreparedDataset(
        data_yaml=yaml_path,
        ground_truth=dataset.ground_truth("val"),
        audit=dataset.audit(),
        hashes=dataset.hashes,
        class_names=dataset.class_names,
        source_kind="yolo_boxes",
        source_path=yaml_path,
        resolved_boxes=dataset,
    )
    loader = _YoloDataModule(dataset=prepared, cfg=None, protocol=Protocol()).val_dataloader()

    assert [Path(item["path"]) for item in loader.dataset] == [
        image.path for image in dataset.eligible_images("val")
    ]
    assert [item["image_id"] for item in loader.dataset] == [0, 1]
