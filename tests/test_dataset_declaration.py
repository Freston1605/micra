"""The YAML box dataset declaration answers one question in one place.

These tests pin the behaviour the five former readers implemented separately:
root resolution, split fallback, membership order, name shapes and the coded
failures that reach the journal. They are the expectation table the refactor
must not regress.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from polyx.core.data import (
    DatasetDeclaration,
    SplitKind,
    declares_real_domain,
    write_declaration,
)
from polyx.core.errors import CodedFailure
from polyx.pipeline.dataset import resolve_box_dataset
from tests.conftest import split_dataset


def write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def touch_images(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def load(tmp_path: Path, data: dict, *, at: str = "dataset") -> DatasetDeclaration:
    return DatasetDeclaration.load(write_yaml(tmp_path / at / "data.yaml", data))


def test_relative_root_and_split_are_resolved_against_the_yaml(tmp_path: Path):
    dataset = tmp_path / "dataset"
    touch_images(dataset / "images" / "train", "b.jpg", "a.jpg")
    touch_images(dataset / "images" / "val", "c.jpg")

    declaration = load(tmp_path, {"path": ".", "train": "images/train", "val": "images/val"})

    assert declaration.path == (dataset / "data.yaml").resolve()
    assert declaration.root == dataset.resolve()
    train = declaration.split("train")
    assert train is not None
    assert train.kind is SplitKind.DIRECTORY
    assert train.path == (dataset / "images" / "train").resolve()
    assert train.exists is True
    assert train.outside_root is False
    assert train.images() == (
        (dataset / "images" / "train" / "a.jpg").resolve(),
        (dataset / "images" / "train" / "b.jpg").resolve(),
    )


def test_split_falls_back_to_the_yaml_folder_when_not_under_the_root(tmp_path: Path):
    dataset = tmp_path / "dataset"
    touch_images(dataset / "images" / "train", "a.jpg")

    declaration = load(
        tmp_path, {"path": "missing-root", "train": "images/train", "val": "images/train"}
    )

    train = declaration.split("train")
    assert train is not None
    assert train.path == (dataset / "images" / "train").resolve()
    assert train.exists is True


def test_absolute_root_and_absolute_split_are_kept(tmp_path: Path):
    root = tmp_path / "elsewhere"
    touch_images(root / "train", "a.png")

    declaration = load(
        tmp_path,
        {"path": str(root), "train": str(root / "train"), "val": str(root / "train")},
    )

    assert declaration.root == root
    train = declaration.split("train")
    assert train is not None
    assert train.path == root / "train"
    assert declaration.split("val").path == root / "train"


def test_absent_split_is_none_and_declared_missing_split_is_empty(tmp_path: Path):
    declaration = load(tmp_path, {"train": "images/train", "val": "images/val"})

    assert declaration.split("test") is None
    train = declaration.split("train")
    assert train is not None
    assert train.kind is SplitKind.DIRECTORY
    assert train.exists is False
    assert train.images() == ()


def test_list_split_keeps_authored_order_and_drops_repeats(tmp_path: Path):
    dataset = tmp_path / "dataset"
    touch_images(dataset / "photos", "first.jpg", "second.jpg")
    (dataset / "train.txt").write_text(
        "# authored by hand\nphotos/second.jpg\nphotos/first.jpg\nphotos/first.jpg\n",
        encoding="utf-8",
    )

    declaration = load(tmp_path, {"path": ".", "train": "train.txt", "val": "train.txt"})

    train = declaration.split("train")
    assert train is not None
    assert train.kind is SplitKind.LIST
    assert train.images() == (
        (dataset / "photos" / "second.jpg").resolve(),
        (dataset / "photos" / "first.jpg").resolve(),
    )


def test_list_entries_resolve_against_the_root_then_the_yaml_folder(tmp_path: Path):
    dataset = tmp_path / "dataset"
    touch_images(tmp_path / "shared", "under-root.jpg")
    touch_images(dataset / "local", "under-yaml.jpg")
    (dataset / "val.txt").write_text(
        "shared/under-root.jpg\nlocal/under-yaml.jpg\n", encoding="utf-8"
    )

    declaration = load(tmp_path, {"path": str(tmp_path), "val": "val.txt"})

    val = declaration.split("val")
    assert val is not None
    assert val.images() == (
        (tmp_path / "shared" / "under-root.jpg").resolve(),
        (dataset / "local" / "under-yaml.jpg").resolve(),
    )


def test_directory_walk_is_recursive_filtered_and_sorted(tmp_path: Path):
    dataset = tmp_path / "dataset"
    touch_images(dataset / "images" / "train" / "nested", "z.JPG", "a.webp")
    (dataset / "images" / "train" / "notes.txt").touch()

    declaration = load(tmp_path, {"train": "images/train", "val": "images/train"})

    train = declaration.split("train")
    assert train is not None
    assert train.images() == (
        (dataset / "images" / "train" / "nested" / "a.webp").resolve(),
        (dataset / "images" / "train" / "nested" / "z.JPG").resolve(),
    )


def test_outside_root_is_a_fact_not_a_refusal(tmp_path: Path):
    touch_images(tmp_path / "outside", "a.jpg")

    declaration = load(
        tmp_path, {"path": ".", "train": str(tmp_path / "outside"), "val": "images/val"}
    )

    train = declaration.split("train")
    assert train is not None
    assert train.outside_root is True
    assert train.images() == ((tmp_path / "outside" / "a.jpg").resolve(),)


def test_non_list_file_split_is_kind_file(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "train.jpg").touch()

    declaration = load(tmp_path, {"train": "train.jpg", "val": "train.jpg"})

    train = declaration.split("train")
    assert train is not None
    assert train.kind is SplitKind.FILE
    assert train.images() == ()


def test_names_accept_a_list_and_a_numeric_mapping(tmp_path: Path):
    listed = load(tmp_path, {"names": ["PET", "PP"], "train": "images/train", "val": "images/val"})
    mapped = DatasetDeclaration.load(
        write_yaml(
            tmp_path / "mapped" / "data.yaml",
            {"names": {"1": "PP", "0": "PET"}, "train": "images/train", "val": "images/val"},
        )
    )

    assert listed.names == {0: "PET", 1: "PP"}
    assert mapped.names == {0: "PET", 1: "PP"}


def test_missing_names_is_empty_and_nc_is_exposed(tmp_path: Path):
    declaration = load(tmp_path, {"nc": "2", "train": "images/train", "val": "images/val"})

    assert declaration.names == {}
    assert declaration.declared_nc == 2


def test_invalid_names_raise_a_coded_failure(tmp_path: Path):
    with pytest.raises(CodedFailure) as raised:
        load(tmp_path, {"names": 3, "train": "images/train", "val": "images/val"})

    assert raised.value.code == "dataset.names_invalid"


def test_sparse_and_non_consecutive_names_are_parsed_not_policed(tmp_path: Path):
    declaration = load(
        tmp_path,
        {"names": {"0": "PET", "2": "LDPE"}, "train": "images/train", "val": "images/val"},
    )

    assert declaration.names == {0: "PET", 2: "LDPE"}


def test_real_domain_is_read_from_the_declaration(tmp_path: Path):
    real = load(tmp_path, {"domain": "real", "train": "images/train", "val": "images/val"})
    nested = DatasetDeclaration.load(
        write_yaml(
            tmp_path / "nested" / "data.yaml",
            {
                "info": {"validation_domain": "real"},
                "train": "images/train",
                "val": "images/val",
            },
        )
    )
    synthetic = load(tmp_path, {"train": "images/train", "val": "images/val"})

    assert real.is_real_domain is True
    assert nested.is_real_domain is True
    assert synthetic.is_real_domain is False
    assert declares_real_domain({"domain": "real_domain"}) is True
    assert declares_real_domain(None) is False


def test_write_round_trips_to_the_same_facts(tmp_path: Path):
    dataset = tmp_path / "dataset"
    touch_images(dataset / "images" / "train", "a.jpg")
    touch_images(dataset / "images" / "val", "b.jpg")
    (dataset / "test.txt").write_text("images/val/b.jpg\n", encoding="utf-8")
    original = DatasetDeclaration.create(
        dataset / "data.yaml",
        root=dataset,
        names={0: "PET", 1: "PP"},
        splits={
            "train": dataset / "images" / "train",
            "val": dataset / "images" / "val",
            "test": dataset / "test.txt",
        },
    )

    write_declaration(original, dataset / "data.yaml")
    written = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    reloaded = DatasetDeclaration.load(dataset / "data.yaml")

    assert written["path"] == str(dataset.resolve())
    assert written["train"] == "images/train"
    assert written["test"] == "test.txt"
    assert written["names"] == ["PET", "PP"]
    assert written["nc"] == 2
    assert reloaded.root == original.root
    assert reloaded.names == original.names
    assert reloaded.declared_nc == 2
    assert reloaded.split("test").kind is SplitKind.LIST
    assert reloaded.split("test").images() == ((dataset / "images" / "val" / "b.jpg").resolve(),)


def test_create_infers_a_missing_list_and_a_directory(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    declaration = DatasetDeclaration.create(
        dataset / "data.yaml",
        root=dataset,
        names=["PET"],
        splits={"train": dataset / "train.txt", "val": dataset / "images" / "val"},
    )

    assert declaration.split("train").kind is SplitKind.LIST
    assert declaration.split("val").kind is SplitKind.DIRECTORY
    assert declaration.split("val").exists is False


def test_missing_yaml_raises_data_yaml_missing(tmp_path: Path):
    with pytest.raises(CodedFailure) as raised:
        DatasetDeclaration.load(tmp_path / "absent" / "data.yaml")

    assert raised.value.code == "data_yaml_missing"
    assert raised.value.parameters["path"] == str((tmp_path / "absent" / "data.yaml").resolve())


def test_malformed_yaml_raises_data_yaml_unreadable(tmp_path: Path):
    path = tmp_path / "dataset" / "data.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("train: [unterminated\n", encoding="utf-8")

    with pytest.raises(CodedFailure) as raised:
        DatasetDeclaration.load(path)

    assert raised.value.code == "data_yaml_unreadable"


def test_non_mapping_yaml_raises_declaration_not_mapping(tmp_path: Path):
    path = write_yaml(tmp_path / "dataset" / "data.yaml", ["not", "a", "mapping"])

    with pytest.raises(CodedFailure) as raised:
        DatasetDeclaration.load(path)

    assert raised.value.code == "dataset.declaration_not_mapping"


def test_unreadable_split_list_raises_a_coded_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "train.txt").write_text("a.jpg\n", encoding="utf-8")
    declaration = load(tmp_path, {"train": "train.txt", "val": "train.txt"})
    train = declaration.split("train")
    assert train is not None

    def refuse(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", refuse)

    with pytest.raises(CodedFailure) as raised:
        train.images()

    assert raised.value.code == "dataset.split_list_unreadable"


def test_training_seam_still_refuses_a_list_split(tmp_path: Path):
    """The declaration represents a list; the headless resolver still refuses it."""
    root = split_dataset(
        tmp_path / "dataset",
        {
            "train": {"train_0": "0 0.5 0.5 0.2 0.2\n"},
            "val": {"val_0": "0 0.5 0.5 0.2 0.2\n"},
        },
        data_yaml=True,
    )
    (root / "train.txt").write_text("images/train/train_0.png\n", encoding="utf-8")
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    data["train"] = "train.txt"
    write_yaml(root / "data.yaml", data)

    with pytest.raises(CodedFailure) as raised:
        resolve_box_dataset(root / "data.yaml", "val")

    assert raised.value.code == "dataset.box_split_not_directory"


def test_training_seam_still_refuses_sparse_class_ids(tmp_path: Path):
    root = split_dataset(
        tmp_path / "dataset",
        {
            "train": {"train_0": "0 0.5 0.5 0.2 0.2\n"},
            "val": {"val_0": "0 0.5 0.5 0.2 0.2\n"},
        },
        data_yaml=True,
    )
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    data["names"] = {0: "PET", 2: "LDPE"}
    write_yaml(root / "data.yaml", data)

    with pytest.raises(CodedFailure) as raised:
        resolve_box_dataset(root / "data.yaml", "val")

    assert raised.value.code == "dataset.invalid_box_classes"
