"""The held-out cohort refuses an image nobody annotated.

Held-out scoring is the final figure of a benchmark, so its denominator must
be the images that were actually reviewed. An empty label file is a reviewed
negative and is scored as a true negative; a missing label file is refused
before the cohort record or any prediction is built (resolution has already
hashed the inputs) instead of being dropped or scored as background.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polyx.pipeline.errors import PipelineError
from polyx.pipeline.heldout import _box_cohort
from tests.conftest import PET_BOX, box_dataset


def test_a_test_fold_with_an_unlabelled_image_is_refused(tmp_path: Path):
    """An unannotated test image is neither dropped nor scored as background."""
    yaml_path = box_dataset(
        tmp_path / "dataset",
        test={"test_boxed": f"{PET_BOX}\n", "test_unlabelled": None},
    )

    with pytest.raises(PipelineError) as raised:
        _box_cohort(yaml_path)

    assert raised.value.code == "heldout.unlabelled_test_image"
    assert raised.value.parameters["count"] == 1
    assert raised.value.parameters["images"] == ["test_unlabelled.png"]


def test_a_fully_labelled_test_fold_builds_one_eligible_cohort(tmp_path: Path):
    """Paths, hashes and ground truth describe the same reviewed test images."""
    yaml_path = box_dataset(
        tmp_path / "dataset",
        test={"test_boxed": f"{PET_BOX}\n", "test_cleared": ""},
    )

    cohort = _box_cohort(yaml_path)

    expected = ["test_boxed.png", "test_cleared.png"]
    assert cohort["images"] == 2
    assert [path.name for path in cohort["paths"]] == expected
    assert [image["file_name"] for image in cohort["ground_truth"]["images"]] == expected
    assert [annotation["image_id"] for annotation in cohort["ground_truth"]["annotations"]] == [0]
    assert {
        "labels/test/test_boxed.txt",
        "labels/test/test_cleared.txt",
    } <= set(cohort["hashes"])
