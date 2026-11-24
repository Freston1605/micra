"""The search space a sweep may declare, and the fields it may never touch."""

from __future__ import annotations

from typing import Any

import pytest

from polyx.pipeline.sweep_space import (
    SEARCHABLE_PROTOCOL_FIELDS,
    validate_direction,
    validate_sampled_parameters,
    validate_search_space,
)


class _Trial:
    """The part of an Optuna trial the guard reads."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params


def test_every_declared_hyperparameter_is_accepted() -> None:
    validate_search_space(f"protocol.{field}" for field in sorted(SEARCHABLE_PROTOCOL_FIELDS))


@pytest.mark.parametrize(
    ("name", "message"),
    [
        # The selection split names its canonical value and the reason.
        ("protocol.split", r"fixed to 'val'"),
        ("protocol.seed", "only declare protocol hyperparameters"),
        ("protocol.augmentation_subset", "only declare protocol hyperparameters"),
        ("protocol.early_stopping.patience", "only declare protocol hyperparameters"),
        ("protocol.confidence", "only declare protocol hyperparameters"),
        ("evaluator", "only declare protocol hyperparameters"),
        ("family.name", "only declare protocol hyperparameters"),
    ],
)
def test_a_fixed_benchmark_field_is_refused(name: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_search_space([name])


def test_a_searched_split_error_names_the_canonical_value_and_the_reason() -> None:
    with pytest.raises(ValueError) as caught:
        validate_search_space(["protocol.split"])
    message = str(caught.value)
    assert "'val'" in message
    assert "held-out" in message


def test_a_custom_search_space_cannot_smuggle_a_fixed_field() -> None:
    """The plugin's callback suggests names after the declared space is checked."""
    with pytest.raises(ValueError, match=r"fixed to 'val'"):
        validate_sampled_parameters(
            [_Trial({"protocol.learning_rate": 0.01, "protocol.split": "val"})]
        )


def test_declared_only_trial_parameters_pass() -> None:
    validate_sampled_parameters([_Trial({"protocol.learning_rate": 0.01})])
    validate_sampled_parameters([_Trial({})])


def test_the_direction_is_normalized_and_must_maximize() -> None:
    assert validate_direction("maximize") == "maximize"
    assert validate_direction("MAXIMIZE") == "maximize"

    with pytest.raises(ValueError, match="must be 'maximize'"):
        validate_direction("minimize")
