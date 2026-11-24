"""The committed Lightning checkpoint is the one authority for recovery.

Each test stops a real execution between Lightning committing an epoch's
checkpoint and the engine writing that epoch's progress heartbeat, then resumes
through the same headless command. The Trainer is never mocked: the proof is
the checkpoint Lightning itself wrote, the number of validations the resumed
process actually ran, and the artifacts the command published.

The progress record is deliberately left stale or removed. If any of its
values still decided an epoch count, a winner or a stop, these tests would
publish a different result.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from polyx.pipeline import lightning_loop
from tests.conftest import build_tiny_dataset
from tests.test_pipeline_local_tracer import (
    _boxes_prediction,
    _FakeYOLO,
    _install_fakes,
    _read,
    _same_weights,
)


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


def _partial_prediction() -> list[dict]:
    """Two of the three validation instances, so with-class F1 is 0.8."""
    return _boxes_prediction()[:2]


def _empty_prediction() -> list[dict]:
    return []


def _overrides(output: Path, data_yaml: Path, *, epochs: int, patience: int = 99) -> list[str]:
    return [
        f"data_yaml={data_yaml}",
        f"output_dir={output}",
        f"protocol.epochs={epochs}",
        "protocol.batch=2",
        "protocol.image_size=64",
        "protocol.warmup_epochs=0",
        "protocol.amp.enabled=false",
        f"protocol.early_stopping.patience={patience}",
    ]


def _interrupt_after_checkpoint(monkeypatch: pytest.MonkeyPatch, *, epoch_completed: int) -> None:
    """Stop the engine just before it writes one epoch's progress heartbeat.

    Lightning has already saved ``last.ckpt`` and the epoch's checkpoint by the
    time ``record_epoch`` runs, so the committed state survives and only the
    heartbeat is missing or one epoch behind.
    """
    real_write = lightning_loop.write_json

    def failing_write(data: dict, path: Path) -> None:
        if (
            Path(path).name == "progress.json"
            and int(data.get("epoch_completed", -1)) == epoch_completed
        ):
            raise RuntimeError("simulated preemption after the checkpoint commit")
        real_write(data, path)

    monkeypatch.setattr(lightning_loop, "write_json", failing_write)


def _committed(output: Path) -> dict:
    """The checkpoint Lightning committed as this execution's resume point."""
    return torch.load(
        output / "checkpoints" / "lightning" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )


def _callback_state(checkpoint: dict, prefix: str) -> dict:
    """One callback's restored state, as Lightning serialised it."""
    for key, value in checkpoint["callbacks"].items():
        if key.startswith(prefix):
            return value
    raise AssertionError(f"the checkpoint holds no {prefix} state")


def test_an_interrupted_epoch_is_not_retrained_and_its_winner_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The heartbeat lags one epoch; the checkpoint decides where to resume."""
    output = tmp_path / "executions" / "authority-mid-run"
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    overrides = _overrides(output, data_yaml, epochs=3)

    with monkeypatch.context() as patch:
        _install_fakes(patch, [_partial_prediction(), _boxes_prediction()])
        _interrupt_after_checkpoint(patch, epoch_completed=1)
        assert _command(*overrides) == 1

    # Proof of the window: epoch 1's checkpoint is committed while the progress
    # record still describes epoch 0 and an older winner.
    committed = _committed(output)
    assert committed["epoch"] == 1
    progress = _read(output / "progress.json")
    assert progress["epochs_completed"] == 1
    assert Path(progress["best_lightning_checkpoint"]).name == "checkpoint_000.ckpt"

    # Epoch 2 is the only epoch left, plus the publication re-score of epoch 1.
    _install_fakes(monkeypatch, [_empty_prediction(), _boxes_prediction()])
    assert _command(*overrides, "resume=true") == 0
    assert _FakeYOLO.validation_calls == 2

    state = _read(output / "training_state.json")
    metrics = _read(output / "metrics.json")
    assert state["epochs_completed"] == 3
    assert state["resumed_from_epoch"] == 2, "epoch 1 must not be retrained"
    assert state["resumed_from"] == str(output / "checkpoints" / "lightning" / "last.ckpt")
    assert metrics["selected_epoch"] == state["early_stopping"]["best_epoch"] == 1
    assert metrics["selection"]["value"] == pytest.approx(1.0)
    assert _same_weights(
        output / "checkpoints" / "best.pt", output / "checkpoints" / "epoch_001.pt"
    )
    # The heartbeat itself catches up from the callbacks, not from its old value.
    heartbeat = _read(output / "progress.json")
    assert heartbeat["best_epoch"] == 1
    assert Path(heartbeat["best_lightning_checkpoint"]).name == "checkpoint_001.ckpt"


def test_a_committed_final_epoch_is_published_without_another_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last epoch's checkpoint names the winner even if the heartbeat does not."""
    output = tmp_path / "executions" / "authority-final-epoch"
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    overrides = _overrides(output, data_yaml, epochs=2)

    with monkeypatch.context() as patch:
        _install_fakes(patch, [_partial_prediction(), _boxes_prediction()])
        _interrupt_after_checkpoint(patch, epoch_completed=1)
        assert _command(*overrides) == 1

    committed = _committed(output)
    assert committed["epoch"] == 1
    progress = _read(output / "progress.json")
    assert progress["epochs_completed"] == 1
    assert Path(progress["best_lightning_checkpoint"]).name == "checkpoint_000.ckpt"

    # Only the publication re-score of the committed winner may run.
    _install_fakes(monkeypatch, [_boxes_prediction()])
    assert _command(*overrides, "resume=true") == 0
    assert _FakeYOLO.validation_calls == 1

    state = _read(output / "training_state.json")
    metrics = _read(output / "metrics.json")
    assert state["epochs_completed"] == 2
    assert metrics["selected_epoch"] == state["early_stopping"]["best_epoch"] == 1
    assert metrics["selection"]["value"] == pytest.approx(1.0)
    assert _same_weights(
        output / "checkpoints" / "best.pt", output / "checkpoints" / "epoch_001.pt"
    )
    assert (output / "completion.json").is_file()


def test_a_committed_checkpoint_without_a_progress_record_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No heartbeat at all: the checkpoint still carries the recovery state."""
    output = tmp_path / "executions" / "authority-no-progress"
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    overrides = _overrides(output, data_yaml, epochs=2)

    with monkeypatch.context() as patch:
        _install_fakes(patch, [_boxes_prediction(), _partial_prediction()])
        # The first epoch's heartbeat is the one that never gets written.
        _interrupt_after_checkpoint(patch, epoch_completed=0)
        assert _command(*overrides) == 1

    assert not (output / "progress.json").exists()
    committed = _committed(output)
    assert committed["epoch"] == 0
    assert "polyx_recovery" in committed, "the checkpoint must carry its fit provenance"

    # The resumed epoch does not improve; the earlier winner must survive.
    _install_fakes(monkeypatch, [_partial_prediction(), _boxes_prediction()])
    assert _command(*overrides, "resume=true") == 0
    assert _FakeYOLO.validation_calls == 2

    state = _read(output / "training_state.json")
    metrics = _read(output / "metrics.json")
    assert state["epochs_completed"] == 2
    assert state["resumed"] is True
    assert metrics["selected_epoch"] == state["early_stopping"]["best_epoch"] == 0
    assert metrics["selection"]["value"] == pytest.approx(1.0)
    assert _same_weights(
        output / "checkpoints" / "best.pt", output / "checkpoints" / "epoch_000.pt"
    )
    # The fit facts come from the checkpoint, so publication does not invent them.
    assert state["precision"]
    assert state["device"]
    assert isinstance(state["weights_verified"], bool)
    assert (output / "completion.json").is_file()


def test_resume_without_a_committed_checkpoint_fails_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A progress heartbeat cannot stand in for a missing checkpoint."""
    output = tmp_path / "executions" / "authority-no-checkpoint"
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    overrides = _overrides(output, data_yaml, epochs=2)

    with monkeypatch.context() as patch:
        _install_fakes(patch, [_partial_prediction(), _boxes_prediction()])
        _interrupt_after_checkpoint(patch, epoch_completed=1)
        assert _command(*overrides) == 1

    assert (output / "progress.json").is_file()
    shutil.rmtree(output / "checkpoints" / "lightning")

    _install_fakes(monkeypatch, [])
    assert _command(*overrides, "resume=true") == 1
    error = capsys.readouterr().err
    assert "committed Lightning checkpoint" in error
    assert _FakeYOLO.validation_calls == 0, "a missing checkpoint must not start a fit"
    assert not (output / "completion.json").exists()
    assert not (output / "training_state.json").exists()


def test_a_missing_committed_winner_is_named_and_nothing_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The winner the committed state names is required, not substituted."""
    output = tmp_path / "executions" / "authority-missing-winner"
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    overrides = _overrides(output, data_yaml, epochs=2)

    with monkeypatch.context() as patch:
        _install_fakes(patch, [_boxes_prediction(), _partial_prediction()])
        _interrupt_after_checkpoint(patch, epoch_completed=1)
        assert _command(*overrides) == 1

    winner = output / "checkpoints" / "lightning" / "checkpoint_000.ckpt"
    assert _callback_state(_committed(output), "ModelCheckpoint")["best_model_path"] == str(winner)
    winner.unlink()

    _install_fakes(monkeypatch, [])
    assert _command(*overrides, "resume=true") == 1
    error = capsys.readouterr().err
    assert "checkpoint_000.ckpt" in error
    assert "missing" in error
    assert _FakeYOLO.validation_calls == 0
    assert not (output / "completion.json").exists()


def test_a_committed_early_stop_is_honoured_without_another_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stopping decision travels in the checkpoint, not in the heartbeat."""
    output = tmp_path / "executions" / "authority-early-stop"
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    overrides = _overrides(output, data_yaml, epochs=5, patience=1)

    with monkeypatch.context() as patch:
        # Epoch 1 is the only non-improving record: patience expires and the
        # checkpoint is saved with that decision in it.
        _install_fakes(patch, [_boxes_prediction(), _empty_prediction()])
        _interrupt_after_checkpoint(patch, epoch_completed=1)
        assert _command(*overrides) == 1

    committed = _committed(output)
    assert committed["epoch"] == 1
    early = _callback_state(committed, "EarlyStopping")
    assert early["wait_count"] == 1
    assert early["stopping_reason"] != 0
    # The heartbeat still describes epoch 0, when nothing had stopped.
    progress = _read(output / "progress.json")
    assert progress["early_stopping"]["stopped_early"] is False

    # Only the publication re-score of the kept epoch 0 may run.
    _install_fakes(monkeypatch, [_boxes_prediction()])
    assert _command(*overrides, "resume=true") == 0
    assert _FakeYOLO.validation_calls == 1

    state = _read(output / "training_state.json")
    metrics = _read(output / "metrics.json")
    assert state["epochs_completed"] == 2
    assert state["early_stopping"]["stopped_early"] is True
    assert state["early_stopping"]["wait_count"] == 1
    assert state["early_stopping"]["best_epoch"] == 0
    assert metrics["selected_epoch"] == 0
    assert (output / "completion.json").is_file()


def test_stale_progress_values_never_decide_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every outcome-bearing field is replaced by the committed checkpoint."""
    output = tmp_path / "executions" / "authority-stale-heartbeat"
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    overrides = _overrides(output, data_yaml, epochs=3)

    with monkeypatch.context() as patch:
        _install_fakes(patch, [_partial_prediction(), _boxes_prediction()])
        _interrupt_after_checkpoint(patch, epoch_completed=1)
        assert _command(*overrides) == 1

    progress = _read(output / "progress.json")
    progress["epochs_completed"] = 99
    progress["best_score"] = 0.0
    progress["best_epoch"] = 7
    progress["resume_checkpoint"] = None
    progress["best_lightning_checkpoint"] = str(tmp_path / "does-not-exist.ckpt")
    progress["stopped_early"] = True
    progress["early_stopping"]["wait_count"] = 99
    (output / "progress.json").write_text(json.dumps(progress), encoding="utf-8")

    # The committed checkpoint says epoch 1, nothing stopped, so epoch 2 runs.
    _install_fakes(monkeypatch, [_empty_prediction(), _boxes_prediction()])
    assert _command(*overrides, "resume=true") == 0
    assert _FakeYOLO.validation_calls == 2

    state = _read(output / "training_state.json")
    metrics = _read(output / "metrics.json")
    assert state["epochs_completed"] == 3
    assert state["resumed_from_epoch"] == 2
    assert state["early_stopping"]["wait_count"] == 1
    assert metrics["selected_epoch"] == 1
    assert metrics["selection"]["value"] == pytest.approx(1.0)
