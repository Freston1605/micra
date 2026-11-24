"""A resume continues the one execution whose checkpoint it holds.

The proof is external: the command's exit status, the identity record on disk,
the bytes of the committed Lightning checkpoint, and the epochs the resumed
process actually ran. Lightning and the framework boundary stay as real as the
sibling resume suites make them; only the model wrapper is a double.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
import torch

from polyx.pipeline.errors import PipelineError
from polyx.pipeline.resume import (
    RECOVERY_KEY,
    RECOVERY_VERSION,
    checkpoint_epoch,
    load_recovery,
)
from tests.conftest import build_tiny_dataset
from tests.test_pipeline_local_tracer import (
    _boxes_prediction,
    _FakeYOLO,
    _install_fakes,
    _read,
)


class _Poison:
    """A non-primitive payload a restricted unpickler must refuse."""

    def __init__(self, marker: str) -> None:
        self.marker = marker


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


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


def _staged_weights(root: Path, payload: bytes = b"initial-weights-v1") -> tuple[Path, Path]:
    """A pinned weight and the manifest that verifies it."""
    directory = root / "weights"
    directory.mkdir(parents=True, exist_ok=True)
    weight = directory / "yolov8n.pt"
    weight.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = directory / "SHA256SUMS"
    manifest.write_text(f"{digest}  yolov8n.pt\n", encoding="utf-8")
    return directory, manifest


def _pin(overrides: list[str], directory: Path, manifest: Path) -> list[str]:
    return [*overrides, f"weights_dir={directory}", f"weights_manifest={manifest}"]


def _committed(output: Path) -> dict:
    return torch.load(
        output / "checkpoints" / "lightning" / "last.ckpt",
        map_location="cpu",
        weights_only=True,
    )


def _callback_state(checkpoint: dict, prefix: str) -> dict:
    for key, value in checkpoint["callbacks"].items():
        if key.startswith(prefix):
            return value
    raise AssertionError(f"the checkpoint holds no {prefix} state")


def _interrupt(output: Path, data_yaml: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    """Leave epoch 0 committed and the next validation preempted."""
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()], interrupt_after=2)
    assert _command(*_overrides(output, data_yaml, epochs=2), *extra) == 1


def test_a_fresh_execution_persists_one_identity_record_and_embeds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "identity-fresh"
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    assert _command(*_overrides(output, data_yaml, epochs=1)) == 0

    identity = _read(output / "execution_identity.json")
    assert identity["family"] == {
        "name": "yolo_detection",
        "framework": "ultralytics",
        "task": "detect",
        "weights": "yolov8n.pt",
    }
    assert identity["weights"] == {
        "declared": "yolov8n.pt",
        "verified": False,
        "identity": "yolov8n.pt",
    }
    assert identity["dataset_fingerprint"] == _read(output / "progress.json")["dataset_fingerprint"]
    assert identity["dataset_fingerprint"]

    committed = _committed(output)
    assert committed["polyx_recovery"]["identity"] == identity
    # The same record is readable from the checkpoint through restricted I/O.
    recovered = load_recovery(output / "checkpoints" / "lightning" / "last.ckpt")
    assert recovered.identity.to_dict() == identity
    assert recovered.epochs_completed == 1
    assert checkpoint_epoch(output / "checkpoints" / "lightning" / "last.ckpt") == 0


def test_a_pinned_weight_is_identified_by_its_verified_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "identity-pinned"
    directory, manifest = _staged_weights(tmp_path)
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    assert (
        _command(
            *_pin(_overrides(output, data_yaml, epochs=1), directory, manifest),
        )
        == 0
    )

    identity = _read(output / "execution_identity.json")
    digest = hashlib.sha256(b"initial-weights-v1").hexdigest()
    assert identity["weights"] == {
        "declared": "yolov8n.pt",
        "verified": True,
        "identity": f"sha256:{digest}",
    }


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ("family.name=another_detector", "family.name"),
        ("family.framework=torchvision", "family.framework"),
        ("family.weights=other.pt", "family.weights"),
    ],
)
def test_resume_refuses_a_changed_family_and_names_the_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
    override: str,
    field: str,
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "identity-family"
    _interrupt(output, data_yaml, monkeypatch)

    _install_fakes(monkeypatch, [])
    capsys.readouterr()
    selected_language("en")
    assert _command(*_overrides(output, data_yaml, epochs=2), override, "resume=true") == 1

    error = capsys.readouterr().err
    assert field in error
    assert "resume" in error
    assert _FakeYOLO.validation_calls == 0, "a refused resume must not start a fit"
    assert not (output / "training_state.json").exists()


def test_resume_refuses_changed_initialization_weights_and_names_the_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "identity-weights"
    directory, manifest = _staged_weights(tmp_path)
    _interrupt(output, data_yaml, monkeypatch, *_pin([], directory, manifest))

    # The same declared name now resolves to different bytes.
    _staged_weights(tmp_path, b"initial-weights-v2")
    _install_fakes(monkeypatch, [])
    capsys.readouterr()
    selected_language("en")
    assert (
        _command(
            *_pin(_overrides(output, data_yaml, epochs=2), directory, manifest),
            "resume=true",
        )
        == 1
    )

    error = capsys.readouterr().err
    assert "initialization weights" in error
    assert _FakeYOLO.validation_calls == 0


def test_resume_refuses_a_changed_dataset_fingerprint_and_names_the_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "identity-dataset"
    _interrupt(output, data_yaml, monkeypatch)

    label = data_yaml.parent / "labels" / "val" / "plate_v1.txt"
    label.write_text(
        label.read_text(encoding="utf-8").replace("0.375", "0.380", 1), encoding="utf-8"
    )

    _install_fakes(monkeypatch, [])
    capsys.readouterr()
    selected_language("en")
    assert _command(*_overrides(output, data_yaml, epochs=2), "resume=true") == 1

    error = capsys.readouterr().err
    assert "dataset fingerprint" in error
    assert _FakeYOLO.validation_calls == 0


def test_a_moved_execution_resumes_from_its_own_checkpoint_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lightning's absolute best_model_path is stale after a move; the file is not."""
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    source = tmp_path / "executions" / "move-source"
    _interrupt(source, data_yaml, monkeypatch)
    recorded = Path(_callback_state(_committed(source), "ModelCheckpoint")["best_model_path"])
    assert recorded.parent == source / "checkpoints" / "lightning"

    moved = tmp_path / "executions" / "move-target"
    shutil.move(str(source), str(moved))
    assert not recorded.exists(), "the moved execution no longer owns the old path"

    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    assert _command(*_overrides(moved, data_yaml, epochs=2), "resume=true") == 0

    state = _read(moved / "training_state.json")
    assert state["epochs_completed"] == 2
    assert state["resumed"] is True
    assert state["resumed_from_epoch"] == 1
    assert state["resumed_from"] == str(moved / "checkpoints" / "lightning" / "last.ckpt")
    assert (moved / "completion.json").is_file()


def test_an_external_checkpoint_reference_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "external-reference"
    _interrupt(output, data_yaml, monkeypatch)

    outside = tmp_path / "outside" / "checkpoint_999.ckpt"
    outside.parent.mkdir()
    outside.write_bytes(b"not this execution's state")
    checkpoint = _committed(output)
    _callback_state(checkpoint, "ModelCheckpoint")["best_model_path"] = str(outside)
    torch.save(checkpoint, output / "checkpoints" / "lightning" / "last.ckpt")

    _install_fakes(monkeypatch, [])
    capsys.readouterr()
    selected_language("en")
    assert _command(*_overrides(output, data_yaml, epochs=2), "resume=true") == 1

    error = capsys.readouterr().err
    assert "outside this execution's checkpoint directory" in error
    assert _FakeYOLO.validation_calls == 0


def test_a_same_named_checkpoint_outside_the_execution_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """A recorded path is verified before its file name is looked up locally.

    The bug: the reference was resolved by file name first, so an external
    ``/elsewhere/checkpoint_000.ckpt`` was silently accepted because this
    execution happened to keep a file of the same name.
    """
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "same-name-outside"
    _interrupt(output, data_yaml, monkeypatch)

    local = output / "checkpoints" / "lightning" / "checkpoint_000.ckpt"
    assert local.is_file(), "the local name that used to shadow the check must exist"
    external = tmp_path / "elsewhere" / "checkpoint_000.ckpt"
    external.parent.mkdir()
    external.write_bytes(b"another execution's state")

    checkpoint = _committed(output)
    _callback_state(checkpoint, "ModelCheckpoint")["best_model_path"] = str(external)
    torch.save(checkpoint, output / "checkpoints" / "lightning" / "last.ckpt")

    _install_fakes(monkeypatch, [])
    capsys.readouterr()
    selected_language("en")
    assert _command(*_overrides(output, data_yaml, epochs=2), "resume=true") == 1

    error = capsys.readouterr().err
    assert "outside this execution's checkpoint directory" in error
    assert _FakeYOLO.validation_calls == 0

    # The refusal keeps an invariant identity: the journal records the code,
    # while the terminal got the translated sentence.
    import os

    journal = Path(os.environ["POLYX_LOG_DIR"]) / "pipeline.jsonl"
    assert '"error_code": "resume.checkpoint_outside"' in journal.read_text(encoding="utf-8")


def test_a_legacy_checkpoint_without_an_identity_is_refused_and_left_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """The old progress-record fallback is superseded: no identity, no resume."""
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "legacy-checkpoint"
    _interrupt(output, data_yaml, monkeypatch)
    last = output / "checkpoints" / "lightning" / "last.ckpt"

    # A checkpoint from before this record existed: protocol, fingerprint and
    # fit are present, but no identity. The progress record still carries the
    # old fallback values, which must not be enough either.
    legacy = torch.load(last, map_location="cpu", weights_only=True)
    metadata = legacy["polyx_recovery"]
    legacy["polyx_recovery"] = {
        "protocol": metadata["identity"]["protocol"],
        "dataset_fingerprint": metadata["identity"]["dataset_fingerprint"],
        "fit": metadata.get("fit"),
    }
    torch.save(legacy, last)
    before = last.read_bytes()

    _install_fakes(monkeypatch, [])
    capsys.readouterr()
    selected_language("en")
    assert _command(*_overrides(output, data_yaml, epochs=2), "resume=true") == 1

    error = capsys.readouterr().err
    assert "no verifiable execution identity" in error
    assert "left untouched" in error
    assert last.is_file()
    assert last.read_bytes() == before
    assert _FakeYOLO.validation_calls == 0
    assert not (output / "training_state.json").exists()
    assert not (output / "completion.json").exists()


def test_a_poisoned_checkpoint_is_refused_without_an_unrestricted_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "poisoned-checkpoint"
    _interrupt(output, data_yaml, monkeypatch)
    last = output / "checkpoints" / "lightning" / "last.ckpt"

    poisoned = torch.load(last, map_location="cpu", weights_only=True)
    poisoned["foreign_payload"] = _Poison("must not execute")
    torch.save(poisoned, last)

    _install_fakes(monkeypatch, [])
    capsys.readouterr()
    selected_language("en")
    assert _command(*_overrides(output, data_yaml, epochs=2), "resume=true") == 1

    error = capsys.readouterr().err
    assert "restricted loading" in error
    assert _FakeYOLO.validation_calls == 0


def test_fit_resume_and_revalidation_request_restricted_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "restricted-calls"
    _interrupt(output, data_yaml, monkeypatch)

    from lightning import Trainer

    seen: list[tuple[str, object]] = []
    real_fit = Trainer.fit
    real_validate = Trainer.validate

    def recording_fit(self, *args, **kwargs):
        seen.append(("fit", kwargs.get("weights_only")))
        return real_fit(self, *args, **kwargs)

    def recording_validate(self, *args, **kwargs):
        seen.append(("validate", kwargs.get("weights_only")))
        return real_validate(self, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", recording_fit)
    monkeypatch.setattr(Trainer, "validate", recording_validate)

    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    assert _command(*_overrides(output, data_yaml, epochs=2), "resume=true") == 0

    assert ("fit", True) in seen, "the resumed fit must request restricted loading"
    assert ("validate", True) in seen, "re-validation must request restricted loading"
    assert all(weights_only is True for _name, weights_only in seen)


def test_staged_weights_initialise_a_new_run_and_never_count_as_a_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "weights-are-not-a-resume"
    staged = output / "checkpoints" / "weights_init.pt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"initialization only")

    _install_fakes(monkeypatch, [])
    capsys.readouterr()
    selected_language("en")
    assert (
        _command(
            f"data_yaml={data_yaml}",
            f"output_dir={output}",
            f"family.weights={staged}",
            "protocol.epochs=1",
            "resume=true",
        )
        == 1
    )

    error = capsys.readouterr().err
    assert "no committed Lightning checkpoint" in error
    assert _FakeYOLO.validation_calls == 0
    assert not (output / "training_state.json").exists()


def test_a_committed_checkpoint_records_the_version_and_a_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real serialized checkpoint states the recovery layout it follows."""
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "recovery-version"
    _interrupt(output, data_yaml, monkeypatch)

    checkpoint = _committed(output)
    metadata = checkpoint[RECOVERY_KEY]
    assert metadata["version"] == RECOVERY_VERSION
    assert Path(metadata["execution_root"]) == output
    assert metadata["callbacks"]["model_checkpoint"]
    assert metadata["callbacks"]["early_stopping"]

    state = load_recovery(
        output / "checkpoints" / "lightning" / "last.ckpt",
        checkpoint_dir=output / "checkpoints" / "lightning",
        root=output,
    )
    assert state.epochs_completed == 1
    assert state.best_epoch == 0
    assert state.identity.family["name"] == "yolo_detection"


def test_a_newer_recovery_layout_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint from a future engine is not read with today's rules."""
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "recovery-future"
    _interrupt(output, data_yaml, monkeypatch)

    checkpoint = _committed(output)
    checkpoint[RECOVERY_KEY]["version"] = RECOVERY_VERSION + 1
    torch.save(checkpoint, output / "checkpoints" / "lightning" / "last.ckpt")

    with pytest.raises(PipelineError) as raised:
        load_recovery(
            output / "checkpoints" / "lightning" / "last.ckpt",
            checkpoint_dir=output / "checkpoints" / "lightning",
            root=output,
        )
    assert raised.value.code == "resume.version_unknown"


def test_the_engine_reads_a_callback_by_its_recorded_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated Lightning key is not the identity recovery depends on."""
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "callback-alias"
    _interrupt(output, data_yaml, monkeypatch)

    checkpoint = _committed(output)
    model_key = next(key for key in checkpoint["callbacks"] if key.startswith("ModelCheckpoint"))
    alias = "kept-checkpoint-under-a-stable-name"
    checkpoint["callbacks"][alias] = checkpoint["callbacks"][model_key]
    checkpoint[RECOVERY_KEY]["callbacks"]["model_checkpoint"] = alias
    torch.save(checkpoint, output / "checkpoints" / "lightning" / "last.ckpt")

    state = load_recovery(
        output / "checkpoints" / "lightning" / "last.ckpt",
        checkpoint_dir=output / "checkpoints" / "lightning",
        root=output,
    )
    assert state.best_epoch == 0

    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    assert _command(*_overrides(output, data_yaml, epochs=2), "resume=true") == 0
    assert _read(output / "training_state.json")["epochs_completed"] == 2


@pytest.mark.parametrize(
    ("reason", "stopped"),
    [
        (0, False),
        (3, True),
        (False, False),
        (True, True),
        ("NOT_STOPPED", False),
        ("EarlyStoppingReason.PATIENCE_EXHAUSTED", True),
        ("non_finite_metric", True),
    ],
)
def test_known_stopping_reason_spellings_are_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: object,
    stopped: bool,
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / f"stopping-{reason}"
    _interrupt(output, data_yaml, monkeypatch)

    checkpoint = _committed(output)
    early = _callback_state(checkpoint, "EarlyStopping")
    early["stopping_reason"] = reason
    torch.save(checkpoint, output / "checkpoints" / "lightning" / "last.ckpt")

    state = load_recovery(
        output / "checkpoints" / "lightning" / "last.ckpt",
        checkpoint_dir=output / "checkpoints" / "lightning",
        root=output,
    )
    assert state.stopped_early is stopped


@pytest.mark.parametrize("reason", [7, "SOMETHING_ELSE", None, {}, ""])
def test_an_unknown_stopping_reason_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: object
) -> None:
    """Guessing would change how the resumed run finishes."""
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "stopping-unknown"
    _interrupt(output, data_yaml, monkeypatch)

    checkpoint = _committed(output)
    early = _callback_state(checkpoint, "EarlyStopping")
    early["stopping_reason"] = reason
    torch.save(checkpoint, output / "checkpoints" / "lightning" / "last.ckpt")

    with pytest.raises(PipelineError) as raised:
        load_recovery(
            output / "checkpoints" / "lightning" / "last.ckpt",
            checkpoint_dir=output / "checkpoints" / "lightning",
            root=output,
        )
    assert raised.value.code == "resume.stopping_unknown"


def test_a_missing_stopping_state_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "stopping-missing"
    _interrupt(output, data_yaml, monkeypatch)

    checkpoint = _committed(output)
    del _callback_state(checkpoint, "EarlyStopping")["stopping_reason"]
    torch.save(checkpoint, output / "checkpoints" / "lightning" / "last.ckpt")

    with pytest.raises(PipelineError) as raised:
        load_recovery(
            output / "checkpoints" / "lightning" / "last.ckpt",
            checkpoint_dir=output / "checkpoints" / "lightning",
            root=output,
        )
    assert raised.value.code == "resume.stopping_unrecorded"


def test_a_structurally_similar_external_checkpoint_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected_language,
) -> None:
    """Matching directory names is not provenance, even when the tree exists."""
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "executions" / "similar-name"
    _interrupt(output, data_yaml, monkeypatch)
    recorded = Path(_callback_state(_committed(output), "ModelCheckpoint")["best_model_path"])

    lookalike = tmp_path / "elsewhere" / "executions" / output.name / "checkpoints" / "lightning"
    lookalike.mkdir(parents=True)
    (lookalike / recorded.name).write_bytes(b"another execution's state")
    checkpoint = _committed(output)
    _callback_state(checkpoint, "ModelCheckpoint")["best_model_path"] = str(
        lookalike / recorded.name
    )
    torch.save(checkpoint, output / "checkpoints" / "lightning" / "last.ckpt")

    _install_fakes(monkeypatch, [])
    capsys.readouterr()
    selected_language("en")
    assert _command(*_overrides(output, data_yaml, epochs=2), "resume=true") == 1
    assert "outside this execution's checkpoint directory" in capsys.readouterr().err
    assert _FakeYOLO.validation_calls == 0
