"""What an interrupted execution can recover, and from where.

Lightning writes the training state: weights, optimizer, scheduler, epoch
progress and the state of the early-stopping and checkpoint callbacks. This
module reads that committed checkpoint as the one authority for recovery:
completed epochs, the kept checkpoint and its score, and whether the loop
stopped early all come from it.

The progress record (``progress.json``) is a heartbeat and provenance aid. It
says how far the last *reported* epoch got, so a human or a lease can tell a
live job from a dead one, but a stale value in it never decides an epoch count,
a winner or a stop. When it carries the protocol, dataset identity and resolved
fit of a run written before the checkpoint stored them, those descriptive facts
are still read from it; the checkpoint wins whenever both are present.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyx.pipeline.artifacts import ArtifactLayout

#: Top-level key of the engine's recovery facts inside a Lightning checkpoint.
RECOVERY_KEY = "polyx_recovery"

#: Epoch-named Lightning checkpoints: ``checkpoint_007.ckpt``.
_EPOCH_CHECKPOINT = re.compile(r"^checkpoint_(\d+)\.ckpt$")


def checkpoint_epoch(path: Path) -> int:
    """The zero-based epoch a Lightning checkpoint holds."""
    import torch

    try:
        loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"cannot read the Lightning checkpoint at {path}: {exc}") from exc
    return int(loaded["epoch"])


@dataclass(frozen=True)
class RecoveryState:
    """The training state the committed Lightning checkpoint carries.

    ``best_checkpoint`` is required: a committed state that names no winner,
    or names a file that is gone, is an inconsistent execution and publication
    refuses it instead of substituting another epoch.
    """

    #: The checkpoint a resumed fit restores (``last.ckpt`` when present).
    checkpoint: Path
    #: Completed epochs, from the checkpoint Lightning committed.
    epochs_completed: int
    #: The winner ``ModelCheckpoint`` kept and the epoch it holds.
    best_checkpoint: Path
    best_score: float | None
    best_epoch: int
    #: The ``EarlyStopping`` decision and patience count stored beside it.
    stopped_early: bool
    wait_count: int
    #: Descriptive provenance stored by this engine; absent in checkpoints
    #: written before the engine embedded it.
    protocol: dict[str, Any] | None = None
    dataset_fingerprint: str | None = None
    fit: dict[str, Any] | None = None


def committed_checkpoint(output_dir: Path) -> RecoveryState | None:
    """Read the newest committed Lightning checkpoint, or ``None`` if none exists."""
    path = _committed_checkpoint_path(ArtifactLayout(Path(output_dir)))
    return None if path is None else load_recovery(path)


def has_committed_checkpoint(output_dir: Path) -> bool:
    """Whether an execution directory holds a Lightning checkpoint to recover from."""
    return _committed_checkpoint_path(ArtifactLayout(Path(output_dir))) is not None


def load_recovery(path: Path) -> RecoveryState:
    """Read the recovery authority out of one committed Lightning checkpoint."""
    import torch

    path = Path(path)
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"cannot read the committed Lightning checkpoint at {path}: {exc}"
        ) from exc
    if not isinstance(saved, dict) or "epoch" not in saved:
        raise RuntimeError(f"{path} is not a Lightning training checkpoint")

    model = _callback_state(saved, "ModelCheckpoint", path)
    early = _callback_state(saved, "EarlyStopping", path)
    named = model.get("best_model_path")
    if not named:
        raise RuntimeError(f"the committed state at {path} names no best checkpoint to publish")
    best = Path(str(named))
    if not best.is_file():
        raise RuntimeError(f"the best checkpoint named by the committed state is missing: {best}")

    metadata = saved.get(RECOVERY_KEY)
    if not isinstance(metadata, dict):
        metadata = {}
    protocol = metadata.get("protocol")
    fingerprint = metadata.get("dataset_fingerprint")
    fit = metadata.get("fit")
    return RecoveryState(
        checkpoint=path,
        epochs_completed=int(saved["epoch"]) + 1,
        best_checkpoint=best,
        best_score=_number(model.get("best_model_score")),
        best_epoch=checkpoint_epoch(best),
        stopped_early=_reason_stopped(early.get("stopping_reason")),
        wait_count=int(early.get("wait_count", 0) or 0),
        protocol=protocol if isinstance(protocol, dict) else None,
        dataset_fingerprint=fingerprint if isinstance(fingerprint, str) else None,
        fit=fit if isinstance(fit, dict) else None,
    )


def _committed_checkpoint_path(layout: ArtifactLayout) -> Path | None:
    """The committed resume point: ``last.ckpt``, else the newest epoch file."""
    directory = layout.lightning_checkpoints
    last = directory / "last.ckpt"
    if last.is_file():
        return last
    if not directory.is_dir():
        return None
    epochs: list[tuple[int, Path]] = []
    for candidate in directory.glob("checkpoint_*.ckpt"):
        match = _EPOCH_CHECKPOINT.match(candidate.name)
        if match:
            epochs.append((int(match.group(1)), candidate))
    if not epochs:
        return None
    return max(epochs)[1]


def _callback_state(saved: dict[str, Any], prefix: str, path: Path) -> dict[str, Any]:
    """One callback's state as Lightning serialised it inside the checkpoint."""
    callbacks = saved.get("callbacks")
    if isinstance(callbacks, dict):
        for key, value in callbacks.items():
            if str(key).startswith(prefix) and isinstance(value, dict):
                return value
    raise RuntimeError(f"the committed checkpoint at {path} holds no {prefix} state")


def _number(value: Any) -> float | None:
    """A serialised scalar, or ``None`` when the checkpoint holds no score."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reason_stopped(reason: Any) -> bool:
    """Whether a serialised ``EarlyStoppingReason`` means the loop stopped.

    Lightning stores the enum's value (``0`` is ``NOT_STOPPED``); the name and
    boolean spellings are accepted so a hand-inspected checkpoint reads the
    same way.
    """
    if isinstance(reason, bool):
        return reason
    if isinstance(reason, str):
        return reason not in {"", "0", "NOT_STOPPED", "EarlyStoppingReason.NOT_STOPPED"}
    try:
        return int(reason) != 0
    except (TypeError, ValueError):
        return bool(reason)


@dataclass(frozen=True)
class ResumeState:
    """Descriptive state read from a progress heartbeat.

    Nothing here decides a training outcome. It names the protocol, dataset
    identity and resolved fit of a run whose checkpoint predates the embedded
    metadata, so an older execution keeps its guards. ``epochs_completed`` and
    ``best_*`` are historical evidence only.
    """

    epochs_completed: int
    resume_checkpoint: Path | None
    best_score: float | None
    stopped_early: bool
    protocol: dict[str, Any] | None
    #: Digest of every dataset file the interrupted execution trained on.
    dataset_fingerprint: str
    #: The fit resolved by the interrupted execution (precision, device,
    #: verified weights, augmentation).
    fit: dict[str, Any] | None = None
    #: Non-improving validation epochs the early-stopping callback counted.
    wait_count: int = 0
    #: The epoch that set the kept checkpoint, and that checkpoint's path.
    best_epoch: int = -1
    best_lightning_checkpoint: Path | None = None

    @classmethod
    def load(cls, path: Path) -> ResumeState:
        """Read a progress heartbeat written by a previous run."""
        path = Path(path)
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unreadable progress snapshot {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError(f"invalid progress snapshot {path}: expected a mapping")
        try:
            resume_checkpoint = raw.get("resume_checkpoint")
            fit = raw.get("fit")
            if fit is not None and not isinstance(fit, dict):
                raise TypeError("fit must be a mapping")
            protocol = raw.get("protocol")
            if protocol is not None and not isinstance(protocol, dict):
                raise TypeError("protocol must be a mapping")
            fingerprint = raw.get("dataset_fingerprint")
            if fingerprint is not None and not isinstance(fingerprint, str):
                raise TypeError("dataset_fingerprint must be a string")
            return cls(
                epochs_completed=int(raw.get("epochs_completed", 0)),
                resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None,
                best_score=raw.get("best_score"),
                stopped_early=bool(raw.get("stopped_early", False)),
                protocol=dict(protocol) if protocol is not None else None,
                dataset_fingerprint=fingerprint or "",
                fit=fit,
                wait_count=int((raw.get("early_stopping") or {}).get("wait_count", 0)),
                best_epoch=int(raw.get("best_epoch", -1)),
                best_lightning_checkpoint=(
                    Path(raw["best_lightning_checkpoint"])
                    if raw.get("best_lightning_checkpoint")
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid progress snapshot {path}: {exc}") from exc
