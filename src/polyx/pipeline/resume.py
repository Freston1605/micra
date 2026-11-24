"""What an interrupted execution can recover, and from where.

Lightning writes the training state: weights, optimizer, scheduler, epoch
progress and the state of the early-stopping and checkpoint callbacks. This
module reads that committed checkpoint as the one authority for recovery:
completed epochs, the kept checkpoint and its score, and whether the loop
stopped early all come from it.

The checkpoint is also bound to one execution. It carries the engine's
identity record — family, initialization weights, protocol and dataset
fingerprint — and every read here is a restricted ``weights_only`` load. A file
that cannot be verified, or a checkpoint reference outside the execution's own
checkpoint directory, is refused rather than read with a wider mode.

The progress record (``progress.json``) is a heartbeat and provenance aid. It
says how far the last *reported* epoch got, so a human or a lease can tell a
live job from a dead one, but a stale value in it never decides an epoch count,
a winner, a stop or whether a checkpoint belongs to this execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyx.core.i18n import N_
from polyx.pipeline.artifacts import ArtifactLayout
from polyx.pipeline.errors import PipelineError
from polyx.pipeline.identity import IDENTITY_KEY, ExecutionIdentity

#: Top-level key of the engine's recovery facts inside a Lightning checkpoint.
RECOVERY_KEY = "polyx_recovery"

#: Epoch-named Lightning checkpoints: ``checkpoint_007.ckpt``.
_EPOCH_CHECKPOINT = re.compile(r"^checkpoint_(\d+)\.ckpt$")


def checkpoint_epoch(path: Path) -> int:
    """The zero-based epoch a Lightning checkpoint holds.

    The read is restricted: a checkpoint the pipeline owns is data plus
    tensors, and a file that needs a wider unpickler is not one of ours.
    """
    import torch

    try:
        loaded = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PipelineError(
            "resume.checkpoint_unreadable",
            N_("cannot read the Lightning checkpoint at %(path)s: %(reason)s"),
            path=str(path),
            reason=str(exc),
        ) from exc
    return int(loaded["epoch"])


def checkpoint_name_epoch(path: Path) -> int | None:
    """The epoch a Lightning checkpoint file name carries, when it carries one.

    ``ModelCheckpoint`` writes ``checkpoint_NNN.ckpt`` with ``NNN`` the
    zero-based epoch, so the heartbeat can report the kept epoch without
    deserialising the whole checkpoint on every validation pass.
    """
    match = _EPOCH_CHECKPOINT.match(Path(path).name)
    return int(match.group(1)) if match else None


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
    #: The execution this checkpoint belongs to.
    identity: ExecutionIdentity
    #: The fit state resolved by the interrupted run.
    fit: dict[str, Any] | None = None


def committed_checkpoint(output_dir: Path) -> RecoveryState | None:
    """Read the newest committed Lightning checkpoint, or ``None`` if none exists."""
    layout = ArtifactLayout(Path(output_dir))
    path = _committed_checkpoint_path(layout)
    if path is None:
        return None
    return load_recovery(path, checkpoint_dir=layout.lightning_checkpoints, root=layout.root)


def has_committed_checkpoint(output_dir: Path) -> bool:
    """Whether an execution directory holds a Lightning checkpoint to recover from."""
    return _committed_checkpoint_path(ArtifactLayout(Path(output_dir))) is not None


def load_recovery(
    path: Path,
    *,
    checkpoint_dir: Path | None = None,
    root: Path | None = None,
) -> RecoveryState:
    """Read the recovery authority out of one committed Lightning checkpoint.

    ``checkpoint_dir`` is the execution's own Lightning checkpoint directory.
    Lightning stores ``best_model_path`` as an absolute path, so a reference
    that names another directory — or another execution's same-named file — is
    refused, never followed.
    """
    import torch

    path = Path(path)
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else path.parent
    execution_root = Path(root) if root is not None else directory.parent.parent
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PipelineError(
            "resume.checkpoint_restricted",
            N_(
                "cannot read the committed Lightning checkpoint at %(path)s with "
                "restricted loading: %(reason)s"
            ),
            path=str(path),
            reason=str(exc),
        ) from exc
    if not isinstance(saved, dict) or "epoch" not in saved:
        raise PipelineError(
            "resume.checkpoint_not_training",
            N_("%(path)s is not a Lightning training checkpoint"),
            path=str(path),
        )

    metadata = saved.get(RECOVERY_KEY)
    identity = (
        ExecutionIdentity.from_dict(metadata.get(IDENTITY_KEY))
        if isinstance(metadata, dict)
        else None
    )
    if identity is None:
        raise PipelineError(
            "execution.identity_unrecorded",
            N_(
                "cannot resume %(path)s: the committed checkpoint at %(checkpoint)s carries "
                "no verifiable execution identity; it is left untouched, so start a new "
                "output directory instead of resuming"
            ),
            path=str(execution_root),
            checkpoint=str(path),
        )

    model = _callback_state(saved, "ModelCheckpoint", path)
    early = _callback_state(saved, "EarlyStopping", path)
    named = model.get("best_model_path")
    if not named:
        raise PipelineError(
            "resume.winner_unrecorded",
            N_("the committed state at %(path)s names no best checkpoint to publish"),
            path=str(path),
        )
    best = _local_checkpoint(str(named), directory, path)
    fit = metadata.get("fit") if isinstance(metadata, dict) else None
    return RecoveryState(
        checkpoint=path,
        epochs_completed=int(saved["epoch"]) + 1,
        best_checkpoint=best,
        best_score=_number(model.get("best_model_score")),
        best_epoch=checkpoint_epoch(best),
        stopped_early=_reason_stopped(early.get("stopping_reason")),
        wait_count=int(early.get("wait_count", 0) or 0),
        identity=identity,
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


def _local_checkpoint(named: str, directory: Path, checkpoint: Path) -> Path:
    """Resolve a recorded checkpoint reference inside the execution's directory.

    Lightning records an absolute path at save time. The reference must name a
    file in this execution's own ``checkpoints/lightning`` directory: a path
    anywhere else is refused, even when a file of the same name happens to
    exist here, because the record would then be describing another
    execution's state. An execution moved to another node keeps that same
    position in the tree and its old directory is gone, so exactly that shape
    is accepted and resolved to the local file.
    """
    recorded = Path(named)
    name = recorded.name
    if not name or not (_EPOCH_CHECKPOINT.match(name) or name == "last.ckpt"):
        raise PipelineError(
            "resume.checkpoint_name_invalid",
            N_(
                "the committed state at %(state)s names %(checkpoint)s, which is not a "
                "Lightning checkpoint file"
            ),
            state=str(checkpoint),
            checkpoint=str(recorded),
        )
    if recorded.parent != directory and not _relocated(recorded.parent, directory):
        raise PipelineError(
            "resume.checkpoint_outside",
            N_(
                "the checkpoint %(checkpoint)s named by %(state)s is outside this "
                "execution's checkpoint directory %(directory)s"
            ),
            checkpoint=str(recorded),
            state=str(checkpoint),
            directory=str(directory),
        )
    candidate = directory / name
    if not candidate.is_file():
        raise PipelineError(
            "resume.checkpoint_missing",
            N_("the best checkpoint named by the committed state is missing: %(checkpoint)s"),
            checkpoint=str(candidate),
        )
    return candidate


def _relocated(recorded_parent: Path, directory: Path) -> bool:
    """Whether a recorded directory is this one before the execution moved.

    Lightning writes the absolute path of the moment it saved. A moved tree
    therefore carries the old prefix, which is this directory's position in
    another checkout — same name, same parent name — and which no longer
    exists. A path that still exists is a live directory that is not this
    execution's checkpoint directory, so the record describes another tree.
    """
    return (
        recorded_parent.name == directory.name
        and recorded_parent.parent.name == directory.parent.name
        and not recorded_parent.exists()
    )


def _callback_state(saved: dict[str, Any], prefix: str, path: Path) -> dict[str, Any]:
    """One callback's state as Lightning serialised it inside the checkpoint."""
    callbacks = saved.get("callbacks")
    if isinstance(callbacks, dict):
        for key, value in callbacks.items():
            if str(key).startswith(prefix) and isinstance(value, dict):
                return value
    raise PipelineError(
        "resume.callback_unrecorded",
        N_("the committed checkpoint at %(path)s holds no %(callback)s state"),
        path=str(path),
        callback=prefix,
    )


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
