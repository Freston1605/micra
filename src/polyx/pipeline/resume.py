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

#: Layout of the recovery block. Version 1 stored the identity and fit without
#: callback keys or the execution root; version 2 adds both, so recovery no
#: longer depends on Lightning's generated callback key spelling or on guessing
#: where a moved execution used to live. A checkpoint from a newer version is
#: refused rather than read with rules it did not follow.
RECOVERY_VERSION = 2

#: Epoch-named Lightning checkpoints: ``checkpoint_007.ckpt``.
_EPOCH_CHECKPOINT = re.compile(r"^checkpoint_(\d+)\.ckpt$")

#: Roles of the callbacks recovery reads, keyed by the name stored in the
#: recovery block beside the exact ``state_key`` Lightning used.
_CALLBACK_CLASSES = {"model_checkpoint": "ModelCheckpoint", "early_stopping": "EarlyStopping"}

#: Every stopping reason Lightning defines, by enum name. Value 0 means the
#: loop did not stop; the rest mean it did. A name or code outside this map is
#: not guessed at.
_STOPPING_REASONS = {
    "NOT_STOPPED": False,
    "STOPPING_THRESHOLD": True,
    "DIVERGENCE_THRESHOLD": True,
    "PATIENCE_EXHAUSTED": True,
    "NON_FINITE_METRIC": True,
}

#: The same reasons by the integer value Lightning serialises.
_STOPPING_CODES = {0: False, 1: True, 2: True, 3: True, 4: True}


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
    if not isinstance(metadata, dict):
        metadata = {}
    version = metadata.get("version", 1)
    if not isinstance(version, int) or not 1 <= version <= RECOVERY_VERSION:
        raise PipelineError(
            "resume.version_unknown",
            N_(
                "cannot resume %(path)s: its committed checkpoint uses recovery layout "
                "%(version)s, and this engine reads up to %(supported)s"
            ),
            path=str(path),
            version=str(version),
            supported=RECOVERY_VERSION,
        )
    identity = ExecutionIdentity.from_dict(metadata.get(IDENTITY_KEY))
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

    model = _callback_state(saved, "model_checkpoint", metadata, path)
    early = _callback_state(saved, "early_stopping", metadata, path)
    named = model.get("best_model_path")
    if not named:
        raise PipelineError(
            "resume.winner_unrecorded",
            N_("the committed state at %(path)s names no best checkpoint to publish"),
            path=str(path),
        )
    recorded_root = metadata.get("execution_root")
    best = _best_checkpoint(
        str(named),
        directory=directory,
        root=execution_root,
        recorded_root=Path(str(recorded_root)) if recorded_root else None,
        checkpoint=path,
    )
    fit = metadata.get("fit") if isinstance(metadata, dict) else None
    return RecoveryState(
        checkpoint=path,
        epochs_completed=int(saved["epoch"]) + 1,
        best_checkpoint=best,
        best_score=_number(model.get("best_model_score")),
        best_epoch=checkpoint_epoch(best),
        stopped_early=_stopping_state(early, path),
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


def _best_checkpoint(
    named: str,
    *,
    directory: Path,
    root: Path,
    recorded_root: Path | None,
    checkpoint: Path,
) -> Path:
    """Resolve a recorded winner inside this execution, or refuse.

    Lightning records an absolute path at save time. The reference is accepted
    only when the file it names belongs to this execution: either it is already
    inside this execution's root, or the checkpoint records the execution root
    it was written in and the same relative position resolves inside this one.
    That is how a moved execution keeps working without trusting a path that
    merely looks similar. A name that happens to exist here is never used to
    rescue a reference from somewhere else.
    """
    recorded = Path(named)
    if not recorded.is_absolute():
        recorded = root / recorded
    if not (_EPOCH_CHECKPOINT.match(recorded.name) or recorded.name == "last.ckpt"):
        raise PipelineError(
            "resume.checkpoint_name_invalid",
            N_(
                "the committed state at %(state)s names %(checkpoint)s, which is not a "
                "Lightning checkpoint file"
            ),
            state=str(checkpoint),
            checkpoint=str(recorded),
        )
    if _inside(recorded, root) and recorded.is_file():
        return recorded.resolve()
    if recorded_root is not None and _inside(recorded, recorded_root):
        candidate = root / recorded.relative_to(recorded_root)
        if _inside(candidate, root) and candidate.is_file():
            return candidate.resolve()
        raise PipelineError(
            "resume.checkpoint_missing",
            N_("the best checkpoint named by the committed state is missing: %(checkpoint)s"),
            checkpoint=str(candidate),
        )
    if recorded.parent == directory and (directory / recorded.name).is_file():
        return (directory / recorded.name).resolve()
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


def _inside(path: Path, root: Path) -> bool:
    """Whether ``path`` resolves inside ``root``."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _callback_state(
    saved: dict[str, Any],
    role: str,
    metadata: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    """One callback's state, resolved by the identity the engine recorded.

    The exact ``state_key`` Lightning used is stored beside the recovery facts,
    so a rename of the generated key does not move the state out of reach. A
    checkpoint without that record is read through the explicit forms Lightning
    is known to have written; several matches are ambiguous and refused.
    """
    callbacks = saved.get("callbacks")
    if not isinstance(callbacks, dict):
        raise PipelineError(
            "resume.callback_unrecorded",
            N_("the committed checkpoint at %(path)s holds no callback state"),
            path=str(path),
        )
    recorded = metadata.get("callbacks")
    if isinstance(recorded, dict):
        key = recorded.get(role)
        if isinstance(key, str) and isinstance(callbacks.get(key), dict):
            return callbacks[key]
    class_name = _CALLBACK_CLASSES[role]
    matches = [
        value
        for key, value in callbacks.items()
        if isinstance(value, dict) and _callback_key_matches(str(key), class_name)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise PipelineError(
            "resume.callback_unrecorded",
            N_("the committed checkpoint at %(path)s holds no %(callback)s state"),
            path=str(path),
            callback=class_name,
        )
    raise PipelineError(
        "resume.callback_ambiguous",
        N_(
            "the committed checkpoint at %(path)s holds more than one %(callback)s "
            "state, so recovery cannot choose which one to use"
        ),
        path=str(path),
        callback=class_name,
    )


def _callback_key_matches(key: str, class_name: str) -> bool:
    """Whether a serialised callback key belongs to ``class_name``."""
    if key == class_name:
        return True
    if key.startswith(f"{class_name}{{") or key.startswith(f"{class_name}["):
        return True
    return key.rsplit(".", 1)[-1].startswith(class_name)


def _number(value: Any) -> float | None:
    """A serialised scalar, or ``None`` when the checkpoint holds no score."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stopping_state(early: dict[str, Any], path: Path) -> bool:
    """Whether the serialised early-stopping state says the loop stopped.

    Lightning stores the reason as the enum's integer value, but a checkpoint
    can also carry the enum name, an enum-like object or a boolean. Every one
    of those is normalised against the reasons Lightning defines; a missing or
    unrecognised state is refused instead of being read as "stopped" or
    "running", either of which would change how the resumed run finishes.
    """
    if "stopping_reason" not in early:
        raise PipelineError(
            "resume.stopping_unrecorded",
            N_("the committed checkpoint at %(path)s records no early-stopping state"),
            path=str(path),
        )
    return _stopping_value(early["stopping_reason"], path)


def _stopping_value(value: Any, path: Path) -> bool:
    """Normalise one stopping-reason spelling, refusing what is not known."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in _STOPPING_CODES:
            return _STOPPING_CODES[value]
        raise _unknown_stopping(value, path)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return _stopping_value(int(text), path)
        name = text.rsplit(".", 1)[-1].strip().upper()
        if name in _STOPPING_REASONS:
            return _STOPPING_REASONS[name]
        raise _unknown_stopping(value, path)
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return _stopping_value(name, path)
    number = getattr(value, "value", None)
    if isinstance(number, int) and not isinstance(number, bool):
        return _stopping_value(number, path)
    raise _unknown_stopping(value, path)


def _unknown_stopping(value: Any, path: Path) -> PipelineError:
    """The refusal for a stopping reason this engine does not recognise."""
    return PipelineError(
        "resume.stopping_unknown",
        N_(
            "the committed checkpoint at %(path)s records an early-stopping state "
            "%(value)s that this engine does not recognise"
        ),
        path=str(path),
        value=repr(value),
    )
