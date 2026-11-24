"""The one Lightning loop: callbacks, checkpoints and the shared score.

Lightning owns everything a training loop does: the epoch loop, the optimizer
and scheduler steps, precision, checkpoint writing, early stopping and
resumption from a checkpoint. This module adds only what the benchmark requires
on top of that lifecycle:

* the shared evaluator's with-class score is logged as ``with_class_score``, so
  Lightning's own ``EarlyStopping`` and ``ModelCheckpoint`` select and keep the
  best epoch;
* every validation epoch writes the small progress record and the
  family-native snapshot the artifact layout promises;
* the kept checkpoint is re-validated before publication, so ``metrics.json``
  and ``best.pt`` always describe the same weights.

Nothing here re-implements a training loop. The two thresholds the benchmark
needs come from the framework as well: ``ModelCheckpoint`` keeps any strictly
better score, while ``EarlyStopping`` measures patience against ``min_delta``,
so noise smaller than the delta neither replaces ``best.pt`` nor keeps a
stagnant run alive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polyx.pipeline.artifacts import ArtifactLayout, append_log, write_json
from polyx.pipeline.config import ExecutionConfig, Protocol, protocol_to_dict
from polyx.pipeline.evaluation import SharedEvaluator
from polyx.pipeline.identity import IDENTITY_KEY, ExecutionIdentity
from polyx.pipeline.resume import (
    RECOVERY_KEY,
    RecoveryState,
    checkpoint_epoch,
    committed_checkpoint,
)
from polyx.pipeline.weights import ResolvedWeights

#: The one metric every family logs and both callbacks monitor.
SCORE_KEY = "with_class_score"

#: Precision strings that mean some form of automatic mixed precision.
_MIXED_PRECISION = frozenset({"16-mixed", "bf16-mixed", "16-true", "bf16-true"})


def amp_active(precision: str) -> bool:
    """Whether the precision Lightning reports is a mixed-precision mode."""
    return precision in _MIXED_PRECISION


def build_optimizer(net: Any, protocol: Protocol) -> Any:
    """Protocol optimizer over a network's parameter groups.

    The weight-decay split (weights decay, biases and normalization do not) is
    part of the benchmark, so every family gets it from this one function.
    """
    import torch

    groups: dict[str, list[Any]] = {"bias": [], "norm": [], "weight": []}
    norms = tuple(value for name, value in torch.nn.__dict__.items() if "Norm" in name)
    for module_name, module in net.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:
                groups["bias"].append(param)
            elif isinstance(module, norms):
                groups["norm"].append(param)
            else:
                groups["weight"].append(param)

    name = protocol.optimizer.lower()
    parameter_groups = [
        {"params": groups["weight"], "weight_decay": protocol.weight_decay},
        {"params": groups["norm"], "weight_decay": 0.0},
        {"params": groups["bias"], "weight_decay": 0.0},
    ]
    non_empty = [group for group in parameter_groups if group["params"]]
    if not non_empty:
        raise ValueError("the network has no trainable parameters")
    first = non_empty[0]
    if name == "sgd":
        optimizer: Any = torch.optim.SGD(
            first["params"],
            lr=protocol.learning_rate,
            momentum=protocol.momentum,
            nesterov=True,
            weight_decay=float(first["weight_decay"]),
        )
    elif name in {"adam", "adamw"}:
        optimizer_class = torch.optim.Adam if name == "adam" else torch.optim.AdamW
        optimizer = optimizer_class(
            first["params"],
            lr=protocol.learning_rate,
            weight_decay=float(first["weight_decay"]),
        )
    else:
        raise ValueError(f"unsupported optimizer in the protocol: {protocol.optimizer}")
    for group in non_empty[1:]:
        optimizer.add_param_group(group)
    return optimizer


def lr_factor(protocol: Protocol) -> Any:
    """Warmup plus the protocol's schedule, as a per-epoch multiplier."""
    import math

    warmup = max(0, protocol.warmup_epochs)
    total = max(1, protocol.epochs)

    def factor(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(1.0, max(0.0, (epoch - warmup) / max(1, total - warmup)))
        if protocol.schedule == "cosine":
            # Decay to one percent, the same floor Ultralytics' cosine uses.
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
        if protocol.schedule == "constant":
            return 1.0
        raise ValueError(f"unsupported learning-rate schedule: {protocol.schedule}")

    return factor


def atomic_checkpoint_io() -> Any:
    """A CheckpointIO whose visible file is always complete.

    Lightning's default writer truncates the destination before writing it. A
    SLURM time limit landing mid-write would leave ``last.ckpt`` as a torn file
    that still looks resumable. Writing beside the destination and renaming
    makes every checkpoint a complete file at the moment its name appears.
    """
    from lightning.pytorch.plugins.io import TorchCheckpointIO

    class AtomicCheckpointIO(TorchCheckpointIO):
        def save_checkpoint(
            self,
            checkpoint: dict[str, Any],
            path: str | Path,
            storage_options: dict[str, Any] | None = None,
        ) -> None:
            destination = Path(path)
            temporary = destination.with_name(f".{destination.name}.tmp")
            super().save_checkpoint(checkpoint, temporary, storage_options)
            os.replace(temporary, destination)

        def remove_checkpoint(
            self, path: str | Path, storage_options: dict[str, Any] | None = None
        ) -> None:
            Path(path).unlink(missing_ok=True)

    return AtomicCheckpointIO()


@dataclass
class ExecutionRecorder:
    """The scientific record Lightning writes around its own checkpoints."""

    layout: ArtifactLayout
    config: ExecutionConfig
    dataset_fingerprint: str
    identity: ExecutionIdentity
    evaluator: SharedEvaluator
    #: The fit state resolved once the trainer exists (precision, device,
    #: verified weights, augmentation); every progress record carries it.
    fit: dict[str, Any] = field(default_factory=dict)

    @property
    def protocol(self) -> Protocol:
        return self.config.protocol

    def record_epoch(
        self,
        *,
        trainer: Any,
        native_checkpoint: Path | None,
        score: float,
    ) -> None:
        """Write the progress heartbeat after Lightning saved this epoch.

        Every outcome-bearing value is read from the callbacks, which Lightning
        has just restored from or written into the checkpoint. The heartbeat
        cannot carry an older winner into a later resume: a resumed process
        starts from the checkpoint, and this record only reports what it saw.
        """
        from lightning.pytorch.callbacks import EarlyStopping

        checkpoint = trainer.checkpoint_callback
        early = _find_callback(trainer, EarlyStopping)
        best_score = getattr(checkpoint, "best_model_score", None)
        best_path = getattr(checkpoint, "best_model_path", "") or None
        best_epoch = checkpoint_epoch(Path(best_path)) if best_path else -1
        progress = {
            "epochs_completed": int(trainer.current_epoch) + 1,
            "epoch_completed": int(trainer.current_epoch),
            "last_checkpoint": str(native_checkpoint) if native_checkpoint else None,
            "resume_checkpoint": (
                str(checkpoint.last_model_path) if checkpoint.last_model_path else None
            ),
            "best_lightning_checkpoint": str(best_path) if best_path else None,
            "best_score": float(best_score) if best_score is not None else None,
            "best_epoch": best_epoch,
            "protocol": protocol_to_dict(self.config),
            "dataset_fingerprint": self.dataset_fingerprint,
            "fit": self.fit,
            "early_stopping": stopping_record(
                self.config,
                best_epoch=best_epoch,
                best_score=float(best_score) if best_score is not None else None,
                stopped_early=stopped_early(early),
                wait_count=int(getattr(early, "wait_count", 0)),
                epochs_completed=int(trainer.current_epoch) + 1,
            ),
        }
        write_json(progress, self.layout.progress)
        append_log(
            self.layout.training_log,
            f"epoch={int(trainer.current_epoch)} with_class_score={score:.6f} "
            f"best={progress['best_score'] if progress['best_score'] is not None else 'none'}",
        )


def stopped_early(early: Any) -> bool:
    """Whether a Lightning ``EarlyStopping`` callback decided to stop.

    The callback records a reason; ``stopped_epoch`` defaults to zero, so the
    reason is the only honest signal. Used for the progress record and for the
    published training state, so both name the same event.
    """
    reason = getattr(early, "stopping_reason", None)
    return bool(reason) and getattr(reason, "name", "NOT_STOPPED") != "NOT_STOPPED"


def stopping_record(
    config: ExecutionConfig,
    *,
    best_epoch: int,
    best_score: float | None,
    stopped_early: bool,
    wait_count: int,
    epochs_completed: int,
) -> dict[str, Any]:
    """The one early-stopping block written to every artifact.

    Progress and training state carry the same keys, so a reviewer never has
    to know which artifact is authoritative for which field.
    """
    stopping = config.protocol.early_stopping
    return {
        "monitor": SCORE_KEY,
        "mode": "max",
        "patience": stopping.patience,
        "min_delta": stopping.min_delta,
        "restore_best": stopping.restore_best,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "stopped_early": stopped_early,
        "wait_count": wait_count,
        "epochs_completed": epochs_completed,
    }


def _find_callback(trainer: Any, kind: type) -> Any:
    for callback in getattr(trainer, "callbacks", []):
        if isinstance(callback, kind):
            return callback
    raise RuntimeError(f"the Lightning trainer is missing its {kind.__name__} callback")


class PolyxModule:
    """The LightningModule behaviour every family shares.

    An adapter subclasses this together with ``lightning.LightningModule`` and
    supplies the model, the two steps and the native export. The optimizer, the
    schedule, the shared score and the artifact record live here, so a family
    cannot silently change them.
    """

    def __init__(
        self, *, protocol: Protocol, evaluator: SharedEvaluator, recorder: ExecutionRecorder
    ) -> None:
        super().__init__()
        self.protocol = protocol
        self.evaluator = evaluator
        self.recorder = recorder
        self.coco_predictions: list[dict] = []
        self.last_metrics: dict[str, Any] | None = None
        self.native_checkpoint_path: Path | None = None

    def configure_optimizers(self) -> Any:
        from torch.optim.lr_scheduler import LambdaLR

        optimizer = build_optimizer(self.net, self.protocol)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_factor(self.protocol))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_validation_start(self) -> None:
        """Start one validation run with an empty prediction set."""
        self.coco_predictions = []

    def on_validation_epoch_end(self) -> None:
        """Score the epoch with the shared evaluator and log the one metric.

        Callbacks' ``on_validation_epoch_end`` runs before this hook, but both
        callbacks act later, at ``on_validation_end``, after Lightning has
        folded the logged value into ``trainer.callback_metrics``. That is why
        the score is logged here and read there.
        """
        self.last_metrics = self.evaluator.evaluate(self.coco_predictions)
        self.log(SCORE_KEY, float(self.last_metrics["selection"]["value"]))

    def on_validation_end(self) -> None:
        """Record the epoch once Lightning has written its checkpoint.

        Callback hooks run before the module's, so by the time this runs the
        checkpoint callback has saved ``last.ckpt`` and, when this epoch is the
        best so far, its own top-k file.
        """
        if not self.fitting():
            return
        metrics = self.last_metrics
        score = float(metrics["selection"]["value"]) if metrics else 0.0
        self.recorder.record_epoch(
            trainer=self.trainer,
            native_checkpoint=self.native_checkpoint_path,
            score=score,
        )

    def fitting(self) -> bool:
        """Whether this hook is running inside ``Trainer.fit``.

        Publication reuses the same module through ``Trainer.validate``; that
        run must recompute the kept epoch's metrics without rewriting the
        progress record or exporting a new epoch snapshot.
        """
        from lightning.pytorch.trainer.states import TrainerFn

        return self.trainer.state.fn is TrainerFn.FITTING

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Carry the recovery facts Lightning does not serialise.

        Lightning restores the loop, optimizer, scheduler and both callbacks
        from the checkpoint it writes. The execution identity and the resolved
        fit are the engine's, so they travel in the same file. One identity
        record covers the family, initialization weights, protocol and dataset
        fingerprint, and a checkpoint without it is refused as legacy.
        """
        checkpoint[RECOVERY_KEY] = {
            IDENTITY_KEY: self.recorder.identity.to_dict(),
            "fit": self.recorder.fit,
        }

    def export_native(self, path: Path, epoch: int | None = None) -> None:
        """Write this family's own checkpoint format. Adapters implement it."""
        raise NotImplementedError


@dataclass
class FamilyRun:
    """What an adapter hands to the one loop: a module and its data."""

    module: PolyxModule
    datamodule: Any
    weights: ResolvedWeights
    augmentation: dict[str, Any]
    device: str


@dataclass
class TrainingOutcome:
    """Observable state of the finished Lightning fit."""

    precision: str
    amp_requested: str
    epochs_completed: int
    amp_active: bool
    device: str
    weights: str
    weights_verified: bool
    augmentation: dict[str, Any]
    best_score: float | None
    best_checkpoint: Path
    best_epoch: int
    stopped_early: bool
    wait_count: int


def build_trainer(config: ExecutionConfig, layout: ArtifactLayout) -> tuple[Any, Any, str]:
    """The one Trainer, its checkpoint callback and the requested precision.

    Lightning may substitute a precision the device supports; the caller
    records ``trainer.precision``, not this request.
    """
    from lightning import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    protocol = config.protocol
    stopping = protocol.early_stopping
    precision = "16-mixed" if protocol.amp.enabled and protocol.amp.dtype == "fp16" else "32-true"
    device = config.backend.device
    accelerator = "cpu" if device in ("cpu", "mps") else "gpu"
    checkpoints = layout.lightning_checkpoints
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoints),
        filename="checkpoint_{epoch:03d}",
        auto_insert_metric_name=False,
        enable_version_counter=False,
        monitor=SCORE_KEY,
        mode="max",
        # Every epoch is kept, as the artifact layout documents, and the best
        # of them is what publication exports.
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    )
    early = EarlyStopping(
        monitor=SCORE_KEY,
        mode="max",
        patience=stopping.patience,
        min_delta=stopping.min_delta,
        verbose=False,
        # The protocol has one stopping rule: patience over the shared score.
        # A non-finite score is reported in the metrics and counted as a
        # non-improvement, not given a second, undocumented stop reason.
        check_finite=False,
        # The score exists only after validation. Checking at
        # ``on_validation_end`` also keeps a resumed run safe: the epoch
        # restored from a checkpoint saved at validation end is not validated
        # again, so a train-epoch-end check would look for a metric this run
        # never logged.
        check_on_train_epoch_end=False,
    )
    trainer = Trainer(
        max_epochs=protocol.epochs,
        precision=precision,
        accelerator=accelerator,
        devices=1,
        callbacks=[early, checkpoint],
        plugins=[atomic_checkpoint_io()],
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=True,
        default_root_dir=str(layout.root),
    )
    return trainer, checkpoint, precision


def run_fit(
    config: ExecutionConfig,
    layout: ArtifactLayout,
    recorder: ExecutionRecorder,
    run: FamilyRun,
    *,
    resume_checkpoint: Path | None = None,
) -> tuple[TrainingOutcome, Any]:
    """Fit one family with Lightning's callbacks doing the selection."""
    trainer, _checkpoint, requested_precision = build_trainer(config, layout)
    recorder.fit = _fit_metadata(
        config=config,
        trainer=trainer,
        device=run.device,
        requested_precision=requested_precision,
        weights=run.weights,
        augmentation=run.augmentation,
    )
    fit_kwargs: dict[str, Any] = {"weights_only": True}
    if resume_checkpoint is not None:
        fit_kwargs["ckpt_path"] = str(resume_checkpoint)
    trainer.fit(run.module, datamodule=run.datamodule, **fit_kwargs)
    outcome = _outcome(
        layout=layout,
        trainer=trainer,
        run=run,
        requested_precision=requested_precision,
    )
    return outcome, trainer


def recovered_outcome(recovery: RecoveryState) -> TrainingOutcome:
    """The state of an execution whose last epoch Lightning already committed.

    The epoch loop is on disk; only the artifacts are missing. Everything that
    decides the publication comes from the committed checkpoint: how many
    epochs completed, the kept checkpoint and its score, and the early-stopping
    decision. The fit facts travel beside them in the same file.
    """
    fit = recovery.fit or {}
    return TrainingOutcome(
        precision=str(fit.get("precision", "")),
        amp_requested=str(fit.get("amp_requested", "")),
        epochs_completed=recovery.epochs_completed,
        amp_active=amp_active(str(fit.get("precision", ""))),
        device=str(fit.get("device", "")),
        weights=str(fit.get("weights", "")),
        weights_verified=bool(fit.get("weights_verified", False)),
        augmentation=fit.get("augmentation") or {},
        best_score=recovery.best_score,
        best_checkpoint=recovery.best_checkpoint,
        best_epoch=recovery.best_epoch,
        stopped_early=recovery.stopped_early,
        wait_count=recovery.wait_count,
    )


def evaluate_best(
    layout: ArtifactLayout,
    run: FamilyRun,
    trainer: Any,
    *,
    best: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Re-score the kept checkpoint and export it as this family's weights.

    The metrics and ``best.pt`` then describe one and the same file, which is
    the property a resumed run could not guarantee when it carried an earlier
    epoch's figures forward.
    """
    if best is None:
        checkpoint = trainer.checkpoint_callback
        best = Path(str(checkpoint.best_model_path or ""))
    if not best.is_file():
        raise RuntimeError("the Lightning loop kept no best checkpoint to publish")
    trainer.validate(
        run.module,
        datamodule=run.datamodule,
        ckpt_path=str(best),
        verbose=False,
        weights_only=True,
    )
    metrics = run.module.last_metrics
    if metrics is None:
        raise RuntimeError("the kept checkpoint produced no evaluation metrics")
    run.module.export_native(layout.best_checkpoint)
    layout.publication_snapshot.unlink(missing_ok=True)
    selected_epoch = checkpoint_epoch(best)
    metrics = {**metrics, "selected_epoch": selected_epoch}
    return metrics, best


def _fit_metadata(
    *,
    config: ExecutionConfig,
    trainer: Any,
    device: str,
    requested_precision: str,
    weights: Any,
    augmentation: dict[str, Any],
) -> dict[str, Any]:
    """The fit state every epoch carries into the progress snapshot."""
    reported = str(trainer.precision)
    return {
        "precision": reported,
        "amp_requested": requested_precision,
        "amp_active": amp_active(reported),
        "device": device,
        "framework": config.family.framework,
        "weights": weights.describe(),
        "weights_verified": weights.verified,
        "augmentation": augmentation,
    }


def _outcome(
    *,
    layout: ArtifactLayout,
    trainer: Any,
    run: FamilyRun,
    requested_precision: str,
) -> TrainingOutcome:
    """The finished fit, as the checkpoint Lightning just committed describes it.

    The callbacks would report the same numbers in memory, but reading the
    committed file keeps one authority for a fresh fit, a resumed fit and a
    recovered publication.
    """
    recovery = committed_checkpoint(layout.root)
    if recovery is None:
        raise RuntimeError("the Lightning loop finished without committing a checkpoint")
    return TrainingOutcome(
        precision=str(trainer.precision),
        amp_requested=requested_precision,
        epochs_completed=recovery.epochs_completed,
        amp_active=amp_active(str(trainer.precision)),
        device=run.device,
        weights=run.weights.describe(),
        weights_verified=run.weights.verified,
        augmentation=run.augmentation,
        best_score=recovery.best_score,
        best_checkpoint=recovery.best_checkpoint,
        best_epoch=recovery.best_epoch,
        stopped_early=recovery.stopped_early,
        wait_count=recovery.wait_count,
    )
