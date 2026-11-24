"""One config-driven training execution on the local backend.

The engine owns the parts that surround training: the dataset audit and its
fingerprint, provenance, the shared evaluator, the artifact layout and the
publication record. The training loop itself is Lightning's, driven by the
callbacks in ``polyx.pipeline.lightning_loop``: a family only supplies its
module and its data module.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from polyx.core.i18n import N_
from polyx.core.paths import ROOT
from polyx.core.provenance import commit_git
from polyx.pipeline.adapters import build_adapter
from polyx.pipeline.artifacts import (
    ArtifactLayout,
    append_log,
    write_json,
    write_json_atomic,
    write_report,
    write_yaml,
)
from polyx.pipeline.coco_import import audit_as_dict
from polyx.pipeline.config import ExecutionConfig, config_to_dict, protocol_to_dict
from polyx.pipeline.dataset import ResolvedBoxDataset, dataset_fingerprint
from polyx.pipeline.errors import PipelineError
from polyx.pipeline.evaluation import SharedEvaluator
from polyx.pipeline.identity import ExecutionIdentity
from polyx.pipeline.lightning_loop import (
    ExecutionRecorder,
    TrainingOutcome,
    build_trainer,
    evaluate_best,
    recovered_outcome,
    run_fit,
    stopping_record,
)
from polyx.pipeline.resume import (
    RecoveryState,
    committed_checkpoint,
    has_committed_checkpoint,
)
from polyx.pipeline.weights import resolve_weights


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON artifact written by this engine."""
    loaded = _try_read_json(path)
    if loaded is None:
        raise RuntimeError(f"cannot read artifact {path}")
    return loaded


def _try_read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON artifact, or ``None`` when it is absent or truncated."""
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _completion_record(
    config: ExecutionConfig,
    *,
    selected_epoch: int,
    selection: dict[str, Any],
    checkpoint: Path,
    epochs_completed: int,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    """The last artifact written, naming the execution as finished."""
    return {
        "status": "completed",
        "protocol": protocol_to_dict(config),
        "dataset_fingerprint": dataset_fingerprint,
        "family": asdict(config.family),
        "backend": asdict(config.backend),
        "selected_epoch": selected_epoch,
        "selection": selection,
        "checkpoint": str(checkpoint),
        "epochs_completed": epochs_completed,
    }


def run_execution(
    config: ExecutionConfig, *, resolved_boxes: ResolvedBoxDataset | None = None
) -> dict[str, Any]:
    """Run one execution and return a small summary of its artifacts.

    A caller that already resolved a box dataset — the campaign, so its cohort
    check and this fit share one file list — hands it in; otherwise the adapter
    resolves it during preparation.
    """
    layout = ArtifactLayout(config.output_dir)
    recovery: RecoveryState | None = None
    if config.resume:
        # The checkpoint Lightning committed is the one recovery authority.
        # The progress record is only a heartbeat: it is never consulted for
        # identity, protocol, dataset or an epoch decision.
        recovery = committed_checkpoint(config.output_dir)
        if recovery is None:
            raise PipelineError(
                "execution.checkpoint_absent",
                N_("resume requested but no committed Lightning checkpoint exists at %(path)s"),
                path=str(layout.lightning_checkpoints),
            )
    else:
        # A fresh execution never reuses an execution directory: an existing
        # record could be inherited as this run's completion, or a stale
        # progress snapshot could resume the previous run's weights.
        existing = [
            path
            for path in (layout.completion, layout.progress, layout.metrics, layout.training_state)
            if path.is_file()
        ]
        if existing or has_committed_checkpoint(config.output_dir):
            raise PipelineError(
                "execution.directory_in_use",
                N_(
                    "%(path)s already holds an execution; pass resume=true to continue it "
                    "or choose a new output directory"
                ),
                path=str(layout.root),
            )

    # Staged or imported weights are an initialization, never a resume point.
    # The identity records the verified digest when the weight is pinned and
    # the declared name otherwise, so a moved node still matches.
    resolved_weights = resolve_weights(
        config.family,
        weights_dir=config.weights_dir,
        manifest=config.weights_manifest,
    )
    declared_identity = ExecutionIdentity.build(
        config, weights=resolved_weights, dataset_fingerprint=""
    )
    if recovery is not None:
        declared_identity.require_static_matches(recovery.identity, path=layout.root)

    adapter = build_adapter(config.family, resolved_boxes=resolved_boxes)
    prepared = adapter.prepare(config)
    audit = prepared.audit
    if not audit.ok:
        codes = [finding.code or "unknown" for finding in audit.findings]
        raise PipelineError(
            "dataset.audit_failed",
            N_("dataset audit failed: %(codes)s"),
            codes=", ".join(codes),
        )
    if config.resume and not prepared.hashes:
        raise PipelineError(
            "execution.dataset_identity_missing",
            N_("cannot verify the dataset of %(path)s: it has no file hashes"),
            path=str(layout.root),
        )
    fingerprint = dataset_fingerprint(prepared.hashes)
    identity = ExecutionIdentity.build(
        config, weights=resolved_weights, dataset_fingerprint=fingerprint
    )
    if recovery is not None:
        identity.require_matches(recovery.identity, path=layout.root)
        if layout.completion.is_file():
            # A completed execution is never rebuilt; the completion record is
            # the one authority, and its identity has just been verified.
            metrics = _read_json(layout.metrics)
            return {
                "output_dir": layout.root,
                "checkpoint": Path(str(_read_json(layout.completion)["checkpoint"])),
                "selection": metrics["selection"],
                "metrics": metrics,
                "training_state": _read_json(layout.training_state),
            }

    layout.create()
    write_yaml(config_to_dict(config), layout.config)
    write_yaml(protocol_to_dict(config), layout.resolved_protocol)
    write_json(identity.to_dict(), layout.execution_identity)
    append_log(
        layout.training_log,
        f"family={config.family.name} backend={config.backend.name} resume={config.resume}",
    )
    if recovery is not None:
        append_log(
            layout.training_log,
            f"resume from epoch={recovery.epochs_completed} checkpoint={recovery.checkpoint}",
        )
    if prepared.conversion is not None:
        append_log(
            layout.training_log,
            "dataset=coco_instances "
            f"conversion={prepared.conversion['root']} "
            f"cache_key={prepared.conversion['cache_key']} "
            f"cached={str(prepared.conversion['cached']).lower()}",
        )

    evaluator = SharedEvaluator(
        prepared.ground_truth,
        config.protocol.split,
        task=config.family.task,
        confidence_threshold=config.protocol.confidence,
    )
    recorder = ExecutionRecorder(
        layout=layout,
        config=config,
        dataset_fingerprint=fingerprint,
        identity=identity,
        evaluator=evaluator,
    )
    run = adapter.build(config, prepared, recorder)

    git = commit_git(ROOT)
    # A container has no source checkout: the build embeds the commit as
    # POLYX_SOURCE_SHA. Keep the repository answer when one is available.
    image_commit = os.environ.get("POLYX_SOURCE_SHA", "").strip()
    if not git.get("commit") and image_commit:
        git = {**git, "commit": image_commit, "source": "image"}
    provenance: dict[str, Any] = {
        "git": git,
        "dataset_hashes": prepared.hashes,
        "dataset_audit": audit_as_dict(audit),
        "data_yaml": (
            str(config.data_yaml)
            if prepared.source_kind == "yolo_boxes" and config.data_yaml is not None
            else ""
        ),
        "dataset_source": {
            "kind": prepared.source_kind,
            "path": str(prepared.source_path),
        },
        "dataset_name": config.dataset_name,
    }
    if prepared.conversion is not None:
        provenance["conversion"] = prepared.conversion

    def publish(
        final_metrics: dict[str, Any],
        training_state: dict[str, Any],
        best_artifact: Path,
        epochs_completed: int,
    ) -> dict[str, Any]:
        """Write every required artifact, then the completion record last."""
        # training_state is written before the metrics it accompanies so a
        # crash during publication leaves the fit record (precision, device,
        # verified weights) recoverable instead of inventing one later.
        write_json(training_state, layout.training_state)
        write_json(final_metrics, layout.metrics)
        write_json(provenance, layout.provenance)
        write_report(layout.report, final_metrics, training_state, config_to_dict(config))
        append_log(
            layout.training_log,
            f"done best={best_artifact} selection={final_metrics['selection']['value']:.6f}",
        )
        # The completion record is the final write: no artifact or log line may
        # fail after the run has already been marked complete.
        write_json_atomic(
            _completion_record(
                config,
                selected_epoch=int(final_metrics.get("selected_epoch", -1)),
                selection=final_metrics["selection"],
                checkpoint=best_artifact,
                epochs_completed=epochs_completed,
                dataset_fingerprint=fingerprint,
            ),
            layout.completion,
        )
        return {
            "output_dir": layout.root,
            "checkpoint": best_artifact,
            "selection": final_metrics["selection"],
            "metrics": final_metrics,
            "training_state": training_state,
        }

    outfit: TrainingOutcome
    if _finished_before_publication(config, recovery):
        # The epoch loop is already on disk; only the artifacts are missing.
        # Publication re-scores the winner the committed checkpoint names
        # instead of rerunning a fit or carrying an earlier epoch's figures
        # forward.
        assert recovery is not None
        outfit = recovered_outcome(recovery)
        trainer, _checkpoint, _requested = build_trainer(config, layout)
    else:
        outfit, trainer = run_fit(
            config,
            layout,
            recorder,
            run,
            resume_checkpoint=_resume_checkpoint(recovery),
        )

    final_metrics, _best = evaluate_best(layout, run, trainer, best=outfit.best_checkpoint)
    training_state = _training_state(config, outfit, layout)
    if recovery is not None:
        training_state["resumed"] = True
        training_state["resumed_from_epoch"] = recovery.epochs_completed
        training_state["resumed_from"] = str(recovery.checkpoint)
    return publish(final_metrics, training_state, layout.best_checkpoint, outfit.epochs_completed)


def _resume_checkpoint(recovery: RecoveryState | None) -> Path | None:
    """The committed Lightning checkpoint a resumed fit must restore from."""
    return None if recovery is None else recovery.checkpoint


def _finished_before_publication(config: ExecutionConfig, recovery: RecoveryState | None) -> bool:
    """Whether the committed checkpoint already holds the execution's last epoch.

    Early stopping is its own finish condition, and both facts come from the
    checkpoint Lightning committed. A progress heartbeat written one epoch
    behind can neither trigger an extra epoch nor hide a committed stop.
    """
    if recovery is None:
        return False
    if recovery.stopped_early:
        return True
    return recovery.epochs_completed >= config.protocol.epochs


def _training_state(
    config: ExecutionConfig, outfit: TrainingOutcome, layout: ArtifactLayout
) -> dict[str, Any]:
    """The fit facts that travel beside the metrics."""
    return {
        "protocol": protocol_to_dict(config),
        "family": asdict(config.family),
        "backend": asdict(config.backend),
        "amp": asdict(config.protocol.amp),
        "precision": outfit.precision,
        "amp_requested": outfit.amp_requested,
        "amp_active": outfit.amp_active,
        "device": outfit.device,
        "framework": config.family.framework,
        "weights": outfit.weights,
        "weights_verified": outfit.weights_verified,
        "epochs_completed": outfit.epochs_completed,
        "augmentation": outfit.augmentation,
        "early_stopping": stopping_record(
            config,
            best_epoch=outfit.best_epoch,
            best_score=outfit.best_score,
            stopped_early=outfit.stopped_early,
            wait_count=outfit.wait_count,
            epochs_completed=outfit.epochs_completed,
        ),
        "best_checkpoint": str(layout.best_checkpoint),
    }
