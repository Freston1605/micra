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
    ResumeState,
    committed_checkpoint,
    has_committed_checkpoint,
)


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


def _dataset_identity(
    hashes: dict[str, str], expected: str | None, root: Path, *, resuming: bool
) -> str:
    """The digest a resumed execution must still match.

    Weights, the best epoch's figures and the early-stopping state describe the
    bytes that trained them. Comparing only the protocol let a relabelled image
    or an edited COCO export keep the old checkpoint and publish provenance for
    the new files. ``expected`` comes from the committed checkpoint, with the
    progress heartbeat as the fallback for executions written before the
    checkpoint carried it. A resume with neither record cannot be verified and
    is refused, as it always was.
    """
    fingerprint = dataset_fingerprint(hashes)
    if not resuming:
        return fingerprint
    if not expected:
        raise PipelineError(
            "execution.dataset_identity_unrecorded",
            N_(
                "cannot verify the dataset of %(path)s: the committed checkpoint records "
                "no dataset identity, so start a new output directory instead of resuming"
            ),
            path=str(root),
        )
    if not hashes:
        raise PipelineError(
            "execution.dataset_identity_missing",
            N_("cannot verify the dataset of %(path)s: it has no file hashes"),
            path=str(root),
        )
    if expected != fingerprint:
        raise PipelineError(
            "execution.dataset_changed",
            N_(
                "the dataset behind %(path)s changed since its checkpoint was written; "
                "start a new output directory instead of resuming over it"
            ),
            path=str(root),
        )
    return fingerprint


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
    if config.resume and layout.completion.is_file():
        # A completed execution is never rebuilt; the completion record is the
        # one authority, and it needs no checkpoint to be re-read.
        metrics = _read_json(layout.metrics)
        return {
            "output_dir": layout.root,
            "checkpoint": Path(str(_read_json(layout.completion)["checkpoint"])),
            "selection": metrics["selection"],
            "metrics": metrics,
            "training_state": _read_json(layout.training_state),
        }
    resume_state: ResumeState | None = None
    recovery: RecoveryState | None = None
    if config.resume:
        # The checkpoint Lightning committed is the one recovery authority.
        # The progress record is only a heartbeat: it may be stale, or absent.
        recovery = committed_checkpoint(config.output_dir)
        if recovery is None:
            raise RuntimeError(
                "resume requested but no committed Lightning checkpoint exists at "
                f"{layout.lightning_checkpoints}"
            )
        if layout.progress.is_file():
            try:
                resume_state = ResumeState.load(layout.progress)
            except RuntimeError:
                # A heartbeat that cannot be read is still only a heartbeat;
                # the checkpoint below carries every value recovery needs.
                resume_state = None
        recorded_protocol = recovery.protocol or (
            resume_state.protocol if resume_state is not None else None
        )
        if recorded_protocol is not None and recorded_protocol != protocol_to_dict(config):
            raise PipelineError(
                "execution.protocol_changed",
                N_("protocol changed since the checkpoint was written"),
            )
    layout.create()
    if not config.resume:
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

    write_yaml(config_to_dict(config), layout.config)
    write_yaml(protocol_to_dict(config), layout.resolved_protocol)
    append_log(
        layout.training_log,
        f"family={config.family.name} backend={config.backend.name} resume={config.resume}",
    )
    if recovery is not None:
        append_log(
            layout.training_log,
            f"resume from epoch={recovery.epochs_completed} checkpoint={recovery.checkpoint}",
        )

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
    if prepared.conversion is not None:
        append_log(
            layout.training_log,
            "dataset=coco_instances "
            f"conversion={prepared.conversion['root']} "
            f"cache_key={prepared.conversion['cache_key']} "
            f"cached={str(prepared.conversion['cached']).lower()}",
        )
    recorded_fingerprint = (
        (recovery.dataset_fingerprint if recovery is not None else None)
        or (resume_state.dataset_fingerprint if resume_state is not None else None)
        or None
    )
    fingerprint = _dataset_identity(
        prepared.hashes, recorded_fingerprint, layout.root, resuming=config.resume
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
