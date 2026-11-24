"""Hydra multirun entry point for the benchmark campaign.

``python -m polyx.pipeline.campaign_app --multirun`` launches one job per model
family inside a single allocation. Overriding ``family`` with a comma-separated
list is what makes it a campaign rather than a single execution; every job reads
the same protocol composition and writes its artifacts under
``<campaign_root>/<run>``. Nothing here is family-specific: the adapters,
evaluator, early stopping and artifact layout are the same ones a single
execution uses.

A campaign is resumable without a separate per-family switch. With
``resume=true`` a job resumes a family whose committed Lightning checkpoint
can be recovered, verifies and skips a family whose execution already finished,
and runs a family that never started normally. Passing the flag to every job
and letting each job decide is what makes resubmission safe. A finished family
is not trusted on its protocol alone: the engine checks the execution identity
and the dataset fingerprint before its published metrics are reused, so a
campaign resubmitted against another dataset or other weights is refused. The
progress heartbeat is not the marker: a checkpoint whose heartbeat was never
written is still resumable.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from hydra import main
from omegaconf import DictConfig, OmegaConf

from polyx.pipeline.artifacts import ArtifactLayout
from polyx.pipeline.config import ExecutionConfig, execution_config_from_raw
from polyx.pipeline.dataset import (
    ResolvedBoxDataset,
    resolve_box_dataset,
    validate_campaign_cohort,
)
from polyx.pipeline.errors import report_failure
from polyx.pipeline.execution import run_execution
from polyx.pipeline.resume import has_committed_checkpoint


def campaign_raw(cfg: DictConfig) -> dict:
    """Resolve one campaign job into the plain mapping the engine validates."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    root = str(raw.get("campaign_root") or "").strip()
    if not root:
        raise ValueError("campaign_root is a required campaign override")
    if not raw.get("output_dir"):
        family = raw.get("family") or {}
        run_name = str(family.get("run_name") or family.get("name") or "")
        raw["output_dir"] = str(Path(root) / run_name)
    return raw


def _resume_decision(config: ExecutionConfig) -> str:
    """Whether this family resumes, verifies a finished record, or starts fresh.

    A family with a committed Lightning checkpoint resumes even when its
    progress heartbeat is missing or stale. One whose execution already
    finished is not skipped on the strength of its protocol alone: the engine
    verifies the whole execution identity — family, initialization weights,
    protocol and dataset fingerprint — and reuses the published metrics only
    when it still matches, so a changed dataset or family weight is refused
    instead of reprinting an old score. Everything else starts normally even
    when the campaign was submitted with ``resume=true``.
    """
    layout = ArtifactLayout(config.output_dir)
    if layout.completion.is_file():
        return "finished"
    if has_committed_checkpoint(config.output_dir):
        # A completed final epoch without a completion record is still resumed:
        # the engine finishes publishing the missing artifacts before retraining.
        return "resume"
    return "start"


@main(config_path="conf", config_name="campaign", version_base="1.3")
def campaign(cfg: DictConfig) -> float:
    """Run one family's training execution and return its with-class score."""
    try:
        return _campaign(cfg)
    except Exception as exc:
        # Hydra prints the exception itself; the journal needs the invariant
        # code and traceback, and a person needs the active language. One
        # boundary serves both, for every failure this job can raise.
        report_failure(exc, stage="campaign")
        raise


def _campaign(cfg: DictConfig) -> float:
    config = execution_config_from_raw(campaign_raw(cfg))
    boxes: ResolvedBoxDataset | None = None
    if config.data_yaml is not None and config.coco_dataset is not None:
        # Both sources travel together in a campaign. They must describe the
        # same real-domain cohort; otherwise the benchmark would vary the
        # dataset as well as the model. The resolved box files are handed to
        # every detection job so the cohort check and the fit read one dataset.
        boxes = resolve_box_dataset(config.data_yaml, config.protocol.split)
        validate_campaign_cohort(
            config.data_yaml, config.coco_dataset, config.protocol.split, resolved_boxes=boxes
        )
    run_name = config.output_dir.name
    finished = False
    if config.resume:
        decision = _resume_decision(config)
        finished = decision == "finished"
        # A family that never started must not be asked to resume. A finished
        # one is handed to the engine too: it is the component that verifies
        # the execution identity and the dataset fingerprint, and it returns
        # the published record without retraining when both still match.
        config = replace(config, resume=decision != "start")

    summary = run_execution(
        config,
        resolved_boxes=boxes if config.family.task == "detect" else None,
    )
    if finished:
        print(f"SKIPPED: {run_name} already finished; keeping its metrics")
        print(f"OUTPUT: {summary['output_dir']}")
        print(f"SELECTION: {summary['selection']['metric']}={summary['selection']['value']:.6f}")
        return float(summary["selection"]["value"])
    print(f"CAMPAIGN_FAMILY: {summary['training_state']['family']['name']}")
    print(f"OUTPUT: {summary['output_dir']}")
    print(f"SELECTION: {summary['selection']['metric']}={summary['selection']['value']:.6f}")
    return float(summary["selection"]["value"])


if __name__ == "__main__":
    campaign()
