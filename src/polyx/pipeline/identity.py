"""The one record that says which training execution a checkpoint belongs to.

A resume is only meaningful for the execution whose checkpoint it holds. The
resolved family, the initialization weights, the protocol and the dataset
fingerprint therefore travel together, both as ``execution_identity.json`` in
the execution directory and inside every Lightning checkpoint the engine
writes. Comparing tensor shapes is not enough: a compatible-looking file can
still be another variant or another initialization, and its weights would then
be published as this execution's result.

Weight identity is the verified SHA-256 digest when the execution runs from
pinned weights, and the declared name otherwise. A digest survives a move to
another node; an unpinned framework default is only as trustworthy as the name
that was declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyx.core.i18n import N_
from polyx.pipeline.config import ExecutionConfig, protocol_to_dict
from polyx.pipeline.errors import PipelineError
from polyx.pipeline.weights import ResolvedWeights

#: The one key every checkpoint uses for its identity record.
IDENTITY_KEY = "identity"


def weights_identity(resolved: ResolvedWeights) -> dict[str, Any]:
    """The initialization-weight half of an execution identity.

    A pinned weight is identified by the digest that was verified against the
    manifest; the declared name travels beside it for a human reader but is
    not what makes two initializations the same. An unpinned weight has no
    digest, so its declared name is the only identity available.
    """
    digest = resolved.digest
    identity = f"sha256:{digest}" if resolved.verified and digest else resolved.name
    return {"declared": resolved.name, "verified": resolved.verified, "identity": identity}


@dataclass(frozen=True)
class ExecutionIdentity:
    """Everything a checkpoint must still match before a resume may use it."""

    family: dict[str, Any]
    weights: dict[str, Any]
    protocol: dict[str, Any]
    dataset_fingerprint: str

    @classmethod
    def build(
        cls,
        config: ExecutionConfig,
        *,
        weights: ResolvedWeights,
        dataset_fingerprint: str,
    ) -> ExecutionIdentity:
        """The identity of the execution this configuration describes."""
        return cls(
            family={
                "name": config.family.name,
                "framework": config.family.framework,
                "task": config.family.task,
                "weights": config.family.weights,
            },
            weights=weights_identity(weights),
            protocol=protocol_to_dict(config),
            dataset_fingerprint=dataset_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        """The JSON form written to the execution and embedded in checkpoints."""
        return {
            "family": dict(self.family),
            "weights": dict(self.weights),
            "protocol": dict(self.protocol),
            "dataset_fingerprint": self.dataset_fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: object) -> ExecutionIdentity | None:
        """Read a persisted identity, or ``None`` when it is not verifiable.

        An incomplete record is not a partial identity: without all four parts
        a checkpoint cannot prove which execution it belongs to, so the record
        is treated like an absent one and the resume is refused.
        """
        if not isinstance(raw, dict):
            return None
        family = raw.get("family")
        weights = raw.get("weights")
        protocol = raw.get("protocol")
        fingerprint = raw.get("dataset_fingerprint")
        if not (
            isinstance(family, dict)
            and isinstance(weights, dict)
            and isinstance(protocol, dict)
            and isinstance(fingerprint, str)
            and fingerprint
        ):
            return None
        if not {"name", "framework", "task", "weights"} <= family.keys():
            return None
        if not {"declared", "verified", "identity"} <= weights.keys():
            return None
        return cls(
            family=dict(family),
            weights=dict(weights),
            protocol=dict(protocol),
            dataset_fingerprint=fingerprint,
        )

    def require_static_matches(self, recorded: ExecutionIdentity, *, path: Path) -> None:
        """Refuse a family, variant, initialization or protocol mismatch.

        These parts are known before the dataset is resolved, so a resume with
        a different model fails before any adapter can interpret the new
        configuration against the old checkpoint.
        """
        for group, label in (
            ("family", "family"),
            ("weights", "initialization weights"),
        ):
            difference = _first_difference(getattr(recorded, group), getattr(self, group), label)
            if difference is not None:
                _refuse(path, difference)
        difference = _first_difference(recorded.protocol, self.protocol, "protocol")
        if difference is not None:
            _refuse(path, difference)

    def require_matches(self, recorded: ExecutionIdentity, *, path: Path) -> None:
        """Refuse any difference from the execution the checkpoint records."""
        self.require_static_matches(recorded, path=path)
        if recorded.dataset_fingerprint != self.dataset_fingerprint:
            _refuse(path, "dataset fingerprint")


def _first_difference(left: Any, right: Any, prefix: str) -> str | None:
    """The dotted field where two identity parts first disagree."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f"{prefix}.{key}"
            difference = _first_difference(left[key], right[key], f"{prefix}.{key}")
            if difference is not None:
                return difference
        return None
    return None if left == right else prefix


def _refuse(path: Path, field: str) -> None:
    """Raise the one resume refusal that always names the mismatched field."""
    raise PipelineError(
        "execution.identity_changed",
        N_(
            "cannot resume %(path)s: %(field)s changed since its checkpoint was written; "
            "start a new output directory instead of resuming over it"
        ),
        path=str(path),
        field=field,
    )
