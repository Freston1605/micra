"""Machine-readable pipeline failures with display-time translation.

A failure that reaches the JSONL journal must carry invariant tokens: a stable
code and its parameters. The message template stays in English for the
traceback; the command translates it only when writing to the terminal, using
the same gettext catalog as the rest of the interface (ADR-0003/0004).

One boundary serves every headless entry point: :func:`report_failure` journals
the traceback under the ``*.error`` alarm convention and prints the translated
sentence, so the CLI, the campaign and the sweeper do not each invent their own
half of it.

The coded exception itself lives in :mod:`polyx.core.errors` so core modules can
raise it. ``PipelineError`` stays as the historical name this package exports.
"""

from __future__ import annotations

import sys
from typing import Any

from polyx.core.errors import CodedFailure
from polyx.core.provenance.observability import event_error, logger_for

#: Legacy name for the shared coded failure. It is an alias, not a subclass:
#: a failure raised in core is the same type the journal boundary recognizes.
PipelineError = CodedFailure


def describe_failure(exc: BaseException) -> str:
    """The sentence a person reads: translated for a coded failure."""
    if isinstance(exc, CodedFailure):
        return exc.display()
    return f"{type(exc).__name__}: {exc}"


def report_failure(exc: BaseException, *, stage: str) -> None:
    """Journal one failure with its traceback and print it translated.

    The journal entry is the invariant half: ``pipeline.error``, the traceback
    (whose message is the untranslated template) and, for a coded failure, its
    stable ``error_code`` and ``error_parameters``. The terminal gets the active
    language, which is the only place a translation belongs.
    """
    extra: dict[str, Any] = {"stage": stage}
    if isinstance(exc, CodedFailure):
        extra["error_code"] = exc.code
        extra["error_parameters"] = exc.parameters
    event_error(logger_for("pipeline"), "pipeline.error", exc, **extra)
    print(f"ERROR: {describe_failure(exc)}", file=sys.stderr)
