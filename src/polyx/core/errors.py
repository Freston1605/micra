"""Coded failures with display-time translation.

A failure that crosses a module seam or reaches the JSONL journal must carry
invariant tokens: a stable code and its parameters. The message template stays
in English for the traceback; the interface translates it with the same
gettext catalog as the rest of the program (ADR-0003/0004).

The class lives in ``core`` because core modules must be able to refuse a
request without importing the pipeline. :mod:`polyx.pipeline.errors` keeps the
journal boundary and the historical ``PipelineError`` name.
"""

from __future__ import annotations

from typing import Any

from polyx.core.i18n import _


class CodedFailure(ValueError):
    """A failure whose machine identity is separate from its display text."""

    def __init__(self, code: str, template: str, **parameters: Any) -> None:
        self.code = code
        self.template = template
        self.parameters = parameters
        super().__init__(template % parameters if parameters else template)

    def display(self) -> str:
        """The operator-facing message, translated in the active language."""
        translated = _(self.template)
        return translated % self.parameters if self.parameters else translated
