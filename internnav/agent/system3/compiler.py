from __future__ import annotations

"""
System 3 Instruction Compiler (non-LLM).

Role in the System 3 architecture:
- **Planner (LLM)** produces a semantic plan: status + free-form instruction.
- **Compiler (this module)** converts that intent into a *System-2-friendly* micro-instruction by
  enforcing a small set of hard constraints:
  - no banned phrases (e.g., "keep X on your left/right")
  - short, unambiguous phrasing (sentence/length limits)
  - safe fallbacks when the planner outputs nothing / invalid text

User preference for this repo:
- **Do not rewrite** the planner's instruction into a different behavior.
- Keep this component to *minimal formatting* only (whitespace + optional punctuation),
  and avoid rule-based "corrections" that change intent.
"""

import re
from dataclasses import dataclass

from .schemas import System3Plan, System3PromptProfile


_MULTISPACE_RE = re.compile(r"\s+")


def _split_sentences(text: str) -> list[str]:
    # A lightweight sentence splitter (good enough for instruction sanitization).
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _normalize(text: str) -> str:
    text = text.strip()
    text = _MULTISPACE_RE.sub(" ", text)
    return text


@dataclass
class InstructionCompiler:
    profile: System3PromptProfile

    def compile(self, plan: System3Plan) -> System3Plan:
        """
        Minimal formatter: normalize whitespace and ensure terminal punctuation.

        Intentionally does NOT:
        - ban phrases
        - force SEARCH fallbacks
        - truncate sentences / length
        """
        plan.instruction = _normalize(plan.instruction or "")
        if plan.instruction and plan.instruction[-1] not in ".!?":
            plan.instruction += "."

        return plan


