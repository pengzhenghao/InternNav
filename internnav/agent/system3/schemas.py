from __future__ import annotations

"""
Shared schemas for System 3.

`System3Plan` is the stable contract passed between modules:
  Planner (LLM) -> Compiler (rules) -> Critic (monitor) -> Agent wrapper.

`System3PromptProfile` is the primary task-adaptation knob:
swap prompts / constraints without changing core logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

System3Status = Literal["EXPLORE", "NAVIGATE", "SEARCH", "VERIFY", "DONE", "ERROR"]


@dataclass
class System3Plan:
    thought: str = ""
    status: System3Status = "SEARCH"
    instruction: str = ""
    change_instruction: bool = True
    discrete_actions: List[str] = field(default_factory=list)

    @classmethod
    def from_tool_json(cls, plan: Dict[str, Any]) -> "System3Plan":
        raw_status = (plan.get("status") or "SEARCH").strip().upper()
        if raw_status not in {"EXPLORE", "NAVIGATE", "SEARCH", "VERIFY", "DONE", "ERROR"}:
            raw_status = "SEARCH"

        raw_change = plan.get("change_instruction", True)
        if isinstance(raw_change, str):
            raw_lower = raw_change.strip().lower()
            change_instruction = raw_lower in ("true", "1", "yes", "y")
        else:
            change_instruction = bool(raw_change)

        discrete_actions = plan.get("discrete_actions", [])
        if not isinstance(discrete_actions, list):
            discrete_actions = []
        discrete_actions = [str(x) for x in discrete_actions]

        return cls(
            thought=str(plan.get("thought", "") or ""),
            status=raw_status,  # type: ignore[assignment]
            instruction=str(plan.get("instruction", "") or ""),
            change_instruction=change_instruction,
            discrete_actions=discrete_actions,
        )


@dataclass
class System3PromptProfile:
    """
    Minimal surface area for task adaptation: swap prompt profile and keep the
    agentic contracts (state/planner/compiler/critic) stable.
    """

    name: str = "default"
    system_prompt: str = ""
    instruction_banlist: List[str] = field(
        default_factory=lambda: [
            # Kept only for optional downstream use; the default compiler in this repo
            # does not enforce/override based on banlists.
        ]
    )
    max_instruction_chars: int = 220
    max_sentences: int = 2


