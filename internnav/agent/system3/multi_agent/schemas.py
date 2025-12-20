from __future__ import annotations

"""
Schemas for multi-agent System 3.

The key addition vs. single-agent:
- explicit milestone list + completion evidence
- a config object that controls how many agent calls happen per step
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from internnav.agent.system3.schemas import System3Plan, System3Status

MilestoneState = Literal["PENDING", "IN_PROGRESS", "DONE", "FAILED"]


@dataclass
class Milestone:
    """
    A milestone is a semantic subgoal. Completion is determined by evidence from perception
    and/or trajectory constraints (as interpreted by the tracker agent).
    """

    id: str
    title: str
    completion_criteria: str = ""


@dataclass
class MilestoneProgress:
    milestone_id: str
    state: MilestoneState = "PENDING"
    evidence: str = ""
    last_update_sys3_call: int = 0


@dataclass
class MultiAgentConfig:
    """
    Controls the multi-agent runtime.

    All agents use OpenAI-compatible API (Gemini via base_url is supported).
    """

    enabled: bool = False

    # Models for each agent role (can be the same model)
    tracker_model: str = ""
    planner_model: str = ""
    writer_model: str = ""
    verifier_model: str = ""

    # Frequency controls (reduce cost/latency)
    tracker_interval_calls: int = 3  # call tracker every N System3 calls
    verifier_on_verify_only: bool = True

    # If True, planner emits System3Plan directly. If False, planner emits "intent" and writer emits final System3Plan.
    writer_enabled: bool = True

    # Debug
    dump_dir: Optional[str] = None
    dump_episode_id: Optional[int] = None
    dump_freq: int = 1

    def with_defaults(self, fallback_model: str) -> "MultiAgentConfig":
        cfg = MultiAgentConfig(**self.__dict__)
        cfg.tracker_model = cfg.tracker_model or fallback_model
        cfg.planner_model = cfg.planner_model or fallback_model
        cfg.writer_model = cfg.writer_model or fallback_model
        cfg.verifier_model = cfg.verifier_model or fallback_model
        return cfg


@dataclass
class PlannerIntent:
    """
    Output of the Planner agent in multi-agent mode (before writing final instruction).
    """

    thought: str = ""
    status: System3Status = "SEARCH"
    change_instruction: bool = True
    # The semantic target / subgoal in plain text (not necessarily System-2 friendly)
    intent: str = ""

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "PlannerIntent":
        raw_status = (obj.get("status") or "SEARCH").strip().upper()
        if raw_status not in {"EXPLORE", "NAVIGATE", "SEARCH", "VERIFY", "DONE", "ERROR"}:
            raw_status = "SEARCH"

        raw_change = obj.get("change_instruction", True)
        if isinstance(raw_change, str):
            raw_lower = raw_change.strip().lower()
            change_instruction = raw_lower in ("true", "1", "yes", "y")
        else:
            change_instruction = bool(raw_change)

        return cls(
            thought=str(obj.get("thought", "") or ""),
            status=raw_status,  # type: ignore[assignment]
            change_instruction=change_instruction,
            intent=str(obj.get("intent", "") or ""),
        )




