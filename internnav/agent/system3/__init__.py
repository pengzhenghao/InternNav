"""
System 3 (VLM-driven) navigation module.

Design goal: keep System 3 as a *modular compiler* from (goal + memory + vision)
into System-2-friendly micro-instructions, without turning it into a monolith.

Modules:
- `state`: persistent, non-LLM memory and counters
- `planner`: one LLM call to produce a structured `System3Plan`
- `compiler`: deterministic instruction sanitization / grammar enforcement
- `critic`: deterministic stuck detection + recovery instruction
- `navigator`: orchestration of the above pieces
"""

from .schemas import System3Plan, System3Status
from .state import System3State
from .navigator import System3Navigator
from .prompt_profiles import System3PromptProfile, DEFAULT_PROFILE
from .multi_agent import MultiAgentConfig, MultiAgentSystem3Navigator, MultiAgentSystem3State, Milestone, MilestoneState

__all__ = [
    "System3Plan",
    "System3Status",
    "System3State",
    "System3Navigator",
    "MultiAgentConfig",
    "MultiAgentSystem3Navigator",
    "MultiAgentSystem3State",
    "Milestone",
    "MilestoneState",
    "System3PromptProfile",
    "DEFAULT_PROFILE",
]


