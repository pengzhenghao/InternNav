"""
Multi-agent System 3 implementation.

This package provides an alternative to the single-orchestrator System 3:
- Multiple LLM "roles" (agents) with small contracts
- A shared external state store with explicit milestone tracking
- Switchable via config flag so the original path remains available
"""

from .schemas import Milestone, MilestoneState, MultiAgentConfig
from .state import MultiAgentSystem3State
from .navigator import MultiAgentSystem3Navigator

__all__ = [
    "Milestone",
    "MilestoneState",
    "MultiAgentConfig",
    "MultiAgentSystem3State",
    "MultiAgentSystem3Navigator",
]




