from __future__ import annotations

"""
System 3 Critic / Monitor (cheap, non-LLM).

Purpose:
- Detect obvious failure modes (no visual change, repeated instruction churn)
- (Disabled by default in this repo) Avoid rule-based overrides that change intent.

This is intentionally lightweight and deterministic so it can run every step
without adding latency or cost.
"""

from dataclasses import dataclass

from .schemas import System3Plan, System3PromptProfile
from .state import System3State


@dataclass
class System3Critic:
    """
    Cheap monitor to prevent instruction churn and recover from obvious stuck states.
    """

    profile: System3PromptProfile
    stuck_no_change_threshold: int = 3
    stuck_same_instruction_threshold: int = 3

    def post_process(self, state: System3State, plan: System3Plan) -> System3Plan:
        """
        No-op by design (user preference): keep the LLM plan as-is.

        We still update state counters (same-instruction detection) so you can
        re-enable heuristic interventions later if desired.
        """
        # Track repeated instruction emission.
        if plan.instruction and plan.instruction == state.last_emitted_instruction:
            state.same_instruction_count += 1
        else:
            state.same_instruction_count = 0

        return plan


