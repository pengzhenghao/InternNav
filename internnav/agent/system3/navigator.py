from __future__ import annotations

"""
System 3 Navigator (orchestrator).

Wires together the modular System 3 pipeline:
  System3State (memory)
    -> System3LLMPlanner (semantic decision + draft instruction)
    -> InstructionCompiler (sanitize to System-2-friendly micro-step)
    -> System3Critic (stuck/churn recovery)
    -> final System3Plan

External interface is intentionally tiny: call `step()` after appending frames to state.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .compiler import InstructionCompiler
from .critic import System3Critic
from .planner import System3LLMPlanner
from .prompt_profiles import DEFAULT_PROFILE
from .schemas import System3Plan, System3PromptProfile
from .state import System3State

logger = logging.getLogger(__name__)


@dataclass
class System3Navigator:
    """
    High-level orchestrator that exposes a small interface:
      input: System3State (+ new frames)
      output: System3Plan

    Internally composed of planner + compiler + critic.
    """

    state: System3State
    model_name: str
    api_key: str
    base_url: str
    profile: System3PromptProfile = DEFAULT_PROFILE

    dump_dir: Optional[str] = None
    dump_freq: int = 1
    dump_episode_id: Optional[int] = None

    def __post_init__(self) -> None:
        # Instantiate modules
        self.planner = System3LLMPlanner(
            user_goal=self.state.user_goal,
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            profile=self.profile,
        )
        self.planner.dump_dir = self.dump_dir
        self.planner.dump_freq = self.dump_freq
        self.planner.dump_episode_id = self.dump_episode_id

        self.compiler = InstructionCompiler(profile=self.profile)
        self.critic = System3Critic(profile=self.profile)

    def step(self) -> Optional[System3Plan]:
        # Planner expects the sub-episode frames (oldest->newest).
        frames = list(self.state.subepisode_frames_b64)
        if not frames:
            return None

        plan = self.planner.plan_next_step(
            frames_b64=frames,
            sys1_steps=self.state.sys1_steps,
            sys2_calls=self.state.sys2_calls,
            sys3_calls=self.state.sys3_calls,
            current_instruction=self.state.current_instruction,
            subepisode_id=self.state.subepisode_id,
        )
        if not plan:
            return None

        # Compile to enforce constraints
        plan = self.compiler.compile(plan)
        # Critic for stuck recovery / churn control
        plan = self.critic.post_process(self.state, plan)

        # Update state trackers used by critic
        self.state.last_emitted_instruction = plan.instruction or self.state.last_emitted_instruction

        logger.info(
            "[Sys3] Plan: status=%s change=%s instr=%s",
            plan.status,
            plan.change_instruction,
            (plan.instruction or "")[:140],
        )
        return plan


