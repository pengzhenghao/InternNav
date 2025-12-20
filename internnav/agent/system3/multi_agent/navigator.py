from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from internnav.agent.system3.schemas import System3Plan

from .agents import MultiAgentLLM
from .schemas import MultiAgentConfig
from .state import MultiAgentSystem3State

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentSystem3Navigator:
    """
    Multi-agent System 3 orchestrator.

    Pipeline:
      1) milestone generation (once at episode start)
      2) milestone tracking (periodic)
      3) planner intent (every step)
      4) instruction writer (optional; otherwise planner writes instruction directly)
    """

    state: MultiAgentSystem3State
    api_key: str
    base_url: str
    cfg: MultiAgentConfig

    def __post_init__(self) -> None:
        self.cfg = self.cfg.with_defaults(fallback_model="")  # model default injected by caller
        self.llm = MultiAgentLLM(api_key=self.api_key, base_url=self.base_url)
        self._initialized = False

        # Gemini "extra_body" compatibility (reuse prior shape in this repo)
        self._extra_body = {
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinking_budget": -1,
                        "include_thoughts": False,
                    }
                }
            }
        }

    def init_episode(self) -> None:
        if self._initialized:
            return
        if not self.cfg.planner_model:
            raise ValueError("MultiAgentConfig must set planner_model (or caller must fill defaults).")

        # Generate milestones once
        ms = self.llm.generate_milestones(self.cfg.tracker_model or self.cfg.planner_model, self.state.user_goal, extra_body=self._extra_body)
        self.state.milestones = ms
        self.state.progress = []
        self.state.current_milestone_idx = 0
        self._initialized = True
        logger.info("[Sys3-MA] Initialized milestones: %d", len(ms))

    def step(self) -> Optional[System3Plan]:
        self.init_episode()
        frames = list(self.state.subepisode_frames_b64)
        if not frames:
            return None

        # Periodically update milestone progress
        if self.cfg.tracker_interval_calls > 0 and (self.state.sys3_calls % self.cfg.tracker_interval_calls == 0):
            self.state.progress = self.llm.update_progress(
                model=self.cfg.tracker_model or self.cfg.planner_model,
                goal=self.state.user_goal,
                milestones=self.state.milestones,
                progress=self.state.progress,
                frames_b64=frames,
                extra_body=self._extra_body,
            )
            # stamp updates
            for p in self.state.progress:
                p.last_update_sys3_call = self.state.sys3_calls
            self.state.advance_if_done()

        cur_ms = self.state.get_current_milestone()

        intent = self.llm.plan_intent(
            model=self.cfg.planner_model,
            goal=self.state.user_goal,
            current_milestone=cur_ms,
            progress=self.state.progress,
            current_instruction=self.state.current_instruction,
            frames_b64=frames,
            extra_body=self._extra_body,
        )
        if not intent:
            return None

        if not self.cfg.writer_enabled:
            # Planner should have written instruction (not implemented in this initial version)
            return System3Plan(thought=intent.thought, status=intent.status, instruction=intent.intent, change_instruction=intent.change_instruction)

        plan = self.llm.write_instruction(
            model=self.cfg.writer_model,
            goal=self.state.user_goal,
            milestone=cur_ms,
            intent=intent,
            frames_b64=frames,
            extra_body=self._extra_body,
        )
        if not plan:
            return None

        # Merge thought: keep planner thought + writer thought
        combined = []
        if intent.thought.strip():
            combined.append("[Planner]\n" + intent.thought.strip())
        if plan.thought.strip():
            combined.append("[Writer]\n" + plan.thought.strip())
        plan.thought = "\n\n".join(combined).strip()
        plan.status = intent.status
        plan.change_instruction = intent.change_instruction

        logger.info(
            "[Sys3-MA] Plan: status=%s change=%s ms=%s instr=%s",
            plan.status,
            plan.change_instruction,
            (cur_ms.id if cur_ms else "none"),
            (plan.instruction or "")[:140],
        )
        return plan




