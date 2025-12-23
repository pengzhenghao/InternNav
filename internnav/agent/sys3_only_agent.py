"""
Sys3-only agent wrapper (no Sys1/Sys2 policy loading).

Why this exists:
- `internnav.agent.system3_agent.System3Agent` inherits `InternVLAN1Agent` and will
  load large navigation policies + weights on init.
- For benchmarking Sys3-style *instruction orchestration* in other simulators
  (e.g., SimWorld), we often want a lightweight wrapper that only depends on
  `internnav.agent.system3` (planner/compiler/critic/state).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from internnav.agent.system3 import DEFAULT_PROFILE, System3Navigator, System3Plan, System3PromptProfile, System3State


def _pil_to_base64_jpeg(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class Sys3OnlyAgentCfg:
    """Minimal config for Sys3-only use outside Habitat."""

    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "EMPTY"
    max_subepisode_frames: int = 8
    # Avoid sharing a mutable profile instance across agents.
    profile: System3PromptProfile = field(default_factory=lambda: System3PromptProfile(**DEFAULT_PROFILE.__dict__))
    dump_dir: Optional[str] = None
    dump_freq: int = 1
    dump_episode_id: Optional[int] = None


class Sys3OnlyAgent:
    """A tiny wrapper that matches the `update_instruction` contract of `System3Agent`."""

    def __init__(self, cfg: Sys3OnlyAgentCfg):
        self.cfg = cfg
        self.navigator: Optional[System3Navigator] = None
        self.state: Optional[System3State] = None
        self.current_instruction: Optional[str] = None

        # Debug signals (mirrors System3Agent fields)
        self.last_sys3_status: Optional[str] = None
        self.last_sys3_thought: Optional[str] = None

        # External counter hook (callers can bump this to reflect "env steps")
        self.sys1_steps: int = 0

    def set_goal(self, goal: str) -> None:
        self.current_instruction = goal
        self.state = System3State(
            user_goal=goal,
            current_instruction=goal,
            max_subepisode_frames=int(self.cfg.max_subepisode_frames),
        )
        self.navigator = System3Navigator(
            state=self.state,
            model_name=self.cfg.model_name,
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,
            profile=self.cfg.profile,
            dump_dir=self.cfg.dump_dir,
            dump_freq=int(self.cfg.dump_freq),
            dump_episode_id=self.cfg.dump_episode_id,
        )

    def update_instruction(self, obs: Dict[str, Any]) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Equivalent contract to `System3Agent.update_instruction`:
        Returns: (instruction or None, status, thought)
        """
        assert self.state is not None and self.navigator is not None, "Call set_goal() first."

        rgb = obs["rgb"]
        if isinstance(rgb, Image.Image):
            image = rgb.convert("RGB")
        else:
            rgb_np = np.asarray(rgb)
            image = Image.fromarray(rgb_np.astype("uint8"), "RGB")

        img_b64 = _pil_to_base64_jpeg(image)

        # Update counters and memory
        self.state.sys1_steps = int(self.sys1_steps)
        self.state.sys3_calls += 1
        self.state.append_frame(img_b64)

        plan: Optional[System3Plan] = self.navigator.step()
        if not plan:
            return self.current_instruction, self.last_sys3_status or "SEARCH", self.last_sys3_thought

        self.last_sys3_status = plan.status
        self.last_sys3_thought = plan.thought

        if plan.status in ("DONE", "ERROR"):
            return None, plan.status, plan.thought

        if plan.change_instruction and plan.instruction and plan.instruction != self.current_instruction:
            self.current_instruction = plan.instruction
            self.state.current_instruction = plan.instruction
            self.state.start_new_subepisode(latest_img_b64=img_b64)

        return self.current_instruction, plan.status, plan.thought


