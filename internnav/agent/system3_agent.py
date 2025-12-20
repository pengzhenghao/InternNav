"""
Habitat-compatible System 3 Agent wrapper.

This file intentionally stays *thin*.
All "agentic" logic lives in `internnav.agent.system3` (state/planner/compiler/critic).

Task adaptation should not require editing this file:
use `AgentCfg.model_settings` overrides (sys3_* keys) to swap prompts/constraints.
"""

import base64
import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from internnav.agent.base import Agent
from internnav.agent.internvla_n1_agent import InternVLAN1Agent
from internnav.agent.system3 import (
    DEFAULT_PROFILE,
    MultiAgentConfig,
    MultiAgentSystem3Navigator,
    MultiAgentSystem3State,
    System3Navigator,
    System3Plan,
    System3PromptProfile,
    System3State,
)
from internnav.configs.agent import AgentCfg

# Configuration (priority: env vars > config > defaults)
DEFAULT_VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://localhost:8080/v1")
DEFAULT_VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _get_profile_from_cfg(model_settings: Dict[str, Any], user_goal: str) -> System3PromptProfile:
    """
    Task adaptation knob: you can supply lightweight overrides from config without editing code.

    Supported keys in model_settings:
      - sys3_profile_name: str
      - sys3_system_prompt: str (optional; if empty uses default template)
      - sys3_instruction_banlist: list[str]
      - sys3_max_instruction_chars: int
      - sys3_max_sentences: int
    """
    profile = System3PromptProfile(**DEFAULT_PROFILE.__dict__)
    profile.name = str(model_settings.get("sys3_profile_name", profile.name))

    sys_prompt = model_settings.get("sys3_system_prompt", None)
    if isinstance(sys_prompt, str) and sys_prompt.strip():
        profile.system_prompt = sys_prompt.replace("{goal}", user_goal)

    banlist = model_settings.get("sys3_instruction_banlist", None)
    if isinstance(banlist, list) and all(isinstance(x, str) for x in banlist):
        profile.instruction_banlist = banlist

    mic = model_settings.get("sys3_max_instruction_chars", None)
    if isinstance(mic, int) and mic > 0:
        profile.max_instruction_chars = mic

    ms = model_settings.get("sys3_max_sentences", None)
    if isinstance(ms, int) and ms > 0:
        profile.max_sentences = ms

    return profile


@Agent.register("system3")
class System3Agent(InternVLAN1Agent):
    """
    Habitat-compatible System 3 agent.

    Contract used by `internnav/habitat_extensions/habitat_vln_evaluator.py`:
      new_instr, status, thought = agent.update_instruction(observations)
    """

    def __init__(self, config: AgentCfg):
        super().__init__(config)

        # Navigator can be single-agent or multi-agent.
        self.navigator: Optional[Any] = None
        self.state: Optional[System3State] = None

        self.current_instruction: Optional[str] = None

        # Prompt dump / debugging
        self.prompt_dump_dir: Optional[str] = None
        self.prompt_dump_freq: int = 1
        self.prompt_dump_episode_id: Optional[int] = None

        # Latest System 3 outputs for visualization/debugging
        self.last_sys3_status: Optional[str] = None
        self.last_sys3_thought: Optional[str] = None
        self.last_sys3_updated_step: Optional[int] = None

        model_settings = config.model_settings or {}
        self.vlm_api_key = os.environ.get("VLLM_API_KEY") or model_settings.get("vlm_api_key") or DEFAULT_VLLM_API_KEY
        self.vlm_base_url = os.environ.get("VLLM_API_URL") or model_settings.get("vlm_api_url") or DEFAULT_VLLM_API_URL
        self.vlm_model_name = os.environ.get("MODEL_NAME") or model_settings.get("vlm_model_name") or DEFAULT_MODEL_NAME

        # Sub-episode / memory config
        self.max_subepisode_frames: int = int(model_settings.get("sys3_max_subepisode_frames", 8))
        # Architecture switch: "single" (default) vs "multi"
        self.sys3_arch: str = str(model_settings.get("sys3_arch", "single")).lower().strip()

        logger.info(
            "System 3 Agent initialized with: vlm_base_url=%s, vlm_model_name=%s, max_subepisode_frames=%d",
            self.vlm_base_url,
            self.vlm_model_name,
            self.max_subepisode_frames,
        )
        logger.info("System 3 architecture: %s", self.sys3_arch)

    def reset(self, reset_index=None):
        super().reset(reset_index)
        self.navigator = None
        self.state = None
        self.current_instruction = None
        self.last_sys3_status = None
        self.last_sys3_thought = None
        self.last_sys3_updated_step = None

    def set_goal(self, goal: str) -> None:
        """
        Initialize System 3 with a high-level goal.
        The goal is also used as the initial local instruction until System 3 refines it.
        """
        model_settings = self.config.model_settings or {}
        profile = _get_profile_from_cfg(model_settings=model_settings, user_goal=goal)

        self.current_instruction = goal

        if self.sys3_arch == "multi":
            # Use dedicated multi-agent state + navigator from `internnav.agent.system3.multi_agent`.
            self.state = MultiAgentSystem3State(
                user_goal=goal,
                current_instruction=goal,
                max_subepisode_frames=self.max_subepisode_frames,
            )

            ms = self.config.model_settings or {}
            ma_cfg = MultiAgentConfig(
                enabled=True,
                tracker_model=str(ms.get("sys3_ma_tracker_model", "")),
                planner_model=str(ms.get("sys3_ma_planner_model", self.vlm_model_name)),
                writer_model=str(ms.get("sys3_ma_writer_model", self.vlm_model_name)),
                verifier_model=str(ms.get("sys3_ma_verifier_model", "")),
                tracker_interval_calls=int(ms.get("sys3_ma_tracker_interval_calls", 3)),
                verifier_on_verify_only=bool(ms.get("sys3_ma_verifier_on_verify_only", True)),
                writer_enabled=bool(ms.get("sys3_ma_writer_enabled", True)),
                dump_dir=self.prompt_dump_dir,
                dump_episode_id=self.prompt_dump_episode_id,
                dump_freq=self.prompt_dump_freq,
            ).with_defaults(fallback_model=self.vlm_model_name)

            self.navigator = MultiAgentSystem3Navigator(
                state=self.state,
                api_key=self.vlm_api_key,
                base_url=self.vlm_base_url,
                cfg=ma_cfg,
            )
        else:
            self.state = System3State(
                user_goal=goal,
                current_instruction=goal,
                max_subepisode_frames=self.max_subepisode_frames,
            )
            self.navigator = System3Navigator(
                state=self.state,
                model_name=self.vlm_model_name,
                api_key=self.vlm_api_key,
                base_url=self.vlm_base_url,
                profile=profile,
                dump_dir=self.prompt_dump_dir,
                dump_freq=self.prompt_dump_freq,
                dump_episode_id=self.prompt_dump_episode_id,
            )

        logger.info("[Sys3] Goal set: %s (profile=%s)", goal, profile.name)

    def set_prompt_dump(self, dump_dir: str, freq: int = 1, episode_id: Optional[int] = None):
        assert dump_dir is not None, "dump_dir cannot be None"
        self.prompt_dump_dir = dump_dir
        self.prompt_dump_freq = max(1, int(freq))
        self.prompt_dump_episode_id = episode_id
        if self.navigator is not None:
            self.navigator.dump_dir = dump_dir
            self.navigator.dump_freq = self.prompt_dump_freq
            self.navigator.dump_episode_id = episode_id
        logger.info("[Sys3] Prompt dump configured: dump_dir=%s freq=%d episode_id=%s", dump_dir, freq, episode_id)

    def update_instruction(self, obs: List[Dict[str, Any]]) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Call System 3 (VLM) to possibly update the instruction.
        Returns: (instruction or None, status, thought)
        """
        if not isinstance(obs, list):
            obs = [obs]
        current_obs = obs[0]

        assert self.state is not None, "System3Agent.set_goal() must be called before update_instruction()."
        assert self.navigator is not None, "System3Agent.set_goal() must be called before update_instruction()."

        # Convert observation to base64 image
        rgb = current_obs["rgb"]
        image = Image.fromarray(rgb.astype("uint8"), "RGB")
        img_b64 = pil_to_base64(image)

        # Update state counters and memory
        self.state.sys1_steps = int(self.episode_step)
        self.state.sys3_calls += 1
        self.state.sys2_calls = int(self.state.sys2_calls)  # kept for future plumbing

        self.state.append_frame(img_b64)

        plan: Optional[System3Plan] = self.navigator.step()
        if not plan:
            # Fail-safe: keep current instruction
            return self.current_instruction, self.last_sys3_status or "SEARCH", self.last_sys3_thought

        self.last_sys3_status = plan.status
        self.last_sys3_thought = plan.thought
        self.last_sys3_updated_step = self.episode_step

        if plan.status in ("DONE", "ERROR"):
            return None, plan.status, plan.thought

        # Apply instruction update policy
        new_instruction = plan.instruction
        if plan.change_instruction and new_instruction and new_instruction != self.current_instruction:
            self.current_instruction = new_instruction
            self.state.current_instruction = new_instruction
            # Reset sub-episode frames since we're issuing a new instruction.
            self.state.start_new_subepisode(latest_img_b64=img_b64)

        return self.current_instruction, plan.status, plan.thought

    # --- Hooks where we can count "System 2 calls" in future, without changing evaluator code. ---
    def predict_goal(self, inputs):
        self.state.sys2_calls += 1 if self.state is not None else 0
        model = self.policy.model
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, top_k=None, top_p=None, temperature=None)
        return output_ids

    def generate_local_actions(self, output_ids, pixel_values, image_grid_thw, images_dp, depths_dp):
        model = self.policy.model
        with torch.no_grad():
            traj_latents = model.generate_latents(output_ids, pixel_values, image_grid_thw)
            dp_actions = model.generate_traj(traj_latents, images_dp, depths_dp, use_async=True)
        return dp_actions


