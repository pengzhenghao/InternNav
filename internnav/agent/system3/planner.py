from __future__ import annotations

"""
System 3 LLM Planner (one model call).

This module is responsible for:
- constructing a multimodal prompt (counters + current instruction + sub-episode frames)
- calling an OpenAI-compatible endpoint with a tool schema
- parsing the tool JSON into `System3Plan`

Important design constraint:
- Keep this as **one call per System 3 step**.
  Downstream modules (compiler/critic) should enforce constraints without extra LLM calls.
"""

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .prompt_profiles import build_default_system_prompt
from .schemas import System3Plan, System3PromptProfile

logger = logging.getLogger(__name__)


STEP_OUTPUT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "output_navigation_plan",
        "description": "Output the navigation plan, status, and thought process.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["EXPLORE", "NAVIGATE", "SEARCH", "VERIFY", "DONE", "ERROR"],
                },
                "instruction": {"type": "string"},
                "change_instruction": {"type": "boolean"},
                "discrete_actions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["thought", "status", "instruction", "change_instruction"],
        },
    },
}


def _redact_image_urls(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "image_url":
                obj[k] = "<image>"
            else:
                _redact_image_urls(v)
    elif isinstance(obj, list):
        for item in obj:
            _redact_image_urls(item)
    return obj


@dataclass
class System3LLMPlanner:
    """
    Planner is responsible for *one* model call that returns a structured plan.
    Compiler/Critic then enforce constraints without additional LLM calls.
    """

    user_goal: str
    model_name: str
    api_key: str
    base_url: str
    profile: System3PromptProfile

    dump_dir: Optional[str] = None
    dump_episode_id: Optional[int] = None
    dump_freq: int = 1
    _human_log_history: List[str] = field(default_factory=list)
    _system_prompt: str = ""

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # Do NOT mutate the profile in-place (profiles may be shared/cached by the caller).
        self._system_prompt = self.profile.system_prompt or build_default_system_prompt(self.user_goal)
        self._base_messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt}]

    def build_user_message(
        self,
        frames_b64: List[str],
        query: str,
        sys1_steps: int,
        sys2_calls: int,
        sys3_calls: int,
        current_instruction: Optional[str],
        subepisode_id: Optional[int],
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        content.append(
            {
                "type": "text",
                "text": (
                    f"System 1 steps so far: {sys1_steps}. "
                    f"System 2 calls so far: {sys2_calls}. "
                    f"System 3 calls so far: {sys3_calls}."
                ),
            }
        )

        if current_instruction:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Current navigation instruction being executed is:\n"
                        f"\"{current_instruction}\""
                    ),
                }
            )
        else:
            content.append(
                {
                    "type": "text",
                    "text": "No navigation instruction has been issued yet. Propose the FIRST short navigation instruction.",
                }
            )

        if subepisode_id is not None:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"You are currently in sub-episode #{subepisode_id}. "
                        f"This sub-episode contains {len(frames_b64)} visual frames, from oldest to newest."
                    ),
                }
            )

        for idx, frame_b64 in enumerate(frames_b64):
            desc = "latest frame" if idx == len(frames_b64) - 1 else f"earlier frame {idx + 1}"
            content.append({"type": "text", "text": f"Visual {idx + 1}: {desc}."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})

        content.append({"type": "text", "text": query})
        return {"role": "user", "content": content}

    def _dump_messages(self, messages: List[Dict[str, Any]], response_text: str, plan: System3Plan, step_idx: int) -> None:
        if not self.dump_dir:
            return
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            no_image_messages = _redact_image_urls(copy.deepcopy(messages))
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "step": step_idx,
                "response_text": response_text,
                "plan": plan.__dict__,
                "messages": no_image_messages,
            }
            episode_str = f"{self.dump_episode_id:04d}" if self.dump_episode_id is not None else "0000"
            step_path = os.path.join(self.dump_dir, f"{episode_str}.json")
            with open(step_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to dump System3 messages: %s", e)

    def plan_next_step(
        self,
        frames_b64: List[str],
        sys1_steps: int,
        sys2_calls: int,
        sys3_calls: int,
        current_instruction: Optional[str],
        subepisode_id: Optional[int],
    ) -> Optional[System3Plan]:
        if not frames_b64:
            return None

        query = (
            "Please decide whether to KEEP the current navigation instruction or CHANGE it, "
            "and use the tool to output your decision."
        )

        user_msg = self.build_user_message(
            frames_b64=frames_b64,
            query=query,
            sys1_steps=sys1_steps,
            sys2_calls=sys2_calls,
            sys3_calls=sys3_calls,
            current_instruction=current_instruction,
            subepisode_id=subepisode_id,
        )
        messages = self._base_messages + [user_msg]

        logger.info(
            "[Sys3] LLM call: model=%s frames=%d sys1=%d sys2=%d sys3=%d subep=%s",
            self.model_name,
            len(frames_b64),
            sys1_steps,
            sys2_calls,
            sys3_calls,
            str(subepisode_id),
        )

        try:
            # Some OpenAI-compatible endpoints (notably Google Gemini's OpenAI compatibility)
            # reject unknown top-level fields like `"google"`. In this repo, we previously
            # used a nested shape: {"extra_body": {"google": {...}}}, so we keep that for
            # compatibility and also add a fallback retry without extra fields.
            extra_body_payload = {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "thinking_budget": -1,
                            "include_thoughts": True,
                        }
                    }
                }
            }

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1200,
                tools=[STEP_OUTPUT_TOOL],
                tool_choice={"type": "function", "function": {"name": "output_navigation_plan"}},
                extra_body=extra_body_payload,
            )
        except Exception as e:
            # Fallback: retry without extra_body if endpoint rejects it (common 400 shape on Gemini).
            msg = str(e)
            if "Unknown name \"google\"" in msg or "Unknown name 'google'" in msg or "INVALID_ARGUMENT" in msg:
                logger.warning("System3 LLM call rejected extra fields; retrying without extra_body.")
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1200,
                        tools=[STEP_OUTPUT_TOOL],
                        tool_choice={"type": "function", "function": {"name": "output_navigation_plan"}},
                    )
                except Exception as e2:
                    logger.error("System3 LLM call failed (after retry): %s", e2)
                    return None
            else:
                logger.error("System3 LLM call failed: %s", e)
                return None

        msg = completion.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            # Gemini/OpenAI-compatible endpoints sometimes ignore tool_choice and emit JSON in content.
            raw_content = (getattr(msg, "content", None) or "").strip()
            if raw_content:
                try:
                    start = raw_content.find("{")
                    end = raw_content.rfind("}") + 1
                    if start != -1 and end > start:
                        parsed = json.loads(raw_content[start:end])
                        plan = System3Plan.from_tool_json(parsed)
                        logger.warning("[Sys3] Model skipped tool call; parsed plan from message.content.")
                        return plan
                except Exception:
                    pass
            logger.error("Model did not call the required tool and no JSON plan could be parsed from content.")
            if raw_content:
                logger.error("Raw content (trunc): %s", raw_content[:400])
            return None

        tool_call = tool_calls[0]
        raw_args = tool_call.function.arguments
        try:
            parsed = json.loads(raw_args)
        except Exception:
            logger.error("Failed to parse tool JSON: %s", raw_args)
            return None

        plan = System3Plan.from_tool_json(parsed)
        step_idx = 0
        self._dump_messages(messages, raw_args, plan, step_idx)
        return plan


