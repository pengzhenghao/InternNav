from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from internnav.agent.system3.schemas import System3Plan

from .schemas import Milestone, MilestoneProgress, PlannerIntent

logger = logging.getLogger(__name__)


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except Exception:
        return None


def _call_with_tool_fallback(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tool: Dict[str, Any],
    tool_name: str,
    max_tokens: int = 1000,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Call a model with tool_choice, and fall back to parsing JSON from message.content
    if tool calls are missing (Gemini sometimes does this).
    Returns (parsed_json, raw_debug_text).
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            extra_body=extra_body,
        )
    except Exception as e:
        return None, f"call_error: {e}"

    msg = completion.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        raw_args = tool_calls[0].function.arguments
        try:
            return json.loads(raw_args), raw_args
        except Exception:
            return None, raw_args

    raw_content = (getattr(msg, "content", None) or "").strip()
    parsed = _safe_json_extract(raw_content)
    return parsed, raw_content[:800]


MILESTONE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "output_milestones",
        "description": "Return a milestone list for the goal (ordered, small number).",
        "parameters": {
            "type": "object",
            "properties": {
                "milestones": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "completion_criteria": {"type": "string"},
                        },
                        "required": ["id", "title", "completion_criteria"],
                    },
                }
            },
            "required": ["milestones"],
        },
    },
}


TRACKER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "output_milestone_progress",
        "description": "Update milestone progress states with evidence based on recent observations.",
        "parameters": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "milestone_id": {"type": "string"},
                            "state": {"type": "string", "enum": ["PENDING", "IN_PROGRESS", "DONE", "FAILED"]},
                            "evidence": {"type": "string"},
                        },
                        "required": ["milestone_id", "state", "evidence"],
                    },
                }
            },
            "required": ["updates"],
        },
    },
}


INTENT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "output_planner_intent",
        "description": "Output next high-level intent and status for navigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "status": {"type": "string", "enum": ["EXPLORE", "NAVIGATE", "SEARCH", "VERIFY", "DONE", "ERROR"]},
                "intent": {"type": "string"},
                "change_instruction": {"type": "boolean"},
            },
            "required": ["thought", "status", "intent", "change_instruction"],
        },
    },
}


PLAN_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "output_navigation_plan",
        "description": "Output the final System-2-friendly instruction.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "status": {"type": "string", "enum": ["EXPLORE", "NAVIGATE", "SEARCH", "VERIFY", "DONE", "ERROR"]},
                "instruction": {"type": "string"},
                "change_instruction": {"type": "boolean"},
            },
            "required": ["thought", "status", "instruction", "change_instruction"],
        },
    },
}


@dataclass
class MultiAgentLLM:
    api_key: str
    base_url: str

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_milestones(
        self,
        model: str,
        goal: str,
        context: str = "",
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> List[Milestone]:
        prompt_content = f'Goal: "{goal}"\n'
        if context:
            prompt_content += f"Context/History: {context}\n"
        prompt_content += "Return milestones."

        messages = [
            {
                "role": "system",
                "content": (
                    "You generate a short ordered milestone list for a navigation instruction.\n"
                    "Return 3-7 milestones. Each milestone should be a concrete subgoal.\n"
                    "Milestones must be ordered and not redundant.\n"
                    "If context is provided, adapt the milestones to the current situation."
                ),
            },
            {"role": "user", "content": prompt_content},
        ]
        parsed, raw = _call_with_tool_fallback(self.client, model, messages, MILESTONE_TOOL, "output_milestones", extra_body=extra_body)
        if not parsed:
            logger.warning("Milestone generation failed; using single milestone fallback. raw=%s", raw)
            return [Milestone(id="m1", title=goal, completion_criteria="Be within ~3m of the goal target.")]

        out = []
        for i, m in enumerate(parsed.get("milestones", []) or []):
            try:
                out.append(
                    Milestone(
                        id=str(m.get("id") or f"m{i+1}"),
                        title=str(m.get("title") or "").strip(),
                        completion_criteria=str(m.get("completion_criteria") or "").strip(),
                    )
                )
            except Exception:
                continue
        return out or [Milestone(id="m1", title=goal, completion_criteria="Be within ~3m of the goal target.")]

    def update_progress(
        self,
        model: str,
        goal: str,
        milestones: List[Milestone],
        progress: List[MilestoneProgress],
        frames_b64: List[str],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> List[MilestoneProgress]:
        # Build a compact text summary of milestones/progress
        ms_lines = []
        for m in milestones:
            p = next((x for x in progress if x.milestone_id == m.id), None)
            st = p.state if p else "PENDING"
            ms_lines.append(f"- {m.id}: {m.title} | state={st} | criteria={m.completion_criteria}")
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": f'Goal: "{goal}"'},
            {"type": "text", "text": "Milestones:\n" + "\n".join(ms_lines)},
            {"type": "text", "text": "Update milestone states based on the recent frames."},
        ]
        for idx, b64 in enumerate(frames_b64[-4:]):  # cap
            content.append({"type": "text", "text": f"Frame {idx+1} (recent)."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a milestone progress tracker. Update states with brief evidence.\n"
                    "Mark a milestone as DONE if completion criteria are met.\n"
                    "Mark a milestone as FAILED if it is impossible or the agent is persistently stuck/blocked."
                ),
            },
            {"role": "user", "content": content},
        ]
        parsed, raw = _call_with_tool_fallback(
            self.client, model, messages, TRACKER_TOOL, "output_milestone_progress", max_tokens=800, extra_body=extra_body
        )
        if not parsed:
            logger.warning("Tracker failed; keeping progress unchanged. raw=%s", raw)
            return progress

        updates = []
        for u in parsed.get("updates", []) or []:
            try:
                updates.append(
                    MilestoneProgress(
                        milestone_id=str(u.get("milestone_id") or ""),
                        state=str(u.get("state") or "PENDING"),
                        evidence=str(u.get("evidence") or ""),
                    )
                )
            except Exception:
                continue
        # Merge into existing
        out = list(progress)
        for upd in updates:
            found = False
            for i, p in enumerate(out):
                if p.milestone_id == upd.milestone_id:
                    out[i] = upd
                    found = True
                    break
            if not found and upd.milestone_id:
                out.append(upd)
        return out

    def plan_intent(
        self,
        model: str,
        goal: str,
        current_milestone: Optional[Milestone],
        progress: List[MilestoneProgress],
        current_instruction: Optional[str],
        frames_b64: List[str],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[PlannerIntent]:
        ms = current_milestone.title if current_milestone else "(none)"
        prog_map = {p.milestone_id: f"{p.state}: {p.evidence}" for p in progress}
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": f'Goal: "{goal}"'},
            {"type": "text", "text": f"Current milestone: {ms}"},
            {"type": "text", "text": f"Progress snapshot: {json.dumps(prog_map, ensure_ascii=False)[:800]}"},
        ]
        if current_instruction:
            content.append({"type": "text", "text": f'Current instruction: "{current_instruction}"'})
        for idx, b64 in enumerate(frames_b64[-4:]):
            content.append({"type": "text", "text": f"Frame {idx+1} (recent)."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": "Decide next status and a short semantic intent for what to do next."})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Planner agent. You decide status and next semantic intent.\n"
                    "Be milestone-aware. Do not write long instructions; write intent only."
                ),
            },
            {"role": "user", "content": content},
        ]
        parsed, raw = _call_with_tool_fallback(self.client, model, messages, INTENT_TOOL, "output_planner_intent", max_tokens=900, extra_body=extra_body)
        if not parsed:
            logger.warning("Planner intent failed. raw=%s", raw)
            return None
        return PlannerIntent.from_json(parsed)

    def write_instruction(
        self,
        model: str,
        goal: str,
        milestone: Optional[Milestone],
        intent: PlannerIntent,
        frames_b64: List[str],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[System3Plan]:
        ms = milestone.title if milestone else "(none)"
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": f'Goal: "{goal}"'},
            {"type": "text", "text": f"Current milestone: {ms}"},
            {"type": "text", "text": f"Planner status: {intent.status}"},
            {"type": "text", "text": f"Planner intent: {intent.intent}"},
        ]
        for idx, b64 in enumerate(frames_b64[-2:]):
            content.append({"type": "text", "text": f"Frame {idx+1} (recent)."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append(
            {
                "type": "text",
                "text": (
                    "Write ONE System-2-friendly micro-instruction (short imperative sentences, concrete landmarks). "
                    "Do not include abstract reasoning."
                ),
            }
        )

        messages = [
            {"role": "system", "content": "You are the Instruction Writer agent."},
            {"role": "user", "content": content},
        ]
        parsed, raw = _call_with_tool_fallback(self.client, model, messages, PLAN_TOOL, "output_navigation_plan", max_tokens=700, extra_body=extra_body)
        if not parsed:
            logger.warning("Writer failed. raw=%s", raw)
            return None
        return System3Plan.from_tool_json(parsed)




