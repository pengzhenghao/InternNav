import os
import logging
import torch
import requests
import json
import time
import base64
import copy
from typing import List, Dict, Optional, Any
from io import BytesIO
from PIL import Image
from openai import OpenAI
import numpy as np

from internnav.agent.base import Agent
from internnav.agent.internvla_n1_agent import InternVLAN1Agent
from internnav.configs.agent import AgentCfg

# Configuration
DEFAULT_VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://localhost:8080/v1")
DEFAULT_VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY") 
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(os.path.basename(__file__))

# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

STEP_OUTPUT_TOOL = {
    "type": "function",
    "function": {
        "name": "output_navigation_plan",
        "description": "Output the navigation plan, status, and thought process.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Reflect on current environment, the next milestone, the progress towards the milestone, visibility of target, and what should be done next. Note down the information served as the history information in future."
                },
                "status": {
                    "type": "string",
                    "enum": ["EXPLORE","NAVIGATE", "SEARCH", "VERIFY", "DONE", "ERROR"],
                    "description": "The current status. 'EXPLORE': Exploring the environment to collect useful visual cues. 'NAVIGATE': Navigating to a visible target. 'SEARCH': Looking for the goal/landmarks. 'VERIFY': At goal, confirming success. 'DONE': Goal achieved. 'ERROR': Stuck or unable to proceed for 5+ steps."
                },
                "instruction": {
                    "type": "string",
                    "description": "A short, rule-based navigation instruction describing the next step for System 2, in the style of R2R instructions (e.g., 'Exit the bedroom and turn left. Walk straight passing the gray couch and stop near the rug.'). Use simple imperative sentences that mention actions and concrete landmarks; avoid chatty language or self-references."
                },
                "change_instruction": {
                    "type": "boolean",
                    "description": "True if the instruction should be changed, False to keep the current one."
                },
                # "discrete_actions": {
                #     "type": "array",
                #     "items": {
                #         "type": "string",
                #         "enum": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP", "LOOK_UP", "LOOK_DOWN"]
                #     },
                #     "description": "Optional list of discrete actions to execute directly, bypassing the local planner. Use this for precise short-term control."
                # }
            },
            "required": ["thought", "status", "instruction", "change_instruction"]
        }
    }
}

ACTION_NAME_TO_ID = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "LOOK_UP": 4,
    "LOOK_DOWN": 5,
}

def redact_image_urls(obj):
    """
    Recursively traverse a list or dict. 
    If a dict key is "image_url", set its value to '<image>'.
    Modifies the object in place.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "image_url":
                obj[k] = '<image>'
            else:
                redact_image_urls(v)
    elif isinstance(obj, list):
        for item in obj:
            redact_image_urls(item)
    # recursively traverse the object
    return obj

class VLMNavigator:
    """
    Isolates all VLM-related logic:
    - System prompt definition
    - Message construction (including image input)
    - Model call
    - Response parsing and compact history management
    """

    def __init__(
        self, 
        user_goal: str, 
        api_key: str = None, 
        base_url: str = None, 
        model_name: str = None
    ):
        self.api_key = api_key or DEFAULT_VLLM_API_KEY
        self.base_url = base_url or DEFAULT_VLLM_API_URL
        self.model_name = model_name or DEFAULT_MODEL_NAME
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.user_goal = user_goal
        self.history: List[Dict] = []
        # Human-readable, cumulative prompt/response snapshots for quick inspection
        self.human_log_history: List[str] = []
        self.last_instruction = None
        self.dump_dir: Optional[str] = None
        self.dump_freq: int = 1
        self.dump_episode_id: Optional[int] = None
        self._init_history()
        logger.info(
            f"[INIT] VLMNavigator -> model={self.model_name}, goal='{self.user_goal}'"
        )

    def _init_history(self):
        system_prompt = f"""You are an advanced autonomous robot agent. 
The user has given you a high-level goal: "{self.user_goal}".

Architecture context:
- System 1: low-level controller that executes the local navigation actions from System 2.
- System 2: local navigation planner that follows short, concrete text instructions (e.g., "Turn left 1 meter, then go to the door").
- System 3 (you): VLM that observes vision, maintains intent, and issues the next concise instruction to System 2.

You have access to a robot's visual feed. You control a local navigation system (System 2) that takes short, simple text descriptions of where to go next. Reaching within 3 meters of the final goal counts as success. If the goal is a specific target (e.g., "the red door", "the table with a laptop"), treat success as being clearly at that target and within roughly 3 meters of it. You will be told the running counters for System 2 and System 3 calls; use them when pacing your decisions.

Status Definitions:
- EXPLORE: You are exploring the environment to collect useful visual cues. Write down the information in the thought field.
- NAVIGATE: You see the next subgoal or target and are moving towards it.
- SEARCH: You cannot see the next subgoal/target, so you are looking around or exploring to find it. Use this whenever you are mainly rotating or probing the environment to locate the goal or key landmarks.
- VERIFY: You believe you have reached the final goal, but you are performing a final check (e.g., looking around) to confirm.
- DONE: You have verified that the goal is accomplished.
- ERROR: You are stuck, lost, or unable to make progress for a significant number of steps (e.g., ~5 System 3 calls with no visual change or progress), and cannot recover.

Strategic Guidelines:
1. Search First & Local Exploration: If you lack a clear understanding of the environment or the goal location, perform local exploration first to record useful visual cues (e.g., landmarks, layout) before committing to a long path. Do not assume the path is forward if uncertain. Issue instructions to look around (e.g., "Turn left and look around") to orient yourself.
2. Verify Targets: Ensure the target you are heading towards is actually the correct one. If the goal is "door" but you are facing a staircase, check your surroundings first.
3. Reflect on Progress: In your "thought", explicitly evaluate if your previous actions brought you closer to the high-level goal. If not, adjust your strategy (e.g., from moving to searching).
4. Subgoals & Phrasing: Break the user goal into immediate, concrete subgoals. Use these as the instruction for System 2. IMPORTANT: Avoid complex spatial constraints like "keeping X on your left". System 2 often interprets "left" or "right" in such phrases as immediate turn commands. Instead, specify the target or direction directly (e.g., "Walk forward past the table", "Go to the white door").
5. Anti-Premature Success (critical): It is very unlikely that you are already at the destination at the very beginning, or after only rotating in place without translating. You MUST NOT enter VERIFICATION/DONE unless you have strong positive visual evidence of the goal AND evidence that you are physically close (roughly within 3 meters). If you have not passed any clear milestones or have not made meaningful forward progress, assume you are NOT there yet and continue SEARCHING/NAVIGATING.
6. Termination Condition First (critical): Before entering VERIFICATION, explicitly state in your "thought" a concrete termination condition for THIS goal: what visual cues must be true (e.g., "I am under the arch", "I am beside the red door") AND what prior path constraints must have been met (e.g. "I have already passed the pool"). The termination condition is not just "seeing the target" but "seeing the target AFTER completing the sequence". Then in VERIFICATION, actively check those cues. If the cues are not satisfied, do NOT output DONE—issue the next corrective instruction to satisfy them.
7. Verification Stage (efficient): When you believe you have reached the goal, DO NOT immediately output DONE. Switch to "VERIFICATION" status and perform a targeted check. CRITICAL: In the instruction field during verification, DO NOT explain what you are doing (e.g., "verifying the goal"). System 2 is not trained for abstract language. Just give the concrete movement command (e.g., "Turn around", "Walk to the table").
8. Instruction Style (System 2 friendly): Write instructions in a rule-based navigation style similar to R2R instructions. Reuse the same language/landmarks from the user's original goal as much as possible if you know what to do. Prefer short imperative sentences that mention actions and concrete landmarks: "Exit the room and turn right.", "Walk straight past the table and stop near the chair." Avoid chatty language, self-references, or references to System 2/3; only describe what the robot should do.
9. Sequential Execution: User goals are often strict sequences (e.g., "Go past X, walk between Y and Z, then stop at W"). You MUST execute these stages in order. Do not identify the final goal (W) until you have visually confirmed passing the intermediate landmarks (X, Y, Z). If you see W but haven't passed Y/Z, it is likely the wrong location or you are approaching from the wrong side.

10. Camera Field of View (HFOV=79°): The robot's camera has a relatively narrow field of view (79°). This means objects may disappear from the side edges sooner than expected when you are passing them. To "pass" an object or go "through" a doorway, you typically need to move further forward than you might intuitively think to ensure you have actually cleared it. If you are unsure whether you have passed a landmark, verify by looking around or moving further forward.

Your Loop:
1. Analyze the visual observations and reflect on the milestones and your progress.
2. Decide the next status (EXPLORE, SEARCH, NAVIGATE, VERIFY, DONE, ERROR) and instruction.
3. Use the tool to output your decision.
"""
        self.history.append({"role": "system", "content": system_prompt})

    def build_user_message(
        self,
        frames_b64: List[str],
        query: str = "What should I do next?",
        sys1_steps: int = 0,
        sys2_calls: int = 0,
        sys3_calls: int = 0,
        current_instruction: Optional[str] = None,
        subepisode_id: Optional[int] = None,
    ) -> Dict:
        """Prepare the multimodal user message for the VLM."""
        # frames_b64 is a list of base64-encoded JPEG images from the current sub-episode,
        # ordered from oldest to newest. The last frame is always the latest observation.
        if not frames_b64:
            return {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "No visual frames are available. Please respond conservatively.",
                    },
                    {"type": "text", "text": query},
                ],
            }

        num_frames = len(frames_b64)
        content: List[Dict[str, Any]] = []

        # High-level counters and context
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

        # Explicitly state the current instruction (or lack of one)
        if current_instruction:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Current navigation instruction being executed by System 2 "
                        f"is:\n\"{current_instruction}\""
                    ),
                }
            )
        else:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "No navigation instruction has been issued yet. "
                        "You must propose the FIRST short navigation instruction for System 2."
                    ),
                }
            )

        # Sub-episode context
        if subepisode_id is not None:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"You are currently in sub-episode #{subepisode_id}. "
                        f"This sub-episode contains {num_frames} visual frames, "
                        "from oldest to newest, describing what happened since the current instruction was issued."
                    ),
                }
            )

        # Attach all frames in temporal order
        for idx, frame_b64 in enumerate(frames_b64):
            if idx == num_frames - 1:
                desc = "latest frame (most recent observation in this sub-episode)"
            else:
                desc = f"earlier frame {idx + 1} of {num_frames} in this sub-episode"
            content.append(
                {
                    "type": "text",
                    "text": f"Visual {idx + 1}: {desc}.",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                    },
                }
            )

        # Final query/instruction to the model
        content.append({"type": "text", "text": query})

        return {
            "role": "user",
            "content": content,
        }

    def _parse_response(self, tool_call) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from the tool call arguments.
        Returns a dict with keys: thought, status, instruction, note, change_instruction.
        """
        try:
            arguments = tool_call.function.arguments
            plan = json.loads(arguments)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool arguments: {arguments}")
            return None

        # Safe parsing of optional boolean field "change_instruction"
        # Tool use should enforce boolean, but good to be safe if model hallucinates string
        raw_change = plan.get("change_instruction", True)
        if isinstance(raw_change, str):
            raw_lower = raw_change.strip().lower()
            change_instruction = raw_lower in ("true", "1", "yes", "y")
        else:
            change_instruction = bool(raw_change)

        return {
            "thought": plan.get("thought", ""),
            "status": plan.get("status", "NAVIGATING"),
            "instruction": plan.get("instruction", ""),
            "change_instruction": change_instruction,
            "discrete_actions": plan.get("discrete_actions", []),
        }

    def plan_next_step(
        self,
        img_b64: str,
        sys1_steps: Optional[int] = None,
        sys2_calls: Optional[int] = None,
        sys3_calls: Optional[int] = None,
        current_instruction: Optional[str] = None,
        subepisode_id: Optional[int] = None,
        history_imgs: Optional[List[str]] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Main VLM entrypoint:
        - builds messages
        - calls the model with tools
        - parses and logs the plan
        - appends a compact textual summary to history
        """
        if not img_b64:
            return None

        # Keep the user query minimal; rely on message history for context
        s2_calls = sys2_calls if sys2_calls is not None else 0
        s3_calls = sys3_calls if sys3_calls is not None else 0
        s1_steps = sys1_steps if sys1_steps is not None else 0
        query = (
            "Please reason and decide whether to KEEP the current navigation instruction or CHANGE it, "
            "and use the tool to output your decision."
        )

        # Decide which frames to send: the provided history (sub-episode) or just the latest image.
        if history_imgs:
            frames_b64 = history_imgs
            # Ensure the latest image is at the end of the sequence.
            if frames_b64[-1] != img_b64:
                frames_b64 = list(frames_b64) + [img_b64]
        else:
            frames_b64 = [img_b64]

        user_msg = self.build_user_message(
            frames_b64,
            query=query,
            sys1_steps=s1_steps,
            sys2_calls=s2_calls,
            sys3_calls=s3_calls,
            current_instruction=current_instruction,
            subepisode_id=subepisode_id,
        )
        messages = self.history + [user_msg]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2000,
                tools=[STEP_OUTPUT_TOOL],
                tool_choice={"type": "function", "function": {"name": "output_navigation_plan"}},
                extra_body={
                    "extra_body": {
                        "google": {
                            "thinking_config": {
                                "thinking_budget": -1,
                                "include_thoughts": True
                            }
                        }
                    }
                }
            )
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            return None

        response_message = completion.choices[0].message
        
        # Log usage
        if hasattr(completion, "usage"):
            logger.info(f"[Sys3] Token Usage: {completion.usage}")
            
        # Capture any natural language reasoning/content that occurred before the tool call
        # Some models output "thinking" traces in the content field even when calling tools.
        content_reasoning = response_message.content or ""

        # Attempt to capture 'reasoning_content' from Gemini/OpenAI-compatible thinking models
        # Note: This field might be in `reasoning_content` (OpenAI style) or `extra_fields`.
        if hasattr(response_message, "reasoning_content") and response_message.reasoning_content:
             content_reasoning += f"\n\n[Reasoning Trace]:\n{response_message.reasoning_content}"
             
        # Parse out <thought>...</thought> tags if present in the content
        if "<thought>" in content_reasoning and "</thought>" in content_reasoning:
            try:
                start_idx = content_reasoning.find("<thought>")
                end_idx = content_reasoning.find("</thought>")
                if start_idx != -1 and end_idx != -1:
                    thought_content = content_reasoning[start_idx + len("<thought>"):end_idx]
                    content_reasoning = content_reasoning.replace(f"<thought>{thought_content}</thought>", "").strip()
                    content_reasoning += f"\n\n[Reasoning Trace]:\n{thought_content}"
            except Exception as e:
                logger.warning(f"Failed to parse thought tags: {e}")

        tool_calls = response_message.tool_calls
        
        if not tool_calls:
            logger.error("Model did not call the required tool.")
            return None
            
        # We expect exactly one tool call due to tool_choice enforcement
        tool_call = tool_calls[0]
        response_text = tool_call.function.arguments # For logging purposes
        logger.debug(f"[Sys3] Raw VLM Tool Args: {response_text}")
        if content_reasoning:
            logger.debug(f"[Sys3] Raw VLM Content (Reasoning): {content_reasoning}")

        plan = self._parse_response(tool_call)
        if not plan:
            return None

        # Merge content reasoning with structured thought
        tool_thought = plan.get("thought", "")
        combined_thought_parts = []
        if content_reasoning.strip():
            combined_thought_parts.append("[Reasoning Trace]:\n" + content_reasoning.strip())
        if tool_thought.strip():
            combined_thought_parts.append("[Thought]:\n" + tool_thought.strip())
        
        plan["thought"] = "\n\n".join(combined_thought_parts)

        thought = plan["thought"]
        status = plan["status"]
        instruction = plan["instruction"]
        change_instruction = plan["change_instruction"]

        # TODO: step should include sys2 and sys1 and sys3 steps.
        step_idx = self.history.__len__() // 2  # each assistant add bumps this

        # Persist the full messages for debugging / inspection
        self._dump_messages(
            messages,
            response_text,
            plan,
            step_idx,
            sys1_steps=sys1_steps,
            sys2_calls=sys2_calls,
            sys3_calls=sys3_calls,
        )

        logger.info(
            f"\n[Sys3] Step {step_idx}\n"
            f"  Status      : {status}\n"
            f"  Instruction : {instruction}\n"
            f"  Thought     : {thought}\n"
        )

        # Maintain a compact text-only summary to keep context small
        # NOTE: We DO NOT add the 'thought' to the history, only the instruction/status.
        if change_instruction:
            self.history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": (
                                f"[Step {step_idx} | {time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                f"Sys2 calls: {s2_calls}. Sys3 calls: {s3_calls}. "
                                f"Status: {status}. Instruction: {instruction}. "
                                f"Thought: {thought}."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            },
                        },
                    ],
                }
            )
            # Update history trackers
            self.last_instruction = instruction

        return plan

    def _dump_messages(
        self,
        messages: List[Dict],
        response_text: str,
        plan: Dict,
        step_idx: int,
        sys1_steps: Optional[int] = None,
        sys2_calls: Optional[int] = None,
        sys3_calls: Optional[int] = None,
    ) -> None:
        """
        Dump the full messages to disk.
        - latest.json : always overwritten with the latest request
        - <episode>_step_XXXX.json : saved every dump_freq steps (default every step)
        """
        try:
            os.makedirs(self.dump_dir, exist_ok=True)   
            no_image_messages = redact_image_urls(copy.deepcopy(messages))
            payload = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "step": step_idx    ,
                "response_text": response_text,
                "plan": plan,
                "sys1_steps": sys1_steps,
                "sys2_calls": sys2_calls,
                "sys3_calls": sys3_calls,
                "messages": no_image_messages,
            }
            episode_str = (
                f"{self.dump_episode_id:04d}"
                if self.dump_episode_id is not None
                else "0000"
            )
            step_path = os.path.join(self.dump_dir, f"{episode_str}.json")
            # print("in _dump_messages, step_path:", step_path)
            with open(step_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            # Also persist a human-readable version for quick inspection
            sep = "=" * 72
            sub_sep = "-" * 56
            human_lines = [
                sep,
                f"System 3 Prompt Log | Step {step_idx} | {payload['timestamp']}",
                sep,
                f"Sys1 steps : {sys1_steps}",
                f"Sys2 calls : {sys2_calls}",
                f"Sys3 calls : {sys3_calls}",
                "",
                "Messages (model input)",
                sub_sep,
            ]

            for idx, msg in enumerate(no_image_messages):
                role = msg.get("role", "unknown")
                human_lines.append(f"[Message {idx}] role={role}")
                content = msg.get("content", "")

                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            human_lines.append(f"    - {part}")
                            continue

                        ptype = part.get("type", "text")
                        if ptype == "text":
                            human_lines.append(f"    - text: {part.get('text', '')}")
                        elif ptype == "image_url":
                            image_val = part.get("image_url", "<image>")
                            if isinstance(image_val, dict):
                                image_val = image_val.get("url", "<image>")
                            human_lines.append(f"    - image: {image_val}")
                        else:
                            human_lines.append(f"    - {ptype}: {part}")
                else:
                    human_lines.append(f"    - {content}")

            human_lines.extend(
                [
                    "",
                    "Model response (raw)",
                    sub_sep,
                    response_text,
                    "",
                    "Parsed plan",
                    sub_sep,
                    json.dumps(plan, ensure_ascii=False, indent=2),
                    "",
                    sep,
                ]
            )

            # Keep a cumulative stack so previous steps stay visible
            self.human_log_history.append("\n".join(human_lines))
            human_path = os.path.splitext(step_path)[0] + ".txt"
            with open(human_path, "w", encoding="utf-8") as f_txt:
                f_txt.write("\n\n".join(self.human_log_history))
        except Exception as e:
            logger.error(f"[Sys3] Failed to write prompt log: {e}")

def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@Agent.register('system3')
class System3Agent(InternVLAN1Agent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        self.navigator: Optional[VLMNavigator] = None
        self.current_instruction = None
        self.prompt_dump_dir: Optional[str] = None
        self.prompt_dump_freq: int = 1
        self.prompt_dump_episode_id: Optional[int] = None
        self.sys3_call_count: int = 0
        self.sys2_call_count: int = 0
        # Latest System 3 outputs for visualization/debugging
        self.last_sys3_status: Optional[str] = None
        self.last_sys3_thought: Optional[str] = None
        self.last_sys3_updated_step: Optional[int] = None
        # Throttle System3 invocations to avoid over-calling VLM.
        # Defaults: allow a new Sys3 call every 2 macro steps.
        self.sys3_interval_steps: int = config.model_settings.get("sys3_interval_steps", 8)
        self.last_sys3_step: int = -self.sys3_interval_steps
        # If True, force a System 3 call on the next step, regardless of interval.
        # Used when System 2 reports that its current local plan is DONE (STOP).
        self.force_sys3_next: bool = False
        # Sub-episode management for System 3
        self.subepisode_id: int = 0
        self.subepisode_frames_b64: List[str] = []
        self.max_subepisode_frames: int = config.model_settings.get("sys3_max_subepisode_frames", 8)
        
        # Load System 3 config from config, fallback to env
        model_settings = config.model_settings
        
        # Priority: Env Vars (Launch Config) > Config File > Default
        self.vlm_api_key = os.environ.get("VLLM_API_KEY") or model_settings.get("vlm_api_key") or DEFAULT_VLLM_API_KEY
        self.vlm_base_url = os.environ.get("VLLM_API_URL") or model_settings.get("vlm_api_url") or DEFAULT_VLLM_API_URL
        self.vlm_model_name = os.environ.get("MODEL_NAME") or model_settings.get("vlm_model_name") or DEFAULT_MODEL_NAME
        logger.info(f"System 3 Agent initialized with: "
                    f"vlm_api_key={self.vlm_api_key}, "
                    f"vlm_base_url={self.vlm_base_url}, "
                    f"vlm_model_name={self.vlm_model_name}")
        
    def reset(self, reset_index=None):
        super().reset(reset_index)
        self.navigator = None
        self.current_instruction = None
        self.subepisode_id = 0
        self.subepisode_frames_b64 = []
        self.force_sys3_next = False
        self.sys3_call_count = 0
        self.sys2_call_count = 0
        self.last_sys3_status = None
        self.last_sys3_thought = None
        self.last_sys3_updated_step = None

    def update_instruction(self, obs: List[Dict[str, Any]]):
        """
        Call System 3 (VLM) to update the instruction.
        Returns: (instruction, status, thought)
        """
        logger.info(f"[Sys3] Update instruction. Sys2 calls: {self.sys2_call_count}. Sys3 calls: {self.sys3_call_count}.")
        if not isinstance(obs, list):
            obs = [obs]
        current_obs = obs[0]
        rgb = current_obs['rgb']
        image = Image.fromarray(rgb.astype('uint8'), 'RGB')
        img_b64 = pil_to_base64(image)
        
        self.subepisode_frames_b64.append(img_b64)
        if len(self.subepisode_frames_b64) > self.max_subepisode_frames:
            # Uniformly select max_subepisode_frames frames over the available range
            total_frames = len(self.subepisode_frames_b64)
            if total_frames > self.max_subepisode_frames:
                # Uniform spacing, endpoints included
                indices = np.linspace(0, total_frames - 1, num=self.max_subepisode_frames, dtype=int)
                subepisode_frames_b64 = [self.subepisode_frames_b64[i] for i in indices]
        else:
            subepisode_frames_b64 = self.subepisode_frames_b64
            
        assert self.navigator
        
        # interval_ok = (self.episode_step - self.last_sys3_step) >= self.sys3_interval_steps
        # should_call_sys3 = self.force_sys3_next or interval_ok or self.episode_step == 0
        # if should_call_sys3:

        self.sys3_call_count += 1
        plan = self.navigator.plan_next_step(
            img_b64,
            sys1_steps=self.episode_step,
            sys2_calls=self.sys2_call_count,
            sys3_calls=self.sys3_call_count,
            current_instruction=self.current_instruction,
            subepisode_id=self.subepisode_id,
            history_imgs=subepisode_frames_b64,
        )
        self.last_sys3_step = self.episode_step
        self.force_sys3_next = False
        
        if plan:
            status = plan.get("status")
            change_instruction = plan.get("change_instruction", True)
            new_instruction = plan.get("instruction")
            discrete_actions = plan.get("discrete_actions", [])
            
            self.last_sys3_status = status
            self.last_sys3_thought = plan.get("thought")
            self.last_sys3_updated_step = self.episode_step
            
            if status == "DONE":
                return None, "DONE", plan.get("thought")
            
            if status == "ERROR":
                logger.info("[Sys3] Status is ERROR. Treating as DONE (termination).")
                return None, "ERROR", plan.get("thought")
            
            if change_instruction and new_instruction != self.current_instruction:
                self.current_instruction = new_instruction
                self.subepisode_id += 1
                self.subepisode_frames_b64 = [img_b64] # Start new sub-episode
                        
        return self.current_instruction, self.last_sys3_status, self.last_sys3_thought

    def predict_goal(self, inputs):
        """
        Call System 2 to predict the goal.
        Args:
            inputs: Dict of inputs (pixel_values, etc.) prepared by processor
        Returns:
            output_ids
        """
        model = self.policy.model
        with torch.no_grad():
             output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, top_k=None, top_p=None, temperature=None)
        return output_ids

    def generate_local_actions(self, output_ids, pixel_values, image_grid_thw, images_dp, depths_dp):
        """
        Call System 1 to generate local actions.
        """
        model = self.policy.model
        with torch.no_grad():
            traj_latents = model.generate_latents(output_ids, pixel_values, image_grid_thw)
            dp_actions = model.generate_traj(
                traj_latents, images_dp, depths_dp, use_async=True
            )
        return dp_actions

    def set_goal(self, goal: str):
        """Initialize VLM Navigator with the high-level goal"""
        self.navigator = VLMNavigator(
            user_goal=goal,
            api_key=self.vlm_api_key,
            base_url=self.vlm_base_url,
            model_name=self.vlm_model_name
        )
        if self.prompt_dump_dir:
            self.navigator.dump_dir = self.prompt_dump_dir
            self.navigator.dump_freq = self.prompt_dump_freq
            self.navigator.dump_episode_id = self.prompt_dump_episode_id
        # The high-level goal doubles as the initial instruction until System 3 refines it.
        self.current_instruction = goal
        # Start first sub-episode.
        self.subepisode_id = 0
        self.subepisode_frames_b64 = []

    def set_prompt_dump(self, dump_dir: str, freq: int = 1, episode_id: Optional[int] = None):
        """Configure where/how often to dump prompts."""
        assert dump_dir is not None, "dump_dir cannot be None"
        self.prompt_dump_dir = dump_dir
        self.prompt_dump_freq = max(1, freq)
        self.prompt_dump_episode_id = episode_id
        if self.navigator:
            self.navigator.dump_dir = dump_dir
            self.navigator.dump_freq = self.prompt_dump_freq
            self.navigator.dump_episode_id = episode_id
        logger.info(
            f"[Sys3] Prompt dump configured: dump_dir={dump_dir}, freq={self.prompt_dump_freq}, "
            f"episode_id={episode_id}"
        )
