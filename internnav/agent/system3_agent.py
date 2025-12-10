import os
import logging
import requests
import json
import time
import base64
import copy
from typing import List, Dict, Optional, Any
from io import BytesIO
from PIL import Image
from openai import OpenAI

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
        self.last_note = None
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

You have access to a robot's visual feed. You control a local navigation system (System 2) that takes short, simple text descriptions of where to go next.
Reaching within 3 meters of the final goal counts as success.
You will be told the running counters for System 2 and System 3 calls; use them when pacing your decisions.

Strategic Guidelines:
1. Search First: If you cannot clearly see your next milestone or are uncertain about your location relative to the goal, DO NOT assume the path is forward. Issue instructions to look around (e.g., "Turn left 30 degrees to search", "Look around") to orient yourself.
2. Verify Targets: Ensure the target you are heading towards is actually the correct one. If the goal is "door" but you are facing a staircase, check your surroundings first.
3. Reflect on Progress: In your "thought", explicitly evaluate if your previous actions brought you closer to the high-level goal. If not, adjust your strategy (e.g., from moving to searching).
4. Subgoals: Break the user goal into immediate, concrete subgoals. Use these as the instruction for System 2.

Your Loop:
1. Analyze the current image.
2. Determine if the goal is reached (success once within 3 meters).
3. Reflect on current progress and validity of the previous plan.
4. If not reached, plan the immediate next step (Search or Move) by selecting a specific subgoal.
5. Output a navigation instruction for the local system.

Output Format:
You must output a JSON object:
{{
  "thought": "Reflect on progress, visibility of target, and why this new step is chosen...",
  "status": "NAVIGATING" | "SEARCHING" | "DONE",
  "instruction": "Informative text describing the next step (e.g. 'Turn left to scan for the door.')",
  "note": "One-line summary of the current scenario/progress to remind yourself in the next step"
}}
"""
        self.history.append({"role": "system", "content": system_prompt})

    def build_user_message(
        self,
        frames_b64: List[str],
        query: str = "What should I do next?",
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
                        f"(do NOT change it unless truly necessary) is:\n\"{current_instruction}\""
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

        # Remind the model about the desired JSON format including change_instruction.
        content.append(
            {
                "type": "text",
                "text": (
                    "Respond ONLY with a JSON object of the form: "
                    '{"thought": "...", "status": "NAVIGATING" | "SEARCHING" | "DONE", '
                    '"instruction": "...", "note": "...", "change_instruction": true | false}. '
                    'Set "change_instruction" to false if the current instruction should be KEPT.'
                ),
            }
        )

        # Final query/instruction to the model
        content.append({"type": "text", "text": query})

        return {
            "role": "user",
            "content": content,
        }

    def _parse_response(self, response_text: str) -> Optional[Dict[str, str]]:
        """
        Extract JSON from the model output and normalise the plan structure.
        Returns a dict with keys: thought, status, instruction or None on failure.
        """
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start == -1 or end == 0:
                logger.error("Could not find JSON in response:", response_text)
                return None

            json_str = response_text[start:end]
            plan = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error("Failed to parse model JSON")
            return None

        # Safe parsing of optional boolean field "change_instruction"
        raw_change = plan.get("change_instruction", True)
        if isinstance(raw_change, str):
            raw_lower = raw_change.strip().lower()
            change_instruction = raw_lower in ("true", "1", "yes", "y")
        else:
            change_instruction = bool(raw_change)

        return {
            "thought": plan.get("thought", ""),
            "note": plan.get("note", ""),
            "status": plan.get("status", "NAVIGATING"),
            "instruction": plan.get("instruction", ""),
            "change_instruction": change_instruction,
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
        - calls the model
        - parses and logs the plan
        - appends a compact textual summary to history
        """
        if not img_b64:
            return None

        # Keep the user query minimal; rely on message history for context
        s2_calls = sys2_calls if sys2_calls is not None else 0
        s3_calls = sys3_calls if sys3_calls is not None else 0
        query = (
            "Please decide whether to KEEP the current navigation instruction or CHANGE it, "
            "and provide your JSON response."
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
                # temperature=0.1,
            )
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            return None

        response_text = completion.choices[0].message.content
        logger.debug(f"[Sys3] Raw VLM Response: {response_text}")

        plan = self._parse_response(response_text)
        if not plan:
            return None

        thought = plan["thought"]
        status = plan["status"]
        instruction = plan["instruction"]
        note = plan["note"]

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
            f"  Note        : {note}\n"
            f"  Thought     : {thought[:400]}\n"
        )


        # Maintain a compact text-only summary to keep context small
        if instruction or thought:
            self.history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": (
                                f"[Step {step_idx} | {time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                f"Status: {status}. Instruction: {instruction}. Note: {note}. Thought: {thought}. "
                                f"Sys2 calls: {s2_calls}. Sys3 calls: {s3_calls}."
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
            self.last_note = note

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

    def step(self, obs: List[Dict[str, Any]]):
        logger.info(f"[Sys3] Episode step {self.episode_step}. Sys2 calls: {self.sys2_call_count}. Sys3 calls: {self.sys3_call_count}.")
        # Handle single observation (batch size 1)
        current_obs = obs[0]
        rgb = current_obs['rgb'] # numpy array (H, W, 3)
        # Convert current frame once and append to sub-episode history
        image = Image.fromarray(rgb.astype('uint8'), 'RGB')
        img_b64 = pil_to_base64(image)
        self.subepisode_frames_b64.append(img_b64)
        if len(self.subepisode_frames_b64) > self.max_subepisode_frames:
            # Keep only the most recent N frames in this sub-episode
            self.subepisode_frames_b64 = self.subepisode_frames_b64[-self.max_subepisode_frames:]
        # 1. System 3 Logic: Only call if System 2 needs to run
        # We check self.should_infer_s2(self.mode) OR if we are looking down (which forces inference)
        # Note: 'look_down' logic is handled in super().step, but we need to know if we should update instruction now.
        assert self.navigator

        should_call_sys2 = self.should_infer_s2(self.mode) or self.look_down
        interval_ok = (self.episode_step - self.last_sys3_step) >= self.sys3_interval_steps

        # TODO: better logic for calling sys3.
        # If System 2 has just reported that its local plan is DONE (STOP),
        # we set force_sys3_next=True so that System 3 immediately issues a
        # new instruction, regardless of the usual interval throttle.
        should_call_sys3 = should_call_sys2 and (
            self.force_sys3_next or interval_ok or self.episode_step == 0
        )
        if should_call_sys3:
            # Plan next step with System 3, using sub-episode history
            self.sys3_call_count += 1
            plan = self.navigator.plan_next_step(
                img_b64,
                sys1_steps=self.episode_step,
                sys2_calls=self.sys2_call_count,
                sys3_calls=self.sys3_call_count,
                current_instruction=self.current_instruction,
                subepisode_id=self.subepisode_id,
                history_imgs=self.subepisode_frames_b64,
            )
            self.last_sys3_step = self.episode_step
            # We have just made a System 3 call, so clear the force flag.
            self.force_sys3_next = False
            
            if plan:
                status = plan.get("status")
                change_instruction = plan.get("change_instruction", True)
                new_instruction = plan.get("instruction")

                if status == "DONE":
                    logger.info(
                        "[Sys3] Termination requested by System 3 (status=DONE, change_instruction=%s). "
                        "Emitting STOP action (0) to environment.",
                        change_instruction,
                    )
                    return [{'action': [0], 'ideal_flag': True}]

                if change_instruction and new_instruction:
                    # New instruction => start a new sub-episode, but no need to hard-reset Sys2 state.
                    if new_instruction != self.current_instruction:
                        self.current_instruction = new_instruction
                        self.subepisode_id += 1
                        # Start new sub-episode buffer with the latest frame only.
                        self.subepisode_frames_b64 = [img_b64]
                        logger.info(f"[Sys3] New instruction (sub-episode {self.subepisode_id}): {self.current_instruction}")
                else:
                    # Keep current instruction; just log that System 3 chose not to change it.
                    logger.info(f"[Sys3] Keeping current instruction: {self.current_instruction}")
        
        # 2. Update instruction for System 2
        if self.current_instruction:
            current_obs['instruction'] = self.current_instruction

        if should_call_sys2:
            self.sys2_call_count += 1  # track how many times Sys2 inference is requested
        
        # 3. System 2 Logic: Call parent step.
        # NOTE: InternVLAN1Agent uses a special "look_down" mechanism triggered
        # by an internal S2 action=5. On the first call after that, it sets
        # self.look_down=True and returns a NO-OP (-1) action while preparing
        # internal state. On the *next* call with look_down=True, it uses that
        # state to produce the actual navigation action.
        #
        # To preserve that behavior, we mirror the original pattern:
        #   ret = super().step(obs)
        #   if self.look_down: ret = super().step(obs)
        #
        # Our STOP interception logic is applied only to the final 'ret'.
        ret = super().step(obs)
        if self.look_down:
            ret = super().step(obs)

        # Intercept System 2 STOP (action=0) so that it only ends the
        # current sub-episode, not the whole Habitat episode.
        # The full episode should terminate only when System 3 explicitly
        # returns status="DONE" (handled earlier in this method).
        try:
            assert len(ret) == 1, "System 2 should return a list with one element"
            if isinstance(ret, list) and ret and 'action' in ret[0]:
                act = ret[0]['action'][0] if ret[0]['action'] else None
                if act == 0:
                    logger.info(
                        "[Sys3] Intercepting System 2 STOP action (0) as sub-episode completion; "
                        "converting to NO-OP (-1). Full episode termination is controlled by System 3."
                    )
                    ret[0]['action'][0] = -1
                    ret[0]['ideal_flag'] = False
                    # Mark that System 2 has finished its current local plan so that
                    # System 3 is forced to compute a new instruction on the next step.
                    self.force_sys3_next = True
        except Exception as e:
            logger.warning(f"[Sys3] Failed to post-process System 2 action: {e}")

        return ret
