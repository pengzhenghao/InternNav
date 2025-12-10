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
logger = logging.getLogger(__name__)

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
        img_b64: str,
        query: str = "What should I do next?",
        sys2_calls: int = 0,
        sys3_calls: int = 0,
    ) -> Dict:
        """Prepare the multimodal user message for the VLM."""
        
        # TODO: can add system 2 annotated images back here.
        
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the current image."},
                {"type": "text", "text": f"System 2 calls so far: {sys2_calls}. System 3 calls so far: {sys3_calls}."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    },
                },
                {"type": "text", "text": query},
            ],
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

        return {
            "thought": plan.get("thought", ""),
            "note": plan.get("note", ""),
            "status": plan.get("status", "NAVIGATING"),
            "instruction": plan.get("instruction", ""),
        }

    def plan_next_step(
        self,
        img_b64: str,
        sys1_steps: Optional[int] = None,
        sys2_calls: Optional[int] = None,
        sys3_calls: Optional[int] = None,
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
            "Please provide a navigation instruction for the local navigation system. "
            f"System 2 calls so far: {s2_calls}. System 3 calls so far: {s3_calls}."
        )

        user_msg = self.build_user_message(img_b64, query=query, sys2_calls=s2_calls, sys3_calls=s3_calls)
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
        self.current_instruction = goal # Default to original until VLM updates it

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
        
        # 1. System 3 Logic: Only call if System 2 needs to run
        # We check self.should_infer_s2(self.mode) OR if we are looking down (which forces inference)
        # Note: 'look_down' logic is handled in super().step, but we need to know if we should update instruction now.
        assert self.navigator

        should_call_sys2 = self.should_infer_s2(self.mode) or self.look_down
        interval_ok = (self.episode_step - self.last_sys3_step) >= self.sys3_interval_steps

        # TODO: better logic for calling sys3.
        should_call_sys3 = should_call_sys2 and (interval_ok or self.episode_step == 0)
        if should_call_sys3:
            # Convert numpy array to PIL Image
            image = Image.fromarray(rgb.astype('uint8'), 'RGB')
            img_b64 = pil_to_base64(image)
            
            # Plan next step
            self.sys3_call_count += 1
            plan = self.navigator.plan_next_step(
                img_b64,
                sys1_steps=self.episode_step,
                sys2_calls=self.sys2_call_count,
                sys3_calls=self.sys3_call_count,
            )
            self.last_sys3_step = self.episode_step
            
            if plan:
                if plan.get("status") == "DONE":
                        logger.info("[Sys3] Goal reached signal.")
                        return [{'action': [0], 'ideal_flag': True}]
                        
                if plan.get("instruction"):
                    self.current_instruction = plan.get("instruction")
                    logger.info(f"[Sys3] New instruction: {self.current_instruction}")
        
        # 2. Update instruction for System 2
        if self.current_instruction:
            current_obs['instruction'] = self.current_instruction

        if should_call_sys2:
            self.sys2_call_count += 1  # track how many times Sys2 inference is requested
        
        # 3. System 2 Logic: Call parent step
        ret = super().step(obs)
        if self.look_down:
            ret = super().step(obs)
        return ret
