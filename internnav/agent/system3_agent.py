import os
import logging
import requests
import json
import time
import base64
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
    format="%(asctime)s | AGENT | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("System3Agent")

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
        self._init_history()
        logger.info(
            f"[INIT] VLMNavigator -> model={self.model_name}, goal='{self.user_goal}'"
        )

    def _init_history(self):
        system_prompt = f"""You are an advanced autonomous robot agent. 
The user has given you a high-level goal: "{self.user_goal}".

You have access to a robot's visual feed. You control a local navigation system (System 2) that takes short, simple text descriptions of where to go next.

Your Loop:
1. Analyze the current image.
2. Determine if the goal is reached.
3. If not, plan the immediate next step.
4. Output a navigation instruction for the local system.

Output Format:
You must output a JSON object:
{{
  "thought": "Your reasoning here...",
  "status": "NAVIGATING" | "SEARCHING" | "DONE",
  "instruction": "Short phrase for System 2 (e.g. 'Go to the door')"
}}
"""
        self.history.append({"role": "system", "content": system_prompt})

    def build_user_message(self, img_b64: str, query: str = "What should I do next?") -> Dict:
        """Prepare the multimodal user message for the VLM."""
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    },
                },
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
                logger.error("Could not find JSON in response")
                return None

            json_str = response_text[start:end]
            plan = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error("Failed to parse model JSON")
            return None

        return {
            "thought": plan.get("thought", ""),
            "status": plan.get("status", "NAVIGATING"),
            "instruction": plan.get("instruction", ""),
        }

    def plan_next_step(self, img_b64: str) -> Optional[Dict[str, str]]:
        """
        Main VLM entrypoint:
        - builds messages
        - calls the model
        - parses and logs the plan
        - appends a compact textual summary to history
        """
        if not img_b64:
            return None

        user_msg = self.build_user_message(img_b64)
        messages = self.history + [user_msg]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=300,
                temperature=0.1,
            )
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            return None

        response_text = completion.choices[0].message.content
        logger.info("[VLM] Raw Model Response received")

        plan = self._parse_response(response_text)
        if not plan:
            return None

        thought = plan["thought"]
        status = plan["status"]
        instruction = plan["instruction"]

        logger.info(
            f"[VLM] Parsed plan -> status={status}, instruction='{instruction}', "
            f"thought_preview='{thought[:120]}{'...' if len(thought) > 120 else ''}'"
        )

        # Maintain a compact text-only summary to keep context small
        if instruction or thought:
            self.history.append(
                {
                    "role": "assistant",
                    "content": f"Action taken: {instruction}. Status: {status}. Reason: {thought}",
                }
            )

        return plan

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
        
        # Load System 3 config from env
        self.vlm_api_key = os.environ.get("VLLM_API_KEY", DEFAULT_VLLM_API_KEY)
        self.vlm_base_url = os.environ.get("VLLM_API_URL", DEFAULT_VLLM_API_URL)
        self.vlm_model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
        
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
        self.current_instruction = goal # Default to original until VLM updates it

    def step(self, obs: List[Dict[str, Any]]):
        # Handle single observation (batch size 1)
        current_obs = obs[0]
        rgb = current_obs['rgb'] # numpy array (H, W, 3)
        
        # 1. System 3 Logic: Get instruction from VLM
        if self.navigator:
            # Convert numpy array to PIL Image
            image = Image.fromarray(rgb.astype('uint8'), 'RGB')
            img_b64 = pil_to_base64(image)
            
            # Plan next step
            plan = self.navigator.plan_next_step(img_b64)
            
            if plan:
                if plan.get("status") == "DONE":
                     logger.info("System 3 decided goal is reached.")
                     # We can signal stop by returning a stop action immediately?
                     # InternVLAN1Agent action 0 is STOP.
                     return [{'action': [0], 'ideal_flag': True}]
                     
                if plan.get("instruction"):
                    self.current_instruction = plan.get("instruction")
        
        # 2. Update instruction for System 2
        if self.current_instruction:
            current_obs['instruction'] = self.current_instruction
        
        # 3. System 2 Logic: Call parent step
        return super().step(obs)
