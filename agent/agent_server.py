import os
import time
import json
import base64
import requests
import logging
from openai import OpenAI
from typing import List, Dict, Optional

# Configuration
# VLLM usually serves at port 8000 and is OpenAI compatible
VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://localhost:8080/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY") # vLLM often uses "EMPTY" if no auth
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct") # Adjust to your served model

# System 2 Server (The robot controller)
ROBOT_API_URL = os.environ.get("ROBOT_API_URL", "http://localhost:5802")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | AGENT | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("System3Agent")

class RobotClient:
    """Interface to interact with the System 2 Robot Server"""
    def __init__(self, base_url):
        self.base_url = base_url
        logger.info(f"[INIT] RobotClient -> base_url={self.base_url}")

    def get_latest_observation(self) -> Optional[str]:
        """Fetch latest base64 image from robot"""
        try:
            resp = requests.get(f"{self.base_url}/ui_data", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("image") # base64 string
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
        return None

    def update_instruction(self, instruction: str):
        """Send new navigation command to System 2"""
        try:
            resp = requests.post(
                f"{self.base_url}/set_instruction", 
                json={"instruction": instruction},
                timeout=2
            )
            if resp.status_code == 200:
                logger.info(f"Updated System 2 instruction: '{instruction}'")
                return True
        except Exception as e:
            logger.error(f"Failed to update instruction: {e}")
        return False


class VLMNavigator:
    """
    Isolates all VLM-related logic:
    - System prompt definition
    - Message construction (including image input)
    - Model call
    - Response parsing and compact history management
    """

    def __init__(self, user_goal: str, api_key: Optional[str] = None, base_url: Optional[str] = None, model_name: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key or VLLM_API_KEY, 
            base_url=base_url or VLLM_API_URL
        )
        self.user_goal = user_goal
        self.model_name = model_name or MODEL_NAME
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

class System3Agent:
    def __init__(self, user_goal: str):
        self.robot = RobotClient(ROBOT_API_URL)
        self.user_goal = user_goal
        self.navigator = VLMNavigator(user_goal)
        self.step_count = 0
        logger.info(
            f"[INIT] System3Agent -> goal='{self.user_goal}', "
            f"robot_url={ROBOT_API_URL}, vlm_url={VLLM_API_URL}"
        )

    def step(self):
        # 1. Get Observation
        self.step_count += 1
        logger.info(
            f"[STEP {self.step_count}] ==============================================="
        )

        img_b64 = self.robot.get_latest_observation()
        if not img_b64:
            logger.warning(f"[STEP {self.step_count}] No image received from robot. Waiting...")
            return

        logger.info(f"[STEP {self.step_count}] Captured image. Thinking...")

        # 2. Ask VLM what to do next (prompt + parsing encapsulated in VLMNavigator)
        plan = self.navigator.plan_next_step(img_b64)
        if not plan:
            logger.warning(f"[STEP {self.step_count}] VLM did not return a valid plan.")
            return

        thought = plan.get("thought", "")
        instruction = plan.get("instruction", "")
        status = plan.get("status", "NAVIGATING")

        logger.info(
            f"[STEP {self.step_count}] Plan -> status={status}, "
            f"instruction='{instruction}', "
            f"thought_preview='{thought[:80]}{'...' if len(thought) > 80 else ''}'"
        )

        if status == "DONE":
            logger.info(f"[STEP {self.step_count}] Goal Reached! Stopping.")
            self.robot.update_instruction("STOP")
            return "DONE"

        if instruction:
            self.robot.update_instruction(instruction)

    def run_loop(self, interval=1.0):
        logger.info(
            f"[LOOP] Starting Agent Loop -> goal='{self.user_goal}', interval={interval:.1f}s"
        )
        while True:
            result = self.step()
            if result == "DONE":
                break
            time.sleep(interval)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, required=True, help="High level user goal")
    parser.add_argument("--interval", type=float, default=1.0, help="Loop interval in seconds")
    args = parser.parse_args()

    agent = System3Agent(args.goal)
    try:
        agent.run_loop(args.interval)
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
