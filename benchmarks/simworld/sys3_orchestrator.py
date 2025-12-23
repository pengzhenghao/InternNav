"""
Sys3-orchestrated navigation loop in SimWorld.

This is intentionally minimal:
- Sys3 (VLM) observes the SimWorld camera frames and emits short instructions.
- A tiny geometric executor turns "go to (x, y)" into rotate/step actions.

Once this works end-to-end, you can swap the executor for SimWorld's `LocalPlanner`
or your own dual-system controller.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from typing import Optional, Tuple

import numpy as np


def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr[:, :, ::-1].copy()


_XY_RE = re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)")


def parse_xy(text: str) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    m = _XY_RE.search(text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _wrap_deg(d: float) -> float:
    # map to [-180, 180]
    d = (d + 180.0) % 360.0 - 180.0
    return d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=os.environ.get("SIMWORLD_IP", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIMWORLD_PORT", "9000")))
    parser.add_argument("--timeout_s", type=float, default=float(os.environ.get("SIMWORLD_CONNECT_TIMEOUT_S", "10")))

    parser.add_argument("--vlm_base_url", default=os.environ.get("VLLM_API_URL", "http://localhost:8080/v1"))
    parser.add_argument("--vlm_api_key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--vlm_model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct"))

    parser.add_argument("--goal", default="Go to (1700, -1700).")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.2)
    args = parser.parse_args()

    from simworld.agent.humanoid import Humanoid
    from simworld.communicator.communicator import Communicator
    from simworld.communicator.unrealcv import UnrealCV
    from simworld.utils.vector import Vector

    from internnav.agent.sys3_only_agent import Sys3OnlyAgent, Sys3OnlyAgentCfg

    try:
        ucv = UnrealCV(ip=args.ip, port=args.port, resolution=(640, 360), connect_timeout_s=args.timeout_s)
    except TimeoutError as e:
        print(f"[sys3_orchestrator] Could not connect to UE UnrealCV server: {e}")
        return 2

    comm = Communicator(ucv)

    agent = Humanoid(position=Vector(0, 0), direction=Vector(1, 0))
    comm.spawn_agent(agent, name=None, model_path="/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C", type="humanoid")

    sys3 = Sys3OnlyAgent(
        Sys3OnlyAgentCfg(
            model_name=args.vlm_model_name,
            base_url=args.vlm_base_url,
            api_key=args.vlm_api_key,
            max_subepisode_frames=8,
        )
    )
    sys3.set_goal(args.goal)

    target_xy = parse_xy(args.goal)
    print(f"[sys3_orchestrator] goal={args.goal!r} parsed_target={target_xy}")

    for step_idx in range(args.max_steps):
        # Grab latest camera frame
        img_bgr = comm.get_camera_observation(agent.camera_id, "lit", mode="direct")
        img_rgb = _bgr_to_rgb(img_bgr)

        # Feed Sys3
        sys3.sys1_steps = step_idx
        instr, status, thought = sys3.update_instruction({"rgb": img_rgb})
        if instr:
            target_xy = parse_xy(instr) or target_xy

        print(f"[sys3] step={step_idx:03d} status={status} instr={instr!r}")
        if thought:
            print(f"[sys3] thought={thought[:240]}")

        if status in ("DONE", "ERROR") or instr is None:
            print(f"[sys3_orchestrator] stopping: status={status}")
            break

        # --- Tiny executor: rotate + step toward the latest (x, y) target, if any ---
        if target_xy is not None:
            info = comm.get_position_and_direction(humanoid_ids=[agent.id])
            pos, yaw_deg = info[("humanoid", agent.id)]
            agent.position = pos
            agent.direction = yaw_deg  # setter updates direction vector too

            dx = target_xy[0] - agent.position.x
            dy = target_xy[1] - agent.position.y
            dist = math.sqrt(dx * dx + dy * dy)
            target_yaw = math.degrees(math.atan2(dy, dx))
            yaw_err = _wrap_deg(target_yaw - yaw_deg)

            if dist < 120.0:
                print(f"[executor] near target (dist={dist:.1f}); stopping")
                comm.humanoid_stop(agent.id)
                time.sleep(args.dt)
                continue

            if abs(yaw_err) > 12.0:
                turn_dir = "right" if yaw_err > 0 else "left"
                comm.humanoid_rotate(agent.id, min(30.0, abs(yaw_err)), turn_dir)
            else:
                comm.humanoid_step_forward(agent.id, duration=0.4, direction=0)
        else:
            # No parsed target yet; do a gentle explore step.
            comm.humanoid_step_forward(agent.id, duration=0.25, direction=0)

        time.sleep(args.dt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


