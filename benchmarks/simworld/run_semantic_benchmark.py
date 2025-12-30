"""
Semantic Navigation Benchmark for SimWorld.
Runs InternNav System 3 on specific semantic tasks (e.g., "Find the Grocery Store").
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import re
from typing import Optional, Tuple

import numpy as np

# Add repo root to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
simworld_path = os.path.join(repo_root, "external", "SimWorld")
import sys
if simworld_path not in sys.path:
    sys.path.append(simworld_path)

# Regex for parsing (x, y) from VLM output
_XY_RE = re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)")

def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr[:, :, ::-1].copy()

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
    parser.add_argument("--task_id", required=True, help="ID of the task to run (from tasks.json)")
    parser.add_argument("--tasks_file", default=os.path.join(current_dir, "tasks.json"))
    
    # Connection args
    parser.add_argument("--ip", default=os.environ.get("SIMWORLD_IP", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIMWORLD_PORT", "9000")))
    parser.add_argument("--timeout_s", type=float, default=10.0)

    # VLM args
    parser.add_argument("--vlm_base_url", default=os.environ.get("VLLM_API_URL", "http://localhost:8080/v1"))
    parser.add_argument("--vlm_api_key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--vlm_model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct"))

    # Config
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output_dir", default="output")
    
    args = parser.parse_args()

    # 1. Load Task
    with open(args.tasks_file, 'r') as f:
        tasks = json.load(f)
    
    task = next((t for t in tasks if t["id"] == args.task_id), None)
    if not task:
        print(f"Error: Task ID {args.task_id} not found in {args.tasks_file}")
        print("Available IDs:", [t["id"] for t in tasks])
        return 1
        
    print(f"[{args.task_id}] Instruction: {task['instruction']}")
    print(f"[{args.task_id}] Target Pos: {task['target_pos']}")

    # 2. Setup Agent
    from simworld.agent.humanoid import Humanoid
    from simworld.communicator.communicator import Communicator
    from simworld.communicator.unrealcv import UnrealCV
    from simworld.utils.vector import Vector
    from internnav.agent.sys3_only_agent import Sys3OnlyAgent, Sys3OnlyAgentCfg

    try:
        ucv = UnrealCV(ip=args.ip, port=args.port, resolution=(640, 360), connect_timeout_s=args.timeout_s)
        comm = Communicator(ucv)
    except Exception as e:
        print(f"Could not connect to SimWorld: {e}")
        return 2

    # Spawn Agent at Origin (or task specific start if defined)
    # Note: tasks.json currently doesn't specify start_pos, defaulting to (0,0)
    start_pos = Vector(0, 0)
    agent = Humanoid(position=start_pos, direction=Vector(1, 0))
    comm.spawn_agent(agent, name=None, model_path="/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C", type="humanoid")
    
    # Important: Spawn UE Manager to enable position queries
    from simworld.config import Config
    config = Config()
    # We must wait for UE Manager to be ready
    comm.spawn_ue_manager(config['simworld.ue_manager_path'])
    time.sleep(2.0) # Wait for manager to spawn
    ucv.update_ue_manager(comm.ue_manager_name) # Link agent to manager
    
    # 3. Initialize System 3
    sys3 = Sys3OnlyAgent(
        Sys3OnlyAgentCfg(
            model_name=args.vlm_model_name,
            base_url=args.vlm_base_url,
            api_key=args.vlm_api_key,
            max_subepisode_frames=8,
        )
    )
    
    # Set the high-level semantic goal
    sys3.set_goal(task['instruction'])
    
    # 4. Main Loop
    target_xy = None # Immediate waypoint from VLM
    final_goal_pos = Vector(task['target_pos'][0], task['target_pos'][1])
    success = False
    
    # Logging
    log_dir = os.path.join(args.output_dir, f"{args.task_id}_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log(f"Started task {args.task_id}")

    for step_idx in range(args.max_steps):
        # A. Check Success
        try:
            info = comm.get_position_and_direction(humanoid_ids=[agent.id])
        except Exception as e:
            log(f"Warning: Failed to get agent info: {e}. Retrying...")
            time.sleep(0.5)
            continue
            
        if ("humanoid", agent.id) not in info:
            log("Error: Agent lost!")
            break
            
        pos, yaw_deg = info[("humanoid", agent.id)]
        agent.position = pos
        agent.direction = yaw_deg
        
        dist_to_goal = pos.distance(final_goal_pos)
        if dist_to_goal < 500.0: # 5 meters threshold
            log(f"SUCCESS! Reached goal (dist={dist_to_goal:.1f})")
            success = True
            break
            
        # B. Get Observation
        try:
            img_bgr = comm.get_camera_observation(agent.camera_id, "lit", mode="direct")
        except Exception as e:
             log(f"Warning: Failed to get camera image: {e}")
             time.sleep(0.1)
             continue
             
        img_rgb = _bgr_to_rgb(img_bgr)
        
        # C. System 3 Reasoning
        sys3.sys1_steps = step_idx
        instr, status, thought = sys3.update_instruction({"rgb": img_rgb})
        
        log(f"Step {step_idx}: Dist={dist_to_goal:.1f}, Instr='{instr}', Status={status}")
        
        if instr:
            parsed = parse_xy(instr)
            if parsed:
                target_xy = parsed
            else:
                log(f"Warning: Could not parse coordinates from '{instr}'")

        if status == "DONE":
            log("System 3 declared DONE.")
            # Verify if actually at goal
            if dist_to_goal < 1000.0:
                success = True
            else:
                log(f"System 3 stopped early! Dist={dist_to_goal:.1f}")
            break
            
        # D. Execution (Geometric)
        if target_xy:
            dx = target_xy[0] - pos.x
            dy = target_xy[1] - pos.y
            dist_wp = math.sqrt(dx*dx + dy*dy)
            target_yaw = math.degrees(math.atan2(dy, dx))
            yaw_err = _wrap_deg(target_yaw - yaw_deg)
            
            if dist_wp > 100.0:
                if abs(yaw_err) > 15.0:
                    turn_dir = "right" if yaw_err > 0 else "left"
                    comm.humanoid_rotate(agent.id, min(30.0, abs(yaw_err)), turn_dir)
                else:
                    comm.humanoid_step_forward(agent.id, duration=0.4, direction=0)
            else:
                # Reached waypoint, stop and wait for new instruction
                comm.humanoid_stop(agent.id)
        else:
            # Explore / Wait
            comm.humanoid_step_forward(agent.id, duration=0.25, direction=0)
            
        time.sleep(args.dt)

    # 5. Cleanup
    log(f"Finished. Success: {success}")
    # comm.disconnect() # Optional
    return 0 if success else 1

if __name__ == "__main__":
    main()
