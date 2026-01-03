"""
Human Control Interface for SimWorld.
Allows controlling the agent via keyboard (W/A/S/D) to explore the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import cv2
import numpy as np
import json

# Add external/SimWorld to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
simworld_path = os.path.join(repo_root, "external", "SimWorld")
if simworld_path not in sys.path:
    sys.path.append(simworld_path)

from simworld.config import Config
from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
from simworld.utils.vector import Vector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", help="Optional task ID to set goal context", default=None)
    parser.add_argument("--tasks_file", default=os.path.join(current_dir, "tasks.json"))
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    # Load Task Info if provided
    task_instr = "Explore Mode"
    target_pos = None
    
    if args.task_id:
        with open(args.tasks_file, 'r') as f:
            tasks = json.load(f)
        task = next((t for t in tasks if t["id"] == args.task_id), None)
        if task:
            task_instr = task['instruction']
            target_pos = Vector(task['target_pos'][0], task['target_pos'][1])
            print(f"Task: {task_instr}")

    # Connect
    try:
        ucv = UnrealCV(ip=args.ip, port=args.port, connect_timeout_s=5.0)
        comm = Communicator(ucv)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Cleanup & Spawn
    print("Initializing environment...")
    # comm.clear_env() # Optional: might want to keep existing env if just attaching?
    # Better to clear to ensure clean state for controlling 'our' agent
    try:
        comm.clear_env() 
    except: 
        pass
    time.sleep(1.0)

    # 3.5 Generate World (CRITICAL for non-empty environment)
    progen_world_path = os.path.join(simworld_path, 'simworld/data/progen_world.json')
    ue_assets_path = os.path.join(simworld_path, 'simworld/data/ue_assets.json')
    
    if os.path.exists(progen_world_path) and os.path.exists(ue_assets_path):
        print(f"Generating world from {progen_world_path}...")
        comm.generate_world(progen_world_path, ue_assets_path, run_time=True)
        # Wait for generation to settle
        time.sleep(5.0)
    else:
        print("Warning: progen_world.json or ue_assets.json not found. World may be empty!")

    config = Config()
    start_pos = Vector(0, 0)
    agent = Humanoid(position=start_pos, direction=Vector(1, 0), communicator=comm, config=config)
    comm.spawn_agent(agent, name=None, model_path="/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C", type="humanoid")
    
    comm.spawn_ue_manager(config['simworld.ue_manager_path'])
    time.sleep(2.0)
    ucv.update_ue_manager(comm.ue_manager_name)
    
    print("\n" + "="*40)
    print("CONTROLS:")
    print("  W: Move Forward")
    print("  A: Rotate Left")
    print("  D: Rotate Right")
    print("  Q: Quit")
    print("="*40 + "\n")

    while True:
        # Get Image
        try:
            img = comm.get_camera_observation(agent.camera_id, "lit")
            if img is None:
                continue
                
            # Add UI Overlay
            display_img = img.copy()
            cv2.putText(display_img, f"Goal: {task_instr}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Get Position
            info = comm.get_position_and_direction(humanoid_ids=[agent.id])
            if ("humanoid", agent.id) in info:
                pos, yaw = info[("humanoid", agent.id)]
                pos_text = f"Pos: ({pos.x:.0f}, {pos.y:.0f})"
                cv2.putText(display_img, pos_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if target_pos:
                    dist = pos.distance(target_pos)
                    dist_text = f"Dist: {dist:.1f}"
                    cv2.putText(display_img, dist_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("SimWorld Human Control", display_img)
            
        except Exception as e:
            # print(f"Error loop: {e}")
            pass

        # Handle Input
        key = cv2.waitKey(100) & 0xFF # 100ms delay ~ 10fps loop
        
        if key == ord('q'):
            break
        elif key == ord('w'):
            comm.humanoid_step_forward(agent.id, duration=0.2, direction=0)
        elif key == ord('a'):
            comm.humanoid_rotate(agent.id, 15.0, "left")
        elif key == ord('d'):
            comm.humanoid_rotate(agent.id, 15.0, "right")
            
    cv2.destroyAllWindows()
    comm.disconnect()

if __name__ == "__main__":
    main()

