"""
Visualize an Expert (Rule-based) run on a specific task.
This script uses the ground-truth map to navigate the agent to the goal,
recording a video of the scenario.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import numpy as np
import cv2
import datetime

# Add external/SimWorld to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
simworld_path = os.path.join(repo_root, "external", "SimWorld")
if simworld_path not in sys.path:
    sys.path.append(simworld_path)

from simworld.config import Config
from simworld.map.map import Map
from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
from simworld.utils.vector import Vector

def _wrap_deg(d: float) -> float:
    d = (d + 180.0) % 360.0 - 180.0
    return d

class VideoLogger:
    def __init__(self, output_path):
        self.video_path = output_path
        self.vis_frames = []
        
    def capture_frame(self, image):
        if image is not None and isinstance(image, np.ndarray):
            # SimWorld returns BGR via direct mode (cv2 compatible)
            self.vis_frames.append(image)

    def save(self, fps=10):
        if not self.vis_frames:
            print("No frames captured.")
            return
            
        height, width, layers = self.vis_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))
        
        for frame in self.vis_frames:
            video.write(frame)
        
        video.release()
        print(f"Video saved to {self.video_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--tasks_file", default=os.path.join(current_dir, "tasks.json"))
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--output_dir", default="output_expert")
    args = parser.parse_args()

    # 1. Load Task
    with open(args.tasks_file, 'r') as f:
        tasks = json.load(f)
    
    task = next((t for t in tasks if t["id"] == args.task_id), None)
    if not task:
        print(f"Error: Task {args.task_id} not found.")
        return

    print(f"Running Expert on: {task['instruction']}")
    target_pos = Vector(task['target_pos'][0], task['target_pos'][1])

    # 2. Setup
    config = Config()
    # Check possible road file locations
    possible_paths = [
        os.path.join(simworld_path, 'simworld/data/predefined_roads.json'),
        os.path.join(simworld_path, 'simworld/data/sample_data/road.json'),
    ]
    roads_path = None
    for p in possible_paths:
        if os.path.exists(p):
            roads_path = p
            break
            
    if not roads_path:
        print("Error: Could not find any roads JSON file in SimWorld data directory.")
        return
        
    print(f"Using road map: {roads_path}")
    if 'map' not in config.config:
        config.config['map'] = {}
    config.config['map']['input_roads'] = roads_path
    
    sim_map = Map(config)
    sim_map.initialize_map_from_file()
    
    try:
        ucv = UnrealCV(ip=args.ip, port=args.port, connect_timeout_s=5.0)
        comm = Communicator(ucv)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 3. Spawn & Reset
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

    # Start at Origin (0,0) as per benchmark
    start_pos = Vector(0, 0)
    agent = Humanoid(position=start_pos, direction=Vector(1, 0), map=sim_map, communicator=comm, config=config)
    comm.spawn_agent(agent, name=None, model_path="/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C", type="humanoid")
    time.sleep(2.0)
    
    # Teleport to ensure exact start
    actor_name = comm.get_humanoid_name(agent.id)
    ucv.set_location([start_pos.x, start_pos.y, 100], actor_name)

    # 4. Plan Path (Oracle)
    start_node = sim_map.get_closest_node(start_pos)
    goal_node = sim_map.get_closest_node(target_pos)
    
    # --- HIGHLIGHT TARGET FOR VISUALIZATION ---
    if "target_id" in task and task["target_id"]:
        print(f"Highlighting target object: {task['target_id']}")
        try:
            # Set to bright red (R=255, G=0, B=0)
            ucv.set_color(task['target_id'], [255, 0, 0])
        except Exception as e:
            print(f"Warning: Failed to highlight target: {e}")
    # ------------------------------------------

    path_nodes = sim_map.get_shortest_path(start_node, goal_node)
    if not path_nodes:
        print("Error: No path found by Oracle!")
        return
        
    waypoints = [n.position for n in path_nodes][1:] # Skip start
    waypoints.append(target_pos) # Ensure we go exactly to target
    
    print(f"Path plan: {len(waypoints)} waypoints.")

    # 5. Execute & Record
    os.makedirs(args.output_dir, exist_ok=True)
    video_path = os.path.join(args.output_dir, f"expert_{args.task_id}.mp4")
    logger = VideoLogger(video_path)
    
    MAX_STEPS = 2000
    DT = 0.1
    current_wp_idx = 0
    
    # --- RESTARTING LOGIC TO INCLUDE UE MANAGER SPAWN CORRECTLY ---
    print("Spawning UE Manager...")
    comm.spawn_ue_manager(config['simworld.ue_manager_path'])
    time.sleep(2.0)
    ucv.update_ue_manager(comm.ue_manager_name)
    
    # Re-Teleport just in case
    ucv.set_location([start_pos.x, start_pos.y, 100], actor_name)
    
    print("Starting navigation loop...")
    
    success = False
    
    for step in range(MAX_STEPS):
        # 1. Capture
        try:
            img = comm.get_camera_observation(agent.camera_id, "lit")
            logger.capture_frame(img)
        except Exception as e:
            print(f"Warning [Step {step}]: Capture failed: {e}")
            pass
            
        # 2. State
        try:
            info = comm.get_position_and_direction(humanoid_ids=[agent.id])
            if ("humanoid", agent.id) not in info:
                 print(f"Warning [Step {step}]: Agent {agent.id} not found in info.")
                 time.sleep(0.1)
                 continue
                 
            current_pos, current_yaw = info[("humanoid", agent.id)]
            agent.position = current_pos
            agent.direction = current_yaw
        except Exception as e:
            print(f"Error [Step {step}]: State update failed: {e}")
            time.sleep(0.1)
            continue
            
        # 3. Check Goal
        dist_to_goal = current_pos.distance(target_pos)
        if step % 10 == 0:
            print(f"Step {step}/{MAX_STEPS}: Pos=({current_pos.x:.1f}, {current_pos.y:.1f}), DistToGoal={dist_to_goal:.1f}")
            
        if dist_to_goal < 200.0:
            print("Reached Goal!")
            success = True
            break
            
        # 4. Control
        if current_wp_idx < len(waypoints):
            target = waypoints[current_wp_idx]
            dist_to_wp = current_pos.distance(target)
            
            if dist_to_wp < 150.0:
                print(f"Reached Waypoint {current_wp_idx} (Dist={dist_to_wp:.1f})")
                current_wp_idx += 1
                if current_wp_idx >= len(waypoints):
                    continue
                target = waypoints[current_wp_idx]
            
            # PID
            dx = target.x - current_pos.x
            dy = target.y - current_pos.y
            target_yaw = math.degrees(math.atan2(dy, dx))
            yaw_err = _wrap_deg(target_yaw - current_yaw)
            
            if abs(yaw_err) > 10.0:
                turn_dir = "right" if yaw_err > 0 else "left"
                comm.humanoid_rotate(agent.id, min(30.0, abs(yaw_err)), turn_dir)
            else:
                comm.humanoid_step_forward(agent.id, duration=DT*2, direction=0)
        
        time.sleep(DT)
        
    logger.save()
    comm.disconnect()

if __name__ == "__main__":
    main()

