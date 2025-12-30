"""
Minimal benchmark script for SimWorld.
This script runs a navigation task without using an LLM, 
simulating a "perfect" agent using the ground-truth map.
It calculates and reports standard metrics (Success Rate, SPL) and supports logging/video generation.
"""

from __future__ import annotations

import math
import os
import sys
import time
import argparse
import numpy as np
import json
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
    # map to [-180, 180]
    d = (d + 180.0) % 360.0 - 180.0
    return d

class BenchmarkLogger:
    def __init__(self, output_dir, save_video=False):
        self.run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(output_dir, f"benchmark_{self.run_ts}")
        self.save_video = save_video
        self.vis_frames = []
        
        os.makedirs(self.output_path, exist_ok=True)
        self.log_file = os.path.join(self.output_path, "log.txt")
        self.metrics_file = os.path.join(self.output_path, "metrics.json")
        self.video_path = os.path.join(self.output_path, "video.mp4")
        
        print(f"[benchmark] Logging to {self.output_path}")

    def log(self, message):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def log_metrics(self, metrics):
        with open(self.metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        self.log(f"[benchmark] Metrics saved to {self.metrics_file}")

    def capture_frame(self, image):
        if self.save_video and image is not None:
            # Assuming image is RGB numpy array (H, W, 3)
            # Ensure it's in BGR for OpenCV
            if isinstance(image, np.ndarray):
                # NOTE: SimWorld get_camera_observation returns BGR by default when using "direct" mode (via unrealcv.get_image)
                # because opencv reads are BGR.
                # However, our other scripts (smoke_test) suggest it might be returning BGR and needing conversion for PIL.
                # Let's verify:
                # If image is BGR:
                #   We need BGR for VideoWriter. So we should NOT convert.
                # If image is RGB:
                #   We need BGR for VideoWriter. So we should convert RGB2BGR.
                
                # Based on user feedback, the previous video had wrong colors (blue/orange swap).
                # This means we likely swapped channels when we shouldn't have, or vice versa.
                # Previous code: bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # If input was already BGR, then this swapped it to RGB, and VideoWriter (expecting BGR) wrote RGB data,
                # resulting in swapped colors on playback.
                
                # So if input is BGR, we should just use it as is.
                # self.vis_frames.append(image)
                
                # Let's assume input from comm.get_camera_observation is BGR (standard OpenCV format from SimWorld).
                self.vis_frames.append(image)
                
                # Note: If we are sending this to VLM (System 3), we MUST convert BGR to RGB.
                # This script is just a benchmark without VLM, but for future reference:
                # RGB_for_VLM = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def save_video_file(self, fps=10):
        if self.save_video and self.vis_frames:
            height, width, layers = self.vis_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))
            
            for frame in self.vis_frames:
                video.write(frame)
            
            video.release()
            self.log(f"[benchmark] Video saved to {self.video_path}")
        elif self.save_video:
            self.log("[benchmark] No frames captured for video.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="output", help="Directory to save logs and results")
    parser.add_argument("--save_video", action="store_true", help="Save video of the navigation")
    args = parser.parse_args()

    # Initialize Logger
    logger = BenchmarkLogger(args.output_dir, args.save_video)

    # 1. Configuration & Map Setup
    logger.log("[benchmark] Setting up configuration...")
    config = Config()
    
    # Locate roads.json
    roads_path = os.path.join(simworld_path, 'simworld/data/roads.json')
    if not os.path.exists(roads_path):
        logger.log(f"[benchmark] Error: roads.json not found at {roads_path}")
        return
    
    # Update config directly
    if 'map' not in config.config:
        config.config['map'] = {}
    config.config['map']['input_roads'] = roads_path
    
    logger.log("[benchmark] Loading Map...")
    sim_map = Map(config)
    sim_map.initialize_map_from_file()
    logger.log(f"[benchmark] Map loaded with {len(sim_map.nodes)} nodes.")

    # 2. Connection
    logger.log(f"[benchmark] Connecting to SimWorld at {args.ip}:{args.port}...")
    try:
        ucv = UnrealCV(ip=args.ip, port=args.port, connect_timeout_s=5.0)
    except Exception as e:
        logger.log(f"[benchmark] Failed to connect: {e}")
        logger.log("[benchmark] Please ensure the SimWorld UE server is running.")
        return
    
    comm = Communicator(ucv)

    # Clean up environment from previous runs
    logger.log("[benchmark] Clearing environment...")
    try:
        comm.clear_env()
    except Exception as e:
        logger.log(f"[benchmark] Warning: clear_env failed: {e}")
    time.sleep(1.0)

    # 3.5 Spawn UE Manager (Required for getting agent positions)
    # The UE Manager handles global state queries
    logger.log("[benchmark] Spawning UE Manager...")
    comm.spawn_ue_manager(config['simworld.ue_manager_path'])
    time.sleep(2.0)

    # Wait for UE Manager to be ready
    logger.log("[benchmark] Waiting for UE Manager...")
    for _ in range(10):
        try:
            res = comm.unrealcv.get_informations(comm.ue_manager_name)
            if res and res.strip():
                 # Try to validate JSON
                 try:
                     json.loads(res)
                     logger.log("[benchmark] UE Manager ready (valid JSON).")
                     break
                 except json.JSONDecodeError:
                     logger.log("[benchmark] UE Manager responding but invalid JSON yet.")
        except:
            pass
        time.sleep(1.0)

    # 3. Define Task (Start -> Goal)
    # We pick two random nodes that are connected (reachable)
    import random
    random.seed(args.seed)
    
    logger.log("[benchmark] Generating task...")
    start_node = sim_map.get_random_node()
    
    # Try to find a reachable goal at least some distance away
    goal_node = None
    ideal_path_nodes = None
    
    for _ in range(20):
        candidate = sim_map.get_random_node(exclude=[start_node])
        if start_node.position.distance(candidate.position) < 500: # Too close
            continue
            
        path = sim_map.get_shortest_path(start_node, candidate)
        if path:
            goal_node = candidate
            ideal_path_nodes = path
            break
            
    if not goal_node:
        logger.log("[benchmark] Could not generate a valid task (start->goal path). Retrying...")
        return

    # Calculate Ideal Path Length (Geodesic Distance)
    ideal_distance = 0.0
    for i in range(len(ideal_path_nodes) - 1):
        ideal_distance += ideal_path_nodes[i].position.distance(ideal_path_nodes[i+1].position)
        
    logger.log(f"[benchmark] Task: Start {start_node.position} -> Goal {goal_node.position}")
    logger.log(f"[benchmark] Ideal Geodesic Distance: {ideal_distance:.2f} cm")

    # 4. Spawn Agent
    # We use Humanoid
    # Important: Reset ID counter if we cleared env
    Humanoid._id_counter = 0
    Humanoid._camera_id_counter = 1
    
    agent = Humanoid(position=start_node.position, direction=Vector(1, 0), map=sim_map, communicator=comm, config=config)
    logger.log(f"[benchmark] Created agent with ID: {agent.id}")
    
    comm.spawn_agent(agent, name=None, model_path="/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C", type="humanoid")
    time.sleep(2.0)
    
    # 4.5 Register Agent with UE Manager
    logger.log("[benchmark] Updating UE Manager...")
    try:
        ucv.update_ue_manager(comm.ue_manager_name)
    except Exception as e:
        logger.log(f"[benchmark] Warning: update_ue_manager failed: {e}")
        
    time.sleep(1.0)
    
    # Force teleport to start to be precise (using Z=100 as safe default)
    actor_name = comm.get_humanoid_name(agent.id)
    ucv.set_location([start_node.position.x, start_node.position.y, 100], actor_name)
    
    # 5. Navigation Loop (The "Agent")
    logger.log("[benchmark] Starting navigation...")
    
    path_points = [n.position for n in ideal_path_nodes]
    # Skip the first point (start)
    waypoints = path_points[1:]
    
    actual_distance = 0.0
    last_pos = start_node.position # We assume we start exactly there
    success = False
    
    # Thresholds
    WAYPOINT_THRESHOLD = 150.0 # cm
    GOAL_THRESHOLD = 150.0
    MAX_STEPS = 500
    DT = 0.1
    
    step_count = 0
    current_waypoint_idx = 0
    
    while step_count < MAX_STEPS:
        step_count += 1
        
        # 0. Capture Frame (if video enabled)
        if args.save_video:
            try:
                # Capture RGB frame
                # Important: In SimWorld, the camera ID corresponds to the agent's camera
                # Check if camera exists?
                # The camera is part of the agent BP.
                img = comm.get_camera_observation(agent.camera_id, "lit")
                logger.capture_frame(img)
            except Exception as e:
                # logger.log(f"Warning: Failed to capture frame: {e}")
                pass

        # 1. Update Agent State
        # The agent ID is likely 0, but let's be robust
        try:
             info = comm.get_position_and_direction(humanoid_ids=[agent.id])
        except Exception as e:
             # print(f"[benchmark] Error getting/parsing info: {e}")
             time.sleep(0.5)
             continue
        
        # If agent 0 is not found, maybe the ID reset logic is tricky?
        # Let's try to find ANY humanoid in the info
        if ("humanoid", agent.id) not in info:
             logger.log(f"[benchmark] Warning: Agent {agent.id} not found. Available keys: {list(info.keys())}")
             
             # Fallback: if there is exactly one humanoid, use it
             humanoid_keys = [k for k in info.keys() if k[0] == "humanoid"]
             if len(humanoid_keys) == 1:
                 logger.log(f"[benchmark] Found alternative humanoid: {humanoid_keys[0]}. Updating agent ID.")
                 agent.id = humanoid_keys[0][1]
                 info = comm.get_position_and_direction(humanoid_ids=[agent.id])
             else:
                 # Try once more after a short sleep
                 time.sleep(0.5)
                 info = comm.get_position_and_direction(humanoid_ids=[agent.id])
                 if ("humanoid", agent.id) not in info:
                     logger.log(f"[benchmark] Error: Agent {agent.id} lost. Exiting.")
                     break
        
        current_pos, current_yaw = info[("humanoid", agent.id)]
        
        # Update agent object (SimWorld agents need this manually updated)
        agent.position = current_pos
        agent.direction = current_yaw
        
        # 2. Track Metrics
        dist_moved = current_pos.distance(last_pos)
        actual_distance += dist_moved
        last_pos = current_pos
        
        # 3. Check Goal
        dist_to_goal = current_pos.distance(goal_node.position)
        if dist_to_goal < GOAL_THRESHOLD:
            logger.log(f"[benchmark] Reached goal! Final dist: {dist_to_goal:.1f}")
            success = True
            break
            
        # 4. Navigation Logic (Follow Waypoints)
        if current_waypoint_idx < len(waypoints):
            target = waypoints[current_waypoint_idx]
            dist_to_wp = current_pos.distance(target)
            
            if dist_to_wp < WAYPOINT_THRESHOLD:
                logger.log(f"[benchmark] Reached waypoint {current_waypoint_idx+1}/{len(waypoints)}")
                current_waypoint_idx += 1
                if current_waypoint_idx >= len(waypoints):
                    # Should be goal
                    continue
                target = waypoints[current_waypoint_idx]
                
            # Move towards target
            dx = target.x - current_pos.x
            dy = target.y - current_pos.y
            target_yaw = math.degrees(math.atan2(dy, dx))
            yaw_err = _wrap_deg(target_yaw - current_yaw)
            
            # Simple P-controller for rotation
            if abs(yaw_err) > 10.0:
                turn_dir = "right" if yaw_err > 0 else "left"
                turn_angle = min(30.0, abs(yaw_err))
                comm.humanoid_rotate(agent.id, turn_angle, turn_dir)
            else:
                comm.humanoid_step_forward(agent.id, duration=DT*2, direction=0)
                
        else:
            # No waypoints left? Should have reached goal.
            pass
            
        time.sleep(DT)

    # 6. Compute Metrics
    logger.log("-" * 40)
    logger.log("[benchmark] RESULTS")
    logger.log("-" * 40)
    
    # Get final collision stats
    try:
        h_col, o_col, b_col, v_col = comm.get_collision_number(agent.id)
        total_collisions = h_col + o_col + b_col + v_col
    except Exception as e:
        logger.log(f"[benchmark] Warning: Could not get collision stats: {e}")
        total_collisions = -1
        h_col, o_col, b_col, v_col = -1, -1, -1, -1
    
    # Success Rate (Binary)
    sr = 1.0 if success else 0.0
    
    # Navigation Error (Final distance to goal)
    nav_error = dist_to_goal
    
    # SPL (Success weighted by Path Length)
    # SPL = Success * (Ideal_Distance / max(Actual_Distance, Ideal_Distance))
    spl = 0.0
    if success:
        spl = ideal_distance / max(actual_distance, ideal_distance)
        
    logger.log(f"Success Rate:     {sr:.1f}")
    logger.log(f"SPL:              {spl:.4f}")
    logger.log(f"Navigation Error: {nav_error:.2f} cm")
    logger.log(f"Trajectory Len:   {actual_distance:.2f} cm (Ideal: {ideal_distance:.2f} cm)")
    logger.log(f"Total Steps:      {step_count}")
    logger.log(f"Collisions:       {total_collisions} (Human: {h_col}, Obj: {o_col}, Bldg: {b_col}, Veh: {v_col})")
    
    metrics = {
        "success_rate": sr,
        "spl": spl,
        "navigation_error": nav_error,
        "trajectory_length": actual_distance,
        "ideal_distance": ideal_distance,
        "total_steps": step_count,
        "collisions": {
            "total": total_collisions,
            "human": h_col,
            "object": o_col,
            "building": b_col,
            "vehicle": v_col
        }
    }
    
    logger.log_metrics(metrics)
    
    if success:
        logger.log("[benchmark] PASSED")
    else:
        logger.log("[benchmark] FAILED (Did not reach goal)")

    # Save video if enabled
    if args.save_video:
        logger.save_video_file()

    # Clean up connection to avoid hanging
    logger.log("[benchmark] Closing connection...")
    comm.disconnect()
    
    # Wait briefly for threads to spin down (optional)
    time.sleep(0.5)
    
    # Force exit to avoid hanging threads
    os._exit(0)

if __name__ == "__main__":
    main()
