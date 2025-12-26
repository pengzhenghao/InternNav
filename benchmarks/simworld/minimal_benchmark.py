"""
Minimal benchmark script for SimWorld.
This script runs a navigation task without using an LLM, 
simulating a "perfect" agent using the ground-truth map.
It calculates and reports standard metrics (Success Rate, SPL).
"""

from __future__ import annotations

import math
import os
import sys
import time
import argparse
import numpy as np

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Configuration & Map Setup
    print("[benchmark] Setting up configuration...")
    config = Config()
    
    # Locate roads.json
    roads_path = os.path.join(simworld_path, 'simworld/data/roads.json')
    if not os.path.exists(roads_path):
        print(f"[benchmark] Error: roads.json not found at {roads_path}")
        return
    # config['map.input_roads'] = roads_path # This fails because Config doesn't support assignment
    
    # Update config directly
    if 'map' not in config.config:
        config.config['map'] = {}
    config.config['map']['input_roads'] = roads_path
    
    print("[benchmark] Loading Map...")
    sim_map = Map(config)
    sim_map.initialize_map_from_file()
    print(f"[benchmark] Map loaded with {len(sim_map.nodes)} nodes.")

    # 2. Connection
    print(f"[benchmark] Connecting to SimWorld at {args.ip}:{args.port}...")
    try:
        ucv = UnrealCV(ip=args.ip, port=args.port, connect_timeout_s=5.0)
    except Exception as e:
        print(f"[benchmark] Failed to connect: {e}")
        print("[benchmark] Please ensure the SimWorld UE server is running.")
        return
    
    comm = Communicator(ucv)

    # 3.5 Spawn UE Manager (Required for getting agent positions)
    # The UE Manager handles global state queries
    print("[benchmark] Spawning UE Manager...")
    comm.spawn_ue_manager(config['simworld.ue_manager_path'])
    time.sleep(1.0)

    # 3. Define Task (Start -> Goal)
    # We pick two random nodes that are connected (reachable)
    import random
    random.seed(args.seed)
    
    print("[benchmark] Generating task...")
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
        print("[benchmark] Could not generate a valid task (start->goal path). Retrying...")
        return

    # Calculate Ideal Path Length (Geodesic Distance)
    ideal_distance = 0.0
    for i in range(len(ideal_path_nodes) - 1):
        ideal_distance += ideal_path_nodes[i].position.distance(ideal_path_nodes[i+1].position)
        
    print(f"[benchmark] Task: Start {start_node.position} -> Goal {goal_node.position}")
    print(f"[benchmark] Ideal Geodesic Distance: {ideal_distance:.2f} cm")

    # 4. Spawn Agent
    # We use Humanoid
    agent = Humanoid(position=start_node.position, direction=Vector(1, 0), map=sim_map, communicator=comm, config=config)
    
    # Cleanup previous agents just in case? 
    # SimWorld doesn't have a clear "reset" for agents, but spawning might reuse ID if we restarted script?
    # Actually Humanoid._id_counter increments. 
    # Ideally we should clear the scene but let's just spawn a new one.
    
    comm.spawn_agent(agent, name=None, model_path="/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C", type="humanoid")
    time.sleep(0.5)
    
    # Force teleport to start to be precise (using Z=100 as safe default)
    actor_name = comm.get_humanoid_name(agent.id)
    ucv.set_location([start_node.position.x, start_node.position.y, 100], actor_name)
    
    # 5. Navigation Loop (The "Agent")
    print("[benchmark] Starting navigation...")
    
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
        
        # 1. Update Agent State
        info = comm.get_position_and_direction(humanoid_ids=[agent.id])
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
            print(f"[benchmark] Reached goal! Final dist: {dist_to_goal:.1f}")
            success = True
            break
            
        # 4. Navigation Logic (Follow Waypoints)
        if current_waypoint_idx < len(waypoints):
            target = waypoints[current_waypoint_idx]
            dist_to_wp = current_pos.distance(target)
            
            if dist_to_wp < WAYPOINT_THRESHOLD:
                print(f"[benchmark] Reached waypoint {current_waypoint_idx+1}/{len(waypoints)}")
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
    print("-" * 40)
    print("[benchmark] RESULTS")
    print("-" * 40)
    
    # Get final collision stats
    h_col, o_col, b_col, v_col = comm.get_collision_number(agent.id)
    total_collisions = h_col + o_col + b_col + v_col
    
    # Success Rate (Binary)
    sr = 1.0 if success else 0.0
    
    # Navigation Error (Final distance to goal)
    nav_error = dist_to_goal
    
    # SPL (Success weighted by Path Length)
    # SPL = Success * (Ideal_Distance / max(Actual_Distance, Ideal_Distance))
    spl = 0.0
    if success:
        spl = ideal_distance / max(actual_distance, ideal_distance)
        
    print(f"Success Rate:     {sr:.1f}")
    print(f"SPL:              {spl:.4f}")
    print(f"Navigation Error: {nav_error:.2f} cm")
    print(f"Trajectory Len:   {actual_distance:.2f} cm (Ideal: {ideal_distance:.2f} cm)")
    print(f"Total Steps:      {step_count}")
    print(f"Collisions:       {total_collisions} (Human: {h_col}, Obj: {o_col}, Bldg: {b_col}, Veh: {v_col})")
    
    if success:
        print("[benchmark] PASSED")
    else:
        print("[benchmark] FAILED (Did not reach goal)")

    # Clean up connection to avoid hanging
    print("[benchmark] Closing connection...")
    comm.disconnect()
    ucv.client.disconnect()

if __name__ == "__main__":
    main()

