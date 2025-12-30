
import json
import os
import sys
import time
import math
import cv2
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

def main():
    # 1. Load Tasks
    tasks_file = os.path.join(current_dir, "tasks.json")
    if not os.path.exists(tasks_file):
        print("Error: tasks.json not found.")
        return

    with open(tasks_file, 'r') as f:
        tasks = json.load(f)
        
    print(f"Loaded {len(tasks)} tasks.")

    # 2. Connect to SimWorld
    try:
        ucv = UnrealCV(ip="127.0.0.1", port=9000, connect_timeout_s=5.0)
        comm = Communicator(ucv)
    except Exception as e:
        print(f"Error connecting to SimWorld: {e}")
        return

    # 3. Load Map (Ground Truth)
    config = Config()
    roads_path = os.path.join(simworld_path, 'simworld/data/roads.json')
    if not os.path.exists(roads_path):
        print(f"Error: roads.json not found at {roads_path}")
        return
        
    if 'map' not in config.config:
        config.config['map'] = {}
    config.config['map']['input_roads'] = roads_path
    
    sim_map = Map(config)
    sim_map.initialize_map_from_file()
    print(f"Map loaded with {len(sim_map.nodes)} nodes.")

    # 4. Verify Each Task
    output_dir = os.path.join(current_dir, "verification_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Spawn a temporary agent for camera captures
    agent = Humanoid(position=Vector(0,0), direction=Vector(1,0))
    comm.spawn_agent(agent, type="humanoid")
    time.sleep(1.0) # Wait for spawn

    results = []

    for task in tasks:
        t_id = task['id']
        t_instr = task['instruction']
        target_pos = Vector(task['target_pos'][0], task['target_pos'][1])
        
        print(f"\nVerifying Task: {t_id}")
        print(f"Instruction: {t_instr}")
        print(f"Target: {target_pos}")
        
        # A. Check Connectivity (Oracle)
        # Find nearest node to (0,0) (Start) and Target
        # Note: SimWorld map uses nodes. We need to find nearest node to coordinates.
        start_node = sim_map.get_nearest_node(Vector(0,0))
        goal_node = sim_map.get_nearest_node(target_pos)
        
        path = sim_map.get_shortest_path(start_node, goal_node)
        
        solvable = path is not None
        path_len = 0.0
        if path:
            # Calculate length
            for i in range(len(path)-1):
                path_len += path[i].position.distance(path[i+1].position)
            print(f"  [Oracle] Path Found! Length: {path_len:.2f} cm")
        else:
            print(f"  [Oracle] NO PATH FOUND (Unreachable?)")

        # B. Capture Visuals (Teleport)
        # 1. Start View
        comm.set_location(agent.id, [0, 0, 100]) # Origin
        comm.set_rotation(agent.id, [0, 0, 0])
        time.sleep(0.5)
        img_start = comm.get_camera_observation(agent.camera_id, "lit")
        
        # 2. Goal View (Teleport near goal, looking AT it)
        # Offset slightly so we see the building
        # Simply move 500cm away in X direction and look back
        goal_view_pos = [target_pos.x - 500, target_pos.y - 500, 100]
        # Look at target
        dx = target_pos.x - goal_view_pos[0]
        dy = target_pos.y - goal_view_pos[1]
        yaw = math.degrees(math.atan2(dy, dx))
        
        comm.set_location(agent.id, goal_view_pos)
        comm.set_rotation(agent.id, [0, yaw, 0])
        time.sleep(0.5)
        img_goal = comm.get_camera_observation(agent.camera_id, "lit")
        
        # Save Images
        cv2.imwrite(os.path.join(output_dir, f"{t_id}_start.jpg"), img_start)
        cv2.imwrite(os.path.join(output_dir, f"{t_id}_goal.jpg"), img_goal)
        print(f"  [Visuals] Saved start/goal images to {output_dir}")
        
        results.append({
            "id": t_id,
            "solvable": solvable,
            "path_length": path_len,
            "images_saved": True
        })

    # Summary
    print("\n--- Verification Summary ---")
    for r in results:
        status = "OK" if r['solvable'] else "FAIL"
        print(f"{r['id']}: {status} (Len: {r['path_length']:.1f})")

if __name__ == "__main__":
    main()


