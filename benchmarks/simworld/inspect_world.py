import json
import os
import sys
import random

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) # benchmarks/simworld
    repo_root = os.path.dirname(os.path.dirname(base_dir)) # InternNav
    simworld_data_dir = os.path.join(repo_root, "external", "SimWorld", "simworld", "data")
    
    progen_path = os.path.join(simworld_data_dir, "progen_world.json")
    desc_path = os.path.join(simworld_data_dir, "description_map.json")
    
    if not os.path.exists(progen_path):
        print(f"Error: {progen_path} not found.")
        return
        
    print(f"Loading world data from {progen_path}...")
    with open(progen_path, 'r') as f:
        world_data = json.load(f)
        
    print(f"Loading descriptions from {desc_path}...")
    with open(desc_path, 'r') as f:
        desc_map = json.load(f)
        
    # Analyze Features
    features = world_data.get("nodes", [])
    print(f"Found {len(features)} nodes in the world.")
    
    buildings = []
    
    for feat in features:
        props = feat.get("properties", {})
        # Use instance_name as the type identifier
        feat_type = feat.get("instance_name", "")
        feat_id = feat.get("id", "")
        
        # Check if it is a building
        # Match instance_name with description map keys
        matched_type = None
        if feat_type in desc_map:
            matched_type = feat_type
        
        if matched_type:
            # Extract location
            loc = props.get("location", {})
            x = loc.get("x")
            y = loc.get("y")
            
            if x is not None and y is not None:
                description = desc_map[matched_type]
                buildings.append({
                    "id": feat_id,
                    "type": matched_type,
                    "description": description,
                    "x": x,
                    "y": y
                })
                
    print(f"Identified {len(buildings)} buildings with descriptions.")
    
    # Group by Description
    grouped = {}
    for b in buildings:
        d = b['description']
        if d not in grouped:
            grouped[d] = []
        grouped[d].append(b)
        
    print("\n--- Available Landmarks ---")
    for desc, items in grouped.items():
        print(f"[{len(items)}] {desc}")
        
    # Generate Task List
    # We want tasks like: "Find the Hospital"
    # We need a start position (random road?) and a goal (building location)
    
    tasks = []
    
    # 1. Hospital Task (if exists)
    hospitals = [b for b in buildings if "Hospital" in b['description']]
    if hospitals:
        target = hospitals[0]
        tasks.append({
            "id": "task_hospital_01",
            "instruction": f"Go to the {target['description']}",
            "target_pos": [target['x'], target['y']],
            "target_id": target['id']
        })
        
    # 2. Grocery Store (if exists)
    groceries = [b for b in buildings if "grocery" in b['description'].lower()]
    if groceries:
        target = groceries[0]
        tasks.append({
            "id": "task_grocery_01",
            "instruction": f"Find the {target['description']}",
            "target_pos": [target['x'], target['y']],
            "target_id": target['id']
        })
        
    # 3. School (if exists)
    schools = [b for b in buildings if "School" in b['description']]
    if schools:
        target = schools[0]
        tasks.append({
            "id": "task_school_01",
            "instruction": f"Navigate to the {target['description']}",
            "target_pos": [target['x'], target['y']],
            "target_id": target['id']
        })

    # 4. Random others
    other_types = [b for b in buildings if b not in hospitals + groceries + schools]
    if other_types:
        for i in range(3):
            target = random.choice(other_types)
            tasks.append({
                "id": f"task_random_{i:02d}",
                "instruction": f"Go to {target['description']}",
                "target_pos": [target['x'], target['y']],
                "target_id": target['id']
            })
            
    # Save Tasks
    task_file = os.path.join(base_dir, "tasks.json")
    with open(task_file, 'w') as f:
        json.dump(tasks, f, indent=2)
        
    print(f"\nGenerated {len(tasks)} tasks in {task_file}")
    for t in tasks:
        print(f"- {t['instruction']} -> ({t['target_pos'][0]:.1f}, {t['target_pos'][1]:.1f})")

if __name__ == "__main__":
    main()

