# SimWorld Learning Notes

## 1. Overview & Architecture
**SimWorld** is an Unreal Engine 5-based simulator for embodied AI, focusing on outdoor environments, procedural city generation, and multi-agent interaction.

- **Architecture**:
  - **Server**: Unreal Engine executable (physics, rendering).
  - **Client**: Python API communicating via `UnrealCV` (TCP port 9000).
  - **Bridge**: `UnrealCV` acts as the bridge between the UE server and Python.

## 2. Benchmarks & Tasks
SimWorld officially provides two main benchmark tasks:

### A. Multimodal Instruction Following (Single-Agent VLN)
- **Goal**: A single robot navigates to a target destination.
- **Input**: 
  - Natural language instructions (e.g., "Go to the red building").
  - Visual hints (images/icons).
  - RGB/Depth/Segmentation camera feeds.
- **Task Definition**: The agent must "ground language into action" and plan a trajectory.
- **Relevance to System 3**: This is the primary target for a VLM-based navigation agent (System 3).

### B. Multi-Robot Search via Communication
- **Goal**: Multiple robots find each other in a city.
- **Mechanism**: They exchange natural language messages to coordinate.

## 3. Low-Level Policy & Action Space
SimWorld provides a "humanoid" or "quadruped" agent with:
- **Low-Level Actions**:
  - `step_forward`, `rotate`, `look_up`, `look_down`.
  - Continuous or discrete parameters (speed, duration, angle).
- **High-Level Helpers**:
  - `LocalPlanner`: A built-in module that can take a coordinate `(x, y)` and drive the agent there, handling basic pathfinding/obstacle avoidance.

## 4. API & Interface
- **Connection**: `Communicator(UnrealCV(ip, port))`
- **Perception**: `comm.get_camera_observation(camera_id, mode="lit")`
- **Control**: 
  - `comm.humanoid_step_forward(agent_id, duration)`
  - `comm.humanoid_rotate(agent_id, angle, direction)`
  - `comm.spawn_agent(...)`

## 5. Review of `sys3_orchestrator.py`
The script `benchmarks/simworld/sys3_orchestrator.py` in the InternNav repo serves as a **"Playground"** or **"Integration Test"** rather than a full benchmark runner.

- **How it works**:
  1. Spawns an agent.
  2. Feeds camera frames to "System 3" (VLM).
  3. System 3 outputs a text instruction (currently parsed as `(x, y)` coordinates).
  4. A simple "geometric executor" (lines 111-138) moves the agent towards that `(x, y)`.
  
- **Gap to Official Benchmark**:
  - It uses a **single hardcoded goal** (`--goal`) instead of loading a dataset of tasks (Start, Goal, Instruction).
  - To fully implement the "SimWorld Benchmark", this script would need to loop over the official validation/test episodes defined by SimWorld (if available publicly) and report success rates/SPL.
  - However, for *developing* System 3 on SimWorld, it is a working and valid starting point.

## 6. Resources
- **Official Docs**: [SimWorld ReadTheDocs](https://simworld.readthedocs.io/en/latest/getting_started/introduction.html)
- **InternNav Benchmark Readme**: `benchmarks/simworld/README.md`

