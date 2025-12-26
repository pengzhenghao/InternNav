## SimWorld Benchmark Playground (for InternNav System 3)

This folder provides a **small integration playground** between:
- **SimWorld's Python client** (UnrealCV-based communication with UE server)
- **InternNav's System 3** (VLM-driven instruction orchestrator)

### 🎯 What This Tests

- **Availability**: Can we install and import SimWorld's Python client in Python 3.8+?
- **Feasibility**: Can we run a loop where System 3 (VLM) consumes SimWorld camera frames, emits navigation instructions, and controls a humanoid agent?
- **Performance**: How well does System 3 navigate in a realistic 3D environment?

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────┐
│  Unreal Engine Server (SimWorld)        │  ← Download & run separately
│  - 3D environment with physics           │
│  - Humanoid character                    │
│  - UnrealCV plugin (port 9000)          │
└─────────────────────────────────────────┘
                    ↕ (UnrealCV protocol over TCP)
┌─────────────────────────────────────────┐
│  Python Client (this repo)               │
│  - smoke_test.py: Test connectivity     │
│  - sys3_orchestrator.py: System 3 loop  │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│  InternNav System 3 (VLM)               │
│  - Observes: camera frames               │
│  - Outputs: navigation instructions      │
│  - Uses: Qwen3-VL or similar            │
└─────────────────────────────────────────┘
```

---

## 🚀 Setup Guide

### Prerequisites

- **Linux or Windows** system
- **GPU**: At least 6 GB VRAM (8 GB+ recommended)
- **Memory**: 32 GB+ recommended
- **Disk**: ~50 GB for SimWorld UE server
- **Python**: 3.8 or later (we've patched SimWorld for 3.8 compatibility)
- **Ports**: 9000 and 9001 (for UnrealCV)

### Step 0: Download SimWorld UE Server

SimWorld requires a separate Unreal Engine server executable. This is the 3D simulator that runs independently.

**Download for Linux:**
```bash
cd ~/Downloads  # or your preferred location
wget https://simworld-release.s3.us-east-1.amazonaws.com/SimWorld-Linux-v0_1_0-Foundation.zip
unzip SimWorld-Linux-v0_1_0-Foundation.zip
cd SimWorld-Linux-v0_1_0-Foundation
chmod +x gym_citynav.sh  # Make the launch script executable
```

**Download for Windows:**
```bash
# Download from:
https://simworld-release.s3.us-east-1.amazonaws.com/SimWorld-Win64-v0_1_0-Foundation.zip
# Unzip it and navigate to the folder
```

---

### Running SimWorld Server (Multiple Options)

The SimWorld server is an Unreal Engine application. You can run it in different modes depending on your needs:

#### Option 1: Windowed Mode (Recommended for Development)

**Linux:**
```bash
cd ~/Downloads/SimWorld-Linux-v0_1_0-Foundation
./gym_citynav.sh -windowed -ResX=1280 -ResY=720
```

**Windows:**
```bash
gym_citynav.exe -windowed -ResX=1280 -ResY=720
```

This opens a 1280x720 window that you can minimize or move to another workspace.

#### Option 2: Headless Mode (No GUI, Best for Servers/Remote)

**Linux only:**
```bash
cd ~/Downloads/SimWorld-Linux-v0_1_0-Foundation
./gym_citynav.sh -RenderOffScreen -nullrhi
```

This runs without any display window. Perfect for:
- Remote servers without display
- Background processes
- CI/CD pipelines
- When you only need camera observations via Python API

**Note:** On Windows, headless mode requires additional setup. Use windowed mode instead.

#### Option 3: Fullscreen Mode (Default, Not Recommended)

**Linux:**
```bash
cd ~/Downloads/SimWorld-Linux-v0_1_0-Foundation
./gym_citynav.sh
```

**Windows:**
```bash
gym_citynav.exe
```

⚠️ **Warning:** This takes over your entire screen and can be disruptive. Press `Alt+F4` (Linux) or `Alt+Enter` (Windows) to exit fullscreen mode once running.

#### Option 4: Custom Resolution + Windowed

```bash
# Linux
./gym_citynav.sh -windowed -ResX=800 -ResY=600

# Windows  
gym_citynav.exe -windowed -ResX=800 -ResY=600
```

#### Other Useful Unreal Engine Command-Line Flags

```bash
-windowed              # Run in windowed mode (not fullscreen)
-ResX=1920 -ResY=1080  # Set window resolution
-RenderOffScreen       # Headless mode (no window at all)
-nullrhi               # Use null rendering hardware interface (headless)
-log                   # Show console log window
-NoVSync               # Disable vertical sync (faster rendering)
-NoSound               # Disable sound (saves resources)
-silent                # Suppress most console output
```

**Example for maximum performance (headless, no sound, no vsync):**
```bash
# Linux
./gym_citynav.sh -RenderOffScreen -nullrhi -NoSound -NoVSync

# For windowed with performance optimizations
./gym_citynav.sh -windowed -ResX=1280 -ResY=720 -NoSound -NoVSync
```

---

### What You Should See

Once the server starts, you should see:
- **Windowed mode:** A Unreal Engine window showing the 3D environment
- **Headless mode:** Just console logs, no window
- **Console output:** Something like:
  ```
  LogInit: Build: ++UE4+Release-4.27
  LogUnrealCV: Starting UnrealCV server on port 9000
  ```

**The viewport size in the UE window may appear small** - this is normal! The actual resolution of camera captures is controlled by your Python code (e.g., `resolution=(640, 360)` in smoke_test.py), not the UE window size.

**Keep this server running** in a separate terminal/window for all subsequent steps.

---

### Step 1: Create Python Environment

From the InternNav repo root:

```bash
cd /path/to/InternNav
python3 -m venv .venv-simworld
source .venv-simworld/bin/activate  # On Windows: .venv-simworld\Scripts\activate
python -m pip install -U pip setuptools wheel
```

---

### Step 2: Install SimWorld Python Client

The SimWorld Python library is already vendored in `external/SimWorld` with compatibility patches applied.

**Patches we've applied:**
- ✅ Fixed circular import between `Humanoid` ↔ `Communicator`
- ✅ Added Python 3.8 compatibility (fixed `list[X]` → `from __future__ import annotations`)
- ✅ Made IPython import optional (no longer required unless using Jupyter)
- ✅ Added `connect_timeout_s` parameter to `UnrealCV`

**Install it:**
```bash
cd /path/to/InternNav
source .venv-simworld/bin/activate
python -m pip install -e external/SimWorld
```

**Verify installation:**
```bash
python -c "from simworld import Communicator; print('✓ SimWorld installed successfully')"
```

---

### Step 3: Smoke Test (Connectivity Check)

With the **UE server running**, test if Python can connect:

```bash
cd /path/to/InternNav
source .venv-simworld/bin/activate
python benchmarks/simworld/smoke_test.py --ip 127.0.0.1 --port 9000 --timeout_s 5
```

**Expected output if server is running:**
```
[smoke_test] OK. Saved one RGB frame to benchmarks/simworld/out_smoke_rgb.npy with shape=(360, 640, 3) dtype=uint8
```
**Exit code: 0** ← Success!

**Expected output if server is NOT running:**
```
ERROR: Can not connect to ('127.0.0.1', 9000)
[smoke_test] Could not connect to UE UnrealCV server: Failed to connect...
[smoke_test] This is expected if the UE server executable is not running yet.
```
**Exit code: 2** ← Expected when server isn't running

**Note:** The many `ERROR:__init__:252` messages are just verbose logging from the unrealcv library as it retries the connection. They're normal noise when the server isn't available.

---

### Step 4: Run System 3 Orchestrator

This script runs your actual System 3 benchmark:

**What it does:**
1. Connects to SimWorld UE server
2. Spawns a humanoid agent at origin (0, 0)
3. **Runs System 3 loop:**
   - Captures camera frame from SimWorld
   - Feeds frame to System 3 VLM
   - System 3 outputs navigation instruction
   - Simple geometric executor moves humanoid
   - Repeats until goal reached or max steps

**Prerequisites:**
- SimWorld UE server running (from Step 0)
- VLM server running (vLLM with Qwen model)

**Run it:**
```bash
cd /path/to/InternNav
source .venv-simworld/bin/activate

PYTHONPATH="$(pwd)" \
  VLLM_API_URL=http://localhost:8080/v1 \
  VLLM_API_KEY=EMPTY \
  MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct \
  python benchmarks/simworld/sys3_orchestrator.py \
    --goal "Go to (1700, -1700)."
```

**Command-line options:**
```bash
--ip 127.0.0.1              # SimWorld server IP
--port 9000                 # SimWorld UnrealCV port
--timeout_s 10              # Connection timeout
--vlm_base_url http://...   # Your VLM API endpoint
--vlm_api_key EMPTY         # VLM API key
--vlm_model_name ...        # Model name for VLM
--goal "Go to (x, y)."      # Navigation goal
--max_steps 50              # Max navigation steps
--dt 0.2                    # Time between steps (seconds)
```

**Expected output:**
```
[sys3_orchestrator] goal='Go to (1700, -1700).' parsed_target=(1700.0, -1700.0)
[sys3] step=000 status=RUNNING instr='Go to (1700, -1700)'
[sys3] thought=I can see an open area ahead. I should move forward to reach the target...
[executor] rotating left by 25.0 degrees
[sys3] step=001 status=RUNNING instr='Continue forward'
[executor] stepping forward
...
[executor] near target (dist=85.3); stopping
[sys3_orchestrator] stopping: status=DONE
```

---

## 📁 What Each Script Does

### `smoke_test.py`
**Purpose:** Verify Python-SimWorld connectivity
- Imports SimWorld Python library
- Attempts to connect to UE server (with timeout)
- Spawns one humanoid
- Captures one camera frame
- Saves frame to `benchmarks/simworld/out_smoke_rgb.npy`

**When to use:** Testing setup, debugging connection issues

### `sys3_orchestrator.py`
**Purpose:** Benchmark System 3 navigation in SimWorld
- Connects to UE server
- Runs System 3 in a perception-action loop
- Uses `Sys3OnlyAgent` (lightweight wrapper, no Sys1/Sys2 policies)
- Simple geometric executor: rotates toward target, steps forward
- Stops when goal reached or max steps exceeded

**When to use:** Benchmarking System 3 instruction generation and navigation

**Key components:**
- **System 3**: `internnav/agent/sys3_only_agent.py`
  - Wraps `internnav/agent/system3/` (planner/compiler/critic)
  - Feeds RGB frames to VLM
  - Outputs navigation instructions
- **Executor**: Lines 111-138 in `sys3_orchestrator.py`
  - Parses instructions like "Go to (x, y)"
  - Rotates humanoid toward target
  - Steps forward when aligned

---

## 🔧 Customization & Next Steps

### Option 1: Better Executor (Path Planning)
Replace the simple geometric executor with SimWorld's `LocalPlanner`:

```python
from simworld.local_planner.local_planner import LocalPlanner

# In sys3_orchestrator.py, replace lines 111-138 with:
local_planner = LocalPlanner(
    agent=agent,
    communicator=comm,
    config=your_config
)
local_planner.execute(instruction)
```

Benefits: Obstacle avoidance, smoother paths, better handling of complex environments

### Option 2: Full System 3 Agent (Sys1+2+3)
Use the complete agent with all subsystems:

```python
from internnav.agent.system3_agent import System3Agent

# Replace Sys3OnlyAgent with System3Agent
# This loads Sys1 (low-level policy) and Sys2 (mid-level policy) as well
```

Benefits: End-to-end evaluation, hierarchical decision-making

### Option 3: Custom Scenarios
- **Different goals:** Change `--goal "Go to (x, y)."`
- **Add obstacles:** Use SimWorld's scene generation API
- **Multiple agents:** Spawn additional humanoids/vehicles
- **Traffic simulation:** Enable SimWorld's traffic system
- **Custom prompts:** Modify `System3PromptProfile` in System 3 config

### Option 4: Evaluation Metrics
Add logging for:
- Steps to goal
- Success rate
- Collision count (use `comm.get_collision_number()`)
- Path efficiency
- VLM query latency

---

## 🐛 Troubleshooting

### "Connection refused" errors
- **Issue:** SimWorld UE server not running
- **Fix:** Start the server executable (Step 0) before running Python scripts

### "ImportError: circular import"
- **Issue:** Using upstream SimWorld without our patches
- **Fix:** Make sure you installed from `external/SimWorld` (our patched version)

### "TypeError: 'type' object is not subscriptable"
- **Issue:** Python 3.8 with modern type hints (`list[X]`)
- **Fix:** Our patches add `from __future__ import annotations` to fix this

### "ModuleNotFoundError: No module named 'IPython'"
- **Issue:** Old version without optional IPython import
- **Fix:** Our patches make IPython optional. Reinstall from `external/SimWorld`

### VLM not responding
- **Issue:** VLM server not running or wrong URL
- **Fix:** Check `VLLM_API_URL` and ensure your vLLM server is running

### Humanoid not moving
- **Issue:** Instructions not being parsed or executor logic failing
- **Fix:** Check System 3 output format matches `(x, y)` regex pattern

### Small viewport/black screen in UE window
- **Issue:** The UE window shows a small viewport (e.g., 640px wide) with black areas around it
- **This is NORMAL!** ✅ The viewport size is controlled by Python code (`resolution=(640, 360)` in smoke_test.py)
- **Fix (if desired):** Change resolution in your Python scripts, or resize the UE window. The black areas are just unused window space and don't affect functionality.

### Fullscreen mode is annoying
- **Issue:** SimWorld takes over your entire screen
- **Fix:** Run with `-windowed` flag: `./gym_citynav.sh -windowed -ResX=1280 -ResY=720`
- **Or:** Run headless (no window): `./gym_citynav.sh -RenderOffScreen -nullrhi`

---

## 📚 Related Files in InternNav

| Path | Purpose |
|------|---------|
| `internnav/agent/sys3_only_agent.py` | Lightweight System 3 wrapper (VLM only) |
| `internnav/agent/system3_agent.py` | Full agent (Sys1+Sys2+Sys3) |
| `internnav/agent/system3/` | Core System 3 modules |
| `internnav/agent/system3/planner.py` | High-level planning |
| `internnav/agent/system3/compiler.py` | Instruction compilation |
| `internnav/agent/system3/critic.py` | Progress evaluation |
| `internnav/agent/system3/schemas.py` | Data structures |
| `internnav/agent/system3/prompt_profiles.py` | VLM prompt templates |

---

## 🎓 Understanding the Code Flow

```python
# 1. Connect to SimWorld
ucv = UnrealCV(ip="127.0.0.1", port=9000, connect_timeout_s=10)
comm = Communicator(ucv)

# 2. Spawn humanoid
agent = Humanoid(position=Vector(0, 0), direction=Vector(1, 0))
comm.spawn_agent(agent, type="humanoid")

# 3. Initialize System 3
sys3 = Sys3OnlyAgent(config)
sys3.set_goal("Go to (1700, -1700).")

# 4. Main loop
for step in range(max_steps):
    # Get observation from SimWorld
    img = comm.get_camera_observation(agent.camera_id, "lit")
    
    # System 3 processes observation → instruction
    instruction, status, thought = sys3.update_instruction({"rgb": img})
    
    # Executor moves humanoid based on instruction
    target_xy = parse_xy(instruction)
    comm.humanoid_rotate(agent.id, angle, direction)
    comm.humanoid_step_forward(agent.id, duration)
    
    # Check if done
    if status == "DONE":
        break
```

---

## 📖 Additional Resources

- **SimWorld Documentation:** https://simworld.readthedocs.io/
- **SimWorld GitHub:** https://github.com/SimWorld-AI/SimWorld
- **SimWorld Paper:** https://arxiv.org/abs/2512.01078
- **UnrealCV Protocol:** http://unrealcv.org/

---

## ✅ Quick Start Checklist

- [ ] Downloaded and unzipped SimWorld UE server
- [ ] Created Python virtual environment (`.venv-simworld`)
- [ ] Installed patched SimWorld: `pip install -e external/SimWorld`
- [ ] Started SimWorld UE server (keep running)
- [ ] Ran smoke test successfully (exit code 0)
- [ ] Started VLM server (vLLM)
- [ ] Ran `sys3_orchestrator.py` and saw System 3 navigate
- [ ] Analyzed System 3 performance and thoughts

**You're ready to benchmark System 3!** 🚀
