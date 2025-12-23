## SimWorld benchmark playground (for InternNav Sys3)

This folder is a **small integration playground** between:
- SimWorld’s **Python client** (UnrealCV-based), and
- InternNav’s **System 3** module (VLM-driven instruction orchestrator).

### What this verifies (availability + feasibility)

- **Availability**: We can install and import SimWorld’s Python client in a clean env.
- **Feasibility**: We can run a loop where a VLM (“Sys3”) consumes SimWorld camera frames and emits micro-instructions, and a simple controller executes them inside SimWorld.

### 0) Prerequisite: SimWorld UE server

SimWorld requires a separate Unreal Engine server executable running locally with UnrealCV enabled.
This repo only sets up the **Python-side** integration.

### 1) Create a dedicated Python env

From the InternNav repo root:

```bash
cd /path/to/InternNav
python3 -m venv .venv-simworld
source .venv-simworld/bin/activate
python -m pip install -U pip setuptools wheel
```

### 2) Install SimWorld (editable)

```bash
cd /path/to/InternNav
source .venv-simworld/bin/activate
mkdir -p external
git clone --depth 1 https://github.com/SimWorld-AI/SimWorld external/SimWorld
python -m pip install -e external/SimWorld
```

Notes:
- We patch a couple of small issues in the vendored `external/SimWorld` copy:
  - circular import (`Humanoid` ↔ `Communicator`)
  - optional connection timeout + lazy IPython import in `UnrealCV`

### 3) Smoke test (connect + spawn + capture one frame)

```bash
cd /path/to/InternNav
source .venv-simworld/bin/activate
python benchmarks/simworld/smoke_test.py --ip 127.0.0.1 --port 9000 --timeout_s 5
```

If the UE server is not running, this exits quickly with an error message (expected).
If your UE server is running on another machine, change `--ip` to that machine’s IP.

### 4) Sys3 orchestrator loop (frames -> Sys3 -> actions)

This script uses `internnav/agent/sys3_only_agent.py` (Sys3-only wrapper) so we don’t load large Sys1/Sys2 policies.

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

### Next step ideas

- Replace the simple geometric executor in `sys3_orchestrator.py` with SimWorld’s `LocalPlanner`.
- Swap `Sys3OnlyAgent` for your full `System3Agent` once you want Sys3+Sys2 end-to-end (requires your InternVLA policy + weights + config).


