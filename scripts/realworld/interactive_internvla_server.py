import argparse
import json
import os
import time
from datetime import datetime
import threading
import io
import base64
import pathlib
from pathlib import Path
import sys

import numpy as np
from flask import Flask, jsonify, request, render_template_string
from PIL import Image, ImageDraw

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

app = Flask(__name__)

# Shared state for UI and Agent
class SharedState:
    def __init__(self):
        self.lock = threading.RLock()
        self.latest_image_b64 = None
        self.latest_depth_b64 = None # New depth
        self.overlay_image_b64 = None # New overlay
        # Default instruction from the original script
        self.instruction = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"
        self.instruction_id = 0  # Counter for instruction versions
        self.logs = []
        self.model_logs = []  # New detailed model log
        self.last_update_time = 0
        self.last_action_info = {}

    def add_log(self, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Adding log: {message}")
        with self.lock:
            ts = datetime.now().strftime('%H:%M:%S')
            log_entry = f"[{ts}] {message}"
            self.logs.insert(0, log_entry)
            if len(self.logs) > 100:
                self.logs.pop()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Leave add_log")

    def add_model_log(self, message):
        with self.lock:
            ts = datetime.now().strftime('%H:%M:%S')
            log_entry = f"[{ts}] {message}"
            self.model_logs.insert(0, log_entry)
            if len(self.model_logs) > 50:
                self.model_logs.pop()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Adding log: {message}")
        with self.lock:
            ts = datetime.now().strftime('%H:%M:%S')
            log_entry = f"[{ts}] {message}"
            self.logs.insert(0, log_entry)
            if len(self.logs) > 100:
                self.logs.pop()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Leave add_log")
    
    def set_image(self, img_bytes):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Setting image")
        with self.lock:
            # Convert raw bytes to base64 for HTML display
            self.latest_image_b64 = base64.b64encode(img_bytes).decode('utf-8')
            self.last_update_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Leave set_image")
        
    def set_depth(self, img_bytes):
        with self.lock:
            self.latest_depth_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
    def get_last_update_time(self): 
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Getting last update time")
        with self.lock:
            return self.last_update_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Leave get_last_update_time")

    def get_instruction_data(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Getting instruction data")
        with self.lock:
            return self.instruction, self.instruction_id
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Leave get_instruction_data")

    def set_instruction(self, new_instruction):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Setting instruction to: {new_instruction}")
        with self.lock:
            self.instruction = new_instruction
            self.instruction_id += 1
            self.add_log(f"Instruction updated to <b>inst{self.instruction_id}</b>: {new_instruction[:50]}...")
            return self.instruction_id
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Leave set_instruction")

state = SharedState()
agent = None
args = None

# Variables from original script
idx = 0
start_time = time.time()
output_dir = ''

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>InternVLA Interface</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f6f9; }
        .log-box { height: 250px; overflow-y: auto; background: #fff; border: 1px solid #dee2e6; padding: 10px; font-family: monospace; font-size: 0.9rem; }
        .log-entry { margin-bottom: 4px; border-bottom: 1px solid #eee; padding-bottom: 2px; }
        .image-container { position: relative; width: 100%; min-height: 300px; background: #000; }
        #robot-image { width: 100%; display: block; }
        #overlay-image { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; opacity: 0.8; }
        .status-badge { font-size: 0.8rem; }
    </style>
</head>
<body>
<nav class="navbar navbar-dark bg-dark mb-4">
    <div class="container-fluid">
        <span class="navbar-brand mb-0 h1">InternVLA Real-Time Control</span>
        <span class="text-light" id="connection-status">Waiting for data...</span>
    </div>
</nav>

<div class="container-fluid px-4">
    <div class="row">
        <!-- Left Column: Robot View -->
        <div class="col-lg-7 mb-4">
            <div class="card">
                <div class="card-header bg-primary text-white d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">Robot View</h5>
                    <span id="last-updated" class="badge bg-light text-dark">No Data</span>
                </div>
                <div class="card-body text-center p-2 bg-dark">
                    <div class="image-container">
                        <img id="robot-image" src="" alt="Robot Camera Feed">
                        <img id="overlay-image" src="" style="display:none;">
                    </div>
                </div>
                <div class="card-footer">
                     <small class="text-muted" id="image-info">Waiting for first frame...</small>
                </div>
            </div>
            
            <!-- Depth View -->
            <div class="card mt-4">
                <div class="card-header bg-warning text-dark">
                    <h5 class="mb-0">Depth View</h5>
                </div>
                <div class="card-body text-center p-2 bg-dark">
                    <div class="image-container">
                        <img id="depth-image" src="" alt="Depth Camera Feed" style="width: 100%; display: block;">
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Column: Controls & Logs -->
        <div class="col-lg-5">
            <!-- Instruction Panel -->
            <div class="card mb-4">
                <div class="card-header bg-success text-white">
                    <h5 class="mb-0">Control Instruction</h5>
                </div>
                <div class="card-body">
                    <div class="mb-3">
                        <label for="instruction" class="form-label">
                            Current Goal Instruction: <span class="badge bg-info text-dark" id="inst-tag">inst0</span>
                        </label>
                        <textarea class="form-control" id="instruction" rows="4"></textarea>
                    </div>
                    <div class="d-grid gap-2">
                        <button class="btn btn-success" onclick="updateInstruction()">Update Instruction</button>
                    </div>
                </div>
            </div>

            <!-- Logs Panel -->
            <div class="card mb-4">
                <div class="card-header bg-secondary text-white d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">System Logs</h5>
                    <button class="btn btn-sm btn-light" onclick="clearLogs('logs')">Clear</button>
                </div>
                <div class="card-body p-0">
                    <div id="logs" class="log-box"></div>
                </div>
            </div>

            <!-- Model Details Panel -->
            <div class="card">
                <div class="card-header bg-info text-dark d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">Model Details</h5>
                    <button class="btn btn-sm btn-light" onclick="clearLogs('model-logs')">Clear</button>
                </div>
                <div class="card-body p-0">
                    <div id="model-logs" class="log-box"></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    let currentInstruction = "";
    
    function fetchUpdates() {
        fetch('/ui_data')
            .then(response => response.json())
            .then(data => {
                // ... (data processing logic) ...
                
                // Update Image
                if (data.image) {
                    document.getElementById('robot-image').src = 'data:image/jpeg;base64,' + data.image;
                    
                    const overlay = document.getElementById('overlay-image');
                    if (data.overlay_image) {
                        overlay.src = 'data:image/png;base64,' + data.overlay_image;
                        overlay.style.display = 'block';
                    } else {
                        overlay.style.display = 'none';
                    }
                    
                    if (data.depth_image) {
                        document.getElementById('depth-image').src = 'data:image/jpeg;base64,' + data.depth_image;
                    }

                    document.getElementById('last-updated').innerText = 'Last: ' + new Date().toLocaleTimeString();
                    document.getElementById('connection-status').innerText = 'Connected';
                    document.getElementById('connection-status').className = 'text-success fw-bold';
                    document.getElementById('image-info').innerText = 'Resolution: ' + data.resolution;
                    
                    // Clear stale warning if image is fresh
                    const warning = document.getElementById('stale-warning');
                    if (warning) warning.style.display = 'none';
                }
                
                // Check if image is stale
                const now = Date.now() / 1000;
                const lastUpdate = data.last_update_time;
                if (now - lastUpdate > 1.0) {
                     let warning = document.getElementById('stale-warning');
                     if (!warning) {
                         warning = document.createElement('div');
                         warning.id = 'stale-warning';
                         warning.className = 'alert alert-warning position-absolute top-50 start-50 translate-middle';
                         warning.style.zIndex = '1000';
                         document.querySelector('.card-body').appendChild(warning);
                         // Ensure parent is relative for absolute positioning
                         document.querySelector('.card-body').style.position = 'relative'; 
                     }
                     warning.style.display = 'block';
                     warning.innerText = `Image lost for ${Math.round(now - lastUpdate)} seconds`;
                }
                
                // Update Logs
                const logsDiv = document.getElementById('logs');
                logsDiv.innerHTML = data.logs.map(log => `<div class="log-entry">${log}</div>`).join('');
                
                // Update Model Logs
                const modelLogsDiv = document.getElementById('model-logs');
                if (data.model_logs) {
                    modelLogsDiv.innerHTML = data.model_logs.map(log => `<div class="log-entry">${log}</div>`).join('');
                }

                // Sync Instruction (only if not focused to avoid overwriting user typing)
                if (document.activeElement.id !== 'instruction' && currentInstruction !== data.instruction) {
                    document.getElementById('instruction').value = data.instruction;
                    currentInstruction = data.instruction;
                }
                
                // Update Instruction Tag
                if (data.instruction_id !== undefined) {
                    document.getElementById('inst-tag').innerText = 'inst' + data.instruction_id;
                }
            })
            .catch(err => {
                console.error(err);
                document.getElementById('connection-status').innerText = 'Disconnected';
                document.getElementById('connection-status').className = 'text-danger fw-bold';
            })
            .finally(() => {
                // Schedule next update ONLY after current one finishes
                setTimeout(fetchUpdates, 500);
            });
    }

    function updateInstruction() {
        const newInst = document.getElementById('instruction').value;
        const btn = document.querySelector('button[onclick="updateInstruction()"]');
        const originalText = "Update Instruction";
        
        btn.disabled = true;
        btn.innerText = "Updating...";

        // Create a timeout controller
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout

        fetch('/set_instruction', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({instruction: newInst}),
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeoutId);
            if (!response.ok) throw new Error("Network response was not ok");
            return response.json();
        })
        .then(data => {
            currentInstruction = newInst;
            const logsDiv = document.getElementById('logs');
            const tempLog = document.createElement('div');
            tempLog.className = 'log-entry text-success';
            tempLog.innerText = `[UI] Instruction update sent...`;
            logsDiv.insertBefore(tempLog, logsDiv.firstChild);
        })
        .catch(err => {
            if (err.name === 'AbortError') {
                console.error('Update request timed out');
            } else {
                console.error(err);
            }
        })
        .finally(() => {
             // ALWAYS re-enable button
             btn.disabled = false;
             btn.innerText = originalText;
        });
    }

    function clearLogs(id) {
        document.getElementById(id).innerHTML = '';
    }

    // Initial load
    fetch('/ui_data').then(r => r.json()).then(d => {
        document.getElementById('instruction').value = d.instruction;
        currentInstruction = d.instruction;
    });

    // Start polling loop
    fetchUpdates();
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/ui_data")
def ui_data():
    # print(f"[{datetime.now().strftime('%H:%M:%S')}] Getting UI data")
    
    # Copy data out of the lock critical section to minimize blocking
    img = None
    depth_img = None
    overlay_img = None
    logs = []
    instr = ""
    iid = 0
    lut = 0
    
    # Try non-blocking acquisition first just to see, but here we do standard blocking acquire
    if not state.lock.acquire(timeout=2.0):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] FAILED TO ACQUIRE LOCK in ui_data after 2s")
        # Return last known state or error? Let's just return empty/None to avoid hanging browser
        return jsonify({"error": "server_busy_locked"}), 503

    try:
        #         print(f"[{datetime.now().strftime('%H:%M:%S')}] Acquired lock in ui_data")
        img = state.latest_image_b64
        depth_img = state.latest_depth_b64
        overlay_img = state.overlay_image_b64
        logs = list(state.logs) # Create a shallow copy
        model_logs = list(state.model_logs)
        instr = state.instruction
        iid = state.instruction_id
        lut = state.last_update_time
    finally:
        state.lock.release()
    
    # print(f"[{datetime.now().strftime('%H:%M:%S')}] Released lock in ui_data, jsonifying...")
    
    data = jsonify({
        "image": img,
        "depth_image": depth_img,
        "overlay_image": overlay_img,
        "logs": logs,
        "model_logs": model_logs,
        "instruction": instr,
        "instruction_id": iid,
        "resolution": "Unknown", # could add resolution info if saved
        "last_update_time": lut
    })
    # print(f"[{datetime.now().strftime('%H:%M:%S')}] Leave ui_data")
    return data

@app.route("/set_instruction", methods=['POST'])
def set_instruction():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Received /set_instruction request")
    try:
        data = request.json
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data in /set_instruction: {data}")
        if 'instruction' in data:
            state.set_instruction(data['instruction'])
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Instruction set successfully")
            return jsonify({"status": "success"})
        return jsonify({"status": "error"}), 400
    except Exception as e:
        print(f"Error in set_instruction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time
    
    # --- Data Ingestion ---
    start_process_time = time.time()
    
    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)

    # Read bytes for UI and Model
    img_bytes = image_file.read()
    image_file.seek(0) # Reset pointer for PIL
    
    # Update UI state immediately
    state.set_image(img_bytes)
    
    # Process Image for Model
    image = Image.open(io.BytesIO(img_bytes))
    image = image.convert('RGB')
    image = np.asarray(image)
    resolution = f"{image.shape[1]}x{image.shape[0]}"

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth_raw = np.asarray(depth)
    
    # Visualization: Normalize to 0-255
    try:
        depth_vis = depth_raw.astype(np.float32)
        d_min, d_max = depth_vis.min(), depth_vis.max()
        if d_max > d_min:
            depth_vis = (depth_vis - d_min) / (d_max - d_min) * 255.0
        else:
            depth_vis = np.zeros_like(depth_vis)
            
        depth_vis_pil = Image.fromarray(depth_vis.astype(np.uint8))
        buf = io.BytesIO()
        depth_vis_pil.save(buf, format='JPEG')
        state.set_depth(buf.getvalue())
    except Exception as e:
        print(f"Error creating depth visualization: {e}")

    depth = depth_raw.astype(np.float32) / 10000.0
    
    read_time = time.time() - start_process_time
    # state.add_log(f"Received request. Read time: {read_time:.3f}s. Res: {resolution}")

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    # Use dynamic instruction from SharedState
    instruction, inst_id = state.get_instruction_data()
    
    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        state.add_log("<b>RESET MODEL</b>")
        agent.reset()

    idx += 1
    look_down = False
    t0 = time.time()
    
    # --- Agent Inference ---
    dual_sys_output = agent.step(
        image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
    )
    
    # Dual system logic (look down check)
    if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
        state.add_log("Action [5] detected, triggering look_down step...")
        look_down = True
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
        )

    # --- Response Construction ---
    json_output = {}
    log_msg = f"Step {idx} (inst{inst_id}): "
    
    # Visualization Overlay Logic
    overlay_b64 = None
    
    # Create overlay canvas early so all visualization paths can draw on it
    overlay = Image.new('RGBA', (image.shape[1], image.shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    has_overlay_content = False
    
    # Get action map dynamically from agent if available, else fallback
    action_map = {}
    if hasattr(agent, 'actions2idx'):
        # Invert the map: {'STOP': [0]} -> {0: 'STOP'}
        # Note: actions2idx values are lists like [0], so we take the first element
        for k, v in agent.actions2idx.items():
            if isinstance(v, list) and len(v) > 0:
                action_map[v[0]] = k
            else:
                action_map[v] = k

    if dual_sys_output.output_action is not None:
        json_output['discrete_action'] = dual_sys_output.output_action
        
        # Translate discrete action to text using dynamic map
        action_texts = [action_map.get(a, f"Unknown({a})") for a in dual_sys_output.output_action]
        action_str = ", ".join(action_texts)
        
        log_msg += f"Action <b>{action_str}</b> ({dual_sys_output.output_action})"
        state.add_model_log(f"Discrete Action: {action_str} {dual_sys_output.output_action}")

    elif hasattr(dual_sys_output, 'output_trajectory'):
        traj = dual_sys_output.output_trajectory
        json_output['trajectory'] = traj.tolist()
        log_msg += f"Trajectory generated ({len(json_output['trajectory'])} points)"
        
        # Detailed Model Log
        sample_str = ""
        if len(traj) > 0:
            # Sample first and last point
            sample_str = f"Start: {np.round(traj[0], 2)}, End: {np.round(traj[-1], 2)}"
        
        state.add_model_log(f"Trajectory Shape: {traj.shape}, Type: Continuous")
        state.add_model_log(f"Sample: {sample_str}")
            
        if dual_sys_output.output_pixel is not None:
            json_output['pixel_goal'] = dual_sys_output.output_pixel
            log_msg += ", Pixel Goal Set"
            state.add_model_log(f"Pixel Goal: {dual_sys_output.output_pixel}")
    else:
        log_msg += "No action/trajectory"
        state.add_model_log("No valid output from model.")
    
    # --- Consolidated Visualization Logic ---
    # 1. Draw Reference Pixel (Red) - ALWAYS check
    if hasattr(dual_sys_output, 'reference_output_pixel') and dual_sys_output.reference_output_pixel is not None:
        px = dual_sys_output.reference_output_pixel
        state.add_model_log(f"Reference Pixel Goal (Raw): {px}")
        
        if len(px) > 0:
            # Model provides [row, col]; convert to (x=col, y=row) for PIL
            y_raw, x_raw = px[0], px[1]
            x, y = x_raw, y_raw
            state.add_model_log(f"Scaled Pixel Goal: [{x:.1f}, {y:.1f}]")
            
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                r = 10
                draw.ellipse((x-r, y-r, x+r, y+r), outline='red', width=3)
                draw.line((x-r, y, x+r, y), fill='red', width=3)
                draw.line((x, y-r, x, y+r), fill='red', width=3)
                has_overlay_content = True
            else:
                state.add_model_log(
                    f"WARNING: Ref Pixel [{x:.1f}, {y:.1f}] out of bounds for {image.shape[1]}x{image.shape[0]}"
                )

    # 2. Draw Final Pixel Goal (Cyan) - Only if trajectory/pixel goal output exists
    if hasattr(dual_sys_output, 'output_pixel') and dual_sys_output.output_pixel is not None:
        px = dual_sys_output.output_pixel
        state.add_model_log(f"Visualizing Pixel Goal: {px}")
        x, y = px[0], px[1]
        
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            r = 10
            draw.ellipse((x-r, y-r, x+r, y+r), outline='cyan', width=3)
            draw.line((x-r, y, x+r, y), fill='cyan', width=3)
            draw.line((x, y-r, x, y+r), fill='cyan', width=3)
            has_overlay_content = True
        else:
             state.add_model_log(f"WARNING: Final Pixel {px} out of bounds")

    # Save if content exists
    if has_overlay_content:
        buf = io.BytesIO()
        overlay.save(buf, format='PNG')
        overlay_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Update state with new overlay (and old image since we don't have new raw bytes here easily without reading again, 
    # BUT wait: set_image updates the image. We should update the overlay separately or update set_image to take optional overlay)
    # Actually, set_image was called at start. We can add a method set_overlay.
    # For now, let's modify set_image to accept overlay or add set_overlay method.
    
    # Better: Update the state with the computed overlay
    with state.lock:
        state.overlay_image_b64 = overlay_b64

    t1 = time.time()
    generate_time = t1 - t0
    
    state.add_log(f"{log_msg} ({generate_time:.3f}s)")
    
    # print(f"dual sys step {generate_time}")
    # print(f"json_output {json_output}")
    
    return jsonify(json_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default=CHECKPOINT_DIR / "InternVLA-N1")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=8)
    parser.add_argument("--port", type=int, default=5801)
    args = parser.parse_args()

    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    
    print("Initializing InternVLA Agent...")
    agent = InternVLAN1AsyncAgent(args)
    
    # Warmup
    print("Warming up agent...")
    agent.step(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640), dtype=np.uint8),
        np.eye(4),
        "hello",
    )
    agent.reset()
    print("Agent ready.")
    print(f"Server starting on port {args.port}...")
    print(f"Access the UI at http://0.0.0.0:{args.port}/")
    
    app.run(host='0.0.0.0', port=args.port, threaded=True)
