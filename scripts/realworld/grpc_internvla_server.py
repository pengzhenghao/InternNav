import argparse
import json
import os
import time
from datetime import datetime
import threading
import io
import base64
import pathlib
import sys
import logging
from concurrent import futures
import traceback

import numpy as np
import grpc
from flask import Flask, jsonify, render_template_string
from PIL import Image, ImageDraw

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))
# Add protos dir to path so generated code imports work
sys.path.insert(0, str(project_root / 'scripts/realworld/protos'))

# Import generated gRPC code
import internvla_stream_pb2
import internvla_stream_pb2_grpc

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

ROOT_DIR = project_root
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Shared State (Same as HTTP server) ---
class SharedState:
    def __init__(self):
        self.lock = threading.RLock()
        self.latest_image_b64 = None
        self.latest_depth_b64 = None # New depth
        self.overlay_image_b64 = None
        self.instruction = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"
        self.instruction_id = 0
        self.logs = []
        self.model_logs = []
        self.last_update_time = 0

    def add_log(self, message):
        with self.lock:
            ts = datetime.now().strftime('%H:%M:%S')
            self.logs.insert(0, f"[{ts}] {message}")
            if len(self.logs) > 100: self.logs.pop()

    def add_model_log(self, message):
        with self.lock:
            ts = datetime.now().strftime('%H:%M:%S')
            self.model_logs.insert(0, f"[{ts}] {message}")
            if len(self.model_logs) > 50: self.model_logs.pop()
    
    def set_image(self, img_bytes):
        with self.lock:
            self.latest_image_b64 = base64.b64encode(img_bytes).decode('utf-8')
            self.last_update_time = time.time()
            
    def set_depth(self, img_bytes):
        with self.lock:
            self.latest_depth_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
    def get_instruction_data(self):
        with self.lock:
            return self.instruction, self.instruction_id

    def set_instruction(self, new_instruction):
        with self.lock:
            self.instruction = new_instruction
            self.instruction_id += 1
            self.add_log(f"Instruction updated to inst{self.instruction_id}")
            return self.instruction_id

state = SharedState()
agent = None
args = None

# --- gRPC Servicer ---
class InternVLAStreamServicer(internvla_stream_pb2_grpc.InternVLAStreamServicer):
    def __init__(self, agent):
        self.agent = agent
        self.idx = 0
        self.start_time = time.time()
        self.output_dir = ''
        
    def Stream(self, request_iterator, context):
        client_id = "unknown"
        logger.info("New client connected to stream.")
        
        try:
            for msg in request_iterator:
                response = None
                
                if msg.HasField('heartbeat'):
                    # Echo heartbeat
                    response = internvla_stream_pb2.ServerMessage(
                        heartbeat=internvla_stream_pb2.Heartbeat(
                            seq=msg.heartbeat.seq,
                            client_id="server"
                        )
                    )
                    yield response
                    
                elif msg.HasField('instruction'):
                    # Client updating instruction? 
                    # Usually server UI sets instruction, but let's allow client too if needed.
                    state.set_instruction(msg.instruction.text)
                    
                elif msg.HasField('frame'):
                    # Main inference loop
                    frame = msg.frame
                    client_id = frame.client_id
                    
                    # 1. Update State & UI
                    state.set_image(frame.image_jpeg)
                    
                    # 2. Reset if requested
                    if frame.reset:
                        self.idx = 0
                        self.start_time = time.time()
                        self.output_dir = f'output/runs_grpc_{datetime.now().strftime("%m-%d-%H%M")}'
                        os.makedirs(self.output_dir, exist_ok=True)
                        state.add_log("<b>RESET MODEL</b>")
                        self.agent.reset()

                    self.idx += 1
                    
                    # 3. Decode Images
                    try:
                        image = Image.open(io.BytesIO(frame.image_jpeg)).convert('RGB')
                        image_np = np.asarray(image)
                        
                        depth = Image.open(io.BytesIO(frame.depth_png)).convert('I')
                        depth_raw = np.asarray(depth)

                        # Visualization
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
                            logger.error(f"Depth viz error: {e}")
                        
                        depth_np = depth_raw.astype(np.float32) / 10000.0
                        
                        # 4. Get Instruction
                        instruction, inst_id = state.get_instruction_data()
                        
                        # 5. Inference
                        # Assuming camera pose is fixed/identity for now as in original script
                        camera_pose = np.eye(4)
                        
                        t0 = time.time()
                        # Pass look_down=False initially
                        output = self.agent.step(
                            image_np, depth_np, camera_pose, instruction, 
                            intrinsic=args.camera_intrinsic, look_down=False
                        )
                        
                        # Check for look down trigger
                        if output.output_action == [5]:
                            state.add_log("Action [5] detected, triggering look_down step...")
                            output = self.agent.step(
                                image_np, depth_np, camera_pose, instruction, 
                                intrinsic=args.camera_intrinsic, look_down=True
                            )
                        
                        t1 = time.time()
                        
                        # 6. Construct Response
                        action_msg = internvla_stream_pb2.Action()
                        action_msg.seq = frame.seq
                        
                        log_parts = []
                        
                        if output.output_action is not None:
                            action_msg.discrete_action.extend(output.output_action)
                            log_parts.append(f"Act: {output.output_action}")
                            
                        if hasattr(output, 'output_trajectory') and output.output_trajectory is not None:
                            # Flatten trajectory
                            traj_flat = output.output_trajectory.flatten().tolist()
                            action_msg.trajectory.extend(traj_flat)
                            log_parts.append(f"Traj len: {len(output.output_trajectory)}")
                            
                        if hasattr(output, 'output_pixel') and output.output_pixel is not None:
                            action_msg.pixel_goal.extend(output.output_pixel)
                        
                        # Visualization for UI (Overlay)
                        overlay_b64 = self._generate_overlay(image_np, output)
                        if overlay_b64:
                            action_msg.overlay_png_b64 = overlay_b64
                            with state.lock:
                                state.overlay_image_b64 = overlay_b64
                        
                        action_msg.log = f"Step {self.idx} (inst{inst_id}): " + ", ".join(log_parts) + f" ({t1-t0:.3f}s)"
                        state.add_log(action_msg.log)
                        
                        yield internvla_stream_pb2.ServerMessage(action=action_msg)
                        
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        traceback.print_exc()
                        # Send error log back?
        except Exception as e:
            logger.error(f"Stream error with client {client_id}: {e}")
            traceback.print_exc()
        finally:
            logger.info(f"Stream closed for client {client_id}")

    def _generate_overlay(self, image_np, output):
        # Similar to original script's overlay logic
        try:
            overlay = Image.new('RGBA', (image_np.shape[1], image_np.shape[0]), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            has_content = False
            
            # Draw Reference Pixel (Red)
            if hasattr(output, 'reference_output_pixel') and output.reference_output_pixel is not None:
                px = output.reference_output_pixel
                if len(px) >= 2:
                    y, x = px[0], px[1] # row, col -> y, x
                    # Swap for drawing? Original script: "y_raw, x_raw = px[0], px[1]; x, y = x_raw, y_raw"
                    # Wait, px is [row, col]. PIL uses (x, y). So x is col, y is row.
                    # x_raw = px[1], y_raw = px[0].
                    r = 10
                    draw.ellipse((x-r, y-r, x+r, y+r), outline='red', width=3)
                    has_content = True

            # Draw Final Pixel Goal (Cyan)
            if hasattr(output, 'output_pixel') and output.output_pixel is not None:
                px = output.output_pixel
                if len(px) >= 2:
                    x, y = px[0], px[1] # Original script used this directly? 
                    # Let's check original script: "x, y = px[0], px[1]" for output_pixel.
                    # Be consistent with original.
                    r = 10
                    draw.ellipse((x-r, y-r, x+r, y+r), outline='cyan', width=3)
                    has_content = True
            
            if has_content:
                buf = io.BytesIO()
                overlay.save(buf, format='PNG')
                return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Overlay generation failed: {e}")
        return None

# --- Flask UI Routes ---
# Reusing the HTML template from the original file would be good, 
# but for brevity let's just use the same template logic or import it?
# The original file has HTML_TEMPLATE. Let's define a similar one.

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>InternVLA gRPC Server</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <meta http-equiv="refresh" content="60"> <!-- Auto refresh page every 60s just in case -->
    <style>
        body { background-color: #f4f6f9; }
        .log-box { height: 250px; overflow-y: auto; background: #fff; border: 1px solid #dee2e6; padding: 10px; font-family: monospace; font-size: 0.9rem; }
        .image-container { position: relative; width: 100%; min-height: 300px; background: #000; }
        #robot-image { width: 100%; display: block; }
        #overlay-image { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; opacity: 0.8; }
    </style>
</head>
<body>
<nav class="navbar navbar-dark bg-dark mb-4">
    <div class="container-fluid"><span class="navbar-brand mb-0 h1">InternVLA gRPC Server</span></div>
</nav>
<div class="container-fluid px-4">
    <div class="row">
        <div class="col-lg-7 mb-4">
            <div class="card">
                <div class="card-header bg-primary text-white">Robot View</div>
                <div class="card-body bg-dark p-0">
                    <div class="image-container">
                        <img id="robot-image" src="" alt="Waiting for stream...">
                        <img id="overlay-image" src="" style="display:none;">
                    </div>
                </div>
            </div>
            
            <div class="card mt-4">
                <div class="card-header bg-warning text-dark">Depth View</div>
                <div class="card-body bg-dark p-0">
                    <div class="image-container">
                        <img id="depth-image" src="" alt="Waiting for depth stream..." style="width: 100%; display: block;">
                    </div>
                </div>
            </div>
        </div>
        <div class="col-lg-5">
            <div class="card mb-4">
                <div class="card-header bg-success text-white">Instruction</div>
                <div class="card-body">
                    <textarea class="form-control mb-2" id="instruction" rows="4"></textarea>
                    <button class="btn btn-success w-100" onclick="updateInstruction()">Update Instruction</button>
                </div>
            </div>
            <div class="card">
                <div class="card-header bg-secondary text-white">Logs</div>
                <div class="card-body p-0"><div id="logs" class="log-box"></div></div>
            </div>
        </div>
    </div>
</div>
<script>
    function fetchUpdates() {
        fetch('/ui_data').then(r => r.json()).then(data => {
            if(data.image) {
                document.getElementById('robot-image').src = 'data:image/jpeg;base64,' + data.image;
                const ov = document.getElementById('overlay-image');
                if(data.overlay_image) { ov.src = 'data:image/png;base64,' + data.overlay_image; ov.style.display = 'block'; }
                else { ov.style.display = 'none'; }
                
                if (data.depth_image) {
                    document.getElementById('depth-image').src = 'data:image/jpeg;base64,' + data.depth_image;
                }
            }
            document.getElementById('logs').innerHTML = data.logs.map(l => `<div>${l}</div>`).join('');
            if(document.activeElement.id !== 'instruction' && document.getElementById('instruction').value !== data.instruction) {
                document.getElementById('instruction').value = data.instruction;
            }
        }).finally(() => setTimeout(fetchUpdates, 500));
    }
    
    function updateInstruction() {
        const txt = document.getElementById('instruction').value;
        fetch('/set_instruction', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({instruction: txt})
        });
    }
    
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
    img = None
    depth = None
    overlay = None
    logs = []
    instr = ""
    with state.lock:
        img = state.latest_image_b64
        depth = state.latest_depth_b64
        overlay = state.overlay_image_b64
        logs = list(state.logs)
        instr = state.instruction
    return jsonify({
        "image": img,
        "depth_image": depth,
        "overlay_image": overlay,
        "logs": logs,
        "instruction": instr
    })

@app.route("/set_instruction", methods=['POST'])
def set_instruction():
    data = request.json
    if 'instruction' in data:
        state.set_instruction(data['instruction'])
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

from flask import request # Import moved here to avoid circularity if any, though likely fine at top

def serve_grpc(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    internvla_stream_pb2_grpc.add_InternVLAStreamServicer_to_server(
        InternVLAStreamServicer(agent), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"gRPC server started on port {port}")
    return server

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default=str(CHECKPOINT_DIR / "InternVLA-N1"))
    parser.add_argument("--port", type=int, default=5801) # gRPC port
    parser.add_argument("--http_port", type=int, default=5802) # Flask port
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
    
    # Start gRPC Server
    grpc_server = serve_grpc(args.port)
    
    # Start Flask App
    print(f"Starting UI on http://0.0.0.0:{args.http_port}")
    try:
        app.run(host='0.0.0.0', port=args.http_port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        grpc_server.stop(0)
