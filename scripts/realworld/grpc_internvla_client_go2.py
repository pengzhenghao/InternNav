"""
Unitree Go2 Robot Navigation Client with InternVLA Integration (gRPC Streaming Version)

This module provides a ROS2 node for controlling a Unitree Go2 robot using the InternVLA
navigation system via gRPC bidirectional streaming.

Key Features:
    - Real-time RGB and depth image processing
    - gRPC streaming for lower latency and better connection handling
    - MPC and PID control modes
    - Dry run mode
"""

import copy
import io
import json
import logging
import math
import os
import threading
import time
import sys
import pathlib
from collections import deque
from enum import Enum
import queue

import numpy as np
import rclpy
import grpc
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from sensor_msgs.msg import Image, CompressedImage

# user-specific
from controllers import Mpc_controller, PID_controller
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock

# Additional ROS2 message types
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

# unitree related
from unitree_go.msg import SportModeState
from unitree_api.msg import Request

# ROS2 service for toggling dry run mode
from example_interfaces.srv import SetBool

# Add project root to path for proto imports
# Determine path to protos based on current script location, valid for both host and container
current_dir = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir / 'protos'))

import internvla_stream_pb2
import internvla_stream_pb2_grpc

# ============================================================================
# Configuration Constants
# ============================================================================

TOPIC_RGB_IMAGE = "/camera/camera/color/image_raw"
TOPIC_DEPTH_IMAGE = "/camera/camera/depth/image_rect_raw"
TOPIC_ODOMETRY = "/sportmodestate"
TOPIC_CONTROL_COMMAND = "/api/sport/request"
SERVICE_DRY_RUN = "~/set_dry_run"

GRPC_SERVER_ADDRESS = "bolei-gpu05.cs.ucla.edu:5801"
# GRPC_SERVER_ADDRESS = "127.0.0.1:5801"

PID_KP_TRANS = 2.0
PID_KD_TRANS = 0.0
PID_KP_YAW = 1.5
PID_KD_YAW = 0.0
PID_MAX_V = 0.6
PID_MAX_W = 0.5

CONTROL_THREAD_SLEEP = 0.1
PLANNING_THREAD_DESIRED_TIME = 0.3
PLANNING_THREAD_INITIAL_SLEEP = 0.05
PLANNING_THREAD_IDLE_SLEEP = 0.01

CALLBACK_REPORT_INTERVAL = 5.0
FRAME_DATA_MAX_SIZE = 10
DRY_RUN_COMMAND_LOG_INTERVAL = 2.0
ODOM_DOWNSAMPLE_RATIO = 5
IMAGE_TIMEOUT_WARNING = 2.0
IMAGE_TIMEOUT_CHECK_INTERVAL = 1.0

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2

# Global Variables
policy_init = True
mpc = None
pid = PID_controller(
    Kp_trans=PID_KP_TRANS, Kd_trans=PID_KD_TRANS, 
    Kp_yaw=PID_KP_YAW, Kd_yaw=PID_KD_YAW, 
    max_v=PID_MAX_V, max_w=PID_MAX_W
)
grpc_seq = 0
trajs_in_world = None
manager = None
current_control_mode = ControlMode.MPC_Mode
desired_v, desired_w = 0.0, 0.0

rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()

callback_stats = {
    'rgb_depth': {'count': 0, 'last_report_time': time.time()},
    'odom': {'count': 0, 'last_report_time': time.time()},
    'rgb_forward': {'count': 0, 'last_report_time': time.time()},
}

def report_callback_rate(callback_name):
    stats = callback_stats[callback_name]
    stats['count'] += 1
    current_time = time.time()
    elapsed = current_time - stats['last_report_time']
    if elapsed >= CALLBACK_REPORT_INTERVAL:
        rate = stats['count'] / elapsed
        logger.info(f"Callback '{callback_name}' framerate: {rate:.2f} Hz")
        stats['count'] = 0
        stats['last_report_time'] = current_time

# --- gRPC Client Wrapper ---
class GrpcClient:
    def __init__(self, address):
        self.address = address
        self.request_queue = queue.Queue(maxsize=1) # Only keep latest frame
        self.channel = None
        self.stub = None
        self.connected = False
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.inflight = 0
        self.lock = threading.Lock()

    def start(self):
        self.thread.start()
        
    def send_frame(self, rgb_bytes, depth_bytes, reset=False, client_id="go2_client"):
        # Drop frame if we already have one inflight to avoid building up latency
        # We allow 1 inflight: client sends, server processes. 
        # If we send another before response, latency increases.
        # But 'inflight' tracking needs to be accurate.
        with self.lock:
            if self.inflight > 0:
                # logger.debug("Dropping frame, server busy.")
                return

        global grpc_seq
        grpc_seq += 1
        frame = internvla_stream_pb2.Frame(
            image_jpeg=rgb_bytes,
            depth_png=depth_bytes,
            reset=reset,
            seq=grpc_seq,
            client_id=client_id
        )
        msg = internvla_stream_pb2.ClientMessage(frame=frame)
        try:
            # Non-blocking put, remove old if full (though with inflight check, queue shouldn't be full often)
            self.request_queue.put_nowait(msg)
        except queue.Full:
            try:
                self.request_queue.get_nowait()
                self.request_queue.put_nowait(msg)
            except queue.Empty:
                pass

    def _request_generator(self):
        while not self._stop_event.is_set():
            try:
                # Wait for next frame
                msg = self.request_queue.get(timeout=1.0)
                if msg.HasField('frame'):
                    with self.lock:
                        self.inflight += 1
                yield msg
            except queue.Empty:
                # Send heartbeat if idle
                yield internvla_stream_pb2.ClientMessage(
                    heartbeat=internvla_stream_pb2.Heartbeat(seq=int(time.time()), client_id="go2_client")
                )

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                logger.info(f"Connecting to gRPC server at {self.address}...")
                self.channel = grpc.insecure_channel(self.address)
                self.stub = internvla_stream_pb2_grpc.InternVLAStreamStub(self.channel)
                
                responses = self.stub.Stream(self._request_generator())
                self.connected = True
                logger.info("gRPC Stream connected.")
                
                for response in responses:
                    if response.HasField('action'):
                        with self.lock:
                            self.inflight = max(0, self.inflight - 1)
                        process_server_action(response.action)
                    elif response.HasField('heartbeat'):
                        pass # logger.debug("Heartbeat ack")
                        
            except grpc.RpcError as e:
                self.connected = False
                logger.error(f"gRPC Stream error: {e}")
                with self.lock:
                    self.inflight = 0
                time.sleep(2.0) # Backoff
            except Exception as e:
                self.connected = False
                logger.error(f"Unexpected gRPC error: {e}")
                with self.lock:
                    self.inflight = 0
                time.sleep(2.0)

grpc_client = GrpcClient(GRPC_SERVER_ADDRESS)

def process_server_action(action):
    global current_control_mode, trajs_in_world, mpc, manager
    
    # Need access to latest odom for coordinate transform
    odom_rw_lock.acquire_read()
    odom = manager.odom.copy() if manager.odom else None
    odom_rw_lock.release_read()
    
    if odom is None:
        return

    logger.debug(f"Received action: {action.log}")
    
    if len(action.trajectory) > 0:
        # Trajectory Mode
        # Reshape flat list to Nx2 or Nx3?
        # Original code used output_trajectory from model which is (N, 2)
        # Proto sends flattened float
        traj_flat = np.array(action.trajectory)
        trajectory = traj_flat.reshape(-1, 2) 
        
        trajs_in_world = []
        x_, y_, yaw_ = odom[0], odom[1], odom[2]
        w_T_b = np.array([
            [np.cos(yaw_), -np.sin(yaw_), 0, x_],
            [np.sin(yaw_), np.cos(yaw_), 0, y_],
            [0.0, 0.0, 1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        # Original: skip first 3 points?
        for i, pt in enumerate(trajectory):
            if i < 3: continue
            w_P = (w_T_b @ (np.array([pt[0], pt[1], 0.0, 1.0])).T)[:2]
            trajs_in_world.append(w_P)
            
        trajs_in_world = np.array(trajs_in_world)
        manager.last_trajs_in_world = trajs_in_world
        
        mpc_rw_lock.acquire_write()
        if mpc is None:
            mpc = Mpc_controller(np.array(trajs_in_world))
        else:
            mpc.update_ref_traj(np.array(trajs_in_world))
        manager.request_cnt += 1
        mpc_rw_lock.release_write()
        current_control_mode = ControlMode.MPC_Mode
        
    elif len(action.discrete_action) > 0:
        # Discrete Action Mode
        actions = list(action.discrete_action)
        if actions != [5] and actions != [9]:
            manager.incremental_change_goal(actions)
            current_control_mode = ControlMode.PID_Mode

def control_thread():
    logger.info("Control thread started")
    global desired_v, desired_w, current_control_mode, mpc
    
    while True:
        if current_control_mode == ControlMode.MPC_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            
            if mpc is not None and manager is not None and odom is not None:
                # mpc object might be updated by other thread, use local ref?
                # Thread safety: mpc internal state? Mpc_controller.solve seems pure?
                # Better lock while solving or deepcopy?
                mpc_rw_lock.acquire_read()
                try:
                    opt_u_controls, opt_x_states = mpc.solve(np.array(odom))
                except Exception as e:
                    logger.error(f"MPC Solve failed: {e}")
                    opt_u_controls = [[0,0]]
                mpc_rw_lock.release_read()
                
                v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)
                
        elif current_control_mode == ControlMode.PID_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            if homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
                if v < 0.0: v = 0.0
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)

        time.sleep(CONTROL_THREAD_SLEEP)

def planning_thread():
    logger.info("Planning thread started")
    global policy_init
    
    # Start gRPC client
    grpc_client.start()
    
    while True:
        start_time = time.time()
        
        if manager is not None:
            manager.check_image_timeouts()
        
        time.sleep(PLANNING_THREAD_INITIAL_SLEEP)

        if not manager.new_image_arrived:
            time.sleep(PLANNING_THREAD_IDLE_SLEEP)
            continue
        manager.new_image_arrived = False

        if not manager.enable_server:
            time.sleep(PLANNING_THREAD_IDLE_SLEEP)
            continue

        # Prepare Data
        rgb_depth_rw_lock.acquire_read()
        try:
            rgb_bytes_io = manager.rgb_bytes
            depth_bytes_io = manager.depth_bytes
            
            if rgb_bytes_io and depth_bytes_io:
                rgb_data = rgb_bytes_io.getvalue()
                depth_data = depth_bytes_io.getvalue()
            else:
                rgb_data = None
                depth_data = None
        except Exception as e:
            logger.error(f"Failed to copy data: {e}")
            rgb_data = None
        rgb_depth_rw_lock.release_read()
        
        if rgb_data and depth_data:
            # Send to gRPC
            grpc_client.send_frame(rgb_data, depth_data, reset=policy_init)
            if policy_init: policy_init = False
        
        # Maintain pacing
        time.sleep(max(0, PLANNING_THREAD_DESIRED_TIME - (time.time() - start_time)))

class Go2Manager(Node):
    def __init__(self):
        super().__init__('go2_manager')
        
        default_dry_run = os.environ.get('DRY_RUN', '0') == '1'
        self.declare_parameter('dry_run', default_dry_run)
        self.dry_run = self.get_parameter('dry_run').get_parameter_value().bool_value
        
        self.declare_parameter('use_compressed', True)
        self.use_compressed = self.get_parameter('use_compressed').get_parameter_value().bool_value
        
        self.declare_parameter('enable_server', True)
        self.enable_server = self.get_parameter('enable_server').get_parameter_value().bool_value

        # Subscribers
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        if self.use_compressed:
            rgb_topic = TOPIC_RGB_IMAGE + "/compressed"
            depth_topic = TOPIC_DEPTH_IMAGE + "/compressed"
            msg_type = CompressedImage
        else:
            rgb_topic = TOPIC_RGB_IMAGE
            depth_topic = TOPIC_DEPTH_IMAGE
            msg_type = Image
            
        rgb_sub = Subscriber(self, msg_type, rgb_topic, qos_profile=sub_qos)
        depth_sub = Subscriber(self, msg_type, depth_topic, qos_profile=sub_qos)
        
        self.syncronizer = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 1, 0.1)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        
        self.odom_sub = self.create_subscription(SportModeState, TOPIC_ODOMETRY, self.odom_callback, 10)
        self.control_pub = self.create_publisher(Request, TOPIC_CONTROL_COMMAND, 5)
        
        self.create_service(SetBool, SERVICE_DRY_RUN, self.set_dry_run_cb)
        
        self.cv_bridge = CvBridge()
        self.rgb_bytes = None
        self.depth_bytes = None
        self.new_image_arrived = False
        
        self.odom = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None
        self.odom_cnt = 0
        
        self.last_trajs_in_world = None
        self.request_cnt = 0
        
        self.last_rgb_received_time = None
        self.last_depth_received_time = None
        self.last_rgb_warning_time = 0
        self.last_depth_warning_time = 0
        self.last_image_check_time = 0
        
        self.command_count = 0
        self.last_command_log_time = time.time()

    def check_image_timeouts(self):
        t = time.time()
        if t - self.last_image_check_time < IMAGE_TIMEOUT_CHECK_INTERVAL: return
        self.last_image_check_time = t
        
        if self.last_rgb_received_time is None:
            if t - self.last_rgb_warning_time > IMAGE_TIMEOUT_CHECK_INTERVAL:
                logger.warning("Waiting for RGB...")
                self.last_rgb_warning_time = t
        
    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        report_callback_rate('rgb_depth')
        try:
            if self.use_compressed:
                raw_image = self.cv_bridge.compressed_imgmsg_to_cv2(rgb_msg, 'rgb8')
                raw_depth = self.cv_bridge.compressed_imgmsg_to_cv2(depth_msg, 'passthrough')
            else:
                raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
                raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            # Compress RGB
            img = PIL_Image.fromarray(raw_image)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            buf.seek(0)
            
            # Process Depth
            raw_depth[np.isnan(raw_depth)] = 0
            raw_depth[np.isinf(raw_depth)] = 0
            depth_m = raw_depth / 1000.0
            depth_m[depth_m < 0] = 0
            
            depth_processed = (np.clip(depth_m * 10000.0, 0, 65535)).astype(np.uint16)
            depth_img = PIL_Image.fromarray(depth_processed)
            dbuf = io.BytesIO()
            depth_img.save(dbuf, format='PNG', compress_level=1)
            dbuf.seek(0)
            
            rgb_depth_rw_lock.acquire_write()
            self.rgb_bytes = buf
            self.depth_bytes = dbuf
            self.last_rgb_received_time = time.time()
            self.last_depth_received_time = time.time()
            rgb_depth_rw_lock.release_write()
            
            self.new_image_arrived = True
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")

    def odom_callback(self, msg):
        report_callback_rate('odom')
        self.odom_cnt += 1
        if self.odom_cnt % ODOM_DOWNSAMPLE_RATIO != 0: return
        
        odom_rw_lock.acquire_write()
        yaw = msg.imu_state.rpy[2]
        x, y = msg.position[0], msg.position[1]
        v_lin, v_ang = msg.velocity[0], msg.yaw_speed
        
        self.odom = [x, y, yaw]
        self.vel = [v_lin, v_ang]
        
        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [x, y]
        
        if self.odom_cnt == ODOM_DOWNSAMPLE_RATIO:
            self.homo_goal = self.homo_odom.copy()
            logger.info("Goal initialized.")
            
        odom_rw_lock.release_write()

    def incremental_change_goal(self, actions):
        if self.homo_goal is None: return
        # Logic same as original
        homo_goal = self.homo_odom.copy()
        for a in actions:
            if a == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif a == 2:
                # Turn left 15 deg
                ang = math.radians(15)
                R = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0,0,1]])
                homo_goal[:3, :3] = np.dot(R, homo_goal[:3, :3])
            elif a == 3:
                # Turn right 15 deg
                ang = -math.radians(15)
                R = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0,0,1]])
                homo_goal[:3, :3] = np.dot(R, homo_goal[:3, :3])
        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        if self.dry_run:
            self.command_count += 1
            if time.time() - self.last_command_log_time > DRY_RUN_COMMAND_LOG_INTERVAL:
                logger.info(f"[DRY RUN] {self.command_count} cmds sent. Latest: {vx:.2f}, {vyaw:.2f}")
                self.command_count = 0
                self.last_command_log_time = time.time()
        else:
            req = Request()
            req.header.identity.api_id = 1008
            req.parameter = json.dumps({"x": float(vx), "y": float(vy), "z": float(vyaw)})
            self.control_pub.publish(req)

    def set_dry_run_cb(self, request, response):
        self.dry_run = request.data
        response.success = True
        response.message = f"Dry run: {self.dry_run}"
        return response

if __name__ == '__main__':
    t_control = threading.Thread(target=control_thread, daemon=True)
    t_planning = threading.Thread(target=planning_thread, daemon=True)
    
    rclpy.init()
    try:
        manager = Go2Manager()
        t_control.start()
        t_planning.start()
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        if manager: manager.destroy_node()
        rclpy.shutdown()
