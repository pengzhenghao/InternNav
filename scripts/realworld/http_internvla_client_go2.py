"""
Unitree Go2 Robot Navigation Client with InternVLA Integration

This module provides a ROS2 node for controlling a Unitree Go2 robot using the InternVLA
navigation system. It handles sensor data processing, trajectory planning, and robot control
through HTTP-based communication with an inference server.

Key Features:
    - Real-time RGB and depth image processing
    - Odometry tracking and trajectory planning
    - MPC (Model Predictive Control) and PID control modes
    - Dry run mode for safe testing without robot movement
    - Callback rate monitoring and logging

Dry Run Mode:
    Dry run mode allows all data processing and planning to continue normally, but prevents
    actual control commands from being sent to the robot. This is useful for:
    - Testing the navigation pipeline without physical robot movement
    - Debugging planning and control algorithms
    - Validating sensor data processing
    
    Enable dry run mode via:
        1. Environment variable: DRY_RUN=true
        2. ROS parameter: dry_run:=true
        3. ROS2 service: ros2 service call /go2_manager/set_dry_run example_interfaces/srv/SetBool "{data: true}"

ROS Topics:
    Subscribed:
        - /camera/camera/color/image_raw (sensor_msgs/Image): RGB camera feed
        - /camera/camera/aligned_depth_to_color/image_raw (sensor_msgs/Image): Depth camera feed
        - /sportmodestate (unitree_go/SportModeState): Robot odometry and state
    
    Published:
        - /api/sport/request (unitree_api/Request): Robot control commands
    
    Services:
        - ~/set_dry_run (example_interfaces/SetBool): Toggle dry run mode at runtime

Configuration:
    - HTTP inference server URL (default: http://127.0.0.1:5801/eval_dual)
    - Control mode: MPC_Mode (default) or PID_Mode
    - PID controller gains: Kp_trans=2.0, Kp_yaw=1.5, max_v=0.6, max_w=0.5
    - Planning thread desired time: 0.3s
    - Callback rate reporting interval: 5.0s

Threads:
    - control_thread: Executes control commands at 10Hz
    - planning_thread: Processes images and generates trajectories at ~3.3Hz

Author: InternNav Team
"""

import copy
import io
import json
import logging
import math
import os
import threading
import time
from collections import deque
from enum import Enum
from urllib.parse import urlparse

import numpy as np
import rclpy
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from sensor_msgs.msg import Image

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
from unitree_api.msg import RequestHeader

# ROS2 service for toggling dry run mode
from example_interfaces.srv import SetBool


# ============================================================================
# Configuration Constants
# ============================================================================

# ROS Topic Names
TOPIC_RGB_IMAGE = "/camera/camera/color/image_raw"

# TOPIC_DEPTH_IMAGE = "/camera/camera/aligned_depth_to_color/image_raw"
TOPIC_DEPTH_IMAGE = "/camera/camera/depth/image_rect_raw"

TOPIC_ODOMETRY = "/sportmodestate"
TOPIC_CONTROL_COMMAND = "/api/sport/request"

# ROS Service Names
SERVICE_DRY_RUN = "~/set_dry_run"

# HTTP Inference Server
# HTTP_INFERENCE_URL = "http://127.0.0.1:5801/eval_dual"

HTTP_INFERENCE_URL = "http://bolei-gpu05.cs.ucla.edu:5801/eval_dual"

HTTP_TIMEOUT = 100  # seconds

# Control Parameters
PID_KP_TRANS = 2.0
PID_KD_TRANS = 0.0
PID_KP_YAW = 1.5
PID_KD_YAW = 0.0
PID_MAX_V = 0.6  # m/s
PID_MAX_W = 0.5  # rad/s

# Thread Timing Parameters
CONTROL_THREAD_SLEEP = 0.1  # seconds (10 Hz)
PLANNING_THREAD_DESIRED_TIME = 0.3  # seconds (~3.3 Hz)
PLANNING_THREAD_INITIAL_SLEEP = 0.05  # seconds
PLANNING_THREAD_IDLE_SLEEP = 0.01  # seconds

# Callback Rate Reporting
CALLBACK_REPORT_INTERVAL = 5.0  # seconds

# Frame Data Management
FRAME_DATA_MAX_SIZE = 10  # Maximum number of frames to keep in memory

# Dry Run Command Logging
DRY_RUN_COMMAND_LOG_INTERVAL = 2.0  # seconds

# Odometry Processing
ODOM_DOWNSAMPLE_RATIO = 5  # Process every Nth odometry message (reduces processing load)

# Image Timeout Detection
IMAGE_TIMEOUT_WARNING = 2.0  # seconds - warn if no images received for this duration
IMAGE_TIMEOUT_CHECK_INTERVAL = 1.0  # seconds - how often to check for timeouts

# ============================================================================
# Logging Configuration
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

frame_data = {}
frame_idx = 0


class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2


# global variable
policy_init = True
mpc = None
pid = PID_controller(
    Kp_trans=PID_KP_TRANS, 
    Kd_trans=PID_KD_TRANS, 
    Kp_yaw=PID_KP_YAW, 
    Kd_yaw=PID_KD_YAW, 
    max_v=PID_MAX_V, 
    max_w=PID_MAX_W
)
http_idx = -1
first_running_time = 0.0
last_pixel_goal = None
last_s2_step = -1
manager = None
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None

desired_v, desired_w = 0.0, 0.0
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()

# Callback rate tracking
callback_stats = {
    'rgb_depth': {'count': 0, 'last_report_time': time.time()},
    'odom': {'count': 0, 'last_report_time': time.time()},
    'rgb_forward': {'count': 0, 'last_report_time': time.time()},
}


def report_callback_rate(callback_name):
    """Report callback framerate every CALLBACK_REPORT_INTERVAL seconds."""
    stats = callback_stats[callback_name]
    stats['count'] += 1
    current_time = time.time()
    elapsed = current_time - stats['last_report_time']
    
    if elapsed >= CALLBACK_REPORT_INTERVAL:
        rate = stats['count'] / elapsed
        logger.info(f"Callback '{callback_name}' framerate: {rate:.2f} Hz ({stats['count']} calls in {elapsed:.2f}s)")
        stats['count'] = 0
        stats['last_report_time'] = current_time


def dual_sys_eval(image_bytes, depth_bytes, front_image_bytes, url=HTTP_INFERENCE_URL):
    """Send RGB and depth images to inference server and get navigation response.
    
    Returns:
        dict: Response from server containing 'trajectory' or 'discrete_action', or empty dict on error
    """
    logger.debug("Starting dual_sys_eval")
    global policy_init, http_idx, first_running_time
    data = {"reset": policy_init, "idx": http_idx}
    json_data = json.dumps(data)

    policy_init = False
    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
        'depth': ('depth_image', depth_bytes, 'image/png'),
    }
    start = time.time()
    
    try:
        logger.debug(f"Sending POST request to {url}")
        response = requests.post(url, files=files, data={'json': json_data}, timeout=HTTP_TIMEOUT)
        
        # Check if response is valid
        if response.status_code != 200:
            logger.error(
                f"❌ Inference server returned error status {response.status_code}\n"
                f"   URL: {url}\n"
                f"   Response: {response.text[:200]}"
            )
            return {}
        
        # Check if response has content
        if not response.text or len(response.text.strip()) == 0:
            logger.error(
                f"❌ Inference server returned empty response\n"
                f"   URL: {url}\n"
                f"   Status code: {response.status_code}"
            )
            return {}
        
        logger.debug(f"Response received: {response.text[:100]}...")  # Truncate long responses
        
    except ConnectionError as e:
        # Extract port from URL for better error message
        try:
            parsed = urlparse(url)
            port = parsed.port if parsed.port else (443 if parsed.scheme == 'https' else 80)
        except Exception:
            port = "unknown"
        
        logger.error(
            f"❌ Cannot connect to inference server!\n"
            f"   URL: {url}\n"
            f"   Error: Connection refused\n"
            f"   \n"
            f"   💡 Solution: Make sure the inference server is running.\n"
            f"      Start the server with: python scripts/realworld/http_internvla_server.py\n"
            f"      Or check if the server is listening on port {port}"
        )
        return {}
        
    except Timeout as e:
        logger.error(
            f"❌ Request to inference server timed out!\n"
            f"   URL: {url}\n"
            f"   Timeout: {HTTP_TIMEOUT}s\n"
            f"   \n"
            f"   💡 Solution: The server may be overloaded or unresponsive.\n"
            f"      Check server logs and consider increasing HTTP_TIMEOUT if needed."
        )
        return {}
        
    except RequestException as e:
        logger.error(
            f"❌ HTTP request failed!\n"
            f"   URL: {url}\n"
            f"   Error: {str(e)}\n"
            f"   \n"
            f"   💡 Solution: Check network connectivity and server status."
        )
        return {}
        
    except Exception as e:
        logger.error(
            f"❌ Unexpected error during inference request!\n"
            f"   URL: {url}\n"
            f"   Error: {type(e).__name__}: {str(e)}\n"
            f"   \n"
            f"   Please check the error details above."
        )
        logger.debug("Full exception details:", exc_info=True)
        return {}

    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    elapsed = time.time() - start
    logger.debug(f"Request {http_idx} completed in {elapsed:.3f}s")

    # Parse JSON response
    try:
        response_data = json.loads(response.text)
        return response_data
    except json.JSONDecodeError as e:
        logger.error(
            f"❌ Failed to parse server response as JSON!\n"
            f"   URL: {url}\n"
            f"   Response length: {len(response.text)} characters\n"
            f"   Response preview: {response.text[:200]}\n"
            f"   JSON Error: {str(e)}\n"
            f"   \n"
            f"   💡 Solution: Check server logs - the server may have returned an error message."
        )
        return {}


def control_thread():
    logger.info("Control thread started")
    global desired_v, desired_w
    while True:
        global current_control_mode
        if current_control_mode == ControlMode.MPC_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            if mpc is not None and manager is not None and odom is not None:
                local_mpc = mpc
                opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
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
                if v < 0.0:
                    v = 0.0
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)

        time.sleep(CONTROL_THREAD_SLEEP)


def planning_thread():
    logger.info("Planning thread started")
    global trajs_in_world

    while True:
        start_time = time.time()
        
        # Check for image timeouts
        if manager is not None:
            manager.check_image_timeouts()
        
        time.sleep(PLANNING_THREAD_INITIAL_SLEEP)

        if not manager.new_image_arrived:
            time.sleep(PLANNING_THREAD_IDLE_SLEEP)
            continue
        logger.debug("New image arrived, processing")
        manager.new_image_arrived = False
        rgb_depth_rw_lock.acquire_read()
        logger.debug("Acquired read lock for RGB/depth data")
        try:
            # Instead of deepcopying bytes which creates new IO objects, use getvalue() if possible or keep as bytes
            # Here we are copying the IO object which might be inefficient if not needed
            # Let's optimize: only store raw bytes if that's what is needed later, or copy the content
            rgb_bytes_io = manager.rgb_bytes
            depth_bytes_io = manager.depth_bytes
            
            # Create new BytesIO objects with the same content instead of deepcopying the wrapper
            rgb_bytes = io.BytesIO(rgb_bytes_io.getvalue()) if rgb_bytes_io else None
            # print("DEBUG: planning_thread: rgb_bytes copied")
            
            depth_bytes = io.BytesIO(depth_bytes_io.getvalue()) if depth_bytes_io else None
            # print("DEBUG: planning_thread: depth_bytes copied")
            
            # For numpy arrays, copy is fine but ensure we are not holding onto them too long
            # infer_rgb = manager.rgb_image.copy() if manager.rgb_image is not None else None
            # Optimization: Downscale image for inference storage if full res not needed
            if manager.rgb_image is not None:
                infer_rgb = manager.rgb_image.copy()
                # infer_rgb = infer_rgb[::2, ::2, :] # Downscale
            else:
                infer_rgb = None
                
            # print("DEBUG: planning_thread: infer_rgb copied")
            
            # infer_depth = manager.depth_image.copy() if manager.depth_image is not None else None
             # Optimization: Downscale depth
            if manager.depth_image is not None:
                infer_depth = manager.depth_image.copy()
                # infer_depth = infer_depth[::2, ::2] # Downscale
            else:
                infer_depth = None

            # print("DEBUG: planning_thread: infer_depth copied")
            
            rgb_time = manager.rgb_time
            # logger.debug("All copies completed")
        except Exception as e:
            logger.error(f"Failed to copy RGB/depth data: {e}", exc_info=True)
            rgb_depth_rw_lock.release_read()
            continue
            
        rgb_depth_rw_lock.release_read()
        
        odom_rw_lock.acquire_read()
        min_diff = 1e10
        # time_diff = 1e10
        odom_infer = None
        for odom in manager.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
                # time_diff = odom[0] - rgb_time
        # odom_time = manager.odom_timestamp
        odom_rw_lock.release_read()

        if odom_infer is not None and rgb_bytes is not None and depth_bytes is not None:
            global frame_data
            try:
                # Store lightweight data or ensure we clean up
                frame_data[http_idx] = {
                    'infer_rgb': infer_rgb, # Already a copy
                    'infer_depth': infer_depth, # Already a copy
                    'infer_odom': copy.deepcopy(odom_infer),
                }
                logger.debug("Frame data added successfully")
            except Exception as e:
                logger.error(f"Failed to add frame data: {e}", exc_info=True)
                
            if len(frame_data) > FRAME_DATA_MAX_SIZE:
                logger.debug(f"Cleaning up frame_data (current size: {len(frame_data)})")
                # Remove old frames aggressively
                sorted_keys = sorted(frame_data.keys())
                keys_to_remove = sorted_keys[:-FRAME_DATA_MAX_SIZE]  # Keep only latest frames
                for k in keys_to_remove:
                    del frame_data[k]
            
            logger.debug(f"Frame data size: {len(frame_data)}")
            
            response = dual_sys_eval(rgb_bytes, depth_bytes, None)

            # Check if we got a valid response
            if not response or len(response) == 0:
                logger.warning(
                    "⚠️  Received empty response from inference server. "
                    "Skipping this planning cycle. Check server logs for details."
                )
                time.sleep(PLANNING_THREAD_IDLE_SLEEP)
                continue

            global current_control_mode
            traj_len = 0.0
            if 'trajectory' in response:
                trajectory = response['trajectory']
                trajs_in_world = []
                odom = odom_infer
                traj_len = np.linalg.norm(trajectory[-1][:2])
                logger.info(f"Trajectory length: {traj_len:.3f}m")
                for i, traj in enumerate(trajectory):
                    if i < 3:
                        continue
                    x_, y_, yaw_ = odom[0], odom[1], odom[2]

                    w_T_b = np.array(
                        [
                            [np.cos(yaw_), -np.sin(yaw_), 0, x_],
                            [np.sin(yaw_), np.cos(yaw_), 0, y_],
                            [0.0, 0.0, 1.0, 0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    )
                    w_P = (w_T_b @ (np.array([traj[0], traj[1], 0.0, 1.0])).T)[:2]
                    trajs_in_world.append(w_P)
                trajs_in_world = np.array(trajs_in_world)
                logger.debug(f"Updated trajectory at {time.time():.3f}")

                manager.last_trajs_in_world = trajs_in_world
                mpc_rw_lock.acquire_write()
                global mpc
                if mpc is None:
                    mpc = Mpc_controller(np.array(trajs_in_world))
                else:
                    mpc.update_ref_traj(np.array(trajs_in_world))
                manager.request_cnt += 1
                mpc_rw_lock.release_write()
                current_control_mode = ControlMode.MPC_Mode
            elif 'discrete_action' in response:
                actions = response['discrete_action']
                if actions != [5] and actions != [9]:
                    manager.incremental_change_goal(actions)
                    current_control_mode = ControlMode.PID_Mode
            else:
                logger.warning(
                    f"⚠️  Unexpected response format from inference server!\n"
                    f"   Expected 'trajectory' or 'discrete_action' keys, but got: {list(response.keys())}\n"
                    f"   Response: {str(response)[:200]}\n"
                    f"   \n"
                    f"   💡 This may indicate a server-side error or version mismatch."
                )
        else:
            logger.debug(
                f"Skipping planning. odom_infer: {odom_infer is not None}, "
                f"rgb_bytes: {rgb_bytes is not None}, depth_bytes: {depth_bytes is not None}"
            )
            time.sleep(PLANNING_THREAD_IDLE_SLEEP)

        time.sleep(max(0, PLANNING_THREAD_DESIRED_TIME - (time.time() - start_time)))


class Go2Manager(Node):
    def __init__(self):
        super().__init__('go2_manager')

        # Dry run mode: if True, all data flow continues but no control signals are sent to robot
        # Can be set via ROS parameter 'dry_run' or environment variable 'DRY_RUN'
        dry_run_env = os.getenv('DRY_RUN', 'false').lower() in ('true', '1', 'yes')
        self.declare_parameter('dry_run', dry_run_env)
        self.dry_run = self.get_parameter('dry_run').get_parameter_value().bool_value
        
        if self.dry_run:
            logger.warning("=" * 60)
            logger.warning("DRY RUN MODE ENABLED - No control signals will be sent to robot")
            logger.warning("=" * 60)
        else:
            logger.warning("Normal operation mode - Control signals will be sent to robot")

        # Use separate QoS profiles if needed. For now, matching publisher's RELIABLE policy.
        # Note: ApproximateTimeSynchronizer by default uses the subscriber's QoS, 
        # but the subscribers themselves need to be created with compatible QoS.
        # Since we pass the subscribers to the synchronizer, the synchronizer manages the callbacks.
        
        # Explicitly define QoS for subscribers
        # Note: For ApproximateTimeSynchronizer, we MUST use Subscriber from message_filters
        # 
        # About QoS depth vs create_subscription queue size:
        # - depth=10 in QoSProfile: Keeps the last 10 messages in the buffer (with KEEP_LAST policy)
        # - The "1" in create_subscription(..., 1): Convenience parameter that sets default QoS depth to 1
        # - They're related but not the same: depth=10 means buffer 10 messages, queue_size=1 means buffer 1 message
        # - For ApproximateTimeSynchronizer, we need message_filters.Subscriber, not regular subscriptions
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        rgb_down_sub = Subscriber(self, Image, TOPIC_RGB_IMAGE, qos_profile=sub_qos)

        depth_down_sub = Subscriber(self, Image, TOPIC_DEPTH_IMAGE, qos_profile=sub_qos)
        
        self.syncronizer = ApproximateTimeSynchronizer([rgb_down_sub, depth_down_sub], 1, 0.1)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        logger.info(f"RGB/Depth synchronizer registered (RGB: {TOPIC_RGB_IMAGE}, Depth: {TOPIC_DEPTH_IMAGE})")

        # Odometry subscriber (updated to SportModeState)
        # self.odom_sub = self.create_subscription(Odometry, "/odom_bridge", self.odom_callback, qos_profile)
        self.odom_sub = self.create_subscription(SportModeState, TOPIC_ODOMETRY, self.odom_callback, 10)
        logger.info(f"Odometry subscriber registered ({TOPIC_ODOMETRY})")

        # Control publisher (updated to Request)
        # self.control_pub = self.create_publisher(Twist, '/cmd_vel_bridge', 5)
        # publisher
        self.control_pub = self.create_publisher(Request, TOPIC_CONTROL_COMMAND, 5)
        logger.info(f"Control publisher registered ({TOPIC_CONTROL_COMMAND})")
        
        # Service to toggle dry run mode at runtime
        self.dry_run_service = self.create_service(
            SetBool, 
            SERVICE_DRY_RUN, 
            self.set_dry_run_service_callback
        )
        logger.info(f"Dry run toggle service available at '{SERVICE_DRY_RUN}' (use: ros2 service call /go2_manager/set_dry_run example_interfaces/srv/SetBool \"{{data: true}}\")")
        
        # class member variable
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.rgb_forward_image = None
        self.rgb_forward_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0
        
        # Image timeout tracking
        self.last_rgb_received_time = None
        self.last_depth_received_time = None
        self.last_rgb_warning_time = 0.0
        self.last_depth_warning_time = 0.0

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.last_s2_step = -1
        self.last_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None
        
        # Track command statistics for dry run logging
        self.command_count = 0
        self.last_command_log_time = time.time()
        
        # Image monitoring
        self.last_image_check_time = time.time()

    def check_image_timeouts(self):
        """Check if RGB or depth images haven't arrived and log warnings."""
        current_time = time.time()
        
        # Only check periodically to avoid spam
        if current_time - self.last_image_check_time < IMAGE_TIMEOUT_CHECK_INTERVAL:
            return
        
        self.last_image_check_time = current_time
        
        rgb_depth_rw_lock.acquire_read()
        last_rgb = self.last_rgb_received_time
        last_depth = self.last_depth_received_time
        rgb_depth_rw_lock.release_read()
        
        # Check RGB timeout
        if last_rgb is None:
            if current_time - self.last_rgb_warning_time > IMAGE_TIMEOUT_CHECK_INTERVAL:
                logger.warning("⚠️  No RGB images received yet. Waiting for images from camera...")
                self.last_rgb_warning_time = current_time
        else:
            time_since_rgb = current_time - last_rgb
            if time_since_rgb > IMAGE_TIMEOUT_WARNING:
                if current_time - self.last_rgb_warning_time > IMAGE_TIMEOUT_CHECK_INTERVAL:
                    logger.warning(
                        f"⚠️  No RGB images received for {time_since_rgb:.1f} seconds! "
                        f"Last RGB image was {time_since_rgb:.1f}s ago. "
                        f"Check camera connection and topic '{TOPIC_RGB_IMAGE}'"
                    )
                    self.last_rgb_warning_time = current_time
        
        # Check depth timeout
        if last_depth is None:
            if current_time - self.last_depth_warning_time > IMAGE_TIMEOUT_CHECK_INTERVAL:
                logger.warning("⚠️  No depth images received yet. Waiting for images from camera...")
                self.last_depth_warning_time = current_time
        else:
            time_since_depth = current_time - last_depth
            if time_since_depth > IMAGE_TIMEOUT_WARNING:
                if current_time - self.last_depth_warning_time > IMAGE_TIMEOUT_CHECK_INTERVAL:
                    logger.warning(
                        f"⚠️  No depth images received for {time_since_depth:.1f} seconds! "
                        f"Last depth image was {time_since_depth:.1f}s ago. "
                        f"Check camera connection and topic '{TOPIC_DEPTH_IMAGE}'"
                    )
                    self.last_depth_warning_time = current_time

    def rgb_forward_callback(self, rgb_msg):
        report_callback_rate('rgb_forward')
        logger.debug("RGB forward callback received data")
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        logger.debug(f"RGB forward image shape: {raw_image.shape}")
        self.rgb_forward_image = raw_image
        image = PIL_Image.fromarray(self.rgb_forward_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        self.rgb_forward_bytes = image_bytes
        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        report_callback_rate('rgb_depth')
        logger.debug(f"RGB/Depth callback received data at {time.time():.3f}")
        try:
            raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
            logger.debug(f"RGB image shape: {raw_image.shape}")
            # Resize image to lower resolution to save memory
            # Assuming original is high res, let's downscale if needed (e.g. to 640x480 or 320x240)
            # This is a common place for memory explosion if input is 1080p or 4k
            # Uncomment next line if image is large
            # raw_image = raw_image[::2, ::2, :] 
            
            self.rgb_image = raw_image
            
            # Optimization: Don't create PIL Image just to save to bytes if not needed or reuse buffer
            # Creating new BytesIO and saving JPEG every frame is CPU/Memory intensive
            # but required for HTTP transfer. Ensure we close buffers if possible (though BytesIO is memory only)
            
            image = PIL_Image.fromarray(self.rgb_image)
            image_bytes = io.BytesIO()
            # Optimize JPEG quality to reduce size
            image.save(image_bytes, format='JPEG', quality=85)
            image_bytes.seek(0)
            logger.debug("RGB image processed")

            raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            logger.debug(f"Depth image shape: {raw_depth.shape}")
            raw_depth[np.isnan(raw_depth)] = 0
            raw_depth[np.isinf(raw_depth)] = 0
            
            # Downscale depth if rgb is downscaled
            # raw_depth = raw_depth[::2, ::2]
            
            self.depth_image = raw_depth / 1000.0
            self.depth_image -= 0.0
            self.depth_image[np.where(self.depth_image < 0)] = 0
            
            # Optimization: Reuse buffer or reduce precision if possible
            depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
            depth = PIL_Image.fromarray(depth)
            depth_bytes = io.BytesIO()
            # Optimize PNG compression level (default is 6, max 9. 1 is fastest/largest)
            # Or use TIFF/Raw bytes if server supports it for speed
            depth.save(depth_bytes, format='PNG', compress_level=1)
            depth_bytes.seek(0)
            logger.debug("Depth image processed")

            rgb_depth_rw_lock.acquire_write()
            self.rgb_bytes = image_bytes

            self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
            self.last_rgb_time = self.rgb_time
            self.last_rgb_received_time = time.time()  # Track when we actually received it

            self.depth_bytes = depth_bytes
            self.depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec / 1.0e9
            self.last_depth_time = self.depth_time
            self.last_depth_received_time = time.time()  # Track when we actually received it

            rgb_depth_rw_lock.release_write()

            self.new_vis_image_arrived = True
            self.new_image_arrived = True
            logger.debug("RGB/Depth callback completed successfully")
        except Exception as e:
            logger.error(f"Error in rgb_depth_down_callback: {e}", exc_info=True)
            raise

    def odom_callback(self, msg):
        """Process odometry from Go2 SportModeState message.
        
        Args:
            msg: SportModeState message containing:
                - imu_state.rpy[2]: Yaw angle in radians
                - position[0], position[1]: X, Y position
                - velocity[0]: Linear velocity
                - yaw_speed: Angular velocity
        """
        report_callback_rate('odom')
        self.odom_cnt += 1
        
        # Downsample: only process every Nth message to reduce processing load
        if self.odom_cnt % ODOM_DOWNSAMPLE_RATIO != 0:
            return
        
        odom_rw_lock.acquire_write()
        
        # Extract yaw from IMU roll-pitch-yaw (rpy[2] is yaw in radians)
        yaw = msg.imu_state.rpy[2]
        
        # Extract position (x, y)
        x = msg.position[0]
        y = msg.position[1]
        
        # Extract velocities
        linear_vel = msg.velocity[0]
        angular_vel = msg.yaw_speed
        
        logger.debug(
            f"Odometry callback received. Position: ({x:.2f}, {y:.2f}), "
            f"Yaw: {yaw:.2f} rad ({math.degrees(yaw):.1f}°), "
            f"Vel: linear={linear_vel:.2f} m/s, angular={angular_vel:.2f} rad/s"
        )
        
        # Store odom as [x, y, yaw] for compatibility with other parts of the code
        self.odom = [x, y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        self.odom_timestamp = time.time()
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
        
        # Build homogeneous transformation matrix
        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [x, y]
        self.vel = [linear_vel, angular_vel]
        
        # Initialize goal on first processed odometry message
        if self.odom_cnt == ODOM_DOWNSAMPLE_RATIO:
            self.homo_goal = self.homo_odom.copy()
            logger.info("Initialized goal position from first odometry reading")
        
        odom_rw_lock.release_write()

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_odom.copy()
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        """Send control command to robot. In dry run mode, only logs the command without publishing."""
        if self.dry_run:
            self.command_count += 1
            current_time = time.time()
            
            # Log command summary periodically to avoid spam
            if current_time - self.last_command_log_time >= DRY_RUN_COMMAND_LOG_INTERVAL:
                logger.info(
                    f"[DRY RUN] Would have sent {self.command_count} commands. "
                    f"Latest: vx={vx:.3f} m/s, vy={vy:.3f} m/s, vyaw={vyaw:.3f} rad/s"
                )
                self.command_count = 0
                self.last_command_log_time = current_time
            else:
                # Log at debug level for individual commands
                logger.debug(f"[DRY RUN] Command suppressed: vx={vx:.3f}, vy={vy:.3f}, vyaw={vyaw:.3f}")
        else:
            # Normal operation: send command to robot
            request = Twist()
            request.linear.x = vx
            request.linear.y = 0.0
            request.angular.z = vyaw
            
            self.control_pub.publish(request)
            logger.debug(f"Control command sent: vx={vx:.3f} m/s, vy={vy:.3f} m/s, vyaw={vyaw:.3f} rad/s")
    
    def set_dry_run(self, enabled: bool):
        """Toggle dry run mode at runtime."""
        old_mode = self.dry_run
        self.dry_run = enabled
        if enabled and not old_mode:
            logger.warning("=" * 60)
            logger.warning("DRY RUN MODE ENABLED - No control signals will be sent to robot")
            logger.warning("=" * 60)
        elif not enabled and old_mode:
            logger.warning("=" * 60)
            logger.warning("DRY RUN MODE DISABLED - Control signals will now be sent to robot")
            logger.warning("=" * 60)
    
    def set_dry_run_service_callback(self, request, response):
        """ROS2 service callback to toggle dry run mode."""
        self.set_dry_run(request.data)
        response.success = True
        response.message = f"Dry run mode {'enabled' if request.data else 'disabled'}"
        return response


if __name__ == '__main__':
    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread)
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True
    rclpy.init()

    try:
        manager = Go2Manager()

        control_thread_instance.start()
        planning_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        rclpy.shutdown()
