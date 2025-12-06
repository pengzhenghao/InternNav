import copy
import io
import json
import math
import threading
import time
from collections import deque
from enum import Enum

import numpy as np
import rclpy
import requests
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from sensor_msgs.msg import Image

frame_data = {}
frame_idx = 0
# user-specific
from controllers import Mpc_controller, PID_controller
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock


class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2


# global variable
policy_init = True
mpc = None
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
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


def dual_sys_eval(image_bytes, depth_bytes, front_image_bytes, url='http://127.0.0.1:5801/eval_dual'):
    print("DEBUG: dual_sys_eval start")
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
        print(f"DEBUG: sending post request to {url}")
        response = requests.post(url, files=files, data={'json': json_data}, timeout=100)
        print(f"response {response.text}")
    except Exception as e:
        print(f"DEBUG: request failed: {e}")
        return {}

    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    print(f"idx: {http_idx} after http {time.time() - start}")

    return json.loads(response.text)


def control_thread():
    print(f"DEBUG: control_thread started")
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

        time.sleep(0.1)


def planning_thread():
    print(f"DEBUG: planning_thread started")
    global trajs_in_world

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.3
        time.sleep(0.05)

        if not manager.new_image_arrived:
            time.sleep(0.01)
            continue
        print("DEBUG: planning_thread: new image arrived")
        manager.new_image_arrived = False
        rgb_depth_rw_lock.acquire_read()
        print("DEBUG: planning_thread: acquire read lock")
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
            # print("DEBUG: planning_thread: all copies done")
        except Exception as e:
            print(f"DEBUG: planning_thread copy failed: {e}")
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
                # print("DEBUG: planning_thread: frame_data added")
            except Exception as e:
                print(f"DEBUG: planning_thread frame_data add failed: {e}")
                
            if len(frame_data) > 10:  # Reduced from 20 to 10
                # print(f"DEBUG: Cleaning up frame_data (current size: {len(frame_data)})")
                # Remove old frames aggressively
                sorted_keys = sorted(frame_data.keys())
                keys_to_remove = sorted_keys[:-10]  # Keep only latest 10
                for k in keys_to_remove:
                    del frame_data[k]
            
            # print(f"DEBUG: frame_data size: {len(frame_data)}")
            
            response = dual_sys_eval(rgb_bytes, depth_bytes, None)

            global current_control_mode
            traj_len = 0.0
            if 'trajectory' in response:
                trajectory = response['trajectory']
                trajs_in_world = []
                odom = odom_infer
                traj_len = np.linalg.norm(trajectory[-1][:2])
                print(f"traj len {traj_len}")
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
                print(f"{time.time()} update traj")

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
            print(
                f"skip planning. odom_infer: {odom_infer is not None} rgb_bytes: {rgb_bytes is not None} depth_bytes: {depth_bytes is not None}"
            )
            time.sleep(0.1)

        time.sleep(max(0, DESIRED_TIME - (time.time() - start_time)))


class Go2Manager(Node):
    def __init__(self):
        super().__init__('go2_manager')

        # Use separate QoS profiles if needed. For now, matching publisher's RELIABLE policy.
        # Note: ApproximateTimeSynchronizer by default uses the subscriber's QoS, 
        # but the subscribers themselves need to be created with compatible QoS.
        # Since we pass the subscribers to the synchronizer, the synchronizer manages the callbacks.
        
        # Explicitly define QoS for subscribers
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        rgb_down_sub = Subscriber(self, Image, "/camera/camera/color/image_raw", qos_profile=sub_qos)
        
        #depth_down_sub = Subscriber(self, Image, "/camera/camera/aligned_depth_to_color/image_raw")
        depth_down_sub = Subscriber(self, Image, "/camera/camera/depth/image_rect_raw", qos_profile=sub_qos)

        

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.syncronizer = ApproximateTimeSynchronizer([rgb_down_sub, depth_down_sub], 1, 0.1)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        print(f"DEBUG: syncronizer registered")


        # self.odom_sub = self.create_subscription(Odometry, "/odometer_state", self.odom_callback, qos_profile)
        self.odom_sub = self.create_subscription(Odometry, "/odom_bridge", self.odom_callback, qos_profile)
        print(f"DEBUG: odom_sub registered")

        # publisher
        self.control_pub = self.create_publisher(Twist, '/cmd_vel_bridge', 5)
        print(f"DEBUG: control_pub registered")
        
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

    def rgb_forward_callback(self, rgb_msg):
        print(f"DEBUG: rgb_forward_callback received data")
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        print(f"DEBUG: rgb_forward_callback image shape: {raw_image.shape}")
        self.rgb_forward_image = raw_image
        image = PIL_Image.fromarray(self.rgb_forward_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        self.rgb_forward_bytes = image_bytes
        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        print(f"DEBUG: rgb_depth_down_callback received data at {time.time()}")
        try:
            # print("DEBUG: starting rgb conversion")
            raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
            print(f"DEBUG: rgb_depth_down_callback RGB image shape: {raw_image.shape}")
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
            # print("DEBUG: rgb processed")

            # print("DEBUG: starting depth conversion")
            raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            print(f"DEBUG: rgb_depth_down_callback Depth image shape: {raw_depth.shape}")
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
            # print("DEBUG: depth processed")

            rgb_depth_rw_lock.acquire_write()
            self.rgb_bytes = image_bytes

            self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
            self.last_rgb_time = self.rgb_time

            self.depth_bytes = depth_bytes
            self.depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec / 1.0e9
            self.last_depth_time = self.depth_time

            rgb_depth_rw_lock.release_write()

            self.new_vis_image_arrived = True
            self.new_image_arrived = True
            # print("DEBUG: rgb_depth_down_callback exit")
        except Exception as e:
            print(f"DEBUG: Error in rgb_depth_down_callback: {e}")
            raise

    def odom_callback(self, msg):
        self.odom_cnt += 1
        odom_rw_lock.acquire_write()
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        print(f"DEBUG: odom_callback received. Pos: ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f}), Yaw: {yaw:.2f}")
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        self.odom_timestamp = time.time()
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        odom_rw_lock.release_write()

        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()

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
        request = Twist()
        request.linear.x = vx
        request.linear.y = 0.0
        request.angular.z = vyaw

        self.control_pub.publish(request)


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
