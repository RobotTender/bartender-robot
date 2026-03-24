import cv2
import time
import asyncio
import json
import numpy as np
from pathlib import Path
import threading

import rclpy
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Bool, Float64MultiArray
from std_srvs.srv import Trigger
from robotender_msgs.srv import GripperControl
from robotender_msgs.action import PickBottle

import DR_init

from .defines import (
    POSJ_HOME, POSJ_PICK_PLACE_READY,
    GRIPPER_POSITION_OPEN,
    GRIPPER_FORCE_DEFAULT,
    BOTTLE_CONFIG, BOTTLE_ID_MAP,
    PICK_PLACE_X_OFFSET, PICK_PLACE_Y_OFFSET
)

ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 30.0, 30.0

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robotender_pick", namespace="/dsr01")
        # ARCHITECTURAL FIX: Use Reentrant group for all node-level callbacks
        self._default_callback_group = ReentrantCallbackGroup()
        
        self.action_cb_group = ReentrantCallbackGroup()
        self.client_cb_group = ReentrantCallbackGroup()
        
        self.bridge = CvBridge()

        # Mock Mode Parameter
        self.declare_parameter('use_mock_vision', False)
        self.use_mock_vision = self.get_parameter('use_mock_vision').value

        self.get_logger().info(f"Pick Node Starting (Mock: {self.use_mock_vision})")

        self.intrinsics = None
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None
        self.model = None
        self.state = 'IDLE'

        # Heartbeat timer to verify executor health
        # self.heartbeat_timer = self.create_timer(5.0, self._heartbeat_callback)

        self.R = np.array([
            [-0.788489317968,  -0.614148198918, -0.0332653756482],
            [-0.0868309265704,  0.0576098432706,  0.994555929121],
            [-0.608888319515,   0.787085189623,  -0.0987518031922],
        ], dtype=np.float64)
        self.t = np.array([521.115058698, 170.946228749, 834.749571453], dtype=np.float64)

        # Clients for non-motion services
        self.gripper_cb_group = MutuallyExclusiveCallbackGroup()
        self.gripper_move_cli = self.create_client(
            GripperControl, 
            'robotender_gripper/move', 
            callback_group=self.gripper_cb_group
        )
        
        # Vision Subscriptions - using default Reentrant group
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera_1/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera_1/aligned_depth_to_color/image_raw')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera_1/aligned_depth_to_color/camera_info')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.info_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synced_callback)
        
        self.last_pose_pub = self.create_publisher(Float64MultiArray, 'robotender_pick/last_pose', 10)

        # Action Server
        self._action_server = ActionServer(
            self, PickBottle, 'robotender_pick/execute',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_cb_group
        )

        if self.use_mock_vision:
            from types import SimpleNamespace
            self.intrinsics = SimpleNamespace(width=640, height=480, ppx=320.0, ppy=240.0, fx=600.0, fy=600.0)
        else:
            self.load_yolo()

    def _heartbeat_callback(self):
        self.get_logger().info(f"[HEARTBEAT] Node alive. State: {self.state}")

    def load_yolo(self):
        try:
            from ultralytics import YOLO
            MODEL_PATH = Path(__file__).resolve().parents[4] / "detection" / "weights" / "cam_1.pt"
            self.model = YOLO(str(MODEL_PATH))
        except Exception as e:
            self.get_logger().error(f"YOLO Fail: {e}")

    def synced_callback(self, color_msg, depth_msg, info_msg):
        # THROTTLING: Only process images if we are in DETECTING state or if we have no data yet
        if self.state != "DETECTING" and self.latest_cv_color is not None:
            return

        try:
            self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except: return
        
        if self.intrinsics is None and not self.use_mock_vision and rs is not None:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width, self.intrinsics.height = info_msg.width, info_msg.height
            self.intrinsics.ppx, self.intrinsics.ppy = info_msg.k[2], info_msg.k[5]
            self.intrinsics.fx, self.intrinsics.fy = info_msg.k[0], info_msg.k[4]
            self.intrinsics.coeffs = list(info_msg.d)

    def goal_callback(self, goal_request):
        self.get_logger().info(f"[ACTION] Goal received! State: {self.state}")
        if self.state == "RUNNING" or self.state == "DETECTING": 
            self.get_logger().warn(f"[ACTION] Goal REJECTED. Busy in state: {self.state}")
            return GoalResponse.REJECT
        self.get_logger().info("[ACTION] Goal ACCEPTED.")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle): return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        # Local import: Critically uses the node assigned to DR_init
        from DSR_ROBOT2 import movej, set_robot_mode, wait, ROBOT_MODE_AUTONOMOUS

        self.get_logger().info(f'--- [PICK] EXECUTION STARTED for: {goal_handle.request.bottle_name} ---')
        feedback_msg = PickBottle.Feedback()
        result = PickBottle.Result()
        target_name = goal_handle.request.bottle_name
        self.state = "RUNNING"

        try:
            # Step 0: Ensure Autonomous Mode
            self.get_logger().info("Step 0: Ensuring Autonomous Mode...")
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)
            self.get_logger().info("Step 0: Wait completed.")

            # Step 1: Preparing
            self.get_logger().info("Step 1: Preparing (Moving to READY)...")
            feedback_msg.current_state, feedback_msg.progress = "STEP 1: PREPARING", 0.1
            goal_handle.publish_feedback(feedback_msg)
            
            self.get_logger().info(f"Moving to POSJ_PICK_PLACE_READY: {POSJ_PICK_PLACE_READY}")
            movej(POSJ_PICK_PLACE_READY, vel=VELOCITY, acc=ACC)
            self.get_logger().info("movej(READY) returned.")
            
            if goal_handle.is_cancel_requested: 
                self.get_logger().warn("Cancel requested during Step 1.")
                return self._abort(goal_handle, result)

            # Step 2: Detecting
            self.state = "DETECTING"
            self.get_logger().info("Step 2: Detecting bottle. Waiting 1s for camera settle...")
            feedback_msg.current_state, feedback_msg.progress = "STEP 2: DETECTING", 0.3
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1.0) # wait for camera to settle
            
            p_robot = None
            if self.use_mock_vision:
                self.get_logger().info("Using MOCK vision, sleeping 1s...")
                p_robot = np.array([450.0, 50.0, 120.0])
            else:
                self.get_logger().info(f"Using REAL vision for {target_name}...")
                # Try detection up to 3 times
                for attempt in range(3):
                    self.get_logger().info(f"Detection attempt {attempt+1}/3")
                    target_data = self.vision_detect(target_name)
                    if target_data:
                        p_robot = self.calculate_pose(target_data)
                        self.get_logger().info(f"Detected pose in robot frame: {p_robot}")
                        break
                    time.sleep(0.5)
                
                if p_robot is None:
                    self.get_logger().error("Detection failed after 3 attempts.")
                    result.success, result.message = False, "Detection failed"
                    goal_handle.succeed(); self.state = "IDLE"; return result

            # Successful detection - Open Gripper now
            cfg = BOTTLE_CONFIG.get(target_name, {})
            release_force = cfg.get('gripper_force', GRIPPER_FORCE_DEFAULT)
            self.get_logger().info(f"Detection successful. Opening gripper sync for {target_name} (force={release_force})...")
            self._gripper_move_sync(GRIPPER_POSITION_OPEN, release_force)
            time.sleep(2.0)

            self.state = "RUNNING"
            # Step 3: Moving
            self.get_logger().info("Step 3: Moving to bottle approach pose...")
            feedback_msg.current_state, feedback_msg.progress = "STEP 3: MOVING_TO_BOTTLE", 0.5
            goal_handle.publish_feedback(feedback_msg)
            self._approach_logic(p_robot)
            
            if goal_handle.is_cancel_requested: 
                self.get_logger().warn("Cancel requested during Step 3.")
                return self._abort(goal_handle, result)

            # Step 4: Grasping
            self.get_logger().info("Step 4: Executing grasp logic...")
            feedback_msg.current_state, feedback_msg.progress = "STEP 4: GRASPING", 0.7
            goal_handle.publish_feedback(feedback_msg)
            self._grasp_logic(target_name)

            # Step 5: Completed
            self.get_logger().info("Step 5: Pick completed. Moving back to READY pose...")
            feedback_msg.current_state, feedback_msg.progress = "STEP 5: COMPLETED", 1.0
            goal_handle.publish_feedback(feedback_msg)
            movej(POSJ_PICK_PLACE_READY, vel=VELOCITY, acc=ACC)

            result.success, result.message = True, "Pick success"
            result.pick_pose = [float(p) for p in p_robot]
            self.get_logger().info("--- [PICK] SUCCESS. Goal Succeeding. ---")
            goal_handle.succeed()
            
        except Exception as e:
            self.get_logger().error(f"--- [PICK] EXECUTION ERROR: {e} ---")
            import traceback
            self.get_logger().error(traceback.format_exc())
            result.success, result.message = False, str(e)
            goal_handle.succeed()
        finally:
            self.state = "IDLE"
        return result

    def _approach_logic(self, p_robot):
        from DSR_ROBOT2 import movel, get_current_posx
        self.get_logger().info(f"Approach logic for p_robot: {p_robot}")
        curr_pos = list(get_current_posx()[0])
        tx, ty = p_robot[0] + PICK_PLACE_X_OFFSET, p_robot[1] + PICK_PLACE_Y_OFFSET
        
        self.get_logger().info(f"Moving X to {tx}")
        movel([tx, curr_pos[1], curr_pos[2], curr_pos[3], curr_pos[4], curr_pos[5]], vel=[40, 40], acc=[40, 40])
        self.get_logger().info(f"Moving Y to {ty}")
        movel([tx, ty, curr_pos[2], curr_pos[3], curr_pos[4], curr_pos[5]], vel=[40, 40], acc=[40, 40])

    def _grasp_logic(self, target_name):
        from DSR_ROBOT2 import movel, get_current_posx, wait
        cfg = BOTTLE_CONFIG.get(target_name, BOTTLE_CONFIG['soju'])
        self.get_logger().info(f"Grasp logic for {target_name}, closing gripper to {cfg['gripper_pos']}")
        self._gripper_move_sync(cfg['gripper_pos'], cfg['gripper_force'])
        
        # Wait 5s before lifting as requested
        self.get_logger().info("Wait 5.0s before lifting...")
        time.sleep(5.0)

        # Step 4.1: Return sequence (Lift then Y-Retreat)
        self.get_logger().info("Step 4.1: Return sequence (Lift then Y-Retreat)...")
        
        # 1. Lift up 3cm
        curr = list(get_current_posx()[0])
        self.get_logger().info("Lifting bottle (+30mm Z)")
        target_pos_lift = [curr[0], curr[1], curr[2] + 30.0, curr[3], curr[4], curr[5]]
        movel(target_pos_lift, vel=[40, 40], acc=[40, 40])
        
        # 2. Retreat (Y-Exit)
        from .defines import POSJ_PICK_PLACE_READY
        from DSR_ROBOT2 import fkin
        ready_posx = list(fkin(POSJ_PICK_PLACE_READY, ref=0))
        ready_y = ready_posx[1]
        
        curr_lifted = list(get_current_posx()[0])
        self.get_logger().info(f"Retreating Y to {ready_y:.1f}")
        target_pos_retreat = [curr_lifted[0], ready_y, curr_lifted[2], curr_lifted[3], curr_lifted[4], curr_lifted[5]]
        movel(target_pos_retreat, vel=[40, 40], acc=[40, 40])
        
        return True

    def _gripper_move_sync(self, pos, force):
        self.get_logger().info(f"Gripper request (Fire & Forget): pos={pos}, force={force}")
        if not self.gripper_move_cli.wait_for_service(timeout_sec=2.0): 
            self.get_logger().error("Gripper service not available!")
            return False
        req = GripperControl.Request(position=int(pos), force=int(force))
        
        try:
            self.gripper_move_cli.call_async(req)
            self.get_logger().info("Gripper request sent. Sleeping 2.0s for physical motion...")
            time.sleep(2.0)
            self.get_logger().info("Gripper motion wait complete.")
            return True
        except Exception as e:
            self.get_logger().error(f"Gripper call failed: {e}")
            return False

    def vision_detect(self, target_name):
        self.get_logger().info(f"vision_detect called for {target_name}")
        if self.latest_cv_color is None:
            self.get_logger().warn("vision_detect: latest_cv_color is None!")
            return None
        if self.model is None:
            self.get_logger().error("vision_detect: model is None!")
            return None
            
        class_id = str(BOTTLE_CONFIG[target_name]['id'])
        self.get_logger().info(f"Running YOLO inference on class_id: {class_id}...")
        
        img = cv2.flip(self.latest_cv_color.copy(), -1)
        h, w = img.shape[:2]
        results = self.model(img)
        
        self.get_logger().info(f"YOLO inference completed. Found {len(results)} results.")
        for result in results:
            for box in result.boxes:
                detected_class = str(int(box.cls[0].cpu().numpy()))
                if detected_class == class_id:
                    self.get_logger().info("Target class match found!")
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')
                    u_raw, v_raw = w - 1 - int((x1 + x2) / 2), h - 1 - int((y1 + y2) / 2)
                    roi = self.latest_cv_depth_mm[max(0,v_raw-5):min(h,v_raw+6), max(0,u_raw-5):min(w,u_raw+6)]
                    valid = roi[roi > 0]
                    if valid.size > 0: 
                        depth = np.median(valid)
                        self.get_logger().info(f"Valid depth found: {depth}mm")
                        return {"u_raw": u_raw, "v_raw": v_raw, "depth_mm": depth}
        
        self.get_logger().warn("Target bottle not found in current frame.")
        return None

    def calculate_pose(self, data):
        depth_m = data["depth_mm"] / 1000.0
        x_cam = (data["u_raw"] - self.intrinsics.ppx) * depth_m / self.intrinsics.fx
        y_cam = (data["v_raw"] - self.intrinsics.ppy) * depth_m / self.intrinsics.fy
        return (np.array([x_cam * 1000, y_cam * 1000, depth_m * 1000]) @ self.R + self.t)

    def _abort(self, goal_handle, result):
        goal_handle.canceled(); self.state = "IDLE"
        result.success, result.message = False, "Canceled"
        return result

def main(args=None):
    rclpy.init(args=args)
    from rclpy.executors import MultiThreadedExecutor
    
    # 1. Main Action Node
    node = RobotControllerNode()
    
    # 2. ARCHITECTURAL FIX: ISOLATED DOOSAN NODE
    # We create a separate node dedicated to the Doosan library's internal logic.
    # This prevents its state subscribers from deadlocking with our vision/action logic.
    doosan_node = rclpy.create_node('pick_doosan_internal', namespace='/dsr01')
    doosan_node._default_callback_group = ReentrantCallbackGroup()
    
    # Initialize DR_init with the ISOLATED node
    import DR_init
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = "dsr01", "e0509", doosan_node
    
    # 3. Use a larger thread pool
    executor = MultiThreadedExecutor(num_threads=20)
    executor.add_node(node)
    executor.add_node(doosan_node)
    
    try: 
        executor.spin()
    except KeyboardInterrupt: 
        pass
    finally: 
        node.destroy_node()
        doosan_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
