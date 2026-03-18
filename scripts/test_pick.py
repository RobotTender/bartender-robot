#!/usr/bin/env python3
import sys
import os
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import cv2
import json
from ultralytics import YOLO
from pathlib import Path
import pyrealsense2 as rs
from cv_bridge import CvBridge

# ROS 2 messages/services
from std_srvs.srv import Trigger
from robotender_msgs.srv import GripperControl
from dsr_msgs2.srv import MoveJoint, MoveLine, GetCurrentPose, SetRobotMode
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
import message_filters

# Constants
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 30.0, 30.0

from bartender_test.defines import (
    PICK_PLACE_READY, HOME_POSE,
    GRIPPER_POSITION_OPEN, GRIPPER_FORCE_OPEN,
    INDEX_JUICE, INDEX_BEER, INDEX_SOJU,
    GRIPPER_POSITIONS, GRIPPER_FORCES,
    PICK_PLACE_X_OFFSET, PICK_PLACE_Y_OFFSET
)

class PickTester(Node):
    def __init__(self, target_item='beer', test_mode='full'):
        super().__init__('pick_tester', namespace=ROBOT_ID)
        self.target_item = target_item
        self.test_mode = test_mode
        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None
        
        # Redefine Offsets here
        self.OFFSET_X = float(PICK_PLACE_X_OFFSET)
        self.OFFSET_Y = float(PICK_PLACE_Y_OFFSET)

        # Load YOLO
        model_path = Path(__file__).resolve().parents[1] / "detection" / "weights" / "cam_1.pt"
        self.model = YOLO(str(model_path))
        self.object_dict_reverse = {'juice': "0", 'beer': "1", 'soju': "2"}
        self.object_dict = {"0": 'juice', "1": 'beer', "2": 'soju'}

        # Camera calibration
        self.R = np.array([
            [-0.788489317968,  -0.614148198918, -0.0332653756482],
            [-0.0868309265704,  0.0576098432706,  0.994555929121],
            [-0.608888319515,   0.787085189623,  -0.0987518031922],
        ], dtype=np.float64)
        self.t = np.array([521.115058698, 170.946228749, 834.749571453], dtype=np.float64)

        # Service Clients
        self.gripper_move_cli = self.create_client(GripperControl, 'robotender_gripper/move')
        self.movej_cli = self.create_client(MoveJoint, 'motion/move_joint')
        self.movel_cli = self.create_client(MoveLine, 'motion/move_line')
        self.pose_cli = self.create_client(GetCurrentPose, 'system/get_current_pose')
        self.mode_cli = self.create_client(SetRobotMode, 'system/set_robot_mode')

        # Camera Subscribers
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera_1/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera_1/aligned_depth_to_color/image_raw')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera_1/aligned_depth_to_color/camera_info')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.info_sub], 10, 0.1)
        self.ts.registerCallback(self.synced_callback)

        # Publisher for coordination with Place node
        self.last_pose_pub = self.create_publisher(Float64MultiArray, 'robotender_pick/last_pose', 10)

        self.get_logger().info(f"PickTester initialized for target: {target_item}, mode: {test_mode}")

    def synced_callback(self, color_msg, depth_msg, info_msg):
        self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        if self.intrinsics is None:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width, self.intrinsics.height = info_msg.width, info_msg.height
            self.intrinsics.ppx, self.intrinsics.ppy = info_msg.k[2], info_msg.k[5]
            self.intrinsics.fx, self.intrinsics.fy = info_msg.k[0], info_msg.k[4]
            self.intrinsics.model = rs.distortion.brown_conrady if info_msg.distortion_model in ['plumb_bob', 'rational_polynomial'] else rs.distortion.none
            self.intrinsics.coeffs = list(info_msg.d)

    def camera_to_robot(self, point_cam):
        return point_cam @ self.R + self.t

    def _gripper_move(self, pos, force):
        self.get_logger().info(f"Service Call: Gripper Move (Pos={pos}, Force={force})")
        if not self.gripper_move_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Gripper move service not available!")
            return False
        req = GripperControl.Request()
        req.position = int(pos)
        req.force = int(force)
        future = self.gripper_move_cli.call_async(req)
        while rclpy.ok() and not future.done(): time.sleep(0.01)
        return future.result().success

    def _gripper_open(self):
        return self._gripper_move(GRIPPER_POSITION_OPEN, GRIPPER_FORCE_OPEN)

    def _gripper_close(self, item_name):
        # Convert item name to index
        try:
            cls_id_str = self.object_dict_reverse.get(item_name)
            if cls_id_str is None:
                raise ValueError(f"Unknown item: {item_name}")
            
            idx = int(cls_id_str)
            pos = GRIPPER_POSITIONS[idx]
            force = GRIPPER_FORCES[idx]
            return self._gripper_move(pos, force)
        except Exception as e:
            self.get_logger().error(f"Gripper close error: {e}, using default soju settings")
            return self._gripper_move(GRIPPER_POSITIONS[INDEX_SOJU], GRIPPER_FORCES[INDEX_SOJU])

    def _movej(self, pos, vel=VELOCITY, acc=ACC):
        self.movej_cli.wait_for_service()
        req = MoveJoint.Request()
        req.pos = [float(x) for x in pos]; req.vel = float(vel); req.acc = float(acc)
        future = self.movej_cli.call_async(req)
        while rclpy.ok() and not future.done(): time.sleep(0.01)
        return future.result()

    def _movel(self, pos, vel=40.0, acc=ACC):
        self.movel_cli.wait_for_service()
        req = MoveLine.Request()
        req.pos = [float(x) for x in pos]; req.vel = [float(vel)]*2; req.acc = [float(acc)]*2
        future = self.movel_cli.call_async(req)
        while rclpy.ok() and not future.done(): time.sleep(0.01)
        return future.result()

    def _get_posx(self):
        self.pose_cli.wait_for_service()
        future = self.pose_cli.call_async(GetCurrentPose.Request(space_type=1))
        while rclpy.ok() and not future.done(): time.sleep(0.01)
        return list(future.result().pos)

    def run_test(self):
        self.get_logger().info("Setting robot mode to AUTONOMOUS...")
        self.mode_cli.wait_for_service()
        self.mode_cli.call_async(SetRobotMode.Request(robot_mode=1))
        time.sleep(0.5)

        self.get_logger().info(f"--- Test Configuration ---")
        self.get_logger().info(f"Target Item: {self.target_item}")
        self.get_logger().info(f"Test Mode: {self.test_mode}")
        self.get_logger().info(f"Offsets: X={self.OFFSET_X}, Y={self.OFFSET_Y}")
        self.get_logger().info(f"--------------------------")

        # 1. Start from HOME_POSE then PICK_PLACE_READY
        self.get_logger().info("Step 1: Moving to HOME_POSE")
        self._movej(HOME_POSE)
        self._gripper_open()
        time.sleep(1.0)
        
        # 2. Pick the target
        self.get_logger().info(f"Step 2: Detecting and Picking {self.target_item}")
        p_robot = self.detect_object()
        if p_robot is None:
            self.get_logger().error("Object detection failed")
            return

        self.get_logger().info(f"로봇 목표 좌표: X={p_robot[0]:.1f}, Y={p_robot[1]:.1f}, Z={p_robot[2]:.1f}")
        
        # Publish coordinates for Place node synchronization
        pose_msg = Float64MultiArray()
        pose_msg.data = [float(p_robot[0]), float(p_robot[1]), float(p_robot[2])]
        self.last_pose_pub.publish(pose_msg)
        self.get_logger().info("Published last_pick_pose for Place node.")

        self.pick_motion(p_robot, self.target_item, self.test_mode)
        
        print("\n" + "="*50)
        print("PICK TEST SUMMARY")
        print(f"Target Item: {self.target_item}")
        print(f"Detected Pose (X Y Z): {p_robot[0]:.1f} {p_robot[1]:.1f} {p_robot[2]:.1f}")
        print(f"Command for Place: python3 scripts/test_place.py {p_robot[0]:.1f} {p_robot[1]:.1f} {p_robot[2]:.1f}")
        print("="*50 + "\n")

        self.get_logger().info("Test Completed Successfully. Shutting down...")
        time.sleep(1.0)
        os._exit(0) # Force exit the process from the thread

    def detect_object(self):
        while self.latest_cv_color is None or self.intrinsics is None:
            time.sleep(0.1)
        img = cv2.flip(self.latest_cv_color.copy(), -1)
        depth_raw = self.latest_cv_depth_mm.copy()
        h, w = img.shape[:2]
        class_id = self.object_dict_reverse[self.target_item]
        results = self.model(img, verbose=False)
        for result in results:
            for box in result.boxes:
                if str(int(box.cls[0].cpu().numpy())) == class_id and box.conf[0].cpu().numpy() > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')
                    u_flip, v_flip = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    u_raw, v_raw = w - 1 - u_flip, h - 1 - v_flip
                    roi_half = 5
                    x_min, x_max = max(0, u_raw - roi_half), min(w, u_raw + roi_half + 1)
                    y_min, y_max = max(0, v_raw - roi_half), min(h, v_raw + roi_half + 1)
                    depth_roi = depth_raw[y_min:y_max, x_min:x_max]
                    valid_depth = depth_roi[depth_roi > 0]
                    if valid_depth.size == 0: continue
                    depth_m = float(np.median(valid_depth)) / 1000.0
                    x_cam, y_cam, z_cam = (u_raw-self.intrinsics.ppx)*depth_m/self.intrinsics.fx, (v_raw-self.intrinsics.ppy)*depth_m/self.intrinsics.fy, depth_m
                    p_cam = np.array([x_cam*1000, y_cam*1000, z_cam*1000])
                    return self.camera_to_robot(p_cam)
        return None

    def pick_motion(self, p_robot, item_name, mode='full'):
        x, y, z = p_robot
        self.get_logger().info(f"잡기 전 자세 (대상: {item_name}, Mode: {mode})")
        self._movej(PICK_PLACE_READY, vel=VELOCITY, acc=ACC)
        
        current_pos = self._get_posx()
        target_x = x + self.OFFSET_X
        target_y = y + self.OFFSET_Y
        
        # Step 1: X-Alignment
        # Align ONLY the X axis while keeping Y, Z, and orientation from the READY pose.
        target_pos_x = [target_x, current_pos[1], current_pos[2], current_pos[3], current_pos[4], current_pos[5]]
        self.get_logger().info(f"Step 1: X-Alignment to {target_x:.1f} (Offset X: {self.OFFSET_X})")
        self._movel(target_pos_x, vel=40.0, acc=ACC)
        
        if mode == 'x':
            self.get_logger().info("X-Alignment test finished.")
            return

        # Step 2: Y-Entry
        # Enter the bottle's position by changing ONLY the Y axis to target.
        target_pos_y = [target_x, target_y, current_pos[2], current_pos[3], current_pos[4], current_pos[5]]
        self.get_logger().info(f"Step 2: Y-Entry to {target_y:.1f} (Offset Y: {self.OFFSET_Y})")
        self._movel(target_pos_y, vel=40.0, acc=ACC)
        
        if mode == 'y':
            self.get_logger().info("Y-Entry test finished.")
            return

        # Full motion continues
        time.sleep(0.5)
        self._gripper_close(item_name)
        self.get_logger().info("그리퍼가 닫힐 때까지 대기합니다 (5초)...")
        time.sleep(5.0)
        
        # Step 2.5: Lift (3cm)
        target_pos_lift = [target_x, target_y, current_pos[2] + 30.0, current_pos[3], current_pos[4], current_pos[5]]
        self.get_logger().info("Step 2.5: Lifting 3cm")
        self._movel(target_pos_lift, vel=40.0, acc=ACC)

        # Step 3: Retreat (Decoupled reversal) at the lifted height
        target_pos_retreat = [target_x, current_pos[1], current_pos[2] + 30.0, current_pos[3], current_pos[4], current_pos[5]]
        self.get_logger().info("Step 3: Retreat (Y-Exit)")
        self._movel(target_pos_retreat, vel=40.0, acc=ACC)

        self.get_logger().info("Moving to PICK_PLACE_READY then HOME_POSE")
        self._movej(PICK_PLACE_READY, vel=VELOCITY, acc=ACC)
        self._movej(HOME_POSE, vel=VELOCITY, acc=ACC)
        self.get_logger().info("Reached HOME_POSE. Waiting 1s for stability...")
        time.sleep(1.0)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/test_pick.py [juice|beer|soju] [x|y|full]")
        sys.exit(1)
        
    target = sys.argv[1].lower()
    if target not in ['juice', 'beer', 'soju']:
        print(f"Error: Invalid target '{target}'. Choose from [juice, beer, soju]")
        sys.exit(1)

    test_mode = 'full'
    if len(sys.argv) >= 3:
        test_mode = sys.argv[2].lower()
        if test_mode not in ['x', 'y', 'full']:
            print(f"Error: Invalid mode '{test_mode}'. Choose from [x, y, full]")
            sys.exit(1)

    rclpy.init()
    tester = PickTester(target, test_mode)
    executor = MultiThreadedExecutor()
    executor.add_node(tester)
    
    import threading
    # Run the test logic in a separate thread so the executor can spin and handle callbacks
    t = threading.Thread(target=tester.run_test, daemon=True)
    t.start()
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
