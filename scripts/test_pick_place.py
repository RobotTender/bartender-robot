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
from dsr_msgs2.srv import MoveJoint, MoveLine, GetCurrentPose, SetRobotMode
from sensor_msgs.msg import Image, CameraInfo
import message_filters

# Constants
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"

from bartender_test.defines import PICK_PLACE_READY

class PickPlaceTester(Node):
    def __init__(self, target_item='beer'):
        super().__init__('pick_place_tester', namespace=ROBOT_ID)
        self.target_item = target_item
        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None
        
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
        self.gripper_open_cli = self.create_client(Trigger, 'robotender_gripper/open')
        self.gripper_close_cli = self.create_client(Trigger, 'robotender_gripper/close')
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

        self.get_logger().info(f"PickPlaceTester initialized for target: {target_item}")

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

    def _call_trigger(self, cli):
        cli.wait_for_service()
        future = cli.call_async(Trigger.Request())
        while rclpy.ok() and not future.done(): time.sleep(0.01)
        return future.result()

    def _movej(self, pos, vel=60.0, acc=60.0):
        self.movej_cli.wait_for_service()
        req = MoveJoint.Request()
        req.pos = [float(x) for x in pos]; req.vel = float(vel); req.acc = float(acc)
        future = self.movej_cli.call_async(req)
        while rclpy.ok() and not future.done(): time.sleep(0.01)
        return future.result()

    def _movel(self, pos, vel=40.0, acc=40.0):
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

        # 1. Start from PICK_PLACE_READY
        self.get_logger().info("Step 1: Moving to PICK_PLACE_READY")
        self._movej(PICK_PLACE_READY)
        
        # 2. Pick the target
        self.get_logger().info(f"Step 2: Detecting and Picking {self.target_item}")
        p_robot = self.detect_object()
        if p_robot is None:
            self.get_logger().error("Object detection failed")
            return

        self.pick_motion(p_robot)

        # 3. Come back to PICK_PLACE_READY
        self.get_logger().info("Step 3: Returning to PICK_PLACE_READY")
        self._movej(PICK_PLACE_READY)

        # 4. Wait 3 seconds
        self.get_logger().info("Step 4: Waiting for 3 seconds")
        time.sleep(3.0)

        # 5. Place back where it was
        self.get_logger().info("Step 5: Placing back to original coordinates")
        self.place_motion(p_robot)

        # 6. Come back to PICK_PLACE_READY
        self.get_logger().info("Step 6: Final return to PICK_PLACE_READY")
        self._movej(PICK_PLACE_READY)
        self.get_logger().info("Test Completed Successfully")

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
                    return self.camera_to_robot(np.array([x_cam*1000, y_cam*1000, z_cam*1000]))
        return None

    def pick_motion(self, p_robot):
        x, y, z = p_robot
        P_ready = [28.0, -35.0, 100.0, 77.0, 63.0, -154.0]
        self._movej(P_ready, vel=40, acc=40)
        self._call_trigger(self.gripper_open_cli)
        time.sleep(1.0)
        current_pos = self._get_posx()
        target_1 = [x - 20, y - 50, z, current_pos[3], current_pos[4], current_pos[5]]
        target_2 = [x - 20, y + 50, z - 20, current_pos[3], current_pos[4], current_pos[5]]
        self._movel(target_1, vel=40, acc=40)
        self._movel(target_2, vel=40, acc=40)
        time.sleep(0.5)
        self._call_trigger(self.gripper_close_cli)
        time.sleep(3.0)
        # 5. Lift and Retreat
        from bartender_test.defines import HOME_POSE, PICK_PLACE_READY
        self.get_logger().info("Moving to PICK_PLACE_READY...")
        self._movej(PICK_PLACE_READY, vel=40, acc=40)
        self.get_logger().info("Moving to HOME_POSE...")
        self._movej(HOME_POSE, vel=40, acc=40)

    def place_motion(self, p_robot):
        x, y, z = p_robot
        from bartender_test.defines import HOME_POSE
        self._movej(HOME_POSE, vel=40, acc=40)
        current_pos = self._get_posx()
        target_2 = [x - 20, y + 50, z - 20, current_pos[3], current_pos[4], current_pos[5]]
        self._movel(target_2, vel=40, acc=40)
        time.sleep(0.5)
        self._call_trigger(self.gripper_open_cli)
        time.sleep(2.0)
        target_1 = [x - 20, y - 50, z, current_pos[3], current_pos[4], current_pos[5]]
        self._movel(target_1, vel=40, acc=40)

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'beer'
    rclpy.init()
    tester = PickPlaceTester(target)
    executor = MultiThreadedExecutor()
    executor.add_node(tester)
    import threading
    t = threading.Thread(target=tester.run_test, daemon=True)
    t.start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
