#!/usr/bin/env python3
import sys
import os
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import numpy as np

# ROS 2 messages/services
from std_srvs.srv import Trigger
from dsr_msgs2.srv import MoveJoint, MoveLine, GetCurrentPose, SetRobotMode
from std_msgs.msg import Float64MultiArray

# Constants
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 30.0, 30.0

from bartender_test.defines import (
    PICK_PLACE_READY, HOME_POSE,
    PICK_PLACE_X_OFFSET, PICK_PLACE_Y_OFFSET
)

class PlaceTester(Node):
    def __init__(self, test_coords=None):
        super().__init__('place_tester', namespace=ROBOT_ID)
        self.test_coords = test_coords # [x, y, z]
        
        # Service Clients
        self.gripper_open_cli = self.create_client(Trigger, 'robotender_gripper/open')
        self.movej_cli = self.create_client(MoveJoint, 'motion/move_joint')
        self.movel_cli = self.create_client(MoveLine, 'motion/move_line')
        self.pose_cli = self.create_client(GetCurrentPose, 'system/get_current_pose')
        self.mode_cli = self.create_client(SetRobotMode, 'system/set_robot_mode')

        # Subscription to get last pick pose if no coords provided
        self.last_pick_pose = None
        self.pose_sub = self.create_subscription(Float64MultiArray, 'robotender_pick/last_pose', self.pose_cb, 10)

        self.get_logger().info(f"PlaceTester initialized.")

    def pose_cb(self, msg):
        if len(msg.data) >= 3:
            self.last_pick_pose = list(msg.data[:3])

    def _gripper_open(self):
        if not self.gripper_open_cli.wait_for_service(timeout_sec=5.0):
            return False
        future = self.gripper_open_cli.call_async(Trigger.Request())
        while rclpy.ok() and not future.done(): time.sleep(0.01)
        return future.result().success

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

        # Determine target coordinates
        target_coords = self.test_coords
        if target_coords is None:
            self.get_logger().info("Waiting for last_pick_pose from topic...")
            start_wait = time.time()
            while self.last_pick_pose is None and (time.time() - start_wait < 5.0):
                rclpy.spin_once(self, timeout_sec=0.1)
            target_coords = self.last_pick_pose

        if target_coords is None:
            self.get_logger().error("No coordinates provided and no last_pick_pose found! Using default test coords.")
            target_coords = [300.0, 500.0, 650.0]

        self.get_logger().info(f"Target Coordinates: {target_coords}")
        self.place_motion(target_coords)
        
        self.get_logger().info("Test Completed Successfully. Shutting down...")
        time.sleep(1.0)
        os._exit(0)

    def place_motion(self, coords):
        x, y, z_pick = coords
        self.get_logger().info("Placing Motion Started")
        
        # 1. Start from PICK_PLACE_READY
        self.get_logger().info("Step 1: Moving to PICK_PLACE_READY")
        self._movej(PICK_PLACE_READY, vel=VELOCITY, acc=ACC)
        
        # Get current Cartesian pose for relative lift
        curr = self._get_posx()
        z_ready = curr[2]
        
        # 2. Lift up 3cm
        self.get_logger().info("Step 2: Lifting up 3cm")
        z_safe = z_ready + 30.0
        self._movel([curr[0], curr[1], z_safe, curr[3], curr[4], curr[5]], vel=40.0, acc=ACC)

        # 3. X-Alignment (at safe height)
        target_x = x + PICK_PLACE_X_OFFSET
        target_y = y + PICK_PLACE_Y_OFFSET
        
        self.get_logger().info(f"Step 3: X-Alignment to {target_x:.1f}")
        self._movel([target_x, curr[1], z_safe, curr[3], curr[4], curr[5]], vel=40.0, acc=ACC)

        # 4. Y-Alignment (at safe height)
        self.get_logger().info(f"Step 4: Y-Entry to {target_y:.1f}")
        self._movel([target_x, target_y, z_safe, curr[3], curr[4], curr[5]], vel=40.0, acc=ACC)

        # 5. Lift down 2.5cm to original placement height
        z_place = z_safe - 25.0
        self.get_logger().info(f"Step 5: Lifting down 2.5cm to Z: {z_place:.1f}")
        self._movel([target_x, target_y, z_place, curr[3], curr[4], curr[5]], vel=40.0, acc=ACC)
        
        self.get_logger().info("Waiting 0.5s before release...")
        time.sleep(0.5)

        # 6. Release grip
        self.get_logger().info("Step 6: Releasing gripper")
        self._gripper_open()
        
        self.get_logger().info("Waiting 3s after release before retreat...")
        time.sleep(3.0)

        # 7. Reverse: Y-Alignment (Exit) at place height
        self.get_logger().info("Step 7: Retreat (Y-Exit)")
        self._movel([target_x, curr[1], z_place, curr[3], curr[4], curr[5]], vel=40.0, acc=ACC)

        # 8. Return to PICK_PLACE_READY
        self.get_logger().info("Step 8: Returning to PICK_PLACE_READY")
        self._movej(PICK_PLACE_READY, vel=60, acc=60)

        # 9. Return to HOME_POSE
        self.get_logger().info("Step 9: Returning to HOME_POSE")
        self._movej(HOME_POSE, vel=60, acc=60)

def main():
    # Optional: python3 scripts/test_place.py [x y z]
    coords = None
    if len(sys.argv) >= 4:
        try:
            coords = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]
        except ValueError:
            print("Invalid coordinates format. Use: python3 scripts/test_place.py [x y z]")
            sys.exit(1)

    rclpy.init()
    tester = PlaceTester(coords)
    executor = MultiThreadedExecutor()
    executor.add_node(tester)
    
    import threading
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
