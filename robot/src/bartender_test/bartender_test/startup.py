#!/usr/bin/env python3
import rclpy
import sys
import time
from rclpy.node import Node
from dsr_msgs2.srv import SetRobotMode, MoveJoint, DrlStop, GetRobotState, GetDrlState
from .defines import CHEERS_POSE
from .gripper_controller import GripperController

ROBOT_ID = "dsr01"
VELOCITY, ACC = 30, 30

class RobotStartupNode(Node):
    def __init__(self):
        super().__init__("robot_startup_node", namespace=ROBOT_ID)
        self.get_logger().info("Robot Startup Node Initialized.")

    def wait_until_ready(self, step_name):
        state_cli = self.create_client(GetRobotState, '/dsr01/system/get_robot_state')
        self.get_logger().info(f"Checking robot readiness after: {step_name}")
        for i in range(20):
            if state_cli.wait_for_service(timeout_sec=1.0):
                future = state_cli.call_async(GetRobotState.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                res = future.result()
                if res and res.success:
                    if res.robot_state == 1: # STATE_STANDBY
                        self.get_logger().info(f"  [OK] Robot is in STANDBY.")
                        time.sleep(1.0)
                        return True
            self.get_logger().info(f"  [WAIT] Still waiting for Standby state ({i+1}/20)...")
            time.sleep(1.0)
        return False

    def wait_drl_stop(self, timeout=15.0):
        state_cli = self.create_client(GetDrlState, '/dsr01/drl/get_drl_state')
        stop_cli = self.create_client(DrlStop, '/dsr01/drl/drl_stop')
        self.get_logger().info("Waiting for DRL Manager to reach STOP state...")
        start = time.time()
        while time.time() - start < timeout:
            if state_cli.wait_for_service(timeout_sec=1.0):
                future = state_cli.call_async(GetDrlState.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                res = future.result()
                if res and res.success:
                    if res.drl_state == 1 or res.drl_state == 3: # STOP or LAST (Idle)
                        self.get_logger().info(f"  [OK] DRL Manager is ready (state={res.drl_state}).")
                        return True
                    else:
                        self.get_logger().warn(f"  [WAIT] DRL state is {res.drl_state}. Forcing DrlStop...")
                        s_future = stop_cli.call_async(DrlStop.Request(stop_mode=1))
                        rclpy.spin_until_future_complete(self, s_future, timeout_sec=2.0)
            time.sleep(1.0)
        return False

    def run(self):
        try:
            # STEP 1: Stabilization
            self.get_logger().info("--- STEP 1: Stabilization ---")
            time.sleep(5.0)

            # STEP 2: Autonomous Mode & HARD RESET
            self.get_logger().info("--- STEP 2: Robot Mode & Reset ---")
            cli_mode = self.create_client(SetRobotMode, '/dsr01/system/set_robot_mode')
            if cli_mode.wait_for_service(timeout_sec=5.0):
                # Toggle mode to force clear simple alarms
                self.get_logger().info("Resetting Robot Mode...")
                cli_mode.call_async(SetRobotMode.Request(robot_mode=0)) # MANUAL
                time.sleep(1.0)
                cli_mode.call_async(SetRobotMode.Request(robot_mode=1)) # AUTONOMOUS
                time.sleep(1.0)
            
            if not self.wait_until_ready("Set Mode"): return False

            # STEP 3: Move to Cheers Pose
            self.get_logger().info("--- STEP 3: Prime Move (Cheers Pose) ---")
            cli_move = self.create_client(MoveJoint, '/dsr01/motion/move_joint')
            if cli_move.wait_for_service(timeout_sec=5.0):
                self.get_logger().info(f"Moving to CHEERS_POSE: {CHEERS_POSE}")
                req = MoveJoint.Request()
                req.pos = [float(x) for x in CHEERS_POSE]
                req.vel = float(VELOCITY)
                req.acc = float(ACC)
                req.mode = 0 # ABS
                future = cli_move.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            
            if not self.wait_until_ready("Prime Move"): return False

            # STEP 4: Cleanup
            self.get_logger().info("--- STEP 4: DRL Cleanup ---")
            if not self.wait_drl_stop(): return False
            if not self.wait_until_ready("DrlStop"): return False

            # STEP 5: Activate and Open Gripper
            self.get_logger().info("--- STEP 5: Gripper Initialization ---")
            gripper = GripperController(node=self, namespace=ROBOT_ID)
            
            self.get_logger().info("Activating Gripper...")
            if not gripper.activate(force=400):
                self.get_logger().error("Gripper Activation Failed!")
                return False
            
            time.sleep(2.0)
            
            self.get_logger().info("Opening Gripper...")
            if not gripper.move(0):
                self.get_logger().error("Gripper Open Failed!")
                return False
            
            if not self.wait_until_ready("Gripper Init"): return False

            self.get_logger().info("==========================================")
            self.get_logger().info("STARTUP COMPLETE: Robot is at CHEERS_POSE and Gripper is OPEN.")
            self.get_logger().info("==========================================")
            return True

        except Exception as e:
            self.get_logger().error(f"Startup Script Failed: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = RobotStartupNode()
    success = node.run()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
