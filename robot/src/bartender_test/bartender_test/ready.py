#!/usr/bin/env python3
import rclpy
import sys
import time
from rclpy.node import Node
from dsr_msgs2.srv import SetRobotMode, MoveJoint, DrlStop, GetRobotState
from .defines import CHEERS_POSE
from .gripper_controller import GripperController

ROBOT_ID = "dsr01"
VELOCITY, ACC = 30, 30

class RobotReadyNode(Node):
    def __init__(self):
        super().__init__("robot_ready_node", namespace=ROBOT_ID)
        self.get_logger().info("Robot Ready Node Initialized.")

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

    def run(self):
        try:
            # STEP 1: Wait for controller_manager
            self.get_logger().info("--- STEP 1: Stabilization ---")
            self.get_logger().info("Waiting 5s for controller_manager to settle...")
            time.sleep(5.0)

            # STEP 2: Set Robot Mode
            self.get_logger().info("--- STEP 2: Robot Mode ---")
            cli_mode = self.create_client(SetRobotMode, '/dsr01/system/set_robot_mode')
            if cli_mode.wait_for_service(timeout_sec=5.0):
                self.get_logger().info("Setting robot to AUTONOMOUS mode...")
                req = SetRobotMode.Request()
                req.robot_mode = 1 # AUTONOMOUS
                future = cli_mode.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            else:
                self.get_logger().error("SetRobotMode service not available!")
                return False
            
            if not self.wait_until_ready("Set Mode"): return False

            # STEP 3: Prime Motion (Cheers Pose)
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
            else:
                self.get_logger().error("MoveJoint service not available!")
                return False
            
            if not self.wait_until_ready("Prime Move"): return False

            # STEP 4: Clear Task Manager
            self.get_logger().info("--- STEP 4: DRL Cleanup ---")
            cli_stop = self.create_client(DrlStop, '/dsr01/drl/drl_stop')
            if cli_stop.wait_for_service(timeout_sec=5.0):
                self.get_logger().info("Stopping any existing DRL tasks...")
                future = cli_stop.call_async(DrlStop.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            time.sleep(2.0)
            if not self.wait_until_ready("DrlStop"): return False

            # STEP 5: Gripper Initialization (Activate and Open)
            self.get_logger().info("--- STEP 5: Gripper Initialization ---")
            gripper = GripperController(node=self, namespace=ROBOT_ID)
            self.get_logger().info("Activating and Opening Gripper...")
            
            # Consolidated DRL: Init + Open(0) in one go
            init_and_open_code = """
wait(0.5)
flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
modbus_set_slaveid(1)
for i in range(0, 5):
    flange_serial_write(modbus_fc06(256, 1))
    wait(0.2)
    flag, val = recv_check()
    if flag is True:
        flange_serial_write(modbus_fc06(275, 400))
        wait(0.2)
        break
wait(1.0)
gripper_move(0)
flange_serial_close()
"""
            gripper._execute_drl(init_and_open_code, timeout=15.0)
            
            if not self.wait_until_ready("Gripper Init"): return False

            self.get_logger().info("==========================================")
            self.get_logger().info("SUCCESS: Robot is at CHEERS_POSE and Gripper is OPEN.")
            self.get_logger().info("==========================================")
            return True

        except Exception as e:
            self.get_logger().error(f"Ready Script Failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False

def main(args=None):
    rclpy.init(args=args)
    node = RobotReadyNode()
    success = node.run()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
