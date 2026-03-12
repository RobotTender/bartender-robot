#!/usr/bin/env python3
import rclpy
import sys
import time
import threading
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from dsr_msgs2.srv import SetRobotMode, MoveJoint, DrlStop, GetRobotState, GetDrlState
from .defines import CHEERS_POSE
from .gripper_controller import GripperController

ROBOT_ID = "dsr01"
VELOCITY, ACC = 30, 30

class RobotStartupNode(Node):
    def __init__(self):
        super().__init__("robot_startup_node", namespace=ROBOT_ID)
        self.get_logger().info("Robot Startup Node Initialized.")
        
        # Shared clients to avoid redundant creation
        self.state_cli = self.create_client(GetRobotState, '/dsr01/system/get_robot_state')
        self.mode_cli = self.create_client(SetRobotMode, '/dsr01/system/set_robot_mode')
        self.move_cli = self.create_client(MoveJoint, '/dsr01/motion/move_joint')
        self.drl_stop_cli = self.create_client(DrlStop, '/dsr01/drl/drl_stop')

    def call_srv(self, cli, req, timeout=5.0):
        """Helper to call service and wait for result in a thread-safe way."""
        if not cli.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"Service {cli.srv_name} not available!")
            return None
        
        future = cli.call_async(req)
        # Since we are using MultiThreadedExecutor, we can just wait for the future
        # without calling spin_until_future_complete manually.
        start = time.time()
        while not future.done() and time.time() - start < timeout:
            time.sleep(0.1)
        return future.result() if future.done() else None

    def wait_until_ready(self, step_name):
        self.get_logger().info(f"Checking robot readiness after: {step_name}")
        for i in range(20):
            res = self.call_srv(self.state_cli, GetRobotState.Request())
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
            # STEP 1: Stabilization
            self.get_logger().info("--- STEP 1: Stabilization ---")
            time.sleep(5.0)

            # STEP 2: Autonomous Mode & HARD RESET
            self.get_logger().info("--- STEP 2: Robot Mode & Reset ---")
            self.get_logger().info("Resetting Robot Mode (MANUAL -> AUTONOMOUS)...")
            self.call_srv(self.mode_cli, SetRobotMode.Request(robot_mode=0)) # MANUAL
            time.sleep(2.0)
            self.call_srv(self.mode_cli, SetRobotMode.Request(robot_mode=1)) # AUTONOMOUS
            time.sleep(2.0)
            
            if not self.wait_until_ready("Set Mode"): return False

            # STEP 3: Move to Cheers Pose
            self.get_logger().info("--- STEP 3: Prime Move (Cheers Pose) ---")
            self.get_logger().info(f"Moving to CHEERS_POSE: {CHEERS_POSE}")
            req = MoveJoint.Request()
            req.pos = [float(x) for x in CHEERS_POSE]
            req.vel = float(VELOCITY)
            req.acc = float(ACC)
            req.mode = 0 # ABS
            self.call_srv(self.move_cli, req, timeout=15.0)
            
            if not self.wait_until_ready("Prime Move"): return False

            # STEP 4: Gripper Initialization (Handles DRL Cleanup Internally)
            self.get_logger().info("--- STEP 4: Gripper Initialization ---")
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
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False

def main(args=None):
    rclpy.init(args=args)
    node = RobotStartupNode()
    
    # Use MultiThreadedExecutor to allow concurrent service processing
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    # Run the logic in a separate thread so the executor can spin
    thread = threading.Thread(target=node.run, daemon=True)
    thread.start()
    
    try:
        while thread.is_alive():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    
    # Get the result from the thread (success/failure)
    # Since we can't easily get result from thread, we rely on node exit status
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
