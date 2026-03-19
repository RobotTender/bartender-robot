#!/usr/bin/env python3
import rclpy
import sys
import time
import threading
import os
import subprocess
from rclpy.node import Node
from dsr_msgs2.srv import GetRobotState, GetDrlState, DrlStop, SetRobotControl

ROBOT_ID = "dsr01"

class RobotStartupNode(Node):
    def __init__(self):
        super().__init__("robot_startup_node", namespace=ROBOT_ID)
        self.get_logger().info("Robot Startup Node Initialized (Passive Mode).")
        
        self.state_cli = self.create_client(GetRobotState, f'/{ROBOT_ID}/system/get_robot_state')
        self.drl_state_cli = self.create_client(GetDrlState, f'/{ROBOT_ID}/drl/get_drl_state')
        self.drl_stop_cli = self.create_client(DrlStop, f'/{ROBOT_ID}/drl/drl_stop')
        self.control_cli = self.create_client(SetRobotControl, f'/{ROBOT_ID}/system/set_robot_control')

    def call_srv(self, cli, req, timeout=5.0):
        if not cli.wait_for_service(timeout_sec=timeout): return None
        future = cli.call_async(req)
        start = time.time()
        while not future.done() and time.time() - start < timeout: time.sleep(0.1)
        return future.result() if future.done() else None

    def run(self):
        try:
            self.get_logger().info("--- STEP 1: System Readiness Check ---")
            time.sleep(5.0) # Let driver settle
            
            # 1. Reset Safety State only
            self.get_logger().info("Ensuring Safety State is Clear...")
            self.call_srv(self.control_cli, SetRobotControl.Request(robot_control=2))
            
            # 2. Check DRL State
            d_res = self.call_srv(self.drl_state_cli, GetDrlState.Request())
            if d_res and d_res.success and d_res.drl_state != 1:
                self.get_logger().info(f"DRL is in state {d_res.drl_state}. Sending DrlStop...")
                self.call_srv(self.drl_stop_cli, DrlStop.Request(stop_mode=0))
            
            # 3. Final Verification (Wait for Standby)
            self.get_logger().info("Waiting for STANDBY status...")
            for _ in range(10):
                r_res = self.call_srv(self.state_cli, GetRobotState.Request())
                if r_res and r_res.success and r_res.robot_state == 1:
                    self.get_logger().info("==========================================")
                    self.get_logger().info("STARTUP VERIFIED: SYSTEM READY.")
                    self.get_logger().info("==========================================")
                    
                    # Launch background nodes
                    self.get_logger().info("Cleaning up existing logic nodes...")
                    # Specifically target logic nodes to avoid killing this startup process
                    subprocess.run(["pkill", "-9", "-f", "python3 -m bartender_test\.(gripper)"], stderr=subprocess.DEVNULL)
                    time.sleep(0.5)

                    self.get_logger().info("Launching Persistent Logic Nodes...")
                    subprocess.Popen(["python3", "-m", "bartender_test.gripper"], start_new_session=True)
                    time.sleep(1.0)
                    # subprocess.Popen(["python3", "-m", "bartender_test.pick"], start_new_session=True)
                    # time.sleep(1.5)
                    # subprocess.Popen(["python3", "-m", "bartender_test.pour"], start_new_session=True)
                    # time.sleep(0.5)
                    # subprocess.Popen(["python3", "-m", "bartender_test.place"], start_new_session=True)
                    self.get_logger().info("Only gripper logic node spawned in background.")
                    
                    return True
                time.sleep(1.0)
            
            self.get_logger().error("System did not reach STANDBY. Please check Teach Pendant.")
            return False

        except Exception as e:
            self.get_logger().error(f"Startup Failed: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = RobotStartupNode()
    result = {"success": False}
    thread = threading.Thread(target=lambda: result.update({"success": node.run()}), daemon=True)
    thread.start()
    try:
        while thread.is_alive(): rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt: pass
    success = result["success"]
    node.destroy_node(); rclpy.shutdown()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
