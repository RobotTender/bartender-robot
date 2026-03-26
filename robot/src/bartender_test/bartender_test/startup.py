#!/usr/bin/env python3
import rclpy
import sys
import time
import threading
import subprocess
from rclpy.node import Node
from rclpy.action import ActionClient
from dsr_msgs2.srv import GetRobotState, GetDrlState, DrlStop, SetRobotControl, MoveJoint
from robotender_msgs.srv import GripperControl
from robotender_msgs.action import PickBottle, PlaceBottle, PourBottle

ROBOT_ID = "dsr01"

class RobotStartupNode(Node):
    def __init__(self):
        super().__init__("robot_startup_node", namespace=ROBOT_ID)
        self.get_logger().info("Robot Startup Node Initialized (Passive Mode).")
        
        self.state_cli = self.create_client(GetRobotState, f'/{ROBOT_ID}/system/get_robot_state')
        self.drl_state_cli = self.create_client(GetDrlState, f'/{ROBOT_ID}/drl/get_drl_state')
        self.drl_stop_cli = self.create_client(DrlStop, f'/{ROBOT_ID}/drl/drl_stop')
        self.control_cli = self.create_client(SetRobotControl, f'/{ROBOT_ID}/system/set_robot_control')
        self.movej_cli = self.create_client(MoveJoint, f'/{ROBOT_ID}/motion/move_joint')

    def wait_for_gripper_service(self, timeout=10.0):
        gripper_cli = self.create_client(GripperControl, f'/{ROBOT_ID}/robotender_gripper/move')
        return gripper_cli.wait_for_service(timeout_sec=timeout)

    def wait_for_motion_service(self, timeout=15.0):
        return self.movej_cli.wait_for_service(timeout_sec=timeout)

    def wait_for_action_server(self, action_type, action_name, timeout=10.0):
        client = ActionClient(self, action_type, f'/{ROBOT_ID}/{action_name}')
        return client.wait_for_server(timeout_sec=timeout)

    def spawn_logic_nodes(self):
        self.get_logger().info("Cleaning up existing logic nodes...")
        subprocess.run(
            ["pkill", "-9", "-f", r"python3 -m bartender_test\.(gripper|pick|pour|place|manager)"],
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)

        self.get_logger().info("Launching Persistent Logic Nodes...")
        subprocess.Popen(["python3", "-m", "bartender_test.gripper"], start_new_session=True)
        if not self.wait_for_gripper_service(timeout=10.0):
            self.get_logger().error("Gripper service did not become available in time.")
            return False

        subprocess.Popen(["python3", "-m", "bartender_test.pick"], start_new_session=True)
        if not self.wait_for_action_server(PickBottle, 'robotender_pick/execute', timeout=10.0):
            self.get_logger().error("Pick action server did not become available in time.")
            return False

        subprocess.Popen(["python3", "-m", "bartender_test.pour"], start_new_session=True)
        if not self.wait_for_action_server(PourBottle, 'robotender_pour/execute', timeout=10.0):
            self.get_logger().error("Pour action server did not become available in time.")
            return False

        subprocess.Popen(["python3", "-m", "bartender_test.place"], start_new_session=True)
        if not self.wait_for_action_server(PlaceBottle, 'robotender_place/execute', timeout=10.0):
            self.get_logger().error("Place action server did not become available in time.")
            return False

        subprocess.Popen(["python3", "-m", "bartender_test.manager"], start_new_session=True)
        self.get_logger().info("Gripper, Pick, Pour, Place, and Manager logic nodes spawned in background.")
        return True

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

                    self.get_logger().info("Waiting for motion service to become available...")
                    if not self.wait_for_motion_service(timeout=15.0):
                        self.get_logger().error("MoveJoint service did not become available in time.")
                        return False

                    return self.spawn_logic_nodes()
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
