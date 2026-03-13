import rclpy
import sys
import time
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from dsr_msgs2.srv import SetRobotMode
from .gripper_controller import GripperController

class GripperNode(Node):
    def __init__(self, namespace="dsr01"):
        super().__init__('gripper_node', namespace=namespace)
        self.namespace = namespace
        self.get_logger().info(f"Initializing Multi-threaded GripperNode for namespace '{namespace}'...")
        self.get_logger().info("--- ROS2: Starting Gripper Hardware Initialization ---")
        
        self.cb_group = ReentrantCallbackGroup()
        
        try:
            self.gripper_controller = GripperController(node=self, namespace=self.namespace)
            self.srv_open = self.create_service(
                Trigger, f'/{self.namespace}/gripper/open', self.open_callback, 
                callback_group=self.cb_group
            )
            self.srv_close = self.create_service(
                Trigger, f'/{self.namespace}/gripper/close', self.close_callback,
                callback_group=self.cb_group
            )
            self.get_logger().info(f"GripperNode Services ready: /{self.namespace}/gripper/open and close (Trigger).")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize GripperController: {e}")
            raise e

    def open_callback(self, request, response):
        self.get_logger().info("Service Call: OPEN (Force 400, Stroke 0)")
        success = self.gripper_controller.action(force=400, stroke=0)
        response.success = success
        return response

    def close_callback(self, request, response):
        self.get_logger().info("Service Call: CLOSE (Force 800, Stroke 700)")
        success = self.gripper_controller.action(force=800, stroke=700)
        response.success = success
        return response

def main(args=None):
    if len(sys.argv) <= 1:
        rclpy.init(args=args)
        try:
            node = GripperNode()
            executor = MultiThreadedExecutor()
            executor.add_node(node)
            executor.spin()
        except Exception as e:
            print(f"GripperNode Error: {e}")
        finally:
            if rclpy.ok():
                rclpy.shutdown()
        return

    # CLI Client Mode
    cmd = sys.argv[1].lower()
    if cmd not in ['open', 'close']:
        print(f"Unknown command: {cmd}")
        return

    rclpy.init(args=args)
    client_node = rclpy.create_node('gripper_cli_client')
    
    # Wait for drl_start service to ensure Doosan stack is ready
    from dsr_msgs2.srv import DrlStart
    cli = client_node.create_client(DrlStart, '/dsr01/drl/drl_start')
    if not cli.wait_for_service(timeout_sec=5.0):
        print("Doosan drl_start service not available.")
        client_node.destroy_node()
        rclpy.shutdown()
        return
        
    controller = GripperController(node=client_node, namespace="dsr01")
    
    if cmd == 'open':
        force = 400
        if len(sys.argv) > 2:
            force = int(sys.argv[2])
        print(f"Executing Open (Force: {force})...")
        res = controller.action(force=force, stroke=0)
        print(f"Result: {res}")
            
    elif cmd == 'close':
        force = 800
        if len(sys.argv) > 2:
            force = int(sys.argv[2])
        print(f"Executing Close (Force: {force})...")
        res = controller.action(force=force, stroke=700)
        print(f"Result: {res}")
            
    client_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
