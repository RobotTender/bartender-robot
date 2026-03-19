import rclpy
import json
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

# Message/Service/Action Imports
from robotender_msgs.action import PickBottle
from robotender_msgs.srv import GripperControl
from dsr_msgs2.srv import MoveJoint

from .defines import (
    POSJ_HOME, 
    GRIPPER_POSITION_OPEN, 
    GRIPPER_FORCE_OPEN
)

class RobotenderManager(Node):
    def __init__(self):
        super().__init__('robotender_manager', namespace='/dsr01')
        self.callback_group = ReentrantCallbackGroup()
        
        # 1. Service Clients (Relative paths now that we are in /dsr01)
        self.movej_cli = self.create_client(
            MoveJoint, 
            'motion/move_joint',
            callback_group=self.callback_group
        )
        self.gripper_cli = self.create_client(
            GripperControl, 
            'robotender_gripper/move',
            callback_group=self.callback_group
        )
        
        # 2. Action client for Pick node
        self.pick_action_client = ActionClient(
            self,
            PickBottle,
            'robotender_pick/execute',
            callback_group=self.callback_group
        )
        
        self.order_topic = "/bartender/order_detail"
        self.order_sub = None 
        self.is_busy = False # Simple state lock
        
        self.get_logger().info('Robotender Manager initialized.')
        
        # Trigger the initial standby sequence via timer
        self.init_timer = self.create_timer(
            0.5, 
            self.initial_standby_trigger, 
            callback_group=self.callback_group
        )

    async def initial_standby_trigger(self):
        """First time standby setup"""
        self.init_timer.cancel()
        await self.run_standby_sequence()

    async def run_standby_sequence(self):
        """
        Physically ready sign:
        1. Move to POSJ_HOME
        2. Close Gripper (1100, 700)
        3. Enable order listening (if first time)
        """
        self.get_logger().info("--- EXECUTING STANDBY SEQUENCE ---")
        
        # Wait for services
        while rclpy.ok() and not self.movej_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for move_joint service...')
        while rclpy.ok() and not self.gripper_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for gripper service...')

        # 1. Move to POSJ_HOME
        self.get_logger().info("Standby Step 1: Moving to POSJ_HOME...")
        req_move = MoveJoint.Request()
        req_move.pos = [float(x) for x in POSJ_HOME]
        req_move.vel = 30.0
        req_move.acc = 30.0
        await self.movej_cli.call_async(req_move)
        
        # 2. Close Gripper (Standby mode: 1100, 700)
        self.get_logger().info("Standby Step 2: Closing gripper (1100, 700)...")
        req_grip = GripperControl.Request()
        req_grip.position = 1100
        req_grip.force = 700
        await self.gripper_cli.call_async(req_grip)
            
        # 3. Create order subscription only ONCE
        if self.order_sub is None:
            self.get_logger().info("Standby Step 3: Enabling order subscription...")
            self.order_sub = self.create_subscription(
                String,
                self.order_topic,
                self.order_callback,
                10,
                callback_group=self.callback_group
            )
        
        self.is_busy = False
        self.get_logger().info("--- SYSTEM READY ---")

    async def order_callback(self, msg: String):
        if self.is_busy:
            self.get_logger().warn("System is busy, ignoring order.")
            return

        try:
            order_data = json.loads(msg.data)
            self.get_logger().info(f'RECEIVED ORDER: {order_data}')
            
            if "recipe" in order_data:
                self.is_busy = True
                recipe = order_data["recipe"]
                bottle_name = list(recipe.keys())[0]
                
                # Start Pick Action
                self.get_logger().info(f"Initiating PICK action for: {bottle_name}")
                pick_success = await self.send_pick_goal(bottle_name)
                
                if pick_success:
                    self.get_logger().info("Pick completed. (Future steps: Pour -> Place)")
                    
                    # After successful pick, just move to POSJ_HOME as requested
                    self.get_logger().info("Moving to POSJ_HOME...")
                    req_move = MoveJoint.Request()
                    req_move.pos = [float(x) for x in POSJ_HOME]
                    req_move.vel = 30.0
                    req_move.acc = 30.0
                    await self.movej_cli.call_async(req_move)
                    
                    self.is_busy = False
                    self.get_logger().info("Cycle complete. System ready.")
                else:
                    self.get_logger().error("Pick failed. Resetting to standby...")
                    await self.run_standby_sequence()
                
        except Exception as e:
            self.get_logger().error(f'Error in order_callback: {e}')
            self.is_busy = False

    async def send_pick_goal(self, bottle_name):
        if not self.pick_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Pick action server not available!")
            return False

        goal_msg = PickBottle.Goal()
        goal_msg.bottle_name = bottle_name

        self.get_logger().info('Sending Pick goal...')
        goal_handle = await self.pick_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.pick_feedback_callback
        )

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.get_logger().info('Goal accepted. Waiting for result...')
        result_response = await goal_handle.get_result_async()
        result = result_response.result

        if result.success:
            self.get_logger().info(f"--- PICK SUCCESS! Pose: {result.pick_pose} ---")
            return True
        else:
            self.get_logger().error(f"--- PICK FAILED: {result.message} ---")
            return False

    def pick_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'[PICK FEEDBACK] {feedback.current_state} ({feedback.progress*100:.0f}%)')

def main(args=None):
    rclpy.init(args=args)
    node = RobotenderManager()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
