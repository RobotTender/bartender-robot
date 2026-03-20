import rclpy
import json
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from std_srvs.srv import Trigger

# Message/Service/Action Imports
from robotender_msgs.action import PickBottle, PlaceBottle, PourBottle
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

        # 2. Action clients
        self.pick_action_client = ActionClient(
            self,
            PickBottle,
            'robotender_pick/execute',
            callback_group=self.callback_group
        )
        self.place_action_client = ActionClient(
            self,
            PlaceBottle,
            'robotender_place/execute',
            callback_group=self.callback_group
        )
        self.pour_action_client = ActionClient(
            self,
            PourBottle,
            'robotender_pour/execute',
            callback_group=self.callback_group
        )

        # 3. Manual Control Services
        self.pick_trigger_srv = self.create_service(
            Trigger,
            'robotender_manager/pick_bottle',
            self.manual_pick_callback,
            callback_group=self.callback_group
        )
        self.pour_trigger_srv = self.create_service(
            Trigger,
            'robotender_manager/pour_bottle',
            self.manual_pour_callback,
            callback_group=self.callback_group
        )
        self.place_trigger_srv = self.create_service(
            Trigger,
            'robotender_manager/place_bottle',
            self.manual_place_callback,
            callback_group=self.callback_group
        )
        
        self.order_topic = "/bartender/order_detail"
        self.order_sub = None 
        self.is_busy = False # Simple state lock
        self.last_ordered_bottle = None # Store bottle name from order
        self.last_picked_pose = None # Remember where the bottle came from
        
        self.get_logger().info('Robotender Manager initialized.')
        self.get_logger().info('Manual Services: pick_bottle, pour_bottle, place_bottle')
        
        # Trigger the initial standby sequence via timer
        self.init_timer = self.create_timer(
            0.5, 
            self.initial_standby_trigger, 
            callback_group=self.callback_group
        )

    async def manual_pick_callback(self, request, response):
        """Manually trigger the pick node using stored bottle name"""
        if self.is_busy:
            response.success = False
            response.message = "System is busy"
            return response

        if self.last_ordered_bottle is None:
            response.success = False
            response.message = "No bottle ordered! Send an order topic first."
            self.get_logger().error("Manual Pick requested but last_ordered_bottle is None.")
            return response

        self.get_logger().info(f"Manual Pick Triggered for: {self.last_ordered_bottle}")
        
        self.is_busy = True
        success = await self.send_pick_goal(self.last_ordered_bottle)
        
        if success:
            # Move to POSJ_HOME after successful pick
            self.get_logger().info("Moving to POSJ_HOME...")
            req_move = MoveJoint.Request()
            req_move.pos = [float(x) for x in POSJ_HOME]
            req_move.vel = 30.0
            req_move.acc = 30.0
            await self.movej_cli.call_async(req_move)
            response.message = "Pick action completed"
        else:
            self.get_logger().error("Pick failed. Resetting to standby...")
            await self.run_standby_sequence()
            response.message = "Pick action failed"

        self.is_busy = False
        response.success = success
        return response

    async def manual_pour_callback(self, request, response):
        """Manually trigger the pour node using stored bottle name"""
        if self.is_busy:
            response.success = False
            response.message = "System is busy"
            return response

        if self.last_ordered_bottle is None:
            self.get_logger().warn("No bottle ordered! Performing manual pour with default 'soju'.")
            bottle_to_pour = 'soju'
        else:
            bottle_to_pour = self.last_ordered_bottle

        self.get_logger().info(f"Manual Pour Triggered for: {bottle_to_pour}")
        
        self.is_busy = True
        success = await self.send_pour_goal(bottle_to_pour)
        
        if success:
            self.get_logger().info("Pour completed successfully. Moving to POSJ_HOME...")
            req_move = MoveJoint.Request()
            req_move.pos = [float(x) for x in POSJ_HOME]
            req_move.vel = 30.0
            req_move.acc = 30.0
            await self.movej_cli.call_async(req_move)
            response.message = "Pour action completed"
        else:
            self.get_logger().error("Pour failed.")
            response.message = "Pour action failed"

        self.is_busy = False
        response.success = success
        return response

    async def send_pour_goal(self, bottle_name):
        if not self.pour_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Pour action server not available!")
            return False

        goal_msg = PourBottle.Goal()
        goal_msg.bottle_name = bottle_name

        self.get_logger().info('Sending Pour goal...')
        goal_handle = await self.pour_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.pour_feedback_callback
        )

        if not goal_handle.accepted:
            self.get_logger().error('Pour Goal rejected')
            return False

        self.get_logger().info('Pour Goal accepted. Waiting for result...')
        result_response = await goal_handle.get_result_async()
        result = result_response.result

        if result.success:
            self.get_logger().info("--- POUR SUCCESS! ---")
            return True
        else:
            self.get_logger().error(f"--- POUR FAILED: {result.message} ---")
            return False

    def pour_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'[POUR FEEDBACK] {feedback.current_state} ({feedback.progress*100:.0f}%)')

    async def manual_place_callback(self, request, response):
        """Manually trigger the place node using stored coordinates"""
        if self.is_busy:
            response.success = False
            response.message = "System is busy"
            return response

        if self.last_picked_pose is None:
            response.success = False
            response.message = "No pick pose stored! Perform a pick first."
            self.get_logger().error("Manual Place requested but last_picked_pose is None.")
            return response

        self.get_logger().info(f"Manual Place Triggered. Sending pose to PlaceNode: {self.last_picked_pose}")
        
        self.is_busy = True
        success = await self.send_place_goal(self.last_picked_pose)
        
        if success:
            self.get_logger().info("Place completed successfully. Returning to standby...")
        else:
            self.get_logger().error("Place failed. Returning to standby...")
        
        await self.run_standby_sequence()
        
        response.success = success
        response.message = "Place action completed" if success else "Place action failed"
        return response

    async def send_place_goal(self, picked_pose):
        if not self.place_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Place action server not available!")
            return False

        goal_msg = PlaceBottle.Goal()
        goal_msg.picked_pose = [float(x) for x in picked_pose]

        self.get_logger().info('Sending Place goal...')
        goal_handle = await self.place_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.place_feedback_callback
        )

        if not goal_handle.accepted:
            self.get_logger().error('Place Goal rejected')
            return False

        self.get_logger().info('Place Goal accepted. Waiting for result...')
        result_response = await goal_handle.get_result_async()
        result = result_response.result

        if result.success:
            self.get_logger().info("--- PLACE SUCCESS! ---")
            return True
        else:
            self.get_logger().error(f"--- PLACE FAILED: {result.message} ---")
            return False

    def place_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'[PLACE FEEDBACK] {feedback.current_state} ({feedback.progress*100:.0f}%)')

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
        
        # 2. Close Gripper (Standby mode: 1100, 1000)
        self.get_logger().info("Standby Step 2: Closing gripper (1100, 1000)...")
        req_grip = GripperControl.Request()
        req_grip.position = 1100
        req_grip.force = 1000
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
        """Extract and store bottle name from order topic"""
        if self.is_busy:
            self.get_logger().warn("System is busy, ignoring order.")
            return

        try:
            order_data = json.loads(msg.data)
            self.get_logger().info(f'RECEIVED ORDER: {order_data}')
            
            if "recipe" in order_data:
                recipe = order_data["recipe"]
                self.last_ordered_bottle = list(recipe.keys())[0]
                self.get_logger().info(f"Order stored: {self.last_ordered_bottle}. Trigger Pick via Service.")
                
        except Exception as e:
            self.get_logger().error(f'Error in order_callback: {e}')

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
            self.last_picked_pose = list(result.pick_pose)
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
