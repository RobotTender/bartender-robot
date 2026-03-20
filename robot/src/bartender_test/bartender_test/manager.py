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
    GRIPPER_FORCE_OPEN,
    BOTTLE_CONFIG
)

class RobotenderManager(Node):
    def __init__(self):
        super().__init__('robotender_manager', namespace='/dsr01')
        self.callback_group = ReentrantCallbackGroup()
        
        # 1. Service Clients
        self.movej_cli = self.create_client(MoveJoint, 'motion/move_joint', callback_group=self.callback_group)
        self.gripper_cli = self.create_client(GripperControl, 'robotender_gripper/move', callback_group=self.callback_group)

        # 2. Action clients
        self.pick_action_client = ActionClient(self, PickBottle, 'robotender_pick/execute', callback_group=self.callback_group)
        self.place_action_client = ActionClient(self, PlaceBottle, 'robotender_place/execute', callback_group=self.callback_group)
        self.pour_action_client = ActionClient(self, PourBottle, 'robotender_pour/execute', callback_group=self.callback_group)

        # 3. Manual Control Services
        self.pick_trigger_srv = self.create_service(Trigger, 'robotender_manager/pick_bottle', self.manual_pick_callback, callback_group=self.callback_group)
        self.pour_trigger_srv = self.create_service(Trigger, 'robotender_manager/pour_bottle', self.manual_pour_callback, callback_group=self.callback_group)
        self.place_trigger_srv = self.create_service(Trigger, 'robotender_manager/place_bottle', self.manual_place_callback, callback_group=self.callback_group)
        
        self.order_topic = "/bartender/order_detail"
        self.order_sub = None 
        self.is_busy = False 
        self.last_ordered_bottle = None 
        self.last_picked_pose = None 
        
        self.get_logger().info('====================================================')
        self.get_logger().info('   Robotender Manager Node Initialized')
        self.get_logger().info('====================================================')
        
        # Trigger initial standby via a safe one-shot timer
        self.init_timer = self.create_timer(1.0, self.initial_standby_trigger, callback_group=self.callback_group)

    async def manual_pick_callback(self, request, response):
        if self.is_busy:
            response.success, response.message = False, "System is busy"
            return response
        if self.last_ordered_bottle is None:
            response.success, response.message = False, "No bottle ordered!"
            return response

        self.get_logger().info(f"[MANUAL] Pick Triggered for: {self.last_ordered_bottle}")
        self.is_busy = True
        success = await self.send_pick_goal(self.last_ordered_bottle)
        
        if success:
            req_move = MoveJoint.Request()
            req_move.pos, req_move.vel, req_move.acc = [float(x) for x in POSJ_HOME], 30.0, 30.0
            await self.movej_cli.call_async(req_move)
            response.success, response.message = True, "Pick completed."
        else:
            await self.run_standby_sequence()
            response.success, response.message = False, "Pick failed."

        self.is_busy = False
        return response

    async def manual_pour_callback(self, request, response):
        if self.is_busy:
            response.success, response.message = False, "System is busy"
            return response
        if self.last_ordered_bottle is None:
            response.success, response.message = False, "No bottle ordered!"
            return response

        self.get_logger().info(f"[MANUAL] Pour Triggered for: {self.last_ordered_bottle}")
        self.is_busy = True
        success = await self.send_pour_goal(self.last_ordered_bottle)
        
        if success:
            req_move = MoveJoint.Request()
            req_move.pos, req_move.vel, req_move.acc = [float(x) for x in POSJ_HOME], 30.0, 30.0
            await self.movej_cli.call_async(req_move)
            response.success, response.message = True, "Pour completed."
        else:
            response.success, response.message = False, "Pour failed."

        self.is_busy = False
        return response

    async def send_pour_goal(self, bottle_name):
        if not self.pour_action_client.wait_for_server(timeout_sec=5.0):
            return False

        config = BOTTLE_CONFIG.get(bottle_name)
        if config is None: return False
            
        target_volume = float(config.get('pour_target_ml', 100.0))
        goal_msg = PourBottle.Goal()
        goal_msg.bottle_name = bottle_name
        goal_msg.target_volume_ml = target_volume

        goal_handle = await self.pour_action_client.send_goal_async(goal_msg, feedback_callback=self.pour_feedback_callback)
        if not goal_handle.accepted: return False

        result_response = await goal_handle.get_result_async()
        return result_response.result.success

    def pour_feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f'[POUR FB] {fb.current_state} ({fb.progress*100:.0f}%)')

    async def manual_place_callback(self, request, response):
        if self.is_busy:
            response.success, response.message = False, "System is busy"
            return response
        if self.last_picked_pose is None:
            response.success, response.message = False, "No pick pose stored!"
            return response

        self.is_busy = True
        success = await self.send_place_goal(self.last_picked_pose)
        await self.run_standby_sequence()
        response.success, response.message = success, "Place completed."
        return response

    async def send_place_goal(self, picked_pose):
        if not self.place_action_client.wait_for_server(timeout_sec=5.0): return False
        goal_msg = PlaceBottle.Goal()
        goal_msg.picked_pose = [float(x) for x in picked_pose]
        goal_handle = await self.place_action_client.send_goal_async(goal_msg, feedback_callback=self.place_feedback_callback)
        if not goal_handle.accepted: return False
        result_response = await goal_handle.get_result_async()
        return result_response.result.success

    def place_feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f'[PLACE FB] {fb.current_state}')

    async def initial_standby_trigger(self):
        self.init_timer.cancel()
        self.get_logger().info("--- SYSTEM STARTUP ---")
        await self.run_standby_sequence()

    async def run_standby_sequence(self):
        # 1. POSJ_HOME
        self.get_logger().info("[STANDBY] 1/3: Moving to POSJ_HOME...")
        if self.movej_cli.wait_for_service(timeout_sec=2.0):
            req_move = MoveJoint.Request()
            req_move.pos, req_move.vel, req_move.acc = [float(x) for x in POSJ_HOME], 30.0, 30.0
            await self.movej_cli.call_async(req_move)
        else:
            self.get_logger().error("MoveJoint service not available!")

        # 2. Close Gripper
        self.get_logger().info("[STANDBY] 2/3: Closing gripper (Force Mode)...")
        if self.gripper_cli.wait_for_service(timeout_sec=2.0):
            req_grip = GripperControl.Request()
            req_grip.position, req_grip.force = 1100, 1000
            # Safety: Don't await indefinitely if physical gripper is stuck
            try:
                self.gripper_cli.call_async(req_grip)
                self.get_logger().info("   (Gripper command sent)")
            except Exception as e:
                self.get_logger().error(f"Gripper service error: {e}")
        else:
            self.get_logger().error("Gripper service not available!")
            
        # 3. Order Topic
        if self.order_sub is None:
            self.get_logger().info("[STANDBY] 3/3: Opening order subscription...")
            self.order_sub = self.create_subscription(String, self.order_topic, self.order_callback, 10, callback_group=self.callback_group)
        
        self.is_busy = False
        self.get_logger().info("--- SYSTEM READY ---")

    async def order_callback(self, msg: String):
        if self.is_busy: return
        try:
            order_data = json.loads(msg.data)
            if "recipe" in order_data:
                self.last_ordered_bottle = list(order_data["recipe"].keys())[0]
                self.get_logger().info(f"ORDER RECEIVED: {self.last_ordered_bottle}")
        except Exception as e:
            self.get_logger().error(f'Order parsing error: {e}')

    async def send_pick_goal(self, bottle_name):
        if not self.pick_action_client.wait_for_server(timeout_sec=5.0): return False
        goal_msg = PickBottle.Goal()
        goal_msg.bottle_name = bottle_name
        goal_handle = await self.pick_action_client.send_goal_async(goal_msg, feedback_callback=self.pick_feedback_callback)
        if not goal_handle.accepted: return False
        result_response = await goal_handle.get_result_async()
        res = result_response.result
        if res.success: self.last_picked_pose = list(res.pick_pose)
        return res.success

    def pick_feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f'[PICK FB] {fb.current_state}')

def main(args=None):
    rclpy.init(args=args)
    node = RobotenderManager()
    from rclpy.executors import MultiThreadedExecutor
    # Use explicit thread count to prevent deadlock in async chains
    executor = MultiThreadedExecutor(num_threads=8)
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
