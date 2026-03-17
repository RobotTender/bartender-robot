import rclpy
import time
import threading
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger
from std_msgs.msg import Float64MultiArray
from .defines import PICK_PLACE_READY, HOME_POSE

class PlaceNode(Node):
    def __init__(self):
        super().__init__('robotender_place', namespace='/dsr01')
        self.callback_group = ReentrantCallbackGroup()
        
        # State
        self.last_pick_pose = None # [x, y, z]

        # Services
        self.srv = self.create_service(Trigger, 'robotender_place/start', self.place_callback, callback_group=self.callback_group)
        
        # Subscriptions
        self.pose_sub = self.create_subscription(Float64MultiArray, 'robotender_pick/last_pose', self.pose_cb, 10, callback_group=self.callback_group)

        # Clients
        self.gripper_open_cli = self.create_client(Trigger, 'robotender_gripper/open', callback_group=self.callback_group)

        self.get_logger().info('--- Robotender Place Node Initialized ---')

    def pose_cb(self, msg):
        if len(msg.data) >= 3:
            self.last_pick_pose = list(msg.data[:3])
            self.get_logger().info(f"Updated last pick pose: {self.last_pick_pose}")

    def _gripper_open(self):
        if not self.gripper_open_cli.wait_for_service(timeout_sec=5.0):
            return False
        future = self.gripper_open_cli.call_async(Trigger.Request())

        # Add a timeout to wait for the service response
        timeout = 10.0
        start_time = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start_time > timeout:
                self.get_logger().warn("Gripper open service call timed out! Proceeding anyway...")
                return True
            time.sleep(0.01)
        
        if future.done():
            return future.result().success
        return False

    def place_callback(self, request, response):
        self.get_logger().info('!!! Received PLACE signal !!!')
        
        if self.last_pick_pose is None:
            self.get_logger().error("No pick pose recorded! Cannot place.")
            response.success = False
            response.message = "No pick pose recorded"
            return response

        # Run motion in a separate thread to avoid blocking the executor
        threading.Thread(target=self._do_place_motion, daemon=True).start()
        
        response.success = True
        response.message = "Place sequence started in background"
        return response

    def _do_place_motion(self):
        # DSR functional API requires DR_init.__dsr__node to be set before import
        from DSR_ROBOT2 import (movej, movel, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, get_current_posx)
        
        try:
            self.get_logger().info("Place Motion Thread: Starting (Waiting 1s for previous node clear)...")
            time.sleep(1.0) # Ensure previous node's move command is fully processed
            
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)

            # 1. Move to PICK_PLACE_READY
            self.get_logger().info("Step 1: Moving to PICK_PLACE_READY")
            movej(PICK_PLACE_READY, vel=60, acc=60)

            # 2. Move to placement point (derived from last_pick_pose)
            x, y, z = self.last_pick_pose
            current_posx = list(get_current_posx()[0])
            # Following the logic from test_pick_place.py
            target_1 = [x - 20, y - 50, z, current_posx[3], current_posx[4], current_posx[5]]
            target_2 = [x - 20, y + 50, z - 20, current_posx[3], current_posx[4], current_posx[5]]
            
            self.get_logger().info(f"Step 2: Moving to placement point: {target_2}")
            movel(target_2, vel=[40, 40], acc=[40, 40])
            wait(0.5)

            # 3. Open Gripper
            self.get_logger().info("Step 3: Opening gripper to release object")
            success = self._gripper_open()
            if not success:
                self.get_logger().error("Failed to open gripper!")
            
            self.get_logger().info("Waiting 2s for gripper to release...")
            time.sleep(2.0) # Ensure gripper is fully released

            # 4. Retreat to approach point
            self.get_logger().info("Step 4: Retreating to approach point")
            try:
                movel(target_1, vel=[40, 40], acc=[40, 40])
            except Exception as e:
                self.get_logger().warn(f"Retreat movel failed, trying movej to READY: {e}")
                movej(PICK_PLACE_READY, vel=40, acc=40)

            # 5. Return to PICK_PLACE_READY
            self.get_logger().info("Step 5: Returning to PICK_PLACE_READY pose")
            movej(PICK_PLACE_READY, vel=60, acc=60)

            # 6. Final Return to HOME_POSE
            self.get_logger().info("Step 6: Returning to HOME_POSE (End of Place Motion)")
            movej(HOME_POSE, vel=60, acc=60)
            
            self.get_logger().info("Place Motion Thread: Completed successfully")
        except Exception as e:
            self.get_logger().error(f"Place Motion Thread Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PlaceNode()
    # Use empty ROBOT_ID as in pour.py for relative namespace resolution
    ROBOT_ID, ROBOT_MODEL = "", "e0509"
    import DR_init
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    DR_init.__dsr__node = node
    
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
