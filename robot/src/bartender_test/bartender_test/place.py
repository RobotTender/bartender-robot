import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import time

from std_srvs.srv import Trigger
from std_msgs.msg import Float64MultiArray
from .defines import (
    POSJ_PICK_PLACE_READY, POSJ_HOME,
    PICK_PLACE_Z, PICK_PLACE_X_OFFSET, PICK_PLACE_Y_OFFSET,
    GRIPPER_POSITION_OPEN, GRIPPER_FORCE_OPEN
)
from robotender_msgs.srv import GripperControl

# Import Doosan Robot functions
import DR_init
from DSR_ROBOT2 import (
    movej, movel, set_robot_mode, wait, get_current_posx,
    ROBOT_MODE_AUTONOMOUS
)

class PlaceNode(Node):
    def __init__(self):
        super().__init__('robotender_place', namespace='/dsr01')
        self.callback_group = ReentrantCallbackGroup()
        
        # Latest known pick pose (where the bottle came from)
        self.target_xyz = [0.0, 0.0, 0.0]
        
        # Subscriptions
        self.create_subscription(
            Float64MultiArray, 
            'robotender_pick/last_pose', 
            self.last_pose_cb, 
            10,
            callback_group=self.callback_group
        )
        
        # Services
        self.place_srv = self.create_service(
            Trigger, 
            'robotender_place/start', 
            self.place_callback,
            callback_group=self.callback_group
        )
        
        # Service Clients
        self.gripper_cli = self.create_client(
            GripperControl, 
            'robotender_gripper/move',
            callback_group=self.callback_group
        )

        self.get_logger().info('--- Robotender Place Node Initialized ---')
        self.get_logger().info('Service: /dsr01/robotender_place/start')

    def last_pose_cb(self, msg):
        if len(msg.data) >= 3:
            self.target_xyz = list(msg.data[:3])
            # self.get_logger().info(f"Updated place target: {self.target_xyz}")

    async def place_callback(self, request, response):
        self.get_logger().info('--- Starting Placement Sequence ---')
        
        if self.target_xyz == [0.0, 0.0, 0.0]:
            self.get_logger().error("No valid pick pose received yet!")
            response.success = False
            response.message = "No target coordinates"
            return response

        try:
            # Start placement in a separate thread to allow async logic
            threading.Thread(target=self.execute_place_motion).start()
            response.success = True
            response.message = "Placement motion started"
        except Exception as e:
            self.get_logger().error(f"Place Error: {e}")
            response.success = False
            response.message = str(e)
        return response

    def execute_place_motion(self):
        try:
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)

            # 1. Move to POSJ_PICK_PLACE_READY
            self.get_logger().info("Step 1: Moving to POSJ_PICK_PLACE_READY")
            movej(POSJ_PICK_PLACE_READY, vel=60, acc=60)

            # 2. Get current pose for orientation reference
            curr_posx = list(get_current_posx()[0])
            
            # 3. Calculate target poses
            # Apply same offsets used during picking
            tx = self.target_xyz[0] + PICK_PLACE_X_OFFSET
            ty = self.target_xyz[1] + PICK_PLACE_Y_OFFSET
            tz = self.target_xyz[2] # Actual bottle depth
            
            # Step 4: X-Alignment (Maintain height)
            target_x = [tx, curr_posx[1], curr_posx[2], curr_posx[3], curr_posx[4], curr_posx[5]]
            self.get_logger().info(f"Step 4: X-Alignment to {tx:.1f}")
            movel(target_x, vel=[60, 60], acc=[60, 60])

            # Step 5: Y-Entry (Approach bottle spot)
            target_y = [tx, ty, curr_posx[2], curr_posx[3], curr_posx[4], curr_posx[5]]
            self.get_logger().info(f"Step 5: Y-Entry to {ty:.1f}")
            movel(target_y, vel=[60, 60], acc=[60, 60])

            # Step 6: Z-Lower (Place bottle down)
            # Lift was 3cm, so we lower back to original pick height
            target_z = [tx, ty, tz, curr_posx[3], curr_posx[4], curr_posx[5]]
            self.get_logger().info(f"Step 6: Lowering to {tz:.1f}")
            movel(target_z, vel=[30, 30], acc=[30, 30])

            # 7. Release Bottle
            self.get_logger().info("Step 7: Releasing gripper")
            self._gripper_move_sync(GRIPPER_POSITION_OPEN, GRIPPER_FORCE_OPEN)
            wait(2.0)

            # Step 7.5: Retreat Y
            target_retreat = [tx, curr_posx[1], tz, curr_posx[3], curr_posx[4], curr_posx[5]]
            self.get_logger().info("Step 7.5: Retreating Y")
            movel(target_retreat, vel=[60, 60], acc=[60, 60])

            # 8. Return to POSJ_PICK_PLACE_READY
            self.get_logger().info("Step 8: Returning to POSJ_PICK_PLACE_READY")
            movej(POSJ_PICK_PLACE_READY, vel=60, acc=60)

            # 9. Return to POSJ_HOME
            self.get_logger().info("Step 9: Returning to POSJ_HOME (End of Place Motion)")
            movej(POSJ_HOME, vel=60, acc=60)

            self.get_logger().info("--- Placement Sequence Completed ---")

        except Exception as e:
            self.get_logger().error(f"Motion execution failed: {e}")

    def _gripper_move_sync(self, pos, force):
        if not self.gripper_cli.wait_for_service(timeout_sec=2.0):
            return False
        req = GripperControl.Request()
        req.position = int(pos)
        req.force = int(force)
        future = self.gripper_cli.call_async(req)
        # We don't need a while loop here if we just want to fire and forget, 
        # but for safety in motion we wait.
        while rclpy.ok() and not future.done():
            time.sleep(0.1)
        return True

def main(args=None):
    rclpy.init(args=args)
    node = PlaceNode()
    
    # Initialize DR_init
    import DR_init
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = "dsr01", "e0509", node
    
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
