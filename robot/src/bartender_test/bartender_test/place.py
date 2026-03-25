import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import asyncio

from robotender_msgs.srv import GripperControl
from robotender_msgs.action import PlaceBottle
from .defines import (
    POSJ_PICK_PLACE_READY, POSJ_HOME,
    PICK_PLACE_X_OFFSET, PICK_PLACE_Y_OFFSET,
    GRIPPER_POSITION_OPEN,
    GRIPPER_FORCE_DEFAULT,
    BOTTLE_CONFIG
)

# Import DR_init but do NOT import DSR_ROBOT2 at top level
import DR_init

VELOCITY, ACC = 30.0, 30.0

class PlaceNode(Node):
    def __init__(self):
        super().__init__('robotender_place', namespace='/dsr01')
        # ARCHITECTURAL FIX: Use Reentrant group for all node-level callbacks
        self._default_callback_group = ReentrantCallbackGroup()
        self.callback_group = ReentrantCallbackGroup()
        
        # Latest known pick pose (where the bottle came from)
        self.target_xyz = [0.0, 0.0, 0.0]
        self.state = 'IDLE'
        
        # Action Server
        self._action_server = ActionServer(
            self,
            PlaceBottle,
            'robotender_place/execute',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )
        
        # Service Clients
        self.gripper_cli = self.create_client(
            GripperControl, 
            'robotender_gripper/move',
            callback_group=self.callback_group
        )

        self.get_logger().info('--- Robotender Place Action Node Initialized ---')
        self.get_logger().info('Action: /dsr01/robotender_place/execute')

    def goal_callback(self, goal_request):
        if self.state == "RUNNING":
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('--- [PLACE] ENTERING EXECUTE_CALLBACK ---')
        feedback_msg = PlaceBottle.Feedback()
        result = PlaceBottle.Result()
        
        self.target_xyz = list(goal_handle.request.picked_pose)
        bottle_name = goal_handle.request.bottle_name
        self.get_logger().info(f'Place target for {bottle_name} from Manager: {self.target_xyz}')
        
        self.state = "RUNNING"

        # Lookup specific gripper force if defined for this bottle
        release_force = GRIPPER_FORCE_DEFAULT
        config = BOTTLE_CONFIG.get(bottle_name)
        if config and 'gripper_force' in config:
            release_force = config['gripper_force']
            self.get_logger().info(f"Using bottle-specific release force: {release_force}")
        else:
            self.get_logger().info(f"Using default release force: {release_force}")

        # LOCAL IMPORT: Uses the node assigned to DR_init
        from DSR_ROBOT2 import (
            movej, movel, set_robot_mode, wait, get_current_posx,
            fkin, ROBOT_MODE_AUTONOMOUS
        )
        
        try:
            # Step 0: Autonomous Mode
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)

            # STEP 1: Starts with movej(POSJ_PICK_PLACE_READY)
            self.get_logger().info("Step 1: Moving to POSJ_PICK_PLACE_READY")
            feedback_msg.current_state, feedback_msg.progress = "STEP 1: MOVING_TO_READY", 0.1
            goal_handle.publish_feedback(feedback_msg)
            movej(POSJ_PICK_PLACE_READY, vel=VELOCITY, acc=ACC)

            # STEP 2: Approach X alignment first
            self.get_logger().info("Step 2: X-Alignment")
            feedback_msg.current_state, feedback_msg.progress = "STEP 2: X_ALIGNMENT", 0.2
            goal_handle.publish_feedback(feedback_msg)
            
            curr_posx = list(get_current_posx()[0])
            tx = self.target_xyz[0] + PICK_PLACE_X_OFFSET
            target_x = [tx, curr_posx[1], curr_posx[2], curr_posx[3], curr_posx[4], curr_posx[5]]
            movel(target_x, vel=[VELOCITY, VELOCITY], acc=[ACC, ACC])

            # STEP 3: Before approach Y, lift up Z 3cm more first
            self.get_logger().info("Step 3: Lifting Z +30mm")
            feedback_msg.current_state, feedback_msg.progress = "STEP 3: LIFTING_Z", 0.4
            goal_handle.publish_feedback(feedback_msg)
            
            curr = list(get_current_posx()[0])
            target_lift = [curr[0], curr[1], curr[2] + 30.0, curr[3], curr[4], curr[5]]
            movel(target_lift, vel=[VELOCITY, VELOCITY], acc=[ACC, ACC])

            # STEP 4: Then approach Y
            self.get_logger().info("Step 4: Y-Entry")
            feedback_msg.current_state, feedback_msg.progress = "STEP 4: Y_ENTRY", 0.6
            goal_handle.publish_feedback(feedback_msg)
            
            ty = self.target_xyz[1] + PICK_PLACE_Y_OFFSET
            curr_lifted = list(get_current_posx()[0])
            target_y = [curr_lifted[0], ty, curr_lifted[2], curr_lifted[3], curr_lifted[4], curr_lifted[5]]
            movel(target_y, vel=[VELOCITY, VELOCITY], acc=[ACC, ACC])

            # STEP 5: When approach Y is done, lift down 2.75cm
            self.get_logger().info("Step 5: Lowering Z -27.5mm")
            feedback_msg.current_state, feedback_msg.progress = "STEP 5: LOWERING_BOTTLE", 0.7
            goal_handle.publish_feedback(feedback_msg)
            
            curr_at_y = list(get_current_posx()[0])
            target_down = [curr_at_y[0], curr_at_y[1], curr_at_y[2] - 27.5, curr_at_y[3], curr_at_y[4], curr_at_y[5]]
            movel(target_down, vel=[30, 30], acc=[30, 30])

            # STEP 6: Release the gripper
            self.get_logger().info(f"Step 6: Releasing gripper with force {release_force} (Wait 8s)")
            feedback_msg.current_state, feedback_msg.progress = "STEP 6: RELEASING_GRIPPER", 0.8
            goal_handle.publish_feedback(feedback_msg)
            
            self._gripper_move_fire_forget(GRIPPER_POSITION_OPEN, release_force)
            time.sleep(8.0) # Using time.sleep here as it's a long blocking wait

            # STEP 7: Retreat Y
            self.get_logger().info("Step 7: Retreating Y")
            feedback_msg.current_state, feedback_msg.progress = "STEP 7: RETREATING_Y", 0.9
            goal_handle.publish_feedback(feedback_msg)
            
            ready_posx = list(fkin(POSJ_PICK_PLACE_READY, ref=0))
            ready_y = ready_posx[1]
            curr_placed = list(get_current_posx()[0])
            target_retreat = [curr_placed[0], ready_y, curr_placed[2], curr_placed[3], curr_placed[4], curr_placed[5]]
            movel(target_retreat, vel=[VELOCITY, VELOCITY], acc=[ACC, ACC])

            # STEP 8: Move back to pick_place_ready pose
            self.get_logger().info("Step 8: Returning to POSJ_PICK_PLACE_READY")
            feedback_msg.current_state, feedback_msg.progress = "STEP 8: RETURNING_TO_READY", 1.0
            goal_handle.publish_feedback(feedback_msg)
            movej(POSJ_PICK_PLACE_READY, vel=VELOCITY, acc=ACC)
            
            self.get_logger().info("--- [PLACE] SUCCESS. Goal Succeeding. ---")
            result.success, result.message = True, "Place success"
            goal_handle.succeed()
            
        except Exception as e:
            self.get_logger().error(f"--- [PLACE] EXECUTION ERROR: {e} ---")
            result.success, result.message = False, str(e)
            goal_handle.succeed()
        finally:
            self.state = "IDLE"
        
        return result

    def _gripper_move_fire_forget(self, pos, force):
        if not self.gripper_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Gripper service not available!")
            return False
        req = GripperControl.Request(position=int(pos), force=int(force))
        try:
            self.gripper_cli.call_async(req)
            return True
        except Exception as e:
            self.get_logger().error(f"Gripper call failed: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    
    # 1. Main Action Node
    node = PlaceNode()
    
    # 2. ARCHITECTURAL FIX: ISOLATED DOOSAN NODE
    doosan_node = rclpy.create_node('place_doosan_internal', namespace='/dsr01')
    doosan_node._default_callback_group = ReentrantCallbackGroup()
    
    # Initialize DR_init with the ISOLATED node
    import DR_init
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = "dsr01", "e0509", doosan_node
    
    # 3. Use MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=10)
    executor.add_node(node)
    executor.add_node(doosan_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        doosan_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
