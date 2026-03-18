import rclpy
import sys
import threading
import time
import math
import json
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import DR_init

from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger
from dsr_msgs2.srv import MoveStop
from sensor_msgs.msg import JointState
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .defines import (HOME_POSE, CHEERS_POSE, CONTACT_POSE, POUR_HORIZONTAL, POUR_DIAGONAL, POUR_VERTICAL, POLE_POSE,
                            POS_CHEERS, BOTTLE_CONFIG, ORDER_TOPIC)

class ActionNode(Node):
    def __init__(self):
        super().__init__('robotender_pour', namespace='/dsr01')
        self.callback_group = ReentrantCallbackGroup()
        
        # State for Pouring
        self.trigger_received = False
        self.recording = False
        self.current_posx = [0.0] * 6
        self.passed_waypoints = []
        self.target_waypoints = []
        self.current_bottle_type = 'soju' # Default

        # Subscriptions
        self.create_subscription(Empty, 'robotender_snap/trigger', self.trigger_cb, 10, callback_group=self.callback_group)
        from dsr_msgs2.msg import RobotState
        self.create_subscription(RobotState, 'state', self.cb_robot_state, 10, callback_group=self.callback_group)
        self.create_subscription(String, ORDER_TOPIC, self.order_cb, 10, callback_group=self.callback_group)

        # Clients
        self.stop_cli = self.create_client(MoveStop, 'motion/move_stop', callback_group=self.callback_group)

        # Services
        self.pour_srv = self.create_service(Trigger, 'robotender_pour/start', self.pour_callback, callback_group=self.callback_group)
        self.warmup_srv = self.create_service(Trigger, 'robotender_pour/warmup', self.warmup_callback, callback_group=self.callback_group)
        
        # Service Clients
        self.place_cli = self.create_client(Trigger, 'robotender_place/start', callback_group=self.callback_group)

        # Timer for path recording (3Hz)
        self.record_timer = self.create_timer(0.33, self.record_loop, callback_group=self.callback_group)

        self.get_logger().info('--- Robotender Pour Node Initialized ---')
        self.get_logger().info('Services: /dsr01/robotender_pour/start, /dsr01/robotender_pour/warmup')

    def cb_robot_state(self, msg):
        self.current_posx = list(msg.current_posx)

    def order_cb(self, msg: String):
        try:
            items = json.loads(msg.data)
            if "recipe" in items:
                # Get the first item from the recipe
                self.current_bottle_type = [x for x in items["recipe"].keys()][0]
                self.get_logger().info(f"Target bottle type set to: {self.current_bottle_type}")
        except Exception as e:
            self.get_logger().error(f"Order parse error in Pour node: {e}")

    def trigger_cb(self, msg):
        if self.recording:
            self.get_logger().info("!!! POURING TRIGGER DETECTED !!!")
            self.trigger_received = True
            if self.stop_cli.wait_for_service(timeout_sec=0.5):
                req = MoveStop.Request()
                req.stop_mode = 2 # DR_SSTOP
                self.stop_cli.call_async(req)
            else:
                self.get_logger().error("MoveStop service not available!")

    def record_loop(self):
        if self.recording and self.target_waypoints:
            # Check if we've passed the next target waypoint (XYZ only)
            target = self.target_waypoints[0]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.current_posx[:3], target[:3])))
            
            # If within 20mm of the waypoint, consider it "passed"
            if dist < 20.0:
                wp = self.target_waypoints.pop(0)
                self.passed_waypoints.append(wp)
                self.get_logger().info(f"Passed waypoint! Waypoints passed: {len(self.passed_waypoints)}")

    async def warmup_callback(self, request, response):
        self.get_logger().info('Starting Warmup Sequence...')
        from DSR_ROBOT2 import (movej, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait)
        
        try:
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)
            
            poses = [("HOME", HOME_POSE), ("CHEERS", CHEERS_POSE), ("CONTACT", CONTACT_POSE), 
                     ("POUR_HORIZONTAL", POUR_HORIZONTAL), ("POUR_DIAGONAL", POUR_DIAGONAL), 
                     ("POUR_VERTICAL", POUR_VERTICAL), ("POLE", POLE_POSE)]
            
            for name, pose in poses:
                self.get_logger().info(f"Moving to {name}")
                movej(pose, vel=60, acc=60)
            
            response.success = True
            response.message = "Warmup completed"
        except Exception as e:
            self.get_logger().error(f"Warmup Error: {e}")
            response.success = False
            response.message = str(e)
        return response

    async def pour_callback(self, request, response):
        # Get bottle-specific configuration
        config = BOTTLE_CONFIG.get(self.current_bottle_type)
        if config is None:
            self.get_logger().warn(f"Unknown bottle type '{self.current_bottle_type}', defaulting to soju")
            config = BOTTLE_CONFIG['soju']
            active_type = 'soju (fallback)'
        else:
            active_type = self.current_bottle_type

        self.get_logger().info(f'--- Starting Pour Sequence for: {active_type} ---')
        self.get_logger().info(f'Using pos_contact: {config["pos_contact"]}')
        self.get_logger().info(f'Using pos_horizontal: {config["pos_horizontal"]}')

        from DSR_ROBOT2 import (movej, movel, movesx, movesj, posx, posj, fkin, set_robot_mode, 
                                ROBOT_MODE_AUTONOMOUS, wait, get_current_posj)
        
        try:
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)

            # Setup Trajectory Points
            p2 = posx(config['pos_contact'] + [float(x) for x in fkin(CONTACT_POSE, ref=0)][3:])
            p3 = posx(config['pos_horizontal'] + [float(x) for x in fkin(POUR_HORIZONTAL, ref=0)][3:])
            p4 = posx(config['pos_diagonal'] + [float(x) for x in fkin(POUR_DIAGONAL, ref=0)][3:])
            p5 = posx(config['pos_vertical'] + [float(x) for x in fkin(POUR_VERTICAL, ref=0)][3:])
            
            spline_path = [p3, p4, p5]

            # 1. Move to Cheers (Joint Space Start)
            self.get_logger().info("Moving to CHEERS_POSE")
            movej(CHEERS_POSE, vel=60, acc=60)
            
            # 2. Execute Pour Sequence with Waypoint Tracking
            self.trigger_received = False
            self.passed_waypoints = []
            # We track p2, p3, p4, p5. Note: p2 is the first target.
            self.target_waypoints = [config['pos_contact'], config['pos_horizontal'], config['pos_diagonal'], config['pos_vertical']]
            
            self.recording = True
            start_time = time.time()

            # Step A: Linear Move to Contact (Targeting config['pos_contact'])
            self.get_logger().info(f"Approaching contact for {active_type}...")
            movel(p2, vel=[100, 100], acc=[100, 100])
            
            # Step B: Spline Move through Pour Positions
            if not self.trigger_received:
                self.get_logger().info("Executing pouring spline...")
                movesx(spline_path, vel=100, acc=100)
            
            end_time = time.time()
            self.recording = False

            # 3. Handle Recovery or Finish
            if self.trigger_received:
                self.get_logger().info(f"Interrupted. Snapping back...")
                msg = "Pour interrupted and recovered"
            else:
                self.get_logger().info(f"Pour finished naturally. Waiting 3s before manual snap...")
                wait(3.0)
                msg = "Pour completed successfully with manual snap"

            # Waypoint-based Backtrack Logic
            from DSR_ROBOT2 import get_current_posx, fkin, posx
            curr_x = list(get_current_posx()[0])
            
            # reverse_path: current -> [passed waypoints in reverse] -> Cheers
            reverse_path = [posx(curr_x)]
            
            # Add passed waypoints in reverse
            # Note: self.passed_waypoints contains XYZ lists. We need to attach correct orientation.
            # We'll map them back to p2, p3, p4, p5 for orientation.
            wp_map = {
                tuple(config['pos_contact']): p2,
                tuple(config['pos_horizontal']): p3,
                tuple(config['pos_diagonal']): p4,
                tuple(config['pos_vertical']): p5
            }

            for wp_xyz in reversed(self.passed_waypoints):
                target_p = wp_map.get(tuple(wp_xyz))
                if target_p:
                    reverse_path.append(target_p)
            
            # Final point: CHEERS_POSE
            cheers_x = posx([float(x) for x in fkin(CHEERS_POSE, ref=0)])
            reverse_path.append(cheers_x)

            self.get_logger().info(f"Snapping back through {len(reverse_path)-1} waypoints using movesx...")
            movesx(reverse_path, vel=[200, 200], acc=[200, 200])
            
            # Wait for 1 second after reaching cheers_pose
            self.get_logger().info("Reached CHEERS_POSE. Waiting 1s...")
            wait(1.0)

            # Return to HOME_POSE
            self.get_logger().info("Moving to HOME_POSE before signaling Place node...")
            movej(HOME_POSE, vel=60, acc=60)

            response.success = True
            response.message = msg

            # Signal Place node (Always signal after pouring is done, whether natural or interrupted)
            if self.place_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Sending signal to Place node...")
                self.place_cli.call_async(Trigger.Request())
            else:
                self.get_logger().warn("Place node service not available, skipping signal.")

        except Exception as e:
            self.get_logger().error(f"Pour Error: {e}")
            response.success = False
            response.message = str(e)
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    # Use empty ROBOT_ID to force relative resolution under /dsr01 namespace
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
