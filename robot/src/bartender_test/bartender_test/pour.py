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

from .defines import (POSJ_HOME, POSJ_CHEERS, BOTTLE_CONFIG)

class ActionNode(Node):
    def __init__(self):
        super().__init__('robotender_pour', namespace='/dsr01')
        self.callback_group = ReentrantCallbackGroup()
        
        # State for Pouring
        self.trigger_received = False
        self.recording = False
        self.current_posx = [0.0] * 6
        self.current_posj = [0.0] * 6
        self.passed_waypoints = []
        self.target_waypoints = []
        self.periodic_buffer = []  # 5Hz sample buffer
        self.last_checkpoint_xyz = [0.0] * 3
        self.current_bottle_type = 'soju' # Default

        # Subscriptions
        self.create_subscription(Empty, 'robotender_snap/trigger', self.trigger_cb, 10, callback_group=self.callback_group)
        self.create_subscription(JointState, 'joint_states', self.cb_joint_states, 10, callback_group=self.callback_group)

        # TF2 Setup for Pose Tracking
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Clients
        self.stop_cli = self.create_client(MoveStop, 'motion/move_stop', callback_group=self.callback_group)

        # Services
        self.pour_srv = self.create_service(Trigger, 'robotender_pour/start', self.pour_callback, callback_group=self.callback_group)
        self.warmup_srv = self.create_service(Trigger, 'robotender_pour/warmup', self.warmup_callback, callback_group=self.callback_group)
        
        # Service Clients
        self.place_cli = self.create_client(Trigger, 'robotender_place/start', callback_group=self.callback_group)

        # Timer for path recording (20Hz for high-speed tracking)
        self.record_timer = self.create_timer(0.05, self.record_loop, callback_group=self.callback_group)
        self.buffer_count = 0

        self.get_logger().info('--- Robotender Pour Node Initialized ---')
        self.get_logger().info('Services: /dsr01/robotender_pour/start, /dsr01/robotender_pour/warmup')

    def cb_joint_states(self, msg):
        if len(msg.position) >= 6:
            self.current_posj = [math.degrees(v) for v in msg.position[:6]]

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
        if self.recording:
            # 0. Update Current Pose via TF
            try:
                # lookup link_6 relative to base_link
                trans = self.tf_buffer.lookup_transform('base_link', 'link_6', rclpy.time.Time())
                p_x = trans.transform.translation.x * 1000.0
                p_y = trans.transform.translation.y * 1000.0
                p_z = trans.transform.translation.z * 1000.0
                self.current_posx[:3] = [p_x, p_y, p_z]
            except Exception as e:
                return

            # 1. Periodic Buffer (5Hz sampling: 1 every 4 ticks of 20Hz timer)
            self.buffer_count += 1
            if self.buffer_count % 4 == 0:
                # Dynamic import for record_loop (only once per loop)
                from DSR_ROBOT2 import posj
                self.periodic_buffer.append(posj(self.current_posj))

            # 2. Waypoint Detection (Finish Line Logic)
            if self.target_waypoints:
                A = self.last_checkpoint_xyz
                B_data = self.target_waypoints[0]
                B_obj = B_data[0]
                B = list(B_obj[:3])
                P = self.current_posx[:3]
                
                V = [B[0]-A[0], B[1]-A[1], B[2]-A[2]]
                U = [P[0]-A[0], P[1]-A[1], P[2]-A[2]]
                
                v_mag_sq = V[0]**2 + V[1]**2 + V[2]**2
                if v_mag_sq > 0:
                    dot_product = U[0]*V[0] + U[1]*V[1] + U[2]*V[2]
                    progress = dot_product / v_mag_sq
                    
                    if progress >= 1.0:
                        wp_data = self.target_waypoints.pop(0)
                        self.passed_waypoints.append(wp_data)
                        self.last_checkpoint_xyz = B
                        self.get_logger().info(f"[Waypoint Crossed] Total: {len(self.passed_waypoints)}, Coord: {B}, Progress: {progress:.2f}")

    async def warmup_callback(self, request, response):
        self.get_logger().info('Starting Warmup Sequence (using Beer Config)...')
        from DSR_ROBOT2 import (movej, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait)
        
        try:
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)
            
            beer_cfg = BOTTLE_CONFIG['beer']
            poses = [("HOME", POSJ_HOME), ("CHEERS", POSJ_CHEERS), 
                     ("CONTACT", beer_cfg['posj_contact']), 
                     ("POUR_HORIZONTAL", beer_cfg['posj_horizontal']), 
                     ("POUR_DIAGONAL", beer_cfg['posj_diagonal']), 
                     ("POUR_VERTICAL", beer_cfg['posj_vertical'])]
            
            for name, pose in poses:
                self.get_logger().info(f"Moving to {name}")
                movej(pose, vel=30, acc=30)
            
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
        
        # Local import to ensure DR_init node is set
        from DSR_ROBOT2 import (
            movej, movel, movesx, movesj, posx, posj, fkin, 
            set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, 
            get_current_posj, get_current_posx
        )

        try:
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)

            def combine_pos(x_key, j_key):
                xyz = config[x_key][:3]
                j_pose = config[j_key]
                orientation = [float(x) for x in fkin(j_pose, ref=0)][3:]
                return posx(xyz + orientation)

            p2 = combine_pos('posx_contact', 'posj_contact')
            p3 = combine_pos('posx_horizontal', 'posj_horizontal')
            p4 = combine_pos('posx_diagonal', 'posj_diagonal')
            p5 = combine_pos('posx_vertical', 'posj_vertical')
            spline_path = [p3, p4, p5]

            self.get_logger().info("Moving to POSJ_CHEERS")
            movej(POSJ_CHEERS, vel=30, acc=30)
            self.last_checkpoint_xyz = list(get_current_posx()[0])[:3]
            
            self.trigger_received = False
            self.passed_waypoints = []
            self.periodic_buffer = []
            self.target_waypoints = [(p2, config.get('posj_contact')), (p3, config.get('posj_horizontal')), (p4, config.get('posj_diagonal')), (p5, config.get('posj_vertical'))]
            
            self.recording = True
            self.get_logger().info(f"Approaching contact for {active_type}...")
            self.last_checkpoint_xyz = list(get_current_posx()[0])[:3]
            movel(p2, vel=[30, 30], acc=[30, 30])
            
            if not self.trigger_received:
                self.get_logger().info("Executing pouring spline...")
                movesx(spline_path, vel=60, acc=60)
            
            wait(0.1)
            self.recording = False

            if not self.trigger_received and self.target_waypoints:
                while self.target_waypoints:
                    self.passed_waypoints.append(self.target_waypoints.pop(0))

            self.get_logger().info(f"--- Pouring Summary ---")
            self.get_logger().info(f"Total Waypoints Passed: {len(self.passed_waypoints)}")
            self.get_logger().info(f"Buffer Size: {len(self.periodic_buffer)} samples")
            
            if self.trigger_received:
                msg = "Pour interrupted and recovered"
            else:
                self.get_logger().info(f"Pour finished naturally. Waiting 3s before snap...")
                wait(3.0); msg = "Pour completed successfully"

            # --- Hybrid Recovery Logic: ASSEMBLE UNIFIED PATH FOR SMOOTHNESS ---
            curr_j_at_snap = list(get_current_posj())
            raw_path = [posj(curr_j_at_snap)]
            
            # 1. Add Section B (Periodic Samples - Aggressive Thinning for Speed)
            count_b = 0
            if self.trigger_received and self.periodic_buffer:
                # Take only the last 2 samples (very recent and slightly older)
                samples_to_add = min(len(self.periodic_buffer), 2)
                for i in range(1, samples_to_add + 1):
                    raw_path.append(posj(self.periodic_buffer[-i]))
                    count_b += 1

            # 2. Add Section C (Waypoints)
            count_c = 0
            waypoints_to_reverse = list(reversed(self.passed_waypoints))
            if not self.trigger_received and len(waypoints_to_reverse) > 0:
                waypoints_to_reverse.pop(0) # Skip p5 if natural finish

            for _, wp_j in waypoints_to_reverse:
                if wp_j is not None:
                    raw_path.append(posj(wp_j))
                    count_c += 1
            
            # 3. Add Section D (Final Destination: Contact Pose)
            count_d = 0
            contact_j = config.get('posj_contact')
            if contact_j: 
                raw_path.append(posj(contact_j))
                count_d = 1
            
            # --- Apply Aggressive Thinning Filter to Unified Path ---
            reverse_path_j = []
            if raw_path:
                reverse_path_j.append(raw_path[0])
                for i in range(1, len(raw_path)):
                    # 5.0 degree threshold for maximum spline speed
                    if sum(abs(a - b) for a, b in zip(raw_path[i], raw_path[i-1])) > 5.0:
                        reverse_path_j.append(raw_path[i])

            # --- Unified Execution (Fast Snap for Production) ---
            self.get_logger().info(f"--- Executing Unified Recovery (vel=150) ---")
            self.get_logger().info(f"Path Breakdown: Start(1), Samples({count_b}), Waypoints({count_c}), Contact({count_d})")
            self.get_logger().info(f"Final Path: {len(reverse_path_j)} unique points")

            if len(reverse_path_j) > 1:
                # Filter redundant start point relative to real-time pose
                curr_now = list(get_current_posj())
                if sum(abs(a - b) for a, b in zip(reverse_path_j[0], curr_now)) < 0.5:
                    reverse_path_j.pop(0)
                
                if len(reverse_path_j) > 1:
                    movesj(reverse_path_j, vel=150, acc=150)
                else:
                    movej(reverse_path_j[0], vel=150, acc=150)
            elif len(reverse_path_j) == 1:
                movej(reverse_path_j[0], vel=150, acc=150)
            
            self.get_logger().info("Reached CONTACT_POSE. Waiting 1s...")
            wait(1.0)
            self.get_logger().info("Moving to POSJ_CHEERS...")
            movej(POSJ_CHEERS, vel=30, acc=30)
            wait(1.0)
            movej(POSJ_HOME, vel=30, acc=30)

            response.success = True
            response.message = msg
            if self.place_cli.wait_for_service(timeout_sec=1.0):
                self.place_cli.call_async(Trigger.Request())
        except Exception as e:
            self.get_logger().error(f"Pour Error: {e}")
            response.success = False; response.message = str(e)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    ROBOT_ID, ROBOT_MODEL = "", "e0509"
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, node
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
