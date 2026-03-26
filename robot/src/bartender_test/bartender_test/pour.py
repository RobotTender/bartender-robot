import rclpy
import sys
import threading
from threading import Timer
import time
import math
import json
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionServer, CancelResponse, GoalResponse
import DR_init

from std_msgs.msg import Empty, String, Float32
from std_srvs.srv import Trigger
from dsr_msgs2.srv import MoveStop
from sensor_msgs.msg import JointState
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from robotender_msgs.action import PourBottle
from .defines import (POSJ_HOME, POSJ_CHEERS, POSJ_SNAP, BOTTLE_CONFIG,
                            SNAP_VELOCITY, SNAP_ACCELERATION)

class ActionNode(Node):
    def __init__(self):
        super().__init__('robotender_pour', namespace='/dsr01')
        self._default_callback_group = ReentrantCallbackGroup()
        self.callback_group = ReentrantCallbackGroup()
        
        self.trigger_received = False
        self.flow_started_received = False
        self.current_target_volume_ml = 0.0
        self.passed_horizontal = False
        self.recording = False
        self.current_posx = [0.0] * 6
        self.current_posj = [0.0] * 6
        self.passed_waypoints = []
        self.target_waypoints = []
        self.periodic_buffer = [] 
        self.last_checkpoint_xyz = [0.0] * 3
        self.current_bottle_type = 'soju'
        self.state = 'IDLE'

        # VISION TRIGGERS
        self.create_subscription(Empty, 'robotender_snap/trigger', self.trigger_cb, 10, callback_group=self.callback_group)
        self.create_subscription(Empty, 'robotender/flow_started', self.flow_started_cb, 10, callback_group=self.callback_group)
        
        self.create_subscription(JointState, 'joint_states', self.cb_joint_states, 10, callback_group=self.callback_group)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.stop_cli = self.create_client(MoveStop, 'motion/move_stop', callback_group=self.callback_group)

        # HANDSHAKE PUBLISHERS
        self.target_volume_pub = self.create_publisher(Float32, '/detection/cup_target_volume', 10)
        self.pour_status_pub = self.create_publisher(String, 'robotender/pouring_status', 10)
        
        # HANDSHAKE SERVICE CLIENT
        self.prepare_cli = self.create_client(Trigger, 'robotender/prepare_pouring', callback_group=self.callback_group)

        self._action_server = ActionServer(
            self,
            PourBottle,
            'robotender_pour/execute',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )
        
        self.record_timer = self.create_timer(0.05, self.record_loop, callback_group=self.callback_group)
        self.buffer_count = 0

        self.get_logger().info('--- Robotender Pour Action Node Initialized ---')

    def cb_joint_states(self, msg):
        if len(msg.position) >= 6:
            self.current_posj = [math.degrees(v) for v in msg.position[:6]]

    def trigger_cb(self, msg):
        """Standard vision trigger callback (Volume-based)."""
        if self.recording and not self.trigger_received:
            self.get_logger().info("!!! VOLUME TRIGGER DETECTED (Vision) !!!")
            self.trigger_snap()

    def flow_started_cb(self, msg):
        """Trigger snap after a fixed wait time from flow detection."""
        if self.recording and not self.flow_started_received:
            self.get_logger().info("--- FLOW STARTED RECEIVED (Camera Signal) ---")
            self.flow_started_received = True
            
            # Logic: Use target volume to index the pour_wait_time list (50ml increments)
            # Index 0: 50ml, Index 1: 100ml, ..., Index 5: 300ml
            config = BOTTLE_CONFIG.get(self.current_bottle_type, BOTTLE_CONFIG['soju'])
            wait_times = config.get('pour_wait_time', [0.0] * 6)
            
            # Robust index calculation: (volume / 50) - 1, clamped to [0, 5]
            idx = int(max(0, min(5, (self.current_target_volume_ml / 50.0) - 1)))
            wait_time = wait_times[idx]
            
            self.get_logger().info(f"Fixed Wait Snap for {self.current_target_volume_ml}ml: {wait_time}s (index {idx})")
            
            if wait_time > 0:
                Timer(wait_time, self.trigger_snap).start()
            elif wait_time == 0:
                self.trigger_snap()
            else:
                self.get_logger().info("Wait time is negative, skipping auto-snap.")

    def trigger_snap(self):
        """Universal snap trigger: calls MoveStop and sets internal state."""
        if self.recording and not self.trigger_received:
            self.trigger_received = True
            self.get_logger().info("Executing MOVE STOP...")
            if self.stop_cli.wait_for_service(timeout_sec=0.5):
                req = MoveStop.Request()
                req.stop_mode = 2 
                self.stop_cli.call_async(req)
            else:
                self.get_logger().error("MoveStop Service not available during snap!")

    def record_loop(self):
        if self.recording:
            try:
                trans = self.tf_buffer.lookup_transform('base_link', 'link_6', rclpy.time.Time())
                p_x, p_y, p_z = trans.transform.translation.x * 1000.0, trans.transform.translation.y * 1000.0, trans.transform.translation.z * 1000.0
                self.current_posx[:3] = [p_x, p_y, p_z]
            except Exception: return

            self.buffer_count += 1
            if self.buffer_count % 4 == 0:
                from DSR_ROBOT2 import posj
                self.periodic_buffer.append(posj(self.current_posj))

            if self.target_waypoints:
                A = self.last_checkpoint_xyz
                B_obj = self.target_waypoints[0][0]
                B = list(B_obj[:3])
                P = self.current_posx[:3]
                V = [B[0]-A[0], B[1]-A[1], B[2]-A[2]]
                U = [P[0]-A[0], P[1]-A[1], P[2]-A[2]]
                v_mag_sq = V[0]**2 + V[1]**2 + V[2]**2
                if v_mag_sq > 0:
                    dot_product = U[0]*V[0] + U[1]*V[1] + U[2]*V[2]
                    if (dot_product / v_mag_sq) >= 1.0:
                        wp_data = self.target_waypoints.pop(0)
                        self.passed_waypoints.append(wp_data)
                        self.last_checkpoint_xyz = B
                        
                        # contact(1), horizontal(2), diagonal(3), vertical(4)
                        if len(self.passed_waypoints) == 2:
                            self.passed_horizontal = True

    def goal_callback(self, goal_request):
        if self.state == "RUNNING": return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.state = "RUNNING"
        feedback_msg = PourBottle.Feedback()
        result = PourBottle.Result()

        self.current_target_volume_ml = goal_handle.request.target_volume_ml
        self.current_bottle_type = goal_handle.request.bottle_name or 'soju'
        config = BOTTLE_CONFIG.get(self.current_bottle_type, BOTTLE_CONFIG['soju'])

        # --- LOG CONFIG MAPPING ---
        wait_times = config.get('pour_wait_time', [0.0] * 6)
        idx = int(max(0, min(5, (self.current_target_volume_ml / 50.0) - 1)))
        wait_time = wait_times[idx]
        self.get_logger().info(f"--- POUR START: {self.current_bottle_type.upper()} {self.current_target_volume_ml}ml (Mapped Wait: {wait_time}s, idx: {idx}) ---")

        # 1. Notify Target Volume
        vol_msg = Float32(data=float(self.current_target_volume_ml))
        self.target_volume_pub.publish(vol_msg)

        from DSR_ROBOT2 import (movej, movel, movesx, movesj, posx, posj, fkin, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, get_current_posj, get_current_posx)

        try:
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)

            def combine_pos(x_key, j_key):
                xyz = config[x_key][:3]
                orientation = [float(x) for x in fkin(config[j_key], ref=0)][3:]
                return posx(xyz + orientation)

            p2, p3, p4, p5 = combine_pos('posx_contact', 'posj_contact'), combine_pos('posx_horizontal', 'posj_horizontal'), combine_pos('posx_diagonal', 'posj_diagonal'), combine_pos('posx_vertical', 'posj_vertical')

            # MOVE TO STARTING POSITION (CHEERS)
            self.get_logger().info("Moving to POSJ_CHEERS...")
            movej(POSJ_CHEERS, vel=30, acc=30)

            # --- HANDSHAKE: PREPARE POURING SERVICE ---
            self.get_logger().info("Calling PreparePouring Service...")
            if not self.prepare_cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().error("Perception Service not available!")
            else:
                req = Trigger.Request()
                future = self.prepare_cli.call_async(req)
                # Wait for perception to calibrate/snapshot
                start_t = time.time()
                while not future.done() and (time.time() - start_t < 5.0): time.sleep(0.1)
                if future.done() and future.result().success:
                    self.get_logger().info(f"Perception Ready: {future.result().message}")
                else:
                    self.get_logger().warn("Perception not ready or timed out. Proceeding with caution.")

            self.last_checkpoint_xyz = list(get_current_posx()[0])[:3]
            self.trigger_received, self.flow_started_received = False, False
            self.passed_horizontal, self.passed_waypoints, self.periodic_buffer = False, [], []
            self.target_waypoints = [(p2, config.get('posj_contact')), (p3, config.get('posj_horizontal')), (p4, config.get('posj_diagonal')), (p5, config.get('posj_vertical'))]
            
            self.recording = True
            movel(p2, vel=[50, 50], acc=[50, 50])
            
            if not self.trigger_received:
                movesx([p3, p4, p5], vel=config.get('pour_velocity'), acc=config.get('pour_acc'))
            
            wait(0.1)
            self.recording = False
            if not self.trigger_received:
                while self.target_waypoints: self.passed_waypoints.append(self.target_waypoints.pop(0))

            if not self.trigger_received: wait(3.0)
            msg = "Pour interrupted" if self.trigger_received else "Pour completed"

            # --- HANDSHAKE: SEND DONE SIGNAL ---
            self.pour_status_pub.publish(String(data='done'))
            self.get_logger().info("Sent 'done' signal to perception.")

            # RECOVERY MOTION
            if self.trigger_received:
                if self.passed_horizontal:
                    self.get_logger().info("Interrupted AFTER horizontal. Executing POSJ_SNAP...")
                    movej(POSJ_SNAP, vel=SNAP_VELOCITY, acc=SNAP_ACCELERATION)
                else:
                    self.get_logger().info("Interrupted BEFORE horizontal. Executing posj_contact...")
                    contact_pose = config.get('posj_contact')
                    if contact_pose:
                        movej(contact_pose, vel=120, acc=120)
            else:
                # Normal completion recovery
                curr_j = list(get_current_posj())
                raw_path = [posj(curr_j)]
                waypoints_rev = list(reversed(self.passed_waypoints))
                if waypoints_rev: waypoints_rev.pop(0)
                for _, wp_j in waypoints_rev:
                    if wp_j is not None: raw_path.append(posj(wp_j))
                if config.get('posj_contact'): raw_path.append(posj(config.get('posj_contact')))
                
                reverse_path = [raw_path[0]]
                for i in range(1, len(raw_path)):
                    if sum(abs(a-b) for a,b in zip(raw_path[i], raw_path[i-1])) > 5.0: reverse_path.append(raw_path[i])

                if len(reverse_path) > 1:
                    if sum(abs(a-b) for a,b in zip(reverse_path[0], list(get_current_posj()))) < 0.5: reverse_path.pop(0)
                    if len(reverse_path) > 1: movesj(reverse_path, vel=210, acc=210)
                    else: movej(reverse_path[0], vel=150, acc=150)
                elif reverse_path: movej(reverse_path[0], vel=150, acc=150)
            
            wait(1.0); movej(POSJ_CHEERS, vel=30, acc=30)

            # CLEAR PERCEPTION TARGET
            self.target_volume_pub.publish(Float32(data=0.0))
            goal_handle.succeed()
            result.success, result.message = True, msg
            self.state = "IDLE"
            return result
        except Exception as e:
            self.get_logger().error(f"Pour Error: {e}")
            self.target_volume_pub.publish(Float32(data=0.0))
            self.pour_status_pub.publish(String(data='done'))
            goal_handle.abort()
            result.success, result.message = False, str(e)
            self.state = "IDLE"
            return result

def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    doosan_node = rclpy.create_node('pour_doosan_internal', namespace='/dsr01')
    doosan_node._default_callback_group = ReentrantCallbackGroup()
    import DR_init
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = "dsr01", "e0509", doosan_node
    executor = MultiThreadedExecutor(num_threads=20)
    executor.add_node(node); executor.add_node(doosan_node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); doosan_node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
