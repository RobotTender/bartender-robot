import rclpy
import sys
import threading
import time
import math
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import DR_init

from std_msgs.msg import Empty
from std_srvs.srv import Trigger
from dsr_msgs2.srv import MoveStop
from sensor_msgs.msg import JointState
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .defines import (HOME_POSE, CHEERS_POSE, CONTACT_POSE, POUR_HORIZONTAL, POUR_DIAGONAL, POUR_VERTICAL, POLE_POSE,
                            POS2_XYZ, POS3_XYZ, POS4_XYZ)

class ActionNode(Node):
    def __init__(self):
        super().__init__('robotender_pour', namespace='/dsr01')
        self.callback_group = ReentrantCallbackGroup()
        
        # State for Pouring
        self.trigger_received = False
        self.recording = False
        self.current_posj = [0.0] * 6
        self.pouring_point_buffer = []

        # Subscriptions
        self.create_subscription(Empty, 'robotender_snap/trigger', self.trigger_cb, 10, callback_group=self.callback_group)
        self.create_subscription(JointState, 'joint_states', self.cb_joint_states, 10, callback_group=self.callback_group)

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
            from DSR_ROBOT2 import posj
            self.pouring_point_buffer.append(posj(self.current_posj))

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
        self.get_logger().info('Starting Pour Sequence...')
        from DSR_ROBOT2 import (movej, movesx, movesj, posx, posj, fkin, set_robot_mode, 
                                ROBOT_MODE_AUTONOMOUS, wait, get_current_posj)
        
        try:
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)

            # Setup Trajectories
            p1 = posx([float(x) for x in fkin(CHEERS_POSE, ref=0)])
            p2 = posx(POS2_XYZ + [float(x) for x in fkin(CONTACT_POSE, ref=0)][3:])
            p3 = posx(POS3_XYZ + [float(x) for x in fkin(POUR_HORIZONTAL, ref=0)][3:])
            p5 = posx([float(x) for x in fkin(POUR_VERTICAL, ref=0)])
            p4 = posx(list(POS4_XYZ) + [float(x) for x in fkin(POUR_DIAGONAL, ref=0)][3:])
            
            approach_path = [p1, p2]
            pour_path = [p2, p3, p4, p5]

            # 1. Move to Cheers
            movej(CHEERS_POSE, vel=60, acc=60)
            
            # 2. Approach
            self.trigger_received = False
            movesx(approach_path, vel=100, acc=100)
            
            if self.trigger_received:
                 movej(CHEERS_POSE, vel=100, acc=100)
                 response.success = False
                 response.message = "Triggered during approach"
                 return response

            # 3. Pour with Recording
            self.pouring_point_buffer = []
            self.recording = True
            start_time = time.time()
            movesx(pour_path, vel=100, acc=100)
            end_time = time.time()
            self.recording = False
            duration = end_time - start_time

            # 4. Handle Recovery or Finish
            if self.trigger_received:
                self.get_logger().info(f"Interrupted. Snapping back...")
                msg = "Pour interrupted and recovered"
            else:
                self.get_logger().info(f"Pour finished naturally. Waiting 3s before manual snap...")
                wait(3.0)
                msg = "Pour completed successfully with manual snap"

            # Common Backtrack Logic (Snap Recovery or Manual Snap)
            curr_j = get_current_posj()
            if duration < 2.5: target_backtrack_size = 1
            elif duration < 5.0: target_backtrack_size = 2
            else: target_backtrack_size = 3

            recorded_full = self.pouring_point_buffer[::-1]
            num_rec = len(recorded_full)
            reverse_path = [curr_j]
            
            if num_rec > target_backtrack_size:
                step = num_rec // target_backtrack_size
                for i in range(0, num_rec, step):
                    p = recorded_full[i]
                    if any(abs(a-b) > 0.1 for a,b in zip(reverse_path[-1], p)):
                        reverse_path.append(p)
            else:
                reverse_path.extend(recorded_full)
            
            cheers_j = posj(CHEERS_POSE)
            if any(abs(a-b) > 0.1 for a,b in zip(reverse_path[-1], cheers_j)):
                reverse_path.append(cheers_j)

            movesj(reverse_path, vel=200, acc=200)
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
