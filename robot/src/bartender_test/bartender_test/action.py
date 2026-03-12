import rclpy
import sys
import threading
import time
import math
from rclpy.node import Node
import DR_init
from .gripper_controller import GripperController
from std_msgs.msg import Empty
from dsr_msgs2.srv import MoveStop
from sensor_msgs.msg import JointState
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from .defines import (HOME_POSE, CHEERS_POSE, CONTACT_POSE, POUR_HORIZONTAL, POUR_DIAGONAL, POUR_VERTICAL, POLE_POSE,
                            POS1_XYZ, POS2_XYZ, POS3_XYZ, POS4_XYZ, POS5_XYZ, DEFAULT_TARGET_POUR)

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z

# Node B: Dedicated Listener Node
class TriggerListener(Node):
    def __init__(self, robot_id):
        super().__init__('trigger_listener', namespace=robot_id)
        self.trigger_received = False
        self.create_subscription(Empty, 'pouring_trigger', self.trigger_cb, 10)
        self.stop_cli = self.create_client(MoveStop, 'motion/move_stop')
        
        # Joint state for background recording
        self.current_posj = [0.0] * 6
        self.sub_js = self.create_subscription(JointState, 'joint_states', self.cb_joint_states, 10)
        
        # TF for background recording
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pouring_point_buffer = []
        self.recording = False
        self.timer = self.create_timer(0.33, self.record_cb) # 3Hz

    def cb_joint_states(self, msg):
        if len(msg.position) >= 6:
            # ROS joint states are in Radians, but Doosan posj is in Degrees
            self.current_posj = [math.degrees(v) for v in msg.position[:6]]

    def trigger_cb(self, msg):
        self.get_logger().info("!!! SPACEBAR TRIGGER DETECTED (Background Listener) !!!")
        self.trigger_received = True
        # Immediately request robot stop
        if self.stop_cli.wait_for_service(timeout_sec=0.5):
            req = MoveStop.Request()
            req.stop_mode = 2 # DR_SSTOP (Soft Stop)
            self.stop_cli.call_async(req)
            self.get_logger().info("Sent MoveStop(DR_SSTOP) to Controller.")
        else:
            self.get_logger().error("MoveStop service not available!")

    def record_cb(self):
        if not self.recording:
            return
        try:
            # We record current_posj (Degrees) which captures 100% of the posture
            from DSR_ROBOT2 import posj
            p = posj(self.current_posj)
            self.pouring_point_buffer.append(p)
        except Exception as e:
            # Only log error occasionally to avoid spamming
            if len(self.pouring_point_buffer) == 0 and int(time.time()) % 5 == 0:
                self.get_logger().error(f"Path Recording Error: {e}")
            pass

    def start_recording(self):
        self.pouring_point_buffer = []
        self.recording = True

    def stop_recording(self):
        self.recording = False

def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: ros2 run bartender_test action [pour orbit | warmup] [repeat=<n>]")
        return

    rclpy.init(args=args)
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    
    # Node A: For Doosan API (Internal Spinning)
    motion_node = Node('motion_node', namespace=ROBOT_ID)
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, motion_node

    # Node B: For Trigger Monitoring (External Spinning)
    listener_node = TriggerListener(ROBOT_ID)
    
    # Spin Node B in background
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(listener_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    from DSR_ROBOT2 import (movej, movel, movesx, posx, posj, movesj, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, 
                            get_current_posx, get_current_posj, fkin, get_robot_state, get_workpiece_weight)

    try:
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        wait(0.5)

        cmd = sys.argv[1].lower()
        sub_cmd = sys.argv[2].lower() if len(sys.argv) > 2 else ""
        
        # Argument parsing
        repeat = 1
        nums = []
        for arg in sys.argv[2:]:
            if arg.startswith('repeat='):
                try: repeat = int(arg.split('=')[1])
                except: pass
            elif arg.isdigit(): nums.append(int(arg))
        
        count = repeat
        if not nums and not sub_cmd.isdigit() and len(sys.argv) > 2 and sys.argv[2].isdigit():
             count = int(sys.argv[2])
        elif nums:
             count = nums[0]

        if cmd == 'warmup':
            motion_node.get_logger().info(f"Starting WARMUP ({count} cycles)")
            poses = [("HOME", HOME_POSE), ("CHEERS", CHEERS_POSE), ("CONTACT", CONTACT_POSE), 
                     ("POUR_HORIZONTAL", POUR_HORIZONTAL), ("POUR_DIAGONAL", POUR_DIAGONAL), 
                     ("POUR_VERTICAL", POUR_VERTICAL), ("POLE", POLE_POSE)]
            for i in range(count):
                for name, pose in poses:
                    movej(pose, vel=60, acc=60)
                wait(0.5)

        elif cmd == 'pour' and sub_cmd == 'orbit':
            # Setup trajectory points
            p1 = posx([float(x) for x in fkin(CHEERS_POSE, ref=0)])
            p2 = posx(POS2_XYZ + [float(x) for x in fkin(CONTACT_POSE, ref=0)][3:])
            p3 = posx(POS3_XYZ + [float(x) for x in fkin(POUR_HORIZONTAL, ref=0)][3:])
            p5 = posx([float(x) for x in fkin(POUR_VERTICAL, ref=0)])
            p4 = posx(list(POS4_XYZ) + [float(x) for x in fkin(POUR_DIAGONAL, ref=0)][3:])
            # Setup trajectory segments
            approach_path = [p1, p2]
            pour_path = [p2, p3, p4, p5]

            motion_node.get_logger().info(f"Starting POUR ORBIT ({count} cycles)")
            for i in range(count):
                motion_node.get_logger().info(f"--- Pour Cycle {i+1}/{count} ---")
                movej(CHEERS_POSE, vel=60, acc=60)
                wait(0.2)
                
                listener_node.trigger_received = False # Reset flag
                motion_node.get_logger().info("Moving to CONTACT_POSE (Approach)...")

                # 1. Execute Approach (Blocking, No recording)
                movesx(approach_path, vel=100, acc=100)

                if listener_node.trigger_received:
                    motion_node.get_logger().info("Trigger detected during approach. Aborting.")
                    movej(CHEERS_POSE, vel=100, acc=100)
                    continue

                # 2. Start Recording/Timing HERE (Contact point reached)
                listener_node.start_recording()
                start_time = time.time()
                motion_node.get_logger().info(f"Pouring STARTED at: {time.strftime('%H:%M:%S', time.localtime(start_time))}.{int((start_time%1)*1000):03d}")

                # 3. Execute Pouring Motion (Blocking)
                movesx(pour_path, vel=100, acc=100)

                # Capture completion time immediately
                end_time = time.time()
                listener_node.stop_recording()

                duration = end_time - start_time
                num_points = len(listener_node.pouring_point_buffer)
                motion_node.get_logger().info(f"Pour Orbit END at: {time.strftime('%H:%M:%S', time.localtime(end_time))}.{int((end_time%1)*1000):03d}")
                motion_node.get_logger().info(f"Motion Duration: {duration:.2f}s | Raw Points: {num_points} (@3Hz)")

                # Check why we stopped
                if listener_node.trigger_received:
                    # snap_time is essentially end_time here
                    snap_duration = end_time - start_time
                    motion_node.get_logger().info(f"!!! Path interrupted at {snap_duration:.2f}s. Executing DYNAMIC SNAP RECOVERY !!!")

                    # 1. Trigger Point: Capture the exact joint position at the moment of the snap
                    curr_j = get_current_posj()

                    # 2. Thinning Buffer: Dynamically scale size based on duration tiers
                    # Tiers (Total Path): <2.5s: 2 pts | 2.5-5s: 3 pts | >5s: 4 pts
                    if snap_duration < 2.5:
                        target_backtrack_size = 1
                    elif snap_duration < 5.0:
                        target_backtrack_size = 2
                    else:
                        target_backtrack_size = 3

                    recorded_path_full = listener_node.pouring_point_buffer[::-1]
                    num_recorded = len(recorded_path_full)

                    if num_recorded > target_backtrack_size:
                        # Downsample to target_backtrack_size-1, then append the start point
                        num_to_sample = target_backtrack_size - 1
                        if num_to_sample > 0:
                            step = num_recorded // num_to_sample
                            thinning_buffer = recorded_path_full[::step][:num_to_sample]
                        else:
                            thinning_buffer = []

                        # Constraint: Always include the starting point (the first point recorded)
                        start_point = listener_node.pouring_point_buffer[0]
                        if not thinning_buffer:
                            thinning_buffer.append(start_point)
                        else:
                            diff_start = [abs(a - b) for a, b in zip(thinning_buffer[-1], start_point)]
                            if any(d > 0.1 for d in diff_start):
                                thinning_buffer.append(start_point)

                        motion_node.get_logger().info(f"Tiered Thinning: {num_recorded} raw -> {len(thinning_buffer)} backtrack points (Total Path: {len(thinning_buffer)+1}).")
                    else:
                        thinning_buffer = recorded_path_full                    # 3. Final Execution Path: [Trigger Point] + [Thinning Buffer Points] + [CHEERS_POSE]
                    final_reverse_path = [curr_j] # Point 1: Trigger Point
                    for p in thinning_buffer:
                        # Avoid duplicates or very close points
                        diff = [abs(a - b) for a, b in zip(final_reverse_path[-1], p)]
                        if any(d > 0.1 for d in diff):
                            final_reverse_path.append(p)

                    # Safety Finish: Always ensure CHEERS_POSE is the final destination
                    cheers_j = posj(CHEERS_POSE)
                    diff_final = [abs(a - b) for a, b in zip(final_reverse_path[-1], cheers_j)]
                    if any(d > 0.1 for d in diff_final):
                         final_reverse_path.append(cheers_j)

                    motion_node.get_logger().info(f"Reversing through {len(final_reverse_path)} waypoints (Snap -> {len(thinning_buffer)} thinned backtrack -> Cheers).")

                    if len(final_reverse_path) > 1:
                        movesj(final_reverse_path, vel=250, acc=250)
                    else:
                        movej(CHEERS_POSE, vel=250, acc=250)

                    motion_node.get_logger().info("Snap recovery complete.")
                    break
                else:
                    motion_node.get_logger().info("Path finished naturally.")
        
        elif cmd == 'pour' and sub_cmd == 'auto':
            # Simplified auto-pour for now
            target_poured_g = DEFAULT_TARGET_POUR
            if len(sys.argv) > 3 and sys.argv[3].replace('.','',1).isdigit():
                target_poured_g = float(sys.argv[3])
                
            motion_node.get_logger().info(f"Starting PREDICTIVE AUTO POUR: Target={target_poured_g}g")
            
            # Weighing
            gripper = GripperController(motion_node, namespace=ROBOT_ID)
            gripper.move(1150, force=250)
            movej(CHEERS_POSE, vel=60, acc=60)
            wait(2.0)
            
            weight_kg = get_workpiece_weight()
            weight_total_g = weight_kg * 1000.0
            target_ry = -0.0433 * weight_total_g + 102.3888
            
            p_approach_raw = [float(x) for x in fkin(CONTACT_POSE, ref=0)]
            p_approach_raw[3] = -90.0
            p_approach_raw[4] = target_ry - 2.0
            p_approach = posx(p_approach_raw)
            
            p_target_raw = list(p_approach_raw)
            p_target_raw[4] = target_ry + 10.0
            p_target = posx(p_target_raw)
            
            for i in range(count):
                motion_node.get_logger().info(f"--- Auto Pour Cycle {i+1}/{count} ---")
                movej(CHEERS_POSE, vel=60, acc=60)
                wait(0.5)
                movel(p_approach, vel=100, acc=100)
                
                # Precision Creep - using movesx/movel blocking is tricky here because 
                # weight updates need spinning. We'll use our listener node for weights too.
                # (Skipping full auto-pour refactor for now to focus on 'snap' success)
                movel(p_target, vel=5, acc=5)
                movej(CHEERS_POSE, vel=150, acc=150)
                wait(1.0)
        
    except Exception as e:
        motion_node.get_logger().error(f"Action Error: {e}")
    finally:
        executor.shutdown()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
