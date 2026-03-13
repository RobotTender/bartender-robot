import rclpy
import sys
import threading
import time
import math
import json
from rclpy.node import Node
import DR_init
from .gripper_controller import GripperController
from std_msgs.msg import Empty, String
from dsr_msgs2.srv import MoveStop
from sensor_msgs.msg import JointState
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from .defines import (HOME_POSE, CHEERS_POSE, CONTACT_POSE, POUR_HORIZONTAL, POUR_DIAGONAL, POUR_VERTICAL, POLE_POSE,
                            POS1_XYZ, POS2_XYZ, POS3_XYZ, POS4_XYZ, POS5_XYZ, DEFAULT_TARGET_POUR)

ACTION_TOPIC = "/bartender/action_request"

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


def execute_action(cmd: str, count: int, motion_node: Node, listener_node: TriggerListener):
    from DSR_ROBOT2 import (movej, movel, movesx, posx, posj, movesj, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait,
                            get_current_posx, get_current_posj, fkin, get_robot_state, get_workpiece_weight)
    from dsr_msgs2.srv import GetRobotState

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    wait(0.5)

    motion_node.get_logger().info("Ensuring robot is completely stationary before starting action...")
    state_cli = motion_node.create_client(GetRobotState, 'system/get_robot_state')
    if state_cli.wait_for_service(timeout_sec=2.0):
        for _ in range(30):
            req = GetRobotState.Request()
            future = state_cli.call_async(req)
            
            # Correctly spin the node to process the service response
            rclpy.spin_until_future_complete(motion_node, future, timeout_sec=1.0)
            
            if future.done():
                res = future.result()
                if res and res.robot_state == 1: # STATE_STANDBY
                    motion_node.get_logger().info("Robot is in STANDBY state.")
                    break
                else:
                    motion_node.get_logger().info(f"Robot state is {res.robot_state}, waiting for STANDBY (1)...")
            else:
                motion_node.get_logger().warn("Timeout waiting for GetRobotState response.")
            time.sleep(0.5)

    if cmd == 'warmup':
        motion_node.get_logger().info(f"Starting WARMUP ({count} cycles)")
        poses = [("HOME", HOME_POSE), ("CHEERS", CHEERS_POSE), ("CONTACT", CONTACT_POSE),
                 ("POUR_HORIZONTAL", POUR_HORIZONTAL), ("POUR_DIAGONAL", POUR_DIAGONAL),
                 ("POUR_VERTICAL", POUR_VERTICAL), ("POLE", POLE_POSE)]
        for _ in range(count):
            for name, pose in poses:
                motion_node.get_logger().info(f"Moving to {name}...")
                movej(pose, vel=60, acc=60)
            wait(0.5)

    elif cmd == 'pour':
        motion_node.get_logger().info("Calculating Cartesian pouring paths using fkin...")
        
        # Calculate Forward Kinematics
        fkin_cheers = fkin(CHEERS_POSE, ref=0)
        fkin_contact = fkin(CONTACT_POSE, ref=0)
        fkin_horiz = fkin(POUR_HORIZONTAL, ref=0)
        fkin_diag = fkin(POUR_DIAGONAL, ref=0)
        fkin_vert = fkin(POUR_VERTICAL, ref=0)

        motion_node.get_logger().info(f"fkin CHEERS: {fkin_cheers}")
        motion_node.get_logger().info(f"fkin CONTACT: {fkin_contact}")

        # Create hybrid Cartesian poses: exact POS_XYZ + orientation from corresponding Joint Pose
        p1 = posx(POS1_XYZ + [float(x) for x in fkin_cheers][3:])
        p2 = posx(POS2_XYZ + [float(x) for x in fkin_contact][3:])
        p3 = posx(POS3_XYZ + [float(x) for x in fkin_horiz][3:])
        p4 = posx(POS4_XYZ + [float(x) for x in fkin_diag][3:])
        p5 = posx(POS5_XYZ + [float(x) for x in fkin_vert][3:])

        motion_node.get_logger().info(f"Pour Path P2 (CONTACT): {p2}")
        motion_node.get_logger().info(f"Pour Path P3 (HORIZONTAL): {p3}")
        motion_node.get_logger().info(f"Pour Path P4 (DIAGONAL): {p4}")
        motion_node.get_logger().info(f"Pour Path P5 (VERTICAL): {p5}")

        # Approach: Move directly using Joint Space to avoid spline weirdness
        # We know p2 corresponds exactly to CONTACT_POSE
        # Pouring orbit: Contact -> Horizontal -> Diagonal -> Vertical
        pour_path = [p2, p3, p4, p5]

        motion_node.get_logger().info(f"Starting POUR ({count} cycles)")
        for i in range(count):
            motion_node.get_logger().info(f"--- Pour Cycle {i+1}/{count} ---")
            movej(CHEERS_POSE, vel=60, acc=60)
            wait(0.2)

            listener_node.trigger_received = False
            motion_node.get_logger().info("Moving to CONTACT_POSE (Approach via movej)...")
            movej(CONTACT_POSE, vel=60, acc=60)

            if listener_node.trigger_received:
                motion_node.get_logger().info("Trigger detected during approach. Aborting.")
                movej(CHEERS_POSE, vel=100, acc=100)
                continue

            listener_node.start_recording()
            start_time = time.time()
            motion_node.get_logger().info(
                f"Pouring STARTED at: {time.strftime('%H:%M:%S', time.localtime(start_time))}.{int((start_time%1)*1000):03d}"
            )
            movesx(pour_path, vel=100, acc=100)
            end_time = time.time()
            listener_node.stop_recording()

            duration = end_time - start_time
            num_points = len(listener_node.pouring_point_buffer)
            motion_node.get_logger().info(
                f"Pour END at: {time.strftime('%H:%M:%S', time.localtime(end_time))}.{int((end_time%1)*1000):03d}"
            )
            motion_node.get_logger().info(f"Motion Duration: {duration:.2f}s | Raw Points: {num_points} (@3Hz)")

            if listener_node.trigger_received:
                snap_duration = end_time - start_time
                motion_node.get_logger().info(
                    f"!!! Path interrupted at {snap_duration:.2f}s. Executing DYNAMIC SNAP RECOVERY !!!"
                )

                curr_j = get_current_posj()
                if snap_duration < 2.5:
                    target_backtrack_size = 1
                elif snap_duration < 5.0:
                    target_backtrack_size = 2
                else:
                    target_backtrack_size = 3

                recorded_path_full = listener_node.pouring_point_buffer[::-1]
                num_recorded = len(recorded_path_full)

                if num_recorded > target_backtrack_size:
                    num_to_sample = target_backtrack_size - 1
                    if num_to_sample > 0:
                        step = num_recorded // num_to_sample
                        thinning_buffer = recorded_path_full[::step][:num_to_sample]
                    else:
                        thinning_buffer = []

                    start_point = listener_node.pouring_point_buffer[0]
                    if not thinning_buffer:
                        thinning_buffer.append(start_point)
                    else:
                        diff_start = [abs(a - b) for a, b in zip(thinning_buffer[-1], start_point)]
                        if any(d > 0.1 for d in diff_start):
                            thinning_buffer.append(start_point)
                else:
                    thinning_buffer = recorded_path_full

                final_reverse_path = [curr_j]
                for p in thinning_buffer:
                    diff = [abs(a - b) for a, b in zip(final_reverse_path[-1], p)]
                    if any(d > 0.1 for d in diff):
                        final_reverse_path.append(p)

                cheers_j = posj(CHEERS_POSE)
                diff_final = [abs(a - b) for a, b in zip(final_reverse_path[-1], cheers_j)]
                if any(d > 0.1 for d in diff_final):
                    final_reverse_path.append(cheers_j)

                motion_node.get_logger().info(
                    f"Reversing through {len(final_reverse_path)} waypoints (Snap -> {len(thinning_buffer)} thinned backtrack -> Cheers)."
                )
                if len(final_reverse_path) > 1:
                    movesj(final_reverse_path, vel=250, acc=250)
                else:
                    movej(CHEERS_POSE, vel=250, acc=250)

                motion_node.get_logger().info("Snap recovery complete.")
                break
            else:
                motion_node.get_logger().info("Path finished naturally.")
    else:
        motion_node.get_logger().error(f"Unknown command: {cmd}")


class ActionRequestNode(Node):
    def __init__(self, robot_id: str, motion_node: Node, listener_node: TriggerListener):
        super().__init__("action_request_node", namespace=robot_id)
        self.motion_node = motion_node
        self.listener_node = listener_node
        self.busy = False
        self.create_subscription(String, ACTION_TOPIC, self.action_callback, 10)
        self.get_logger().info(f"Listening action requests on {ACTION_TOPIC}")

    def action_callback(self, msg: String):
        if self.busy:
            self.get_logger().warn("Action request ignored because another action is running.")
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Invalid action request JSON: {exc}")
            return

        cmd = str(payload.get("action", "pour")).lower()
        count = int(payload.get("count", 1))

        def run_action():
            self.busy = True
            try:
                execute_action(cmd, count, self.motion_node, self.listener_node)
            except Exception as exc:
                self.motion_node.get_logger().error(f"Action Error: {exc}")
            finally:
                self.busy = False

        threading.Thread(target=run_action, daemon=True).start()

def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: ros2 run bartender_test action [pour | warmup | listen] [repeat=<n>]")
        return

    rclpy.init(args=args)
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    
    # Node A: For Doosan API (Internal Spinning)
    motion_node = Node('motion_node', namespace=ROBOT_ID)
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, motion_node

    # Node B: For Trigger Monitoring (External Spinning)
    listener_node = TriggerListener(ROBOT_ID)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(listener_node)

    try:
        cmd = sys.argv[1].lower()
        if cmd == "listen":
            action_node = ActionRequestNode(ROBOT_ID, motion_node, listener_node)
            executor.add_node(action_node)
            spin_thread = threading.Thread(target=executor.spin, daemon=True)
            spin_thread.start()
            motion_node.get_logger().info("Action listener started.")
            try:
                while rclpy.ok():
                    time.sleep(0.5)
            finally:
                action_node.destroy_node()
        else:
            spin_thread = threading.Thread(target=executor.spin, daemon=True)
            spin_thread.start()

            repeat = 1
            nums = []
            for arg in sys.argv[2:]:
                if arg.startswith('repeat='):
                    try:
                        repeat = int(arg.split('=')[1])
                    except Exception:
                        pass
                elif arg.isdigit():
                    nums.append(int(arg))

            count = repeat
            if not nums and len(sys.argv) > 2 and sys.argv[2].isdigit():
                count = int(sys.argv[2])
            elif nums:
                count = nums[0]

            execute_action(cmd, count, motion_node, listener_node)
    except Exception as e:
        motion_node.get_logger().error(f"Action Error: {e}")
    finally:
        executor.shutdown()
        rclpy.shutdown()
        if 'spin_thread' in locals():
            spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
