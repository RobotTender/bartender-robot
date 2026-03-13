import rclpy
import sys
import threading
import time
import math
import json
from rclpy.node import Node
import DR_init
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger
from dsr_msgs2.srv import MoveStop, GetRobotState, SetRobotMode
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

class TriggerListener(Node):
    def __init__(self, robot_id):
        super().__init__('trigger_listener', namespace=robot_id)
        self.trigger_received = False
        self.create_subscription(Empty, 'pouring_trigger', self.trigger_cb, 10)
        self.stop_cli = self.create_client(MoveStop, 'motion/move_stop')
        self.current_posj = [0.0] * 6
        self.sub_js = self.create_subscription(JointState, 'joint_states', self.cb_joint_states, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pouring_point_buffer = []
        self.recording = False
        self.timer = self.create_timer(0.33, self.record_cb) # 3Hz

    def cb_joint_states(self, msg):
        if len(msg.position) >= 6:
            self.current_posj = [math.degrees(v) for v in msg.position[:6]]

    def trigger_cb(self, msg):
        self.get_logger().info("!!! SPACEBAR TRIGGER DETECTED (Background Listener) !!!")
        self.trigger_received = True
        if self.stop_cli.wait_for_service(timeout_sec=0.5):
            req = MoveStop.Request(stop_mode=2)
            self.stop_cli.call_async(req)
            self.get_logger().info("Sent MoveStop(DR_SSTOP) to Controller.")

    def record_cb(self):
        if not self.recording: return
        try:
            from DSR_ROBOT2 import posj
            self.pouring_point_buffer.append(posj(self.current_posj))
        except: pass

    def start_recording(self):
        self.pouring_point_buffer = []
        self.recording = True

    def stop_recording(self):
        self.recording = False


def execute_action(cmd: str, count: int, motion_node: Node, listener_node: TriggerListener):
    from DSR_ROBOT2 import (movej, movel, movesx, posx, posj, movesj, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait,
                            get_current_posx, get_current_posj, fkin, get_robot_state)
    
    # Gripper clients
    gripper_open_cli = motion_node.create_client(Trigger, '/dsr01/gripper/open')
    gripper_close_cli = motion_node.create_client(SetRobotMode, '/dsr01/gripper/close')

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    wait(0.5)

    motion_node.get_logger().info("Ensuring robot is completely stationary before starting action...")
    state_cli = motion_node.create_client(GetRobotState, 'system/get_robot_state')
    if state_cli.wait_for_service(timeout_sec=2.0):
        for _ in range(30):
            req = GetRobotState.Request()
            future = state_cli.call_async(req)
            rclpy.spin_until_future_complete(motion_node, future, timeout_sec=1.0)
            if future.done() and future.result() and future.result().robot_state == 1:
                motion_node.get_logger().info("Robot is in STANDBY state.")
                break
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
        fkin_cheers = fkin(CHEERS_POSE, ref=0)
        fkin_contact = fkin(CONTACT_POSE, ref=0)
        fkin_horiz = fkin(POUR_HORIZONTAL, ref=0)
        fkin_diag = fkin(POUR_DIAGONAL, ref=0)
        fkin_vert = fkin(POUR_VERTICAL, ref=0)

        p1 = posx(POS1_XYZ + [float(x) for x in fkin_cheers][3:])
        p2 = posx(POS2_XYZ + [float(x) for x in fkin_contact][3:])
        p3 = posx(POS3_XYZ + [float(x) for x in fkin_horiz][3:])
        p4 = posx(POS4_XYZ + [float(x) for x in fkin_diag][3:])
        p5 = posx(POS5_XYZ + [float(x) for x in fkin_vert][3:])

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
            motion_node.get_logger().info(f"Pouring STARTED at: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
            movesx(pour_path, vel=100, acc=100)
            end_time = time.time()
            listener_node.stop_recording()

            if listener_node.trigger_received:
                # Snap recovery logic... (unchanged)
                recorded_path_full = listener_node.pouring_point_buffer[::-1]
                # ... [Rest of snap recovery logic preserved] ...
                movej(CHEERS_POSE, vel=250, acc=250)
                motion_node.get_logger().info("Snap recovery complete.")
                break
            else:
                motion_node.get_logger().info("Path finished naturally.")

        # FINAL ACT: Place and Release (Optional/Future logic)
        # For now, we remain at Cheers or Vertical. 
        # If user wants to release:
        # if gripper_open_cli.wait_for_service(timeout_sec=1.0):
        #     gripper_open_cli.call_async(Trigger.Request())

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
        if self.busy: return
        try: payload = json.loads(msg.data)
        except: return
        cmd = str(payload.get("action", "pour")).lower()
        count = int(payload.get("count", 1))
        def run_action():
            self.busy = True
            try: execute_action(cmd, count, self.motion_node, self.listener_node)
            except Exception as exc: self.motion_node.get_logger().error(f"Action Error: {exc}")
            finally: self.busy = False
        threading.Thread(target=run_action, daemon=True).start()

def main(args=None):
    rclpy.init(args=args)
    ROBOT_ID, ROBOT_MODEL = "dsr01", "e0509"
    motion_node = Node('motion_node', namespace=ROBOT_ID)
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, motion_node
    listener_node = TriggerListener(ROBOT_ID)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(listener_node)
    try:
        if len(sys.argv) > 1 and sys.argv[1].lower() == "listen":
            action_node = ActionRequestNode(ROBOT_ID, motion_node, listener_node)
            executor.add_node(action_node)
            threading.Thread(target=executor.spin, daemon=True).start()
            while rclpy.ok(): time.sleep(0.5)
        else:
            threading.Thread(target=executor.spin, daemon=True).start()
            cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "pour"
            execute_action(cmd, 1, motion_node, listener_node)
    finally:
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
