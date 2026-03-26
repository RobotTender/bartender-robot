#!/usr/bin/env python3
import rclpy
import sys
import math
import os
import csv
import time
import collections
import select
import termios
import tty
from rclpy.node import Node
from std_msgs.msg import Empty, Float64MultiArray, Float32
from sensor_msgs.msg import JointState
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

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

class UnifiedMonitor(Node):
    def __init__(self):
        super().__init__('robotender_monitor', namespace='dsr01')
        
        # Real-time Data (10Hz)
        self.joint_pos = [0.0] * 6
        self.tcp_pose = [0.0] * 6
        self.liquid_volume = 0.0
        
        # Static Data (Update only when stable)
        self.motor_data_stable = [0.0] * 6
        self.tcp_force_stable = [0.0] * 6
        self.weight_jts_stable = 0.0
        self.weight_tcp_stable = 0.0
        
        # Live Buffers
        self.window_size = 20
        self.motor_buffers = [collections.deque(maxlen=self.window_size) for _ in range(6)]
        self.force_buffers = [collections.deque(maxlen=self.window_size) for _ in range(6)]
        self.tare_motor = [0.0] * 6
        self.tare_force = [0.0] * 6
        
        # Stability tracking
        self.is_stable = False
        self.stable_start_time = None
        self.STABILITY_THRESHOLD = 0.001 # rad/s
        self.STABILITY_DURATION = 1.0    # seconds
        
        # Topics
        self.topic_map = {
            'jts_mot': '/rt_topic/actual_motor_torque',
            'tcp_force': '/rt_topic/external_tcp_force'
        }
        
        # Subscriptions
        self.sub_js = self.create_subscription(JointState, 'joint_states', self.cb_joint_states, 10)
        self.sub_mot = self.create_subscription(Float64MultiArray, self.topic_map['jts_mot'], self.cb_mot, 10)
        self.sub_force = self.create_subscription(Float64MultiArray, self.topic_map['tcp_force'], self.cb_force, 10)
        self.sub_volume = self.create_subscription(Float32, '/dsr01/robotender/liquid_volume', self.cb_volume, 10)
        
        self.msg_count = 0
        self.last_msg_time = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.loop)
        
        # Terminal handling for interactive input (Optional if no TTY)
        self.is_tty = sys.stdin.isatty()
        if self.is_tty:
            try:
                self.settings = termios.tcgetattr(sys.stdin)
            except Exception:
                self.is_tty = False
        
        os.system('clear')

    def cb_joint_states(self, msg):
        if len(msg.position) >= 6:
            self.joint_pos = [math.degrees(v) for v in msg.position[:6]]
        if len(msg.velocity) >= 6:
            moving = any(abs(v) > self.STABILITY_THRESHOLD for v in msg.velocity[:6])
            if moving:
                self.is_stable = False
                self.stable_start_time = None
            else:
                if self.stable_start_time is None:
                    self.stable_start_time = time.time()
                elif time.time() - self.stable_start_time > self.STABILITY_DURATION:
                    self.is_stable = True
                    self.update_stable_values()

    def cb_mot(self, msg):
        self.msg_count += 1
        self.last_msg_time = time.time()
        for i in range(len(msg.data)):
            self.motor_buffers[i].append(msg.data[i])

    def cb_force(self, msg):
        for i in range(len(msg.data)):
            self.force_buffers[i].append(msg.data[i])

    def cb_volume(self, msg):
        self.liquid_volume = msg.data

    def update_stable_values(self):
        self.motor_data_stable = [sum(b)/len(b) if b else 0.0 for b in self.motor_buffers]
        self.tcp_force_stable = [sum(b)/len(b) if b else 0.0 for b in self.force_buffers]
        
        # 1. JTS Based Weight
        w_j2 = abs((self.motor_data_stable[1] - self.tare_motor[1]) / (0.4 * 9.81)) * 1000.0
        w_j3 = abs((self.motor_data_stable[2] - self.tare_motor[2]) / (0.4 * 9.81)) * 1000.0
        self.weight_jts_stable = (w_j2 + w_j3) / 2.0

        # 2. TCP Based Weight (Raw sensor)
        net_fz = self.tcp_force_stable[2] - self.tare_force[2]
        self.weight_tcp_stable = abs(net_fz / 9.81) * 1000.0

    def get_tf_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'link_6', rclpy.time.Time())
            x, y, z = trans.transform.translation.x * 1000.0, trans.transform.translation.y * 1000.0, trans.transform.translation.z * 1000.0
            rx, ry, rz = euler_from_quaternion(trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w)
            return [x, y, z, math.degrees(rx), math.degrees(ry), math.degrees(rz)]
        except: return None

    def loop(self):
        pose = self.get_tf_pose()
        if pose: self.tcp_pose = pose
        self.check_input()
        self.display()

    def check_input(self):
        if not self.is_tty:
            return
            
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            key = sys.stdin.read(1)
            if key.lower() == 't':
                for i in range(6):
                    self.tare_motor[i] = sum(self.motor_buffers[i])/len(self.motor_buffers[i]) if self.motor_buffers[i] else 0.0
                    self.tare_force[i] = sum(self.force_buffers[i])/len(self.force_buffers[i]) if self.force_buffers[i] else 0.0
                self.update_stable_values()
                print("\nTARE COMPLETE.")
                time.sleep(0.5)

    def display(self):
        # We only clear the screen if in TTY
        if self.is_tty:
            print("\033[H\033[J", end="")
        
        latency = (time.time() - self.last_msg_time) if self.last_msg_time > 0 else 999.0
        data_status = f"\033[92mOK ({self.msg_count})\033[0m" if latency < 1.0 else f"\033[91mSTALE ({latency:.1f}s)\033[0m"
        status_box = "\033[92m[STABLE]\033[0m" if self.is_stable else "\033[93m[MOVING]\033[0m"
        
        # In launch mode, we prefix the output if no TTY to avoid excessive scrolling
        prefix = ""
        if not self.is_tty:
            # Only print periodically in non-TTY mode to reduce log size
            if self.msg_count % 100 != 0: return 
            prefix = "[Monitor] "

        print(f"{prefix}" + "="*75)
        print(f"{prefix} UNIFIED ROBOT MONITOR | Status: {status_box} | Data: {data_status}")
        print(f"{prefix}" + "="*75)

        # 1. Joints
        print(f"{prefix} [JOINTS] (deg)")
        print(f"{prefix} J1:{self.joint_pos[0]:>8.2f} | J2:{self.joint_pos[1]:>8.2f} | J3:{self.joint_pos[2]:>8.2f}")
        print(f"{prefix} J4:{self.joint_pos[3]:>8.2f} | J5:{self.joint_pos[4]:>8.2f} | J6:{self.joint_pos[5]:>8.2f}")
        print(f"{prefix}" + "-" * 75)

        # 2. Pose
        print(f"{prefix} [POSE] (mm, deg)")
        print(f"{prefix} X :{self.tcp_pose[0]:>8.2f} | Y :{self.tcp_pose[1]:>8.2f} | Z :{self.tcp_pose[2]:>8.2f}")
        print(f"{prefix} RX:{self.tcp_pose[3]:>8.2f} | RY:{self.tcp_pose[4]:>8.2f} | RZ:{self.tcp_pose[5]:>8.2f}")
        print(f"{prefix}" + "-" * 75)

        # 3. Feel
        print(f"{prefix} [FEEL] {status_box if self.is_stable else '(Holding Last Stable Values)'}")
        
        # JTS Sub-section
        net_j2 = self.motor_data_stable[1] - self.tare_motor[1]
        net_j3 = self.motor_data_stable[2] - self.tare_motor[2]
        print(f"{prefix} JTS (Arm&Shoulder) | J2 Net:{net_j2:>7.3f} | J3 Net:{net_j3:>7.3f} Nm")
        
        # TCP Sub-section
        net_fx = self.tcp_force_stable[0] - self.tare_force[0]
        net_fy = self.tcp_force_stable[1] - self.tare_force[1]
        net_fz = self.tcp_force_stable[2] - self.tare_force[2]
        
        tcp_is_dead = all(abs(v) < 0.0001 for v in [net_fx, net_fy, net_fz])
        fx_str = f"{net_fx:>7.3f}" if not tcp_is_dead else "    N/A"
        fy_str = f"{net_fy:>7.3f}" if not tcp_is_dead else "    N/A"
        fz_str = f"{net_fz:>7.3f}" if not tcp_is_dead else "    N/A"
        
        print(f"{prefix} TCP (Fingers)      | Fx:{fx_str} | Fy:{fy_str} | Fz:{fz_str} N")
        
        unified_weight = self.weight_tcp_stable if not tcp_is_dead else self.weight_jts_stable
        source_tag = "(TCP Sensor)" if not tcp_is_dead else "(JTS Estimate)"
        
        print(f"{prefix} UNIFIED WEIGHT     | \033[92m{unified_weight:>7.1f} g\033[0m {source_tag}")
        print(f"{prefix} LIQUID VOLUME    | \033[96m{self.liquid_volume:>7.1f} ml\033[0m (Vision)")
        print(f"{prefix}" + "-" * 75)

        if self.is_tty:
            print(" Press 'T' to Tare | Ctrl+C to stop")
        print(f"{prefix}" + "="*75)

def main():
    rclpy.init()
    node = UnifiedMonitor()
    
    # Conditional terminal setup
    fd = sys.stdin.fileno()
    if node.is_tty:
        tty.setcbreak(fd)
        
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.is_tty:
            termios.tcsetattr(fd, termios.TCSADRAIN, node.settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
