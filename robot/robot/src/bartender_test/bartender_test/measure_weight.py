import rclpy
import sys
import os
import time
from rclpy.node import Node
import DR_init
from .gripper_controller import GripperController

# Constants
TARE_FILE = "/tmp/robot_tare_raw_torque.txt"
CALIB_FILE = "/tmp/robot_weight_multiplier.txt"
HOME_POSE = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
BOTTLE_POSE = [0.0, -45.0, 135.0, 90.0, 0.0, 0.0]

def get_tare_torques():
    if os.path.exists(TARE_FILE):
        with open(TARE_FILE, 'r') as f:
            try: return [float(x) for x in f.read().strip().split(',')]
            except: pass
    return [0.0] * 6

def save_tare_torques(torques):
    with open(TARE_FILE, 'w') as f:
        f.write(','.join([str(x) for x in torques]))

def get_multiplier():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'r') as f:
            try: return float(f.read().strip())
            except: pass
    return 135.0 # Default fallback

def save_multiplier(val):
    with open(CALIB_FILE, 'w') as f:
        f.write(str(val))

def wait_for_stability(node, get_torque_func, max_wait=10.0, threshold=0.02):
    node.get_logger().info(f"Stabilizing (max {max_wait}s)...")
    start_time = time.time()
    last_t = get_torque_func()
    stable_count = 0
    while (time.time() - start_time) < max_wait:
        curr_t = get_torque_func()
        if not isinstance(curr_t, list): 
            time.sleep(0.1)
            continue
        # J2, J3, J5
        diff = sum([abs(curr_t[i] - last_t[i]) for i in [1, 2, 4]])
        if diff < (threshold * 3): stable_count += 1
        else: stable_count = 0
        if stable_count >= 10:
            node.get_logger().info(f"Settled in {time.time() - start_time:.1f}s.")
            return True
        last_t = curr_t
        time.sleep(0.1)
    return False

def sample_torques(get_torque_func, count=100):
    avg = [0.0] * 6
    for _ in range(count):
        t = get_torque_func()
        if isinstance(t, list):
            for j in range(6): avg[j] += t[j]
        time.sleep(0.01)
    return [x/float(count) for x in avg]

def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: [tare | first <force> | again | calibrate <grams>]")
        return

    rclpy.init(args=args)
    node = rclpy.create_node('measure_weight_node', namespace="dsr01")
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = "dsr01", "e0509", node
    from DSR_ROBOT2 import get_joint_torque, set_robot_mode, ROBOT_MODE_AUTONOMOUS, movej, wait

    try:
        cmd = sys.argv[1].lower()
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        gripper = GripperController(node=node, namespace="dsr01")

        if cmd == 'tare':
            node.get_logger().info("GOLD STANDARD TARE...")
            gripper.move(0) 
            movej(HOME_POSE, vel=30, acc=30)
            movej(BOTTLE_POSE, vel=20, acc=20)
            wait_for_stability(node, get_joint_torque)
            save_tare_torques(sample_torques(get_joint_torque))
            print("TARE COMPLETE.")

        elif cmd in ['first', 'again', 'calibrate']:
            if cmd == 'first':
                force = int(sys.argv[2]) if len(sys.argv) >= 3 else 250
                node.get_logger().info(f"FIRST: Sequence Open -> Close with force {force}...")
                # Combined sequence for better reliability
                gripper.move_sequence([0, 700], current=force)
                wait_for_stability(node, get_joint_torque)
            elif cmd == 'again':
                node.get_logger().info("AGAIN: Measuring current state...")
                wait_for_stability(node, get_joint_torque, max_wait=3.0)

            curr_torques = sample_torques(get_joint_torque)
            tare_torques = get_tare_torques()
            total_diff = sum([abs(curr_torques[i] - tare_torques[i]) for i in [1, 2, 4]])
            
            if cmd == 'calibrate':
                if len(sys.argv) < 3:
                    print("Provide real weight in grams.")
                    return
                real_g = float(sys.argv[2])
                new_mult = real_g / total_diff if total_diff > 0 else 135.0
                save_multiplier(new_mult)
                print(f"CALIBRATED Multiplier: {new_mult:.2f}")
            else:
                multiplier = get_multiplier()
                print("-" * 30)
                print(f">> WEIGHT: {total_diff * multiplier:.1f} g")
                print("-" * 30)

    except Exception as e: node.get_logger().error(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
