import rclpy
import sys
import os
import time
import math
from rclpy.node import Node
import DR_init
from .gripper_controller import GripperController

# Constants
TARE_FILE = "/tmp/robot_tare_raw_torque.txt"
HOME_POSE = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
BOTTLE_POSE = [0.0, -45.0, 135.0, 90.0, 0.0, 0.0]

def get_tare_torques():
    if os.path.exists(TARE_FILE):
        with open(TARE_FILE, 'r') as f:
            try:
                return [float(x) for x in f.read().strip().split(',')]
            except:
                return [0.0] * 6
    return [0.0] * 6

def save_tare_torques(torques):
    with open(TARE_FILE, 'w') as f:
        f.write(','.join([str(x) for x in torques]))

def wait_for_stability(node, get_torque_func, max_wait=10.0, threshold=0.02):
    """
    Monitor torques and return as soon as they stop fluctuating.
    threshold: Max change in Nm between samples to consider 'stable'.
    """
    node.get_logger().info(f"Active stabilization (max {max_wait}s)...")
    start_time = time.time()
    last_torques = get_torque_func()
    
    stable_count = 0
    required_stable_samples = 10 # Must be stable for 1 second (10 * 0.1s)
    
    while (time.time() - start_time) < max_wait:
        current_torques = get_torque_func()
        
        # Check change in J2, J3, and J5
        diff_j2 = abs(current_torques[1] - last_torques[1])
        diff_j3 = abs(current_torques[2] - last_torques[2])
        diff_j5 = abs(current_torques[4] - last_torques[4])
        
        if diff_j2 < threshold and diff_j3 < threshold and diff_j5 < threshold:
            stable_count += 1
        else:
            stable_count = 0 # Reset if it moves
            
        if stable_count >= required_stable_samples:
            elapsed = time.time() - start_time
            node.get_logger().info(f"Settled in {elapsed:.1f}s.")
            return True
            
        last_torques = current_torques
        time.sleep(0.1)
        
    node.get_logger().warn("Timed out waiting for stability. Proceeding anyway.")
    return False

def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: ros2 run my_dsr_control measure_weight [tare | first <force> | again]")
        return

    rclpy.init(args=args)
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    
    node = rclpy.create_node('measure_weight_node', namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import get_joint_torque, set_robot_mode, ROBOT_MODE_AUTONOMOUS, movej, wait

    gripper = None
    try:
        cmd = sys.argv[1].lower()
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        wait(0.5)
        gripper = GripperController(node=node, namespace=ROBOT_ID)

        if cmd == 'tare':
            gripper.move(0) 
            movej(HOME_POSE, vel=30, acc=30)
            movej(BOTTLE_POSE, vel=20, acc=20)
            
            wait_for_stability(node, get_joint_torque)
            
            avg_torques = [0.0] * 6
            for _ in range(100):
                t = get_joint_torque()
                for j in range(6): avg_torques[j] += t[j]
                time.sleep(0.01)
            avg_torques = [x/100.0 for x in avg_torques]
            save_tare_torques(avg_torques)
            print(f"TARE COMPLETE. Baseline: J2={avg_torques[1]:.2f}")

        elif cmd == 'first':
            force = int(sys.argv[2]) if len(sys.argv) >= 3 else 250
            gripper.move(0)
            wait(0.5)
            gripper.move(700, current=force)
            
            wait_for_stability(node, get_joint_torque)
            
            curr_torques = [0.0] * 6
            for _ in range(100):
                t = get_joint_torque()
                for j in range(6): curr_torques[j] += t[j]
                time.sleep(0.01)
            curr_torques = [x/100.0 for x in curr_torques]
            
            tare_torques = get_tare_torques()
            diff_j2 = abs(curr_torques[1] - tare_torques[1])
            diff_j3 = abs(curr_torques[2] - tare_torques[2])
            diff_j5 = abs(curr_torques[4] - tare_torques[4])
            
            # Adjusted multiplier for J2+J3+J5
            est_weight_g = (diff_j2 + diff_j3 + diff_j5) * 125 
            
            print("-" * 30)
            print(f">> WEIGHT: {est_weight_g:.1f} g")
            print("-" * 30)

        elif cmd == 'again':
            # Fast measure
            curr_torques = [0.0] * 6
            for _ in range(50): # Fewer samples for 'again'
                t = get_joint_torque()
                for j in range(6): curr_torques[j] += t[j]
                time.sleep(0.01)
            curr_torques = [x/50.0 for x in curr_torques]
            
            tare_torques = get_tare_torques()
            diff_j2 = abs(curr_torques[1] - tare_torques[1])
            diff_j3 = abs(curr_torques[2] - tare_torques[2])
            diff_j5 = abs(curr_torques[4] - tare_torques[4])
            est_weight_g = (diff_j2 + diff_j3 + diff_j5) * 125 
            print(f">> WEIGHT (AGAIN): {est_weight_g:.1f} g")

    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        if gripper:
            gripper.terminate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
