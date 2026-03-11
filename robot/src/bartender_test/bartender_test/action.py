import rclpy
import sys
import threading
import time
from rclpy.node import Node
import DR_init
from .gripper_controller import GripperController
from std_msgs.msg import Empty
from dsr_msgs2.srv import MoveStop
from .defines import (HOME_POSE, CHEERS_POSE, CONTACT_POSE, POUR_HORIZONTAL, POUR_DIAGONAL, POUR_VERTICAL, POLE_POSE,
                            POS1_XYZ, POS2_XYZ, POS3_XYZ, POS4_XYZ, POS5_XYZ, DEFAULT_TARGET_POUR)

# Node B: Dedicated Listener Node
class TriggerListener(Node):
    def __init__(self, robot_id):
        super().__init__('trigger_listener', namespace=robot_id)
        self.trigger_received = False
        self.create_subscription(Empty, 'pouring_trigger', self.trigger_cb, 10)
        self.stop_cli = self.create_client(MoveStop, 'motion/move_stop')

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

    from DSR_ROBOT2 import (movej, movel, movesx, posx, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, 
                            get_current_posx, fkin, get_robot_state, get_workpiece_weight)

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
            path = [p1, p2, p3, p4, p5]

            motion_node.get_logger().info(f"Starting POUR ORBIT ({count} cycles)")
            for i in range(count):
                motion_node.get_logger().info(f"--- Pour Cycle {i+1}/{count} ---")
                movej(CHEERS_POSE, vel=60, acc=60)
                wait(0.2)
                
                listener_node.trigger_received = False # Reset flag
                motion_node.get_logger().info("Starting BLOCKING movesx path. Waiting for path completion or Spacebar...")
                
                # This is a BLOCKING call. It will only return when:
                # 1. Path is finished.
                # 2. move_stop() is called from the listener thread.
                movesx(path, vel=100, acc=100)
                
                # Check why we stopped
                if listener_node.trigger_received:
                    motion_node.get_logger().info("Path interrupted by Trigger. Executing SNAP RECOVERY...")
                    # Phase 6, Step 3: High-Acceleration Return
                    # Max safe velocity (250 deg/s) and acceleration (250 deg/s^2)
                    movej(CHEERS_POSE, vel=250, acc=250)
                    motion_node.get_logger().info("Snap to CHEERS_POSE complete.")
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
