import rclpy
import sys
from rclpy.node import Node
import DR_init
from .gripper_controller import GripperController
from .defines import (HOME_POSE, CHEERS_POSE, CONTACT_POSE, POUR_HORIZONTAL, POUR_DIAGONAL, POUR_VERTICAL, POLE_POSE,
                            POS1_XYZ, POS2_XYZ, POS3_XYZ, POS4_XYZ, POS5_XYZ)

def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: ros2 run bartender_test action [pour orbit | warmup] [repeat=<n>]")
        return

    rclpy.init(args=args)
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    
    # Simple node for Doosan API
    main_node = Node('action_node', namespace=ROBOT_ID)
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, main_node

    from DSR_ROBOT2 import (movej, movel, movesx, posx, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, 
                            get_current_posx, fkin)
    
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
            main_node.get_logger().info(f"Starting WARMUP ({count} cycles)")
            poses = [("HOME", HOME_POSE), ("CHEERS", CHEERS_POSE), ("CONTACT", CONTACT_POSE), 
                     ("POUR_HORIZONTAL", POUR_HORIZONTAL), ("POUR_DIAGONAL", POUR_DIAGONAL), 
                     ("POUR_VERTICAL", POUR_VERTICAL), ("POLE", POLE_POSE)]
            for i in range(count):
                main_node.get_logger().info(f"--- Warmup Cycle {i+1}/{count} ---")
                for name, pose in poses:
                    main_node.get_logger().info(f"Moving to {name}...")
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

            main_node.get_logger().info(f"Starting POUR ORBIT ({count} cycles)")
            for i in range(count):
                main_node.get_logger().info(f"--- Pour Cycle {i+1}/{count} ---")
                movej(CHEERS_POSE, vel=60, acc=60)
                wait(0.2)
                movesx(path, vel=100, acc=100)
        
        elif cmd == 'pour' and sub_cmd == 'auto':
            # Target weight from argument (e.g., 'pour auto 50')
            target_poured_g = DEFAULT_TARGET_POUR
            if len(sys.argv) > 3 and sys.argv[3].replace('.','',1).isdigit():
                target_poured_g = float(sys.argv[3])
                
            main_node.get_logger().info(f"Starting PREDICTIVE AUTO POUR: Target={target_poured_g}g")
            
            # 1. Weighing
            main_node.get_logger().info("Moving to CHEERS_POSE and closing gripper (Force=250)...")
            from .gripper_controller import GripperController
            gripper = GripperController(main_node, namespace=ROBOT_ID)
            gripper.move(1150, force=250) # Ensure firm grasp
            movej(CHEERS_POSE, vel=60, acc=60)
            wait(2.0)
            
            from DSR_ROBOT2 import get_workpiece_weight
            weight_kg = get_workpiece_weight()
            if weight_kg < 0:
                main_node.get_logger().error("Weight sensing failed. Ensure robot is in REAL mode.")
                return
            
            weight_total_g = weight_kg * 1000.0
            from .defines import BOTTLE_EMPTY_WEIGHT
            liquid_start_g = weight_total_g - BOTTLE_EMPTY_WEIGHT
            
            main_node.get_logger().info(f"Sensed TOTAL Weight: {weight_total_g:.1f} g")
            main_node.get_logger().info(f"Calculated LIQUID Start: {liquid_start_g:.1f} g")
            
            # 2. Predicted Angle
            target_ry = -0.0433 * weight_total_g + 102.3888
            main_node.get_logger().info(f"Predicted First Impact Angle: {target_ry:.2f} deg")
            
            # 3. Trajectory Prep
            p_approach_raw = [float(x) for x in fkin(CONTACT_POSE, ref=0)]
            p_approach_raw[3] = -90.0
            p_approach_raw[4] = target_ry - 2.0 # 2-degree buffer for safety
            p_approach = posx(p_approach_raw)
            
            p_target_raw = list(p_approach_raw)
            p_target_raw[4] = target_ry + 10.0 # Allow deep tilt for low volumes
            p_target = posx(p_target_raw)
            
            # 4. Volume Monitoring Setup
            from std_msgs.msg import Float64
            poured_in_cup = 0.0
            def weight_cb(msg):
                nonlocal poured_in_cup
                poured_in_cup = msg.data
            
            sub = main_node.create_subscription(Float64, 'cup_poured_weight', weight_cb, 10)

            for i in range(count):
                main_node.get_logger().info(f"--- Auto Pour Cycle {i+1}/{count} ---")
                movej(CHEERS_POSE, vel=60, acc=60)
                wait(0.5)
                
                # Fast Approach
                movel(p_approach, vel=100, acc=100)
                
                # Precision Creep until first impact or volume match
                main_node.get_logger().info("Stage B: Precision Creep. Waiting for Trigger...")
                # We use a very slow move that we will INTERRUPT or let finish
                # In Doosan ROS, we can poll poured_in_cup
                
                poured_in_cup = 0.0 # Reset
                movel(p_target, vel=5, acc=5, asyncio=True) # Non-blocking tilt
                
                # Poll for volume target
                while rclpy.ok():
                    rclpy.spin_once(main_node, timeout_sec=0.01)
                    if poured_in_cup > 1.0: # Impact detected
                         main_node.get_logger().info(f"Impact detected! Flow: {poured_in_cup:.1f}g")
                         break
                    # If we reach the end of the tilt without impact
                    # (Safety stop handled by movel finishing)
                
                # Continue tilt until target weight is reached
                while rclpy.ok() and poured_in_cup < target_poured_g:
                    rclpy.spin_once(main_node, timeout_sec=0.01)
                    # We can adjust speed here if needed
                
                main_node.get_logger().info(f"Target {target_poured_g}g reached! Current: {poured_in_cup:.1f}g")
                
                # RECOVERY (Phase 5)
                from DSR_ROBOT2 import move_stop, DR_QSTOP_STO
                move_stop() # Stop current creep
                movej(CHEERS_POSE, vel=150, acc=150) # Rapid tilt back
                wait(1.0)
        
    except Exception as e:
        main_node.get_logger().error(f"Action Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
