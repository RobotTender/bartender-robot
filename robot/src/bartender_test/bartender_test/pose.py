import rclpy
import sys
from rclpy.node import Node
import DR_init
from .defines import (HOME_POSE, CHEERS_POSE, POLE_POSE, POUR_HORIZONTAL, 
                            POUR_DIAGONAL, POUR_VERTICAL, CONTACT_POSE, PICK_PLACE_READY,
                            POS1_XYZ, POS2_XYZ, POS3_XYZ, POS4_XYZ, POS5_XYZ)

def main(args=None):
    if len(sys.argv) < 2:
        print("Usage:")
        print("  ros2 run bartender_test pose home|pnp_ready|cheers|contact|pole")
        print("  ros2 run bartender_test pose horizontal|diagonal|vertical")
        print("  ros2 run bartender_test pose pos1|pos2|pos3|pos4|pos5")
        return

    rclpy.init(args=args)
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    node = rclpy.create_node('pose_node', namespace=ROBOT_ID)
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, node

    from DSR_ROBOT2 import (movej, movel, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, get_current_posj, get_current_posx)

    try:
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        wait(0.5)

        cmd = sys.argv[1].lower()
        
        target_joint_pose = None
        target_cart_pose = None

        if cmd == 'home':
            target_joint_pose = HOME_POSE
        elif cmd == 'pnp_ready':
            target_joint_pose = PICK_PLACE_READY
        elif cmd == 'cheers':
            target_joint_pose = CHEERS_POSE
        elif cmd == 'contact':
            target_joint_pose = CONTACT_POSE
        elif cmd in ['horizontal', 'pour_horizontal', 'pour_mid']:
            target_joint_pose = POUR_HORIZONTAL
        elif cmd in ['diagonal', 'pour_diagonal']:
            target_joint_pose = POUR_DIAGONAL
        elif cmd in ['vertical', 'pour_vertical', 'pour_end']:
            target_joint_pose = POUR_VERTICAL
        elif cmd == 'pole':
            target_joint_pose = POLE_POSE
        elif cmd in ['pos1', 'pos2', 'pos3', 'pos4', 'pos5']:
            if cmd == 'pos1': coords = POS1_XYZ
            elif cmd == 'pos2': coords = POS2_XYZ
            elif cmd == 'pos3': coords = POS3_XYZ
            elif cmd == 'pos4': coords = POS4_XYZ
            elif cmd == 'pos5': coords = POS5_XYZ
            
            # Use current orientation, update only XYZ
            current_posx = list(get_current_posx()[0])
            target_cart_pose = current_posx
            target_cart_pose[0] = float(coords[0])
            target_cart_pose[1] = float(coords[1])
            target_cart_pose[2] = float(coords[2])
        else:
            print(f"Unknown pose: {cmd}")
            return

        if target_joint_pose:
            node.get_logger().info(f"Moving to JOINT pose: {cmd} ({target_joint_pose})")
            ret = movej(target_joint_pose, vel=60, acc=60)
        else:
            node.get_logger().info(f"Moving to CARTESIAN pose: {cmd} ({target_cart_pose[:3]})")
            ret = movel(target_cart_pose, vel=[100.0, 100.0], acc=[100.0, 100.0])
            
        node.get_logger().info(f"Move complete with return code: {ret}")

        # Print current status for adjustment
        final_posj = get_current_posj()
        final_posx = get_current_posx()[0]
        node.get_logger().info(f"Final JOINTs: {[round(j, 2) for j in final_posj]}")
        node.get_logger().info(f"Final XYZ: {[round(x, 2) for x in final_posx[:3]]}")
        node.get_logger().info(f"Full posx: {[round(x, 2) for x in final_posx]}")

    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
