import rclpy
import sys
from rclpy.node import Node
import DR_init
from .defines import (HOME_POSE, CHEERS_POSE, POLE_POSE, PICK_PLACE_READY,
                            POS_CHEERS, BOTTLE_CONFIG)

def main(args=None):
    if len(sys.argv) < 2:
        print("Usage:")
        print("  ros2 run bartender_test pose home|pnp_ready|cheers|pole")
        print("  ros2 run bartender_test pose contact|horizontal|diagonal|vertical [juice|beer|soju]")
        return

    rclpy.init(args=args)
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    node = rclpy.create_node('pose_node', namespace=ROBOT_ID)
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, node

    from DSR_ROBOT2 import (movej, movel, posx, posj, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, get_current_posj, get_current_posx, fkin)

    try:
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        wait(0.5)

        cmd = sys.argv[1].lower()
        bottle = sys.argv[2].lower() if len(sys.argv) > 2 else 'soju'
        
        target_joint_pose = None
        target_cart_pose = None

        if cmd == 'home':
            target_joint_pose = HOME_POSE
        elif cmd == 'pnp_ready':
            target_joint_pose = PICK_PLACE_READY
        elif cmd == 'cheers':
            target_joint_pose = CHEERS_POSE
        elif cmd == 'pole':
            target_joint_pose = POLE_POSE
        elif cmd in ['contact', 'horizontal', 'diagonal', 'vertical']:
            # Get bottle-specific configuration
            config = BOTTLE_CONFIG.get(bottle)
            if config is None:
                print(f"Unknown bottle type: {bottle}")
                return
            
            # Map command to config key
            x_key = f'posx_{cmd}'
            j_key = f'posj_{cmd}'
            
            if x_key in config:
                target_cart_pose = config[x_key]
                node.get_logger().info(f"Targeting {cmd} for {bottle} via {x_key}")
            else:
                print(f"Key {x_key} not found for {bottle}")
                return
        else:
            print(f"Unknown pose: {cmd}")
            return

        if target_joint_pose:
            node.get_logger().info(f"Moving to JOINT pose: {cmd} ({target_joint_pose})")
            ret = movej(target_joint_pose, vel=30, acc=30)
        else:
            node.get_logger().info(f"Moving to CARTESIAN pose: {cmd} ({target_cart_pose[:3]})")
            ret = movel(target_cart_pose, vel=[30.0, 30.0], acc=[30.0, 30.0])
            
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
