import rclpy
import sys
from rclpy.node import Node
import DR_init
from .defines import (POSJ_CHEERS, BOTTLE_CONFIG)

def main(args=None):
    if len(sys.argv) < 2:
        print("Usage:")
        print("  ros2 run bartender_test movel <dx_cm> <dy_cm> <dz_cm> [base|tool]  (Relative move)")
        print("  ros2 run bartender_test movel pos_cheers|pos_contact|pos_horizontal|pos_diagonal|pos_vertical (Defaults to soju)")
        print("\nExample:")
        print("  ros2 run bartender_test movel +3 -2 +9 base")
        print("  ros2 run bartender_test movel 0 0 1 tool")
        print("  ros2 run bartender_test movel pos_cheers")
        return

    rclpy.init(args=args)
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    node = rclpy.create_node('movel_node', namespace=ROBOT_ID)
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, node

    from DSR_ROBOT2 import (movel, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, get_current_posx, DR_BASE, DR_TOOL)

    try:
        # Must be in AUTONOMOUS mode to use movel
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        wait(0.5)

        arg1 = sys.argv[1].lower()
        ref_frame = DR_BASE
        
        if arg1 in ['pos_cheers', 'pos_contact', 'pos_horizontal', 'pos_diagonal', 'pos_vertical', 'pos1', 'pos2', 'pos3', 'pos4', 'pos5']:
            # Get current Cartesian position [X, Y, Z, A, B, C]
            current_posx = list(get_current_posx()[0])
            target_posx = list(current_posx)
            
            # Default to soju for manual commands
            soju_cfg = BOTTLE_CONFIG['soju']
            if arg1 in ['pos_cheers', 'pos1']: coords = POSJ_CHEERS
            elif arg1 in ['pos_contact', 'pos2']: coords = soju_cfg['pos_contact']
            elif arg1 in ['pos_horizontal', 'pos3']: coords = soju_cfg['pos_horizontal']
            elif arg1 in ['pos_diagonal', 'pos4']: coords = soju_cfg['pos_diagonal']
            elif arg1 in ['pos_vertical', 'pos5']: coords = soju_cfg['pos_vertical']
            
            target_posx[0] = float(coords[0])
            target_posx[1] = float(coords[1])
            target_posx[2] = float(coords[2])
            node.get_logger().info(f"Moving to absolute marker {arg1}: {target_posx[:3]}")
        else:
            if len(sys.argv) < 4:
                print("Relative move requires at least 3 arguments: dx dy dz (in cm)")
                return
            
            dx_mm = float(sys.argv[1]) * 10.0
            dy_mm = float(sys.argv[2]) * 10.0
            dz_mm = float(sys.argv[3]) * 10.0
            
            if len(sys.argv) >= 5:
                frame_arg = sys.argv[4].lower()
                if frame_arg == 'tool':
                    ref_frame = DR_TOOL
                elif frame_arg == 'base':
                    ref_frame = DR_BASE
                else:
                    print(f"Unknown reference frame '{frame_arg}', defaulting to base")

            if ref_frame == DR_BASE:
                current_posx = list(get_current_posx()[0])
                target_posx = list(current_posx)
                target_posx[0] += dx_mm
                target_posx[1] += dy_mm
                target_posx[2] += dz_mm
                node.get_logger().info(f"Moving relative (BASE): shift by {dx_mm}, {dy_mm}, {dz_mm} mm")
            else:
                # For DR_TOOL, we pass the relative offset [dx, dy, dz, 0, 0, 0]
                target_posx = [dx_mm, dy_mm, dz_mm, 0.0, 0.0, 0.0]
                node.get_logger().info(f"Moving relative (TOOL): shift by {dx_mm}, {dy_mm}, {dz_mm} mm")

        # Execute linear move
        # Using vel=100 mm/s and acc=100 mm/s^2 for safe, smooth movement
        ret = movel(target_posx, vel=[100.0, 100.0], acc=[100.0, 100.0], ref=ref_frame)
        
        node.get_logger().info(f"Move complete with return code: {ret}")
        
        # Print final position
        final_posx = get_current_posx()[0]
        node.get_logger().info(f"Final XYZ (Base): {[round(x, 2) for x in final_posx[:3]]}")

    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
