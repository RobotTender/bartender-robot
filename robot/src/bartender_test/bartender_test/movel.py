import rclpy
import sys
from rclpy.node import Node
import DR_init

def main(args=None):
    if len(sys.argv) != 4:
        print("Usage:")
        print("  ros2 run bartender_test movel <dx_cm> <dy_cm> <dz_cm>")
        print("\nExample (Move +3cm X, -2cm Y, +9cm Z):")
        print("  ros2 run bartender_test movel +3 -2 +9")
        return

    rclpy.init(args=args)
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    node = rclpy.create_node('movel_node', namespace=ROBOT_ID)
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, node

    from DSR_ROBOT2 import (movel, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, get_current_posx)

    try:
        dx_cm = float(sys.argv[1])
        dy_cm = float(sys.argv[2])
        dz_cm = float(sys.argv[3])
        
        # Convert cm to mm
        dx_mm = dx_cm * 10.0
        dy_mm = dy_cm * 10.0
        dz_mm = dz_cm * 10.0

        # Must be in AUTONOMOUS mode to use movel
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        wait(0.5)

        # Get current Cartesian position [X, Y, Z, A, B, C]
        # posx is typically wrapped in a tuple/list, we take the [0] element if it returns a tuple with status, 
        # but in DSR API get_current_posx()[0] gives the coordinates (a list of 6 floats).
        current_posx = get_current_posx()[0]
        
        node.get_logger().info(f"Current posx: {current_posx}")

        # Modify only the X, Y, Z components
        target_posx = list(current_posx)
        target_posx[0] += dx_mm
        target_posx[1] += dy_mm
        target_posx[2] += dz_mm

        node.get_logger().info(f"Target posx (shifting by {dx_mm}mm, {dy_mm}mm, {dz_mm}mm): {target_posx}")

        # Execute linear move keeping orientation exactly the same
        # Using vel=100 mm/s and acc=100 mm/s^2 for safe, smooth movement
        ret = movel(target_posx, vel=[100.0, 100.0], acc=[100.0, 100.0])
        
        node.get_logger().info(f"Move complete with return code: {ret}")

    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
