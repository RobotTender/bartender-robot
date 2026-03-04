import rclpy
import sys
from rclpy.node import Node
from dsr_msgs2.srv import SetRobotMode
import DR_init

def main(args=None):
    # Named poses
    cheers_pose = [0.0, -45.0, 135.0, 90.0, 0.0, 0.0]
    home_pose = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
    pole_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Was 'zero'
    
    target_pose = None
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  ros2 run my_dsr_control robot_move home")
        print("  ros2 run my_dsr_control robot_move cheers")
        print("  ros2 run my_dsr_control robot_move pole")
        print("  ros2 run my_dsr_control robot_move j<num> [rel|abs] <val>")
        return

    rclpy.init(args=args)
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    
    node = rclpy.create_node('robot_move_node', namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movej, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait, get_current_posj

    try:
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        wait(0.5)

        cmd = sys.argv[1].lower()
        
        # --- Handle Poses ---
        if cmd == 'home':
            target_pose = home_pose
        elif cmd == 'cheers':
            target_pose = cheers_pose
        elif cmd == 'pole':
            target_pose = pole_pose
            
        # --- Handle Joint Commands ---
        elif cmd.startswith('j') and len(cmd) <= 2 and cmd[1:].isdigit():
            joint_index = int(cmd[1:]) - 1
            if 0 <= joint_index <= 5 and len(sys.argv) >= 4:
                mode = sys.argv[2].lower()
                val = float(sys.argv[3])
                current_pose = get_current_posj()
                target_pose = list(current_pose)
                if mode == 'rel': target_pose[joint_index] += val
                else: target_pose[joint_index] = val
            else:
                print("Invalid joint command format.")
                return

        # --- Handle raw 6-joint list ---
        elif len(sys.argv) >= 7:
            try:
                target_pose = [float(x) for x in sys.argv[1:7]]
            except:
                print("Invalid joint list.")
                return
        else:
            print(f"Unknown command: {cmd}")
            return

        # Execute
        if target_pose:
            node.get_logger().info(f"Moving to: {target_pose}")
            movej(target_pose, vel=20, acc=20)
            node.get_logger().info("Move complete.")

    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
