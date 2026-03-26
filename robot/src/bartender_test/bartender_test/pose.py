import rclpy
import sys
from rclpy.node import Node
from dsr_msgs2.srv import MoveJoint
from .defines import (POSJ_HOME, POSJ_CHEERS, POSJ_PICK_PLACE_READY, POSJ_SNAP,
                      BOTTLE_CONFIG, SNAP_VELOCITY, SNAP_ACCELERATION)

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('pose_oneshot')
    
    # Create client
    cli = node.create_client(MoveJoint, '/dsr01/motion/move_joint')
    while not cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Service not available, waiting...')

    if len(sys.argv) < 2:
        print("Usage: ros2 run bartender_test pose [home|ready|cheers|snap|contact|horizontal|diagonal|vertical] [juice|beer|soju]")
        return

    pose_name = sys.argv[1].lower()
    bottle_type = sys.argv[2].lower() if len(sys.argv) > 2 else 'soju'
    
    target_joint_pose = None
    target_vel = 60.0
    target_acc = 60.0
    
    try:
        if pose_name == 'home':
            target_joint_pose = POSJ_HOME
        elif pose_name == 'ready':
            target_joint_pose = POSJ_PICK_PLACE_READY
        elif pose_name == 'cheers':
            target_joint_pose = POSJ_CHEERS
        elif pose_name == 'snap':
            target_joint_pose = POSJ_SNAP
            target_vel = SNAP_VELOCITY
            target_acc = SNAP_ACCELERATION
        elif pose_name in ['contact', 'horizontal', 'diagonal', 'vertical']:
            config = BOTTLE_CONFIG.get(bottle_type)
            if config:
                key = f'posj_{pose_name}'
                target_joint_pose = config.get(key)
        
        if target_joint_pose is None:
            print(f"Error: Unknown pose '{pose_name}' or bottle '{bottle_type}'")
            return

        # Send request
        req = MoveJoint.Request()
        req.pos = [float(x) for x in target_joint_pose]
        req.vel = target_vel
        req.acc = target_acc
        
        node.get_logger().info(f"Moving to {pose_name}...")
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() is not None:
            print(f"Successfully moved to {pose_name}")
        else:
            print("Service call failed")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
