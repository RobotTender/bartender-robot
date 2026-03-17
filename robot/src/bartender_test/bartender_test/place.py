import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from .defines import PICK_PLACE_READY

class PlaceNode(Node):
    def __init__(self):
        super().__init__('robotender_place', namespace='/dsr01')
        self.srv = self.create_service(Trigger, 'robotender_place/start', self.place_callback)
        self.get_logger().info('--- Robotender Place Node Initialized ---')

    async def place_callback(self, request, response):
        self.get_logger().info('!!! Received PLACE signal. Moving to PICK_PLACE_READY !!!')
        
        from DSR_ROBOT2 import (movej, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait)
        
        try:
            # 1. Set robot mode to autonomous
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)

            # 2. Move to PICK_PLACE_READY
            self.get_logger().info(f"Moving to PICK_PLACE_READY: {PICK_PLACE_READY}")
            movej(PICK_PLACE_READY, vel=60, acc=60)
            
            response.success = True
            response.message = "Moved to PICK_PLACE_READY"
        except Exception as e:
            self.get_logger().error(f"Place Error: {e}")
            response.success = False
            response.message = str(e)
            
        return response

def main(args=None):
    rclpy.init(args=args)
    node = PlaceNode()
    # Use empty ROBOT_ID to force relative resolution under /dsr01 namespace
    ROBOT_ID, ROBOT_MODEL = "", "e0509"
    import DR_init
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    DR_init.__dsr__node = node
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
