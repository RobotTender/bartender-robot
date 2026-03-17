import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class PlaceNode(Node):
    def __init__(self):
        super().__init__('robotender_place', namespace='/dsr01')
        self.srv = self.create_service(Trigger, 'robotender_place/start', self.place_callback)
        self.get_logger().info('--- Robotender Place Node Initialized ---')

    def place_callback(self, request, response):
        self.get_logger().info('!!! Received PLACE signal from Pour node !!!')
        response.success = True
        response.message = "Place signal received successfully"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = PlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
