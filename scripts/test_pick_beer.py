#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time

class BeerPickTester(Node):
    def __init__(self):
        super().__init__('beer_pick_tester')
        # ORDER_TOPIC is defined as "/bartender/order_detail" (global) in pick.py
        self.publisher_ = self.create_publisher(String, '/bartender/order_detail', 10)
        time.sleep(2)  # Wait for discovery

    def send_pick_request(self):
        # Based on pick.py process_grip, it expects recipe to be a dictionary
        # e.g., {"recipe": {"beer": 1}}
        data = {
            "recipe": {
                "beer": 1
            }
        }
        msg = String()
        msg.data = json.dumps(data)
        self.get_logger().info(f'Publishing pick request to /bartender/order_detail: {msg.data}')
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    tester = BeerPickTester()
    tester.send_pick_request()
    # Give it a moment to ensure the message is sent before shutting down
    time.sleep(1)
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
