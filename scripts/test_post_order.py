#!/usr/bin/env python3
import rclpy
import sys
from rclpy.node import Node
from std_msgs.msg import String
import json
import time

class OrderTester(Node):
    def __init__(self, item='beer'):
        super().__init__('order_tester')
        # ORDER_TOPIC is defined as "/bartender/order_detail" (global) in pick.py
        self.publisher_ = self.create_publisher(String, '/bartender/order_detail', 10)
        self.item = item
        time.sleep(2)  # Wait for discovery

    def send_order_request(self):
        # Based on pick.py process_grip, it expects recipe to be a dictionary
        # e.g., {"recipe": {"beer": 1}}
        data = {
            "recipe": {
                self.item: 1
            }
        }
        msg = String()
        msg.data = json.dumps(data)
        self.get_logger().info(f'Publishing order request to /bartender/order_detail: {msg.data}')
        self.publisher_.publish(msg)

def main(args=None):
    # Default item is beer
    item = 'beer'
    if len(sys.argv) > 1:
        item = sys.argv[1].lower()
        if item not in ['beer', 'soju', 'juice']:
            print(f"Unknown item: {item}. Choose one of [beer, soju, juice]")
            return

    rclpy.init(args=args)
    tester = OrderTester(item)
    tester.send_order_request()
    # Give it a moment to ensure the message is sent before shutting down
    time.sleep(1)
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
