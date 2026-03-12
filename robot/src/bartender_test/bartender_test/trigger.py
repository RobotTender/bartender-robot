import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import sys
import tty
import termios

class TriggerNode(Node):
    def __init__(self):
        super().__init__('trigger_node', namespace='dsr01')
        self.publisher_ = self.create_publisher(Empty, 'pouring_trigger', 10)
        self.get_logger().info("Manual SNAP Trigger Node started.")
        self.get_logger().info("Press SPACEBAR for Manual SNAP (Instant Recovery). Press 'q' to quit.")

    def run(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            while rclpy.ok():
                ch = sys.stdin.read(1)
                if ch == ' ':
                    self.publisher_.publish(Empty())
                    # Using print instead of logger for instant visual feedback
                    sys.stdout.write("\rTrigger: Manual SNAP Triggered!          \n")
                    sys.stdout.flush()
                elif ch.lower() == 'q':
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main(args=None):
    rclpy.init(args=args)
    node = TriggerNode()
    try:
        node.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
