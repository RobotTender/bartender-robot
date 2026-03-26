import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import sys
import tty
import termios
import threading

class TriggerNode(Node):
    def __init__(self):
        super().__init__('robotender_snap', namespace='/dsr01')
        self.publisher_ = self.create_publisher(Empty, 'robotender_snap/trigger', 10)
        
        # Subscribe to our own topic to monitor ANY triggers (Manual or Vision)
        self.subscription = self.create_subscription(
            Empty,
            'robotender_snap/trigger',
            self.trigger_callback,
            10
        )
        
        self.get_logger().info("Snap Trigger Node (Manual + Monitor) started.")
        self.get_logger().info("Press SPACEBAR for Manual SNAP. Monitoring for Vision Trigger from Camera...")

    def trigger_callback(self, msg):
        # This will be called whenever Camera 2 OR the Spacebar sends a trigger
        sys.stdout.write("\r>>> SNAP TRIGGER DETECTED! (Vision or Manual) <<<          \n")
        sys.stdout.flush()

    def run(self):
        # Start ROS 2 spinning in a background thread so it can receive messages
        # while we block on keyboard input
        spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        spin_thread.start()

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            while rclpy.ok():
                ch = sys.stdin.read(1)
                if ch == ' ':
                    # Publishing here will also trigger our own trigger_callback
                    self.publisher_.publish(Empty())
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
        # No need for node.destroy_node() here as we are shutting down
        rclpy.shutdown()

if __name__ == '__main__':
    main()
