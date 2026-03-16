import rclpy
import time
from rclpy.node import Node
from dsr_msgs2.srv import DrlStart
from std_srvs.srv import Trigger
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# DRL Template for immediate execution (One-Shot)
# This handles the serial port and Modbus commands in one go.
DRL_GRIPPER_TEMPLATE = """
def modbus_fc16(slaveid, startaddress, cnt, valuelist):
    data_list = [slaveid, 16, (startaddress >> 8) & 0xFF, startaddress & 0xFF, (cnt >> 8) & 0xFF, cnt & 0xFF, (2 * cnt) & 0xFF]
    for i in range(0, cnt):
        data_list.append((valuelist[i] >> 8) & 0xFF)
        data_list.append(valuelist[i] & 0xFF)
    return bytes(modbus_send_make(bytes(data_list)))

def modbus_fc06(slaveid, address, value):
    data_list = [slaveid, 6, (address >> 8) & 0xFF, address & 0xFF, (value >> 8) & 0xFF, value & 0xFF]
    return bytes(modbus_send_make(bytes(data_list)))

# 1. Open Port
# (Removed initial close to avoid alarm 2018 if already closed)
flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
wait(0.3)

# 2. Torque Enable (Slave 1, Addr 256, Val 1)
flange_serial_write(modbus_fc06(1, 256, 1))
wait(0.2)

# 3. Set Velocity (Slave 1, Addr 270, Val 255 - Max)
flange_serial_write(modbus_fc06(1, 270, 255))
wait(0.2)

# 4. Set Force (Slave 1, Addr 275, Val {force})
flange_serial_write(modbus_fc06(1, 275, {force}))
wait(0.2)

# 5. Move (Slave 1, Addr 282, Val {stroke})
# Stroke: 1100 (Open), 0 (Closed)
flange_serial_write(modbus_fc16(1, 282, 2, [{stroke}, 0]))
wait(1.5)

# 6. Cleanup
flange_serial_close()
"""

class GripperNode(Node):
    def __init__(self):
        super().__init__('robotender_gripper', namespace='/')
        self.callback_group = ReentrantCallbackGroup()
        
        # DRL Start Client (Executes raw DRL code)
        self.drl_cli = self.create_client(
            DrlStart, 
            '/dsr01/drl/drl_start',
            callback_group=self.callback_group
        )
        
        # ROS Services
        self.open_srv = self.create_service(Trigger, '/robotender/gripper/open', self.open_callback, callback_group=self.callback_group)
        self.close_srv = self.create_service(Trigger, '/robotender/gripper/close', self.close_callback, callback_group=self.callback_group)
        
        self.get_logger().info('--- Autonomous Gripper Node (One-Shot Mode) Initialized ---')

    async def execute_drl(self, stroke, force=400):
        if not self.drl_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('DRL Start service not available!')
            return False
        
        code = DRL_GRIPPER_TEMPLATE.format(stroke=stroke, force=force)
        req = DrlStart.Request()
        req.robot_system = 0 # Real/Virtual depending on bringup
        req.code = code
        
        self.get_logger().info(f'Executing Gripper DRL: Stroke={stroke}, Force={force}')
        try:
            future = self.drl_cli.call_async(req)
            result = await future
            return result.success
        except Exception as e:
            self.get_logger().error(f'DRL execution failed: {e}')
            return False

    async def open_callback(self, request, response):
        # Open (0 stroke for this configuration)
        success = await self.execute_drl(0, force=400)
        response.success = success
        response.message = "Open command (One-Shot) finished" if success else "Failed to execute DRL"
        return response

    async def close_callback(self, request, response):
        # Close (1100 stroke for this configuration)
        success = await self.execute_drl(1100, force=800) 
        response.success = success
        response.message = "Close command (One-Shot) finished" if success else "Failed to execute DRL"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GripperNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
