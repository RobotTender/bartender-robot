import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import DrlStart, DrlStop, SetRobotControl, GetLastAlarm
import time

def get_drl_code(stroke, force):
    cmd_val = 0 if stroke == 0 else 700
    
    return f"""
# Configuration (Must match hardware)
BAUD_RATE = 115200 
SLAVE_ID = 9

res = flange_serial_open(BAUD_RATE, DR_EIGHTBITS, DR_PARITY_NONE, DR_STOPBITS_ONE)
if res == -1:
    tp_log("Failed to open flange serial!")

def modbus_fc06(address, value):
    data_list = [SLAVE_ID, 6, (address >> 8) & 0xFF, address & 0xFF, (value >> 8) & 0xFF, value & 0xFF]
    return bytes(modbus_send_make(data_list))

def modbus_fc16(startaddress, cnt, valuelist):
    data_list = [SLAVE_ID, 16, (startaddress >> 8) & 0xFF, startaddress & 0xFF, (cnt >> 8) & 0xFF, cnt & 0xFF, (2 * cnt) & 0xFF]
    for i in range(0, cnt):
        data_list.append((valuelist[i] >> 8) & 0xFF)
        data_list.append(valuelist[i] & 0xFF)
    return bytes(modbus_send_make(data_list))

# 1. Enable Torque
flange_serial_write(modbus_fc06(256, 1))
wait(0.1)

# 2. Set Force
flange_serial_write(modbus_fc06(275, {force}))
wait(0.1)

# 3. Action Command
flange_serial_write(modbus_fc16(282, 2, [{cmd_val}, 0]))

# 4. Cleanup
flange_serial_close()
"""

class GripperController:
    def __init__(self, node: Node, namespace: str = "dsr01", robot_system: int = 1):
        self.node = node
        self.namespace = namespace
        self.robot_system = robot_system 
        self.cli = self.node.create_client(DrlStart, f"/{namespace}/drl/drl_start")
        self.stop_cli = self.node.create_client(DrlStop, f"/{namespace}/drl/drl_stop")
        
        # Stop any old background loops hanging around
        self.node.get_logger().info("Stopping legacy background DRL tasks...")
        self._wait_for_future(self.stop_cli.call_async(DrlStop.Request(stop_mode=1)), timeout=2.0)
        self.node.get_logger().info("Ready for synchronous Action/Pick workflow.")

    def _wait_for_future(self, future, timeout):
        start = time.time()
        while not future.done() and (time.time() - start < timeout):
            time.sleep(0.05)
        return future.result() if future.done() else None

    def action(self, force: int, stroke: int) -> bool:
        cmd_name = "OPEN" if stroke == 0 else f"CLOSE(Force:{force})"
        self.node.get_logger().info(f"Executing Synchronous Gripper Action: {cmd_name}...")
        
        code = get_drl_code(stroke, force)

        
        req = DrlStart.Request(robot_system=self.robot_system, code=code)
        res = self._wait_for_future(self.cli.call_async(req), timeout=5.0)
        
        if res and res.success:
            self.node.get_logger().info(f"Gripper {cmd_name} sent successfully.")
            return True
        
        self.node.get_logger().error(f"Failed to execute Gripper {cmd_name}.")
        return False
