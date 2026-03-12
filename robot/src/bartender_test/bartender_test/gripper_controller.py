import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import DrlStart
import time

# Helper functions for Modbus communication
DRL_HELPER_FUNCTIONS = """
g_slaveid = 0
def modbus_set_slaveid(slaveid):
    global g_slaveid
    g_slaveid = slaveid
def modbus_fc06(address, value):
    global g_slaveid
    data = (g_slaveid).to_bytes(1, byteorder='big')
    data += (6).to_bytes(1, byteorder='big')
    data += (address).to_bytes(2, byteorder='big')
    data += (value).to_bytes(2, byteorder='big')
    return modbus_send_make(data)
def modbus_fc16(startaddress, cnt, valuelist):
    global g_slaveid
    data = (g_slaveid).to_bytes(1, byteorder='big')
    data += (16).to_bytes(1, byteorder='big')
    data += (startaddress).to_bytes(2, byteorder='big')
    data += (cnt).to_bytes(2, byteorder='big')
    data += (2 * cnt).to_bytes(1, byteorder='big')
    for i in range(0, cnt):
        data += (valuelist[i]).to_bytes(2, byteorder='big')
    return modbus_send_make(data)
def recv_check():
    size, val = flange_serial_read(0.1)
    if size > 0:
        return True, val
    else:
        return False, val
def gripper_move(stroke):
    flange_serial_write(modbus_fc16(282, 2, [stroke, 0]))
    wait(0.5)
"""

# Initialization code
DRL_INIT_CODE = """
wait(0.5)
flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
modbus_set_slaveid(1)
for i in range(0, 10):
    flange_serial_write(modbus_fc06(256, 1))
    wait(0.2)
    flag, val = recv_check()
    if flag is True:
        flange_serial_write(modbus_fc06(275, {current}))
        wait(0.2)
        flag, val = recv_check()
        if flag is True:
            break
    wait(0.5)
flange_serial_close()
"""

# Simple movement code
DRL_MOVE_CODE = """
wait(0.1)
flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
modbus_set_slaveid(1)
gripper_move({stroke})
flange_serial_close()
"""

class GripperController:
    def __init__(self, node: Node, namespace: str = "dsr01", robot_system: int = 0):
        self.node = node
        self.robot_system = robot_system
        self.cli = self.node.create_client(DrlStart, f"/{namespace}/drl/drl_start")
        
        # Safe service wait loop
        start = time.time()
        while not self.cli.service_is_ready():
            if time.time() - start > 10.0:
                self.node.get_logger().error("Gripper service not available after 10s!")
                break
            self.node.get_logger().info("Waiting for DRL service...")
            time.sleep(1.0)

    def _execute_drl(self, code, timeout=10.0):
        req = DrlStart.Request()
        req.robot_system = self.robot_system
        req.code = f"{DRL_HELPER_FUNCTIONS}\n{code}"
        future = self.cli.call_async(req)
        
        start = time.time()
        while not future.done() and time.time() - start < timeout:
            time.sleep(0.05)
        return bool(future.result().success) if future.result() else False

    def activate(self, force: int = 400) -> bool:
        """Initializes the gripper once."""
        self.node.get_logger().info(f"Activating Gripper with force={force}...")
        return self._execute_drl(DRL_INIT_CODE.format(current=force), timeout=15.0)

    def move(self, stroke: int) -> bool:
        """Sends a simple movement command."""
        self.node.get_logger().info(f"Gripper Move: stroke={stroke}")
        return self._execute_drl(DRL_MOVE_CODE.format(stroke=stroke))

    def move_sequence(self, sequence, force: int = 400) -> bool:
        """Sends a sequence of strokes."""
        task_code = "wait(0.1)\nflange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)\nmodbus_set_slaveid(1)\n"
        for s in sequence:
            task_code += f"gripper_move({s})\nwait(1.0)\n"
        task_code += "flange_serial_close()"
        return self._execute_drl(task_code, timeout=20.0)

    def terminate(self) -> bool:
        req = DrlStart.Request()
        req.code = "flange_serial_close()"
        self.cli.call_async(req)
        return True
