import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import DrlStart, GetDrlState, DrlStop, GetLastAlarm
import time

# Robust, Single-Task DRL for Gripper Control (Python 3 Compatible)
DRL_HELPER_FUNCTIONS = """
g_slaveid = 1
def modbus_fc06(address, value):
    global g_slaveid
    # Built-in modbus_send_make requires a list to append CRC, 
    # but flange_serial_write requires bytes.
    return bytes(modbus_send_make([g_slaveid, 6, (address >> 8) & 0xFF, address & 0xFF, (value >> 8) & 0xFF, value & 0xFF]))

def modbus_fc16(startaddress, cnt, valuelist):
    global g_slaveid
    data = [g_slaveid, 16, (startaddress >> 8) & 0xFF, startaddress & 0xFF, (cnt >> 8) & 0xFF, cnt & 0xFF, 2 * cnt]
    for val in valuelist:
        data.append((val >> 8) & 0xFF)
        data.append(val & 0xFF)
    return bytes(modbus_send_make(data))

def recv_check():
    size, val = flange_serial_read(0.1)
    return size > 0

def gripper_init(force):
    res = flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
    wait(0.5)
    if res == 0:
        # Torque enable
        flange_serial_write(modbus_fc06(256, 1))
        recv_check()
        wait(0.5)
        # Set Current/Force
        flange_serial_write(modbus_fc06(275, force))
        recv_check()
        wait(0.5)
        flange_serial_close()
        return True
    return False

def gripper_move_and_wait(stroke):
    res = flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
    wait(0.2)
    if res == 0:
        # Send Position command
        flange_serial_write(modbus_fc16(282, 2, [stroke, 0]))
        recv_check()
        wait(0.5)
        
        # Internal Polling for "Moving" (Reg 284)
        for i in range(0, 50):
            # FC03 Read (ID=1, FC=3, Addr=284, Cnt=1)
            flange_serial_write(bytes(modbus_send_make([1, 3, 1, 28, 0, 1])))
            size, val = flange_serial_read(0.1)
            if size >= 5:
                # val is bytes in Python 3, indexable as ints
                moving = val[3] * 256 + val[4]
                if moving == 0:
                    break
            wait(0.1)
        
        flange_serial_close()
        return True
    return False
"""

class GripperController:
    # ... (init and helpers unchanged)
    def __init__(self, node: Node, namespace: str = "dsr01", robot_system: int = 0):
        self.node = node
        self.namespace = namespace
        self.robot_system = robot_system
        self.cli = self.node.create_client(DrlStart, f"/{namespace}/drl/drl_start")
        self.state_cli = self.node.create_client(GetDrlState, f"/{namespace}/drl/get_drl_state")
        self.stop_cli = self.node.create_client(DrlStop, f"/{namespace}/drl/drl_stop")
        self.alarm_cli = self.node.create_client(GetLastAlarm, f"/{namespace}/system/get_last_alarm")

    def _wait_for_future(self, future, timeout):
        try:
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        except Exception as e:
            self.node.get_logger().error(f"Error spinning for future: {e}")
            return None
        return future.result() if future.done() else None

    def wait_drl_ready(self, timeout=30.0, force_stop=True):
        start = time.time()
        while time.time() - start < timeout:
            req = GetDrlState.Request()
            future = self.state_cli.call_async(req)
            res = self._wait_for_future(future, timeout=2.0)
            if res and res.success:
                if res.drl_state in [1, 3]: 
                    self.node.get_logger().info(f"DRL Ready (state={res.drl_state})")
                    return True
                if force_stop:
                    self.node.get_logger().warn(f"DRL busy ({res.drl_state}), stopping...")
                    stop_f = self.stop_cli.call_async(DrlStop.Request(stop_mode=1))
                    self._wait_for_future(stop_f, timeout=2.0)
            else:
                self.node.get_logger().warn("Waiting for DRL state service response...")
            time.sleep(1.0)
        self.node.get_logger().error(f"DRL not ready within {timeout}s.")
        return False

    def _execute_raw(self, code, timeout=30.0):
        self.node.get_logger().info("Checking DRL readiness...")
        if not self.wait_drl_ready(force_stop=True): 
            return False

        self.node.get_logger().info("DRL is ready. Starting task...")
        time.sleep(1.0) # Settle Task Manager

        req = DrlStart.Request(robot_system=self.robot_system, code=f"{DRL_HELPER_FUNCTIONS}\n{code}")
        future = self.cli.call_async(req)
        res = self._wait_for_future(future, timeout=5.0)

        if res and res.success:
            self.node.get_logger().info("DRL Start Service Success. Waiting for completion...")
            time.sleep(1.0) # Wait for task to register as PLAYING
            success = self.wait_drl_ready(timeout=timeout, force_stop=False)
            if success:
                self.node.get_logger().info("DRL Task Completed Successfully.")
            else:
                self.node.get_logger().error("DRL Task failed to complete or timed out.")
            return success

        self.node.get_logger().error(f"DRL Start Service Failed: {res.message if res else 'No response'}")
        return False


    def activate(self, force: int = 400) -> bool:
        self.node.get_logger().info(f"Gripper: Activating (Force={force})...")
        return self._execute_raw(f"gripper_init({force})")

    def move(self, stroke: int) -> bool:
        self.node.get_logger().info(f"Gripper: Moving to {stroke}...")
        return self._execute_raw(f"gripper_move_and_wait({stroke})")

    def terminate(self) -> bool:
        self.cli.call_async(DrlStart.Request(code="flange_serial_close()"))
        return True
