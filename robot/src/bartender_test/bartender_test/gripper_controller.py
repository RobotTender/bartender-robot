import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import DrlStart, GetDrlState, DrlStop, GetLastAlarm
import time

# Robust, Single-Task DRL for Gripper Control (Python 3 Compatible)
DRL_HELPER_FUNCTIONS = """
def gripper_init(force):
    res = flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
    wait(0.5)
    if res == 0:
        # Torque enable (ID=1, FC=6, Addr=256, Val=1)
        flange_serial_write(modbus_send_make(bytes([1, 6, 1, 0, 0, 1])))
        wait(0.5)
        # Set Force (ID=1, FC=6, Addr=275, Val=force)
        f_hi = (force >> 8) & 0xFF
        f_lo = force & 0xFF
        flange_serial_write(modbus_send_make(bytes([1, 6, 1, 19, f_hi, f_lo])))
        wait(0.5)
        flange_serial_close()
        return True
    return False

def gripper_move_and_wait(stroke):
    res = flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
    wait(0.2)
    if res == 0:
        # Send Position command (ID=1, FC=16, Addr=282, Cnt=2, Len=4, Pos_Hi, Pos_Lo, 0, 0)
        s_hi = (stroke >> 8) & 0xFF
        s_lo = stroke & 0xFF
        flange_serial_write(modbus_send_make(bytes([1, 16, 1, 26, 0, 2, 4, s_hi, s_lo, 0, 0])))
        wait(0.5)
        
        # Internal Polling for "Moving" (Reg 284)
        for i in range(0, 50):
            flange_serial_write(modbus_send_make(bytes([1, 3, 1, 28, 0, 1])))
            size, val = flange_serial_read(0.1)
            if size >= 5:
                # val is bytes in Python 3
                moving = val[3] * 256 + val[4]
                if moving == 0:
                    break
            wait(0.1)
        
        flange_serial_close()
        return True
    return False
"""

class GripperController:
    def __init__(self, node: Node, namespace: str = "dsr01", robot_system: int = 0):
        self.node = node
        self.namespace = namespace
        self.robot_system = robot_system
        self.cli = self.node.create_client(DrlStart, f"/{namespace}/drl/drl_start")
        self.state_cli = self.node.create_client(GetDrlState, f"/{namespace}/drl/get_drl_state")
        self.stop_cli = self.node.create_client(DrlStop, f"/{namespace}/drl/drl_stop")
        self.alarm_cli = self.node.create_client(GetLastAlarm, f"/{namespace}/system/get_last_alarm")

    def _wait_for_future(self, future, timeout):
        if self.node.executor is not None:
            start = time.time()
            while not future.done() and time.time() - start < timeout:
                time.sleep(0.05)
            return future.result() if future.done() else None
        else:
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
            return future.result() if future.done() else None

    def wait_drl_ready(self, timeout=10.0, force_stop=True):
        start = time.time()
        while time.time() - start < timeout:
            future = self.state_cli.call_async(GetDrlState.Request())
            res = self._wait_for_future(future, timeout=2.0)
            if res and res.success:
                if res.drl_state in [1, 3]: return True
                if force_stop:
                    self.node.get_logger().warn(f"DRL busy ({res.drl_state}), stopping...")
                    stop_f = self.stop_cli.call_async(DrlStop.Request(stop_mode=1))
                    self._wait_for_future(stop_f, timeout=2.0)
            time.sleep(1.0)
        return False

    def _execute_raw(self, code, timeout=30.0):
        if not self.wait_drl_ready(force_stop=True): return False
        time.sleep(1.0) # Settle Task Manager
        
        req = DrlStart.Request(robot_system=self.robot_system, code=f"{DRL_HELPER_FUNCTIONS}\n{code}")
        future = self.cli.call_async(req)
        res = self._wait_for_future(future, timeout=5.0)
        if res and res.success:
            time.sleep(1.0) # Wait for task to register as PLAYING
            return self.wait_drl_ready(timeout=timeout, force_stop=False)
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
