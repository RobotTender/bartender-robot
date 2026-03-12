import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import DrlStart, GetDrlState, DrlStop, GetLastAlarm
import time

# Helper functions for Modbus communication (DRL Compatible List-based)
DRL_HELPER_FUNCTIONS = """
g_slaveid = 0
def modbus_set_slaveid(slaveid):
    global g_slaveid
    g_slaveid = slaveid

def modbus_fc03(address, cnt):
    global g_slaveid
    data = [
        g_slaveid,
        3,
        (address >> 8) & 0xFF,
        address & 0xFF,
        (cnt >> 8) & 0xFF,
        cnt & 0xFF
    ]
    return modbus_send_make(data)

def modbus_fc06(address, value):
    global g_slaveid
    data = [
        g_slaveid,
        6,
        (address >> 8) & 0xFF,
        address & 0xFF,
        (value >> 8) & 0xFF,
        value & 0xFF
    ]
    return modbus_send_make(data)

def modbus_fc16(startaddress, cnt, valuelist):
    global g_slaveid
    data = [
        g_slaveid,
        16,
        (startaddress >> 8) & 0xFF,
        startaddress & 0xFF,
        (cnt >> 8) & 0xFF,
        cnt & 0xFF,
        2 * cnt
    ]
    for i in range(0, cnt):
        data.append((valuelist[i] >> 8) & 0xFF)
        data.append(valuelist[i] & 0xFF)
    return modbus_send_make(data)

def recv_check():
    size, val = flange_serial_read(0.1)
    if size > 0:
        return True, val
    else:
        return False, val

def gripper_move(stroke):
    flange_serial_write(modbus_fc16(282, 2, [stroke, 0]))
    wait(0.2)
    fail_cnt = 0
    while True:
        flange_serial_write(modbus_fc03(284, 1))
        size, val = flange_serial_read(0.1)
        if size >= 5:
            # val is a list of integers in DRL
            moving = val[3] * 256 + val[4]
            if moving == 0:
                break
            fail_cnt = 0
        else:
            fail_cnt += 1
        if fail_cnt > 20:
            break
        wait(0.1)
"""

# Initialization code
DRL_INIT_CODE = """
wait(0.5)
success = False
for i in range(0, 5):
    res = flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
    if res == 0:
        wait(0.2)
        modbus_set_slaveid(1)
        flange_serial_write(modbus_fc06(256, 1))
        wait(0.5)
        flag, val = recv_check()
        if flag is True:
            flange_serial_write(modbus_fc06(275, {current}))
            wait(0.5)
            flag, val = recv_check()
            if flag is True:
                tp_log("Gripper Activation Success")
                success = True
                flange_serial_close()
                break
        flange_serial_close()
    else:
        tp_log("flange_serial_open failed code: " + str(res))
    
    tp_log("Gripper Activation Retry...")
    wait(1.0)

if success is False:
    tp_log("Gripper Activation Failed")
"""

# Simple movement code
DRL_MOVE_CODE = """
wait(0.1)
res = flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
if res == 0:
    wait(0.2)
    modbus_set_slaveid(1)
    gripper_move({stroke})
    flange_serial_close()
else:
    tp_log("flange_serial_open failed code: " + str(res))
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
        
        # Safe service wait loop
        start = time.time()
        while not self.cli.service_is_ready():
            if time.time() - start > 10.0:
                self.node.get_logger().error("Gripper service not available after 10s!")
                break
            self.node.get_logger().info("Waiting for DRL service...")
            time.sleep(1.0)

    def _wait_for_future(self, future, timeout):
        """Helper to wait for future depending on whether the node is already spinning."""
        if self.node.executor is not None:
            start = time.time()
            while not future.done() and time.time() - start < timeout:
                time.sleep(0.05)
            return future.result() if future.done() else None
        else:
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
            return future.result() if future.done() else None

    def check_alarm(self):
        """Returns True if there is a fatal active alarm."""
        future = self.alarm_cli.call_async(GetLastAlarm.Request())
        res = self._wait_for_future(future, timeout=2.0)
        if res and res.success:
            # level 2 is fatal Alarm
            if res.log_alarm.level >= 2 and res.log_alarm.index != 0:
                self.node.get_logger().error(f"Detected Fatal Robot Alarm: [{res.log_alarm.index}] {res.log_alarm.param[1]}")
                return True
        return False

    def wait_drl_ready(self, timeout=20.0, force_stop=True):
        """Waits until DRL interpreter is STOPPED or IDLE."""
        start = time.time()
        while time.time() - start < timeout:
            if not self.state_cli.service_is_ready():
                time.sleep(0.1)
                continue
            
            future = self.state_cli.call_async(GetDrlState.Request())
            res = self._wait_for_future(future, timeout=2.0)
            
            if res and res.success:
                if res.drl_state == 1 or res.drl_state == 3: # STOP or LAST (Idle)
                    return True
                else:
                    if force_stop:
                        self.node.get_logger().warn(f"DRL is BUSY (state={res.drl_state}). Forcing DrlStop...")
                        stop_req = DrlStop.Request(stop_mode=1)
                        stop_future = self.stop_cli.call_async(stop_req)
                        self._wait_for_future(stop_future, timeout=2.0)
            
            time.sleep(1.0)
        return False

    def _execute_drl(self, code, timeout=30.0):
        # 1. Clear manager BEFORE starting
        if not self.wait_drl_ready(force_stop=True):
            self.node.get_logger().error("CRITICAL: DRL Manager still busy! Attempting DrlStart anyway.")
        
        req = DrlStart.Request()
        req.robot_system = self.robot_system
        req.code = f"{DRL_HELPER_FUNCTIONS}\n{code}"
        future = self.cli.call_async(req)
        
        res = self._wait_for_future(future, timeout=5.0)
        if res and res.success:
            # 2. Wait for completion
            time.sleep(0.5) 
            if self.wait_drl_ready(timeout=timeout, force_stop=False):
                # 3. Final check: was there an alarm during execution?
                if self.check_alarm():
                    return False
                return True
        return False

    def activate(self, force: int = 400) -> bool:
        """Initializes the gripper once."""
        self.node.get_logger().info(f"Activating Gripper with force={force}...")
        return self._execute_drl(DRL_INIT_CODE.format(current=force), timeout=45.0)

    def move(self, stroke: int) -> bool:
        """Sends a simple movement command."""
        self.node.get_logger().info(f"Gripper Move: stroke={stroke}")
        return self._execute_drl(DRL_MOVE_CODE.format(stroke=stroke))

    def move_sequence(self, sequence, force: int = 400) -> bool:
        """Sends a sequence of strokes."""
        task_code = "wait(0.1)\nres = flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)\nif res == 0:\n  wait(0.2)\n  modbus_set_slaveid(1)\n"
        for s in sequence:
            task_code += f"  gripper_move({s})\n"
        task_code += "  flange_serial_close()\nelse:\n  tp_log('open failed')"
        return self._execute_drl(task_code, timeout=60.0)

    def terminate(self) -> bool:
        req = DrlStart.Request()
        req.code = "flange_serial_close()"
        self.cli.call_async(req)
        return True
