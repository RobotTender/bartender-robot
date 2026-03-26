from __future__ import annotations

import importlib
import json
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .core import OBJECT_CLASS_COUNT, ObjectState, RobotState, euler_xyz_deg_to_quat_wxyz, quat_apply_np


@dataclass
class ModbusGripperConfig:
    protocol: str = "flange_serial_fc"  # flange_serial_fc | modbus_tcp_signal
    enabled: bool = True
    auto_register: bool = True
    ip: str = "192.168.137.2"
    port: int = 502
    slave_id: int = 1
    cmd_signal_name: str = "rh_cmd"
    cmd_reg_index: int = 0
    pos_signal_name: str = "rh_pos"
    pos_reg_index: int = 1
    open_raw: int = 0
    close_raw: int = 750
    command_deadband_raw: int = 2
    min_write_interval_s: float = 0.05
    rx_timeout_s: float = 0.5
    ack_timeout_s: float = 0.1

    # flange serial settings (Doosan flange serial + Modbus RTU frame)
    serial_baudrate: int = 57600
    serial_bytesize: int = 8
    serial_parity: str = "N"
    serial_stopbits: int = 1
    serial_probe_retries: int = 5
    serial_probe_wait_s: float = 0.1

    # RH-P12-RN(A) register map used in DART examples
    reg_operating_mode: int = 5
    reg_torque_enable: int = 256
    reg_goal_current: int = 275
    reg_goal_velocity: int = 276
    reg_goal_position: int = 282
    reg_present_position: int = 290
    operating_mode_value: int = 5 << 8  # current-based position mode
    init_goal_current: int = 200
    init_goal_velocity: int = -1  # negative -> skip write
    position_word_order: str = "LO_HI"  # LO_HI | HI_LO


@dataclass
class RealRobotConfig:
    robot_id: str = "dsr01"
    robot_model: str = "e0509"
    robot_host: str = ""
    robot_port: int = 0
    joint_command_mode: str = "servoj"  # servoj | speedj
    servo_time_s: float = 1.0 / 60.0
    servo_vel_deg_s: float = 90.0
    servo_acc_deg_s2: float = 180.0
    speedj_time_s: float = 1.0 / 60.0
    speedj_vel_deg_s: float = 30.0
    speedj_acc_deg_s2: float = 100.0
    speedj_reanchor_to_measured: bool = True
    use_posx_orientation: bool = False
    object_source: str = "fixed"  # fixed | json
    object_json_path: str = "/tmp/object_state.json"
    object_fixed_xyz_m: tuple[float, float, float] = (0.0, 0.65, 1.30)
    object_fixed_up_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    object_fixed_custom_extra: tuple[float, ...] = ()
    object_class_index: int = 0
    gripper: ModbusGripperConfig = field(default_factory=ModbusGripperConfig)


@dataclass
class Ros2Config:
    node_name: str = "sim2real_policy_runner"
    spin_timeout_s: float = 0.01
    init_wait_s: float = 3.0

    # Subscriptions
    joint_state_topic: str = "/joint_states"
    ee_pose_topic: str = "/ee_pose"
    object_pose_topic: str = "/object_pose"
    object_class_topic: str = "/object_class"
    object_extra_topic: str = "/object_extra"
    gripper_state_topic: str = "/gripper/state"

    # Publications
    joint_cmd_topic: str = "/arm/joint_position_cmd"
    gripper_cmd_topic: str = "/gripper/command"
    enable_gripper_output: bool = True

    # Gripper command mode
    gripper_command_mode: str = "topic"  # topic | drl_service
    drl_namespace: str = "dsr01"
    drl_robot_system: int = 0
    drl_service_timeout_s: float = 2.0
    drl_call_timeout_s: float = 5.0
    drl_stroke_open: int = 0
    drl_stroke_close: int = 750
    drl_deadband_stroke: int = 5
    drl_min_command_interval_s: float = 0.2
    drl_slave_id: int = 1
    drl_baudrate: int = 57600
    drl_torque_enable_addr: int = 256
    drl_goal_current_addr: int = 275
    drl_goal_position_addr: int = 282
    drl_init_goal_current: int = 400

    # Fallbacks (when topic data is missing)
    fallback_object_xyz_m: tuple[float, float, float] = (0.0, 0.65, 1.30)
    fallback_object_up_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fallback_object_custom_extra: tuple[float, ...] = ()
    fallback_object_class_index: int = 0


class RobotIO(Protocol):
    def connect(self) -> None: ...

    def read_robot_state(self) -> RobotState: ...

    def read_object_state(self) -> ObjectState: ...

    def send_joint_targets(
        self,
        joint_targets_rad: np.ndarray,
        gripper_close_ratio: float,
        joint_velocity_cmd_rad_s: np.ndarray | None = None,
    ) -> None: ...

    def shutdown(self) -> None: ...


class DummyRobotIO:
    def __init__(self, class_index: int = 0, fixed_custom_extra: tuple[float, ...] = ()):
        self.class_index = int(np.clip(class_index, 0, OBJECT_CLASS_COUNT - 1))
        self.joint_pos = np.array([np.pi / 2, -np.pi / 4, np.pi / 2, 0.0, np.pi / 4, -np.pi / 2], dtype=np.float32)
        self.joint_vel = np.zeros(6, dtype=np.float32)
        self.gripper = 0.0
        self.object_pos = np.array([0.0, 0.65, 1.30], dtype=np.float32)
        self.object_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.object_custom_extra = np.asarray(fixed_custom_extra, dtype=np.float32).reshape(-1)

    def connect(self) -> None:
        return

    def _fake_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        j1, j2, j3 = float(self.joint_pos[0]), float(self.joint_pos[1]), float(self.joint_pos[2])
        ee_pos = np.array(
            [
                0.20 + 0.14 * np.cos(j1),
                0.65 + 0.14 * np.sin(j1),
                1.20 + 0.05 * np.sin(j2) + 0.03 * np.sin(j3),
            ],
            dtype=np.float32,
        )
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return ee_pos, ee_quat

    def read_robot_state(self) -> RobotState:
        ee_pos, ee_quat = self._fake_ee_pose()
        return RobotState(
            joint_pos_rad=self.joint_pos.copy(),
            joint_vel_rad_s=self.joint_vel.copy(),
            gripper_close_ratio=float(self.gripper),
            ee_pos_m=ee_pos,
            ee_quat_wxyz=ee_quat,
        )

    def read_object_state(self) -> ObjectState:
        return ObjectState(
            position_m=self.object_pos.copy(),
            up_vector_w=self.object_up.copy(),
            class_index=self.class_index,
            custom_extra=self.object_custom_extra.copy(),
        )

    def send_joint_targets(
        self,
        joint_targets_rad: np.ndarray,
        gripper_close_ratio: float,
        joint_velocity_cmd_rad_s: np.ndarray | None = None,
    ) -> None:
        target = np.asarray(joint_targets_rad, dtype=np.float32).reshape(-1)
        if target.shape[0] < self.joint_pos.shape[0]:
            padded = np.zeros_like(self.joint_pos, dtype=np.float32)
            padded[: target.shape[0]] = target
            target = padded
        else:
            target = target[: self.joint_pos.shape[0]]

        if joint_velocity_cmd_rad_s is None:
            self.joint_vel = (target - self.joint_pos).astype(np.float32)
        else:
            vel = np.asarray(joint_velocity_cmd_rad_s, dtype=np.float32).reshape(-1)
            if vel.shape[0] < self.joint_vel.shape[0]:
                padded_vel = np.zeros_like(self.joint_vel, dtype=np.float32)
                padded_vel[: vel.shape[0]] = vel
                vel = padded_vel
            else:
                vel = vel[: self.joint_vel.shape[0]]
            self.joint_vel = vel.astype(np.float32)

        self.joint_pos = target.astype(np.float32)
        self.gripper = float(np.clip(gripper_close_ratio, 0.0, 1.0))

    def shutdown(self) -> None:
        return


class ROS2RobotIO:
    """ROS2 topic bridge for lightweight team integration."""

    def __init__(self, cfg: Ros2Config):
        self.cfg = cfg
        self._rclpy: Any | None = None
        self._node: Any | None = None
        self._joint_state_msg: Any | None = None
        self._ee_pose_msg: Any | None = None
        self._object_pose_msg: Any | None = None
        self._object_class_value: int = int(cfg.fallback_object_class_index)
        self._object_extra_value = np.asarray(cfg.fallback_object_custom_extra, dtype=np.float32).reshape(-1)
        self._gripper_state_value: float = 0.0
        self._joint_cmd_pub: Any | None = None
        self._gripper_cmd_pub: Any | None = None
        self._joint_cmd_names: list[str] = []
        self._drl_start_srv_type: Any | None = None
        self._drl_client: Any | None = None
        self._last_drl_stroke: int | None = None
        self._last_drl_cmd_ts = 0.0

    def _import_ros2(self) -> tuple[Any, Any, Any, Any]:
        try:
            rclpy = importlib.import_module("rclpy")
            qos = importlib.import_module("rclpy.qos")
            node_module = importlib.import_module("rclpy.node")
            msg_joint = importlib.import_module("sensor_msgs.msg")
            msg_geom = importlib.import_module("geometry_msgs.msg")
            msg_std = importlib.import_module("std_msgs.msg")
            return rclpy, qos, node_module, (msg_joint, msg_geom, msg_std)
        except Exception as exc:
            raise RuntimeError(
                "ROS2 모듈 import 실패: rclpy/sensor_msgs/geometry_msgs/std_msgs가 필요합니다."
            ) from exc

    def connect(self) -> None:
        rclpy, qos_module, node_module, msgs = self._import_ros2()
        msg_joint, msg_geom, msg_std = msgs
        self._rclpy = rclpy

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = node_module.Node(self.cfg.node_name)
        qos = qos_module.QoSProfile(depth=10)

        self._joint_cmd_pub = self._node.create_publisher(msg_joint.JointState, self.cfg.joint_cmd_topic, qos)
        if bool(self.cfg.enable_gripper_output):
            self._gripper_cmd_pub = self._node.create_publisher(msg_std.Float32, self.cfg.gripper_cmd_topic, qos)
        else:
            self._gripper_cmd_pub = None

        self._node.create_subscription(msg_joint.JointState, self.cfg.joint_state_topic, self._on_joint_state, qos)
        self._node.create_subscription(msg_geom.PoseStamped, self.cfg.ee_pose_topic, self._on_ee_pose, qos)
        self._node.create_subscription(msg_geom.PoseStamped, self.cfg.object_pose_topic, self._on_object_pose, qos)
        self._node.create_subscription(msg_std.Int32, self.cfg.object_class_topic, self._on_object_class, qos)
        self._node.create_subscription(msg_std.Float32MultiArray, self.cfg.object_extra_topic, self._on_object_extra, qos)
        self._node.create_subscription(msg_std.Float32, self.cfg.gripper_state_topic, self._on_gripper_state, qos)

        if bool(self.cfg.enable_gripper_output) and self.cfg.gripper_command_mode == "drl_service":
            self._setup_drl_gripper()

        start = time.perf_counter()
        while time.perf_counter() - start < float(self.cfg.init_wait_s):
            rclpy.spin_once(self._node, timeout_sec=float(self.cfg.spin_timeout_s))
            if self._joint_state_msg is not None and self._ee_pose_msg is not None:
                break

    def _setup_drl_gripper(self) -> None:
        assert self._node is not None and self._rclpy is not None
        try:
            dsr_srv = importlib.import_module("dsr_msgs2.srv")
            self._drl_start_srv_type = dsr_srv.DrlStart
        except Exception as exc:
            raise RuntimeError("dsr_msgs2.srv.DrlStart를 import할 수 없습니다.") from exc

        service_name = f"/{self.cfg.drl_namespace}/drl/drl_start"
        self._drl_client = self._node.create_client(self._drl_start_srv_type, service_name)

        while not self._drl_client.wait_for_service(timeout_sec=float(self.cfg.drl_service_timeout_s)):
            if not self._rclpy.ok():
                raise RuntimeError("ROS2가 종료되어 DrlStart 서비스 연결을 중단합니다.")
            self._node.get_logger().info(f"Waiting for service {service_name} ...")

        init_script = self._make_drl_init_script()
        if not self._send_drl_script(init_script):
            raise RuntimeError("DrlStart gripper init script 실행 실패")

    def _make_drl_base_code(self) -> str:
        return textwrap.dedent(
            f"""
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
                return False, val
            def gripper_move(stroke):
                flange_serial_write(modbus_fc16({int(self.cfg.drl_goal_position_addr)}, 2, [stroke, 0]))
                wait(0.2)
            """
        )

    def _make_drl_init_script(self) -> str:
        task = textwrap.dedent(
            f"""
            flange_serial_open(baudrate={int(self.cfg.drl_baudrate)}, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
            modbus_set_slaveid({int(self.cfg.drl_slave_id)})
            flange_serial_write(modbus_fc06({int(self.cfg.drl_torque_enable_addr)}, 1))
            recv_check()
            flange_serial_write(modbus_fc06({int(self.cfg.drl_goal_current_addr)}, {int(self.cfg.drl_init_goal_current)}))
            recv_check()
            """
        )
        return f"{self._make_drl_base_code()}\n{task}"

    def _make_drl_move_script(self, stroke: int) -> str:
        task = textwrap.dedent(
            f"""
            flange_serial_open(baudrate={int(self.cfg.drl_baudrate)}, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
            modbus_set_slaveid({int(self.cfg.drl_slave_id)})
            gripper_move({int(stroke)})
            """
        )
        return f"{self._make_drl_base_code()}\n{task}"

    def _send_drl_script(self, code: str) -> bool:
        assert self._drl_client is not None and self._drl_start_srv_type is not None and self._rclpy is not None and self._node is not None
        req = self._drl_start_srv_type.Request()
        req.robot_system = int(self.cfg.drl_robot_system)
        req.code = str(code)

        future = self._drl_client.call_async(req)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=float(self.cfg.drl_call_timeout_s))
        result = future.result()
        if result is None:
            return False
        return bool(getattr(result, "success", False))

    def _ratio_to_drl_stroke(self, close_ratio: float) -> int:
        ratio = float(np.clip(close_ratio, 0.0, 1.0))
        stroke = float(self.cfg.drl_stroke_open) + ratio * float(self.cfg.drl_stroke_close - self.cfg.drl_stroke_open)
        return int(round(stroke))

    def _send_drl_gripper(self, close_ratio: float, force: bool = False) -> None:
        stroke = self._ratio_to_drl_stroke(close_ratio)
        now = time.perf_counter()
        if not force and self._last_drl_stroke is not None:
            if abs(stroke - self._last_drl_stroke) <= int(self.cfg.drl_deadband_stroke):
                return
            if (now - self._last_drl_cmd_ts) < float(self.cfg.drl_min_command_interval_s):
                return
        if self._send_drl_script(self._make_drl_move_script(stroke)):
            self._last_drl_stroke = stroke
            self._last_drl_cmd_ts = now

    def _on_joint_state(self, msg: Any) -> None:
        self._joint_state_msg = msg
        if getattr(msg, "name", None):
            self._joint_cmd_names = list(msg.name[:6])

    def _on_ee_pose(self, msg: Any) -> None:
        self._ee_pose_msg = msg

    def _on_object_pose(self, msg: Any) -> None:
        self._object_pose_msg = msg

    def _on_object_class(self, msg: Any) -> None:
        self._object_class_value = int(getattr(msg, "data", self._object_class_value))

    def _on_object_extra(self, msg: Any) -> None:
        data = getattr(msg, "data", None)
        if data is None:
            return
        try:
            self._object_extra_value = np.asarray(list(data), dtype=np.float32).reshape(-1)
        except Exception:
            return

    def _on_gripper_state(self, msg: Any) -> None:
        self._gripper_state_value = float(getattr(msg, "data", self._gripper_state_value))

    def _spin_once(self) -> None:
        assert self._rclpy is not None and self._node is not None
        self._rclpy.spin_once(self._node, timeout_sec=float(self.cfg.spin_timeout_s))

    @staticmethod
    def _pose_to_pos_quat_wxyz(pose_msg: Any) -> tuple[np.ndarray, np.ndarray]:
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        pos = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float32)
        quat = np.array([float(q.w), float(q.x), float(q.y), float(q.z)], dtype=np.float32)
        return pos, quat

    def _get_object_state_from_topics(self) -> ObjectState:
        if self._object_pose_msg is None:
            return ObjectState(
                position_m=np.asarray(self.cfg.fallback_object_xyz_m, dtype=np.float32),
                up_vector_w=np.asarray(self.cfg.fallback_object_up_xyz, dtype=np.float32),
                class_index=int(self._object_class_value),
                custom_extra=self._object_extra_value.copy(),
            )
        pos, quat = self._pose_to_pos_quat_wxyz(self._object_pose_msg)
        up = quat_apply_np(quat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        return ObjectState(
            position_m=pos,
            up_vector_w=up.astype(np.float32),
            class_index=int(self._object_class_value),
            custom_extra=self._object_extra_value.copy(),
        )

    def read_robot_state(self) -> RobotState:
        self._spin_once()
        if self._joint_state_msg is None:
            q = np.zeros(6, dtype=np.float32)
            v = np.zeros(6, dtype=np.float32)
        else:
            q = np.asarray(self._joint_state_msg.position[:6], dtype=np.float32)
            if len(getattr(self._joint_state_msg, "velocity", [])) >= 6:
                v = np.asarray(self._joint_state_msg.velocity[:6], dtype=np.float32)
            else:
                v = np.zeros(6, dtype=np.float32)

        if self._ee_pose_msg is None:
            ee_pos = np.zeros(3, dtype=np.float32)
            ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            ee_pos, ee_quat = self._pose_to_pos_quat_wxyz(self._ee_pose_msg)

        return RobotState(
            joint_pos_rad=q,
            joint_vel_rad_s=v,
            gripper_close_ratio=float(np.clip(self._gripper_state_value, 0.0, 1.0)),
            ee_pos_m=ee_pos,
            ee_quat_wxyz=ee_quat,
        )

    def read_object_state(self) -> ObjectState:
        self._spin_once()
        return self._get_object_state_from_topics()

    def send_joint_targets(
        self,
        joint_targets_rad: np.ndarray,
        gripper_close_ratio: float,
        joint_velocity_cmd_rad_s: np.ndarray | None = None,
    ) -> None:
        assert self._joint_cmd_pub is not None and self._node is not None
        sensor_msgs = importlib.import_module("sensor_msgs.msg")

        joint_msg = sensor_msgs.JointState()
        joint_msg.header.stamp = self._node.get_clock().now().to_msg()
        if self._joint_cmd_names:
            joint_msg.name = list(self._joint_cmd_names)
        joint_msg.position = [float(v) for v in np.asarray(joint_targets_rad, dtype=np.float32).tolist()]
        self._joint_cmd_pub.publish(joint_msg)

        if not bool(self.cfg.enable_gripper_output):
            return

        if self.cfg.gripper_command_mode == "drl_service":
            self._send_drl_gripper(gripper_close_ratio)
        else:
            assert self._gripper_cmd_pub is not None
            std_msgs = importlib.import_module("std_msgs.msg")
            gripper_msg = std_msgs.Float32()
            gripper_msg.data = float(np.clip(gripper_close_ratio, 0.0, 1.0))
            self._gripper_cmd_pub.publish(gripper_msg)

    def shutdown(self) -> None:
        if bool(self.cfg.enable_gripper_output) and self.cfg.gripper_command_mode == "drl_service":
            try:
                self._send_drl_gripper(0.0, force=True)
                self._send_drl_script("flange_serial_close()")
            except Exception:
                pass
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()


class RealRobotIO:
    """Doosan e0509 + Modbus gripper backend."""

    def __init__(self, cfg: RealRobotConfig):
        self.cfg = cfg
        self._dr: Any | None = None
        self._last_gripper_raw: int | None = None
        self._last_gripper_write_ts = 0.0
        self._last_gripper_ratio = 0.0
        self._flange_serial_opened = False
        self._last_speedj_vel_deg_s: np.ndarray | None = None

    def _import_doosan_api(self) -> Any:
        try:
            dr_init = importlib.import_module("DR_init")
            dr_init.__dsr__id = self.cfg.robot_id
            dr_init.__dsr__model = self.cfg.robot_model
            if str(self.cfg.robot_host).strip():
                setattr(dr_init, "__dsr__host", str(self.cfg.robot_host).strip())
            if int(self.cfg.robot_port) > 0:
                setattr(dr_init, "__dsr__port", int(self.cfg.robot_port))
            return importlib.import_module("DSR_ROBOT2")
        except Exception as exc:
            raise RuntimeError(
                "Doosan Python API import 실패: DR_init/DSR_ROBOT2 모듈을 찾지 못했습니다. "
                "로봇 컨트롤러 환경 또는 Doosan SDK 설치를 확인하세요."
            ) from exc

    @staticmethod
    def _modbus_crc16(raw_data: bytes) -> tuple[int, int]:
        crc = 0xFFFF
        for b in raw_data:
            crc ^= int(b)
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        crc &= 0xFFFF
        crc_low = crc & 0xFF
        crc_high = (crc >> 8) & 0xFF
        return crc_low, crc_high

    def _modbus_send_make(self, raw_data: bytes) -> bytes:
        assert self._dr is not None
        if hasattr(self._dr, "modbus_send_make"):
            return bytes(self._dr.modbus_send_make(raw_data))
        crc_low, crc_high = self._modbus_crc16(raw_data)
        return raw_data + bytes([crc_low, crc_high])

    @staticmethod
    def _to_byte_list(resp: Any) -> list[int]:
        if isinstance(resp, (bytes, bytearray)):
            return [int(b) for b in resp]
        return [int(b) for b in resp]

    def _safe_add_modbus_signal(self, *, name: str, reg_type: int, index: int, value: int = 0) -> None:
        assert self._dr is not None
        try:
            self._dr.add_modbus_signal(
                ip=self.cfg.gripper.ip,
                port=int(self.cfg.gripper.port),
                slaveid=int(self.cfg.gripper.slave_id),
                name=name,
                reg_type=reg_type,
                index=int(index),
                value=int(value),
            )
        except Exception:
            return

    def _flange_serial_open(self) -> None:
        assert self._dr is not None
        if self._flange_serial_opened:
            return
        bytesize = int(self.cfg.gripper.serial_bytesize)
        parity: Any = self.cfg.gripper.serial_parity
        stopbits = int(self.cfg.gripper.serial_stopbits)
        if hasattr(self._dr, "DR_EIGHTBITS") and bytesize == 8:
            bytesize = self._dr.DR_EIGHTBITS
        if hasattr(self._dr, "DR_PARITY_NONE") and str(self.cfg.gripper.serial_parity).upper() == "N":
            parity = self._dr.DR_PARITY_NONE
        if hasattr(self._dr, "DR_STOPBITS_ONE") and int(self.cfg.gripper.serial_stopbits) == 1:
            stopbits = self._dr.DR_STOPBITS_ONE
        if not hasattr(self._dr, "flange_serial_open"):
            raise RuntimeError("DSR_ROBOT2에 flange_serial_open 함수가 없습니다.")
        self._dr.flange_serial_open(
            baudrate=int(self.cfg.gripper.serial_baudrate),
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
        )
        self._flange_serial_opened = True

    def _flange_serial_close(self) -> None:
        if not self._flange_serial_opened or self._dr is None:
            return
        if hasattr(self._dr, "flange_serial_close"):
            self._dr.flange_serial_close()
        self._flange_serial_opened = False

    def _flange_flush_rx(self, timeout_s: float = 0.02, max_round: int = 10) -> None:
        assert self._dr is not None
        for _ in range(int(max_round)):
            size, _ = self._dr.flange_serial_read(float(timeout_s))
            if int(size) <= 0:
                break

    def _flange_recv(self, timeout_s: float) -> tuple[bool, list[int]]:
        assert self._dr is not None
        size, resp = self._dr.flange_serial_read(float(timeout_s))
        if int(size) <= 0:
            return False, []
        return True, self._to_byte_list(resp)

    def _flange_modbus_fc03(self, start_addr: int, reg_cnt: int) -> bytes:
        payload = bytearray()
        payload += int(self.cfg.gripper.slave_id).to_bytes(1, byteorder="big")
        payload += (3).to_bytes(1, byteorder="big")
        payload += int(start_addr).to_bytes(2, byteorder="big")
        payload += int(reg_cnt).to_bytes(2, byteorder="big")
        return self._modbus_send_make(bytes(payload))

    def _flange_modbus_fc06(self, addr: int, value: int) -> bytes:
        payload = bytearray()
        payload += int(self.cfg.gripper.slave_id).to_bytes(1, byteorder="big")
        payload += (6).to_bytes(1, byteorder="big")
        payload += int(addr).to_bytes(2, byteorder="big")
        payload += int(value).to_bytes(2, byteorder="big")
        return self._modbus_send_make(bytes(payload))

    def _flange_modbus_fc16(self, start_addr: int, reg_cnt: int, values: list[int]) -> bytes:
        payload = bytearray()
        payload += int(self.cfg.gripper.slave_id).to_bytes(1, byteorder="big")
        payload += (16).to_bytes(1, byteorder="big")
        payload += int(start_addr).to_bytes(2, byteorder="big")
        payload += int(reg_cnt).to_bytes(2, byteorder="big")
        payload += int(2 * reg_cnt).to_bytes(1, byteorder="big")
        for i in range(int(reg_cnt)):
            payload += int(values[i]).to_bytes(2, byteorder="big")
        return self._modbus_send_make(bytes(payload))

    def _flange_write_and_ack(self, packet: bytes, timeout_s: float | None = None) -> bool:
        assert self._dr is not None
        self._flange_flush_rx()
        self._dr.flange_serial_write(packet)
        ok, _ = self._flange_recv(float(self.cfg.gripper.ack_timeout_s if timeout_s is None else timeout_s))
        return ok

    def _flange_modbus_read_regs(self, start_addr: int, reg_cnt: int, timeout_s: float | None = None) -> tuple[bool, list[int]]:
        assert self._dr is not None
        self._flange_flush_rx()
        self._dr.flange_serial_write(self._flange_modbus_fc03(start_addr=start_addr, reg_cnt=reg_cnt))
        time.sleep(0.02)
        ok, buf = self._flange_recv(float(self.cfg.gripper.rx_timeout_s if timeout_s is None else timeout_s))
        if not ok:
            return False, []

        min_len = 3 + 2 * reg_cnt + 2
        if len(buf) < min_len:
            return False, []
        if buf[0] != int(self.cfg.gripper.slave_id):
            return False, []
        if buf[1] != 3:
            return False, []
        byte_count = int(buf[2])
        if byte_count != 2 * int(reg_cnt):
            return False, []
        regs: list[int] = []
        idx = 3
        for _ in range(int(reg_cnt)):
            regs.append((int(buf[idx]) << 8) | int(buf[idx + 1]))
            idx += 2
        return True, regs

    def _flange_read_u32(self, addr: int, word_order: str = "LO_HI") -> tuple[bool, int]:
        ok, regs = self._flange_modbus_read_regs(addr, 2)
        if not ok:
            return False, 0
        if str(word_order).upper() == "HI_LO":
            val = (int(regs[0]) << 16) | int(regs[1])
        else:
            val = (int(regs[1]) << 16) | int(regs[0])
        return True, int(val)

    def _setup_gripper_tcp_signal(self) -> None:
        assert self._dr is not None
        if not self.cfg.gripper.enabled:
            return
        if self.cfg.gripper.auto_register:
            self._safe_add_modbus_signal(
                name=self.cfg.gripper.cmd_signal_name,
                reg_type=self._dr.DR_MODBUS_REG_OUTPUT,
                index=self.cfg.gripper.cmd_reg_index,
                value=self.cfg.gripper.open_raw,
            )
            if self.cfg.gripper.pos_signal_name:
                self._safe_add_modbus_signal(
                    name=self.cfg.gripper.pos_signal_name,
                    reg_type=self._dr.DR_MODBUS_REG_INPUT,
                    index=self.cfg.gripper.pos_reg_index,
                    value=0,
                )
        self._write_gripper_command(raw_cmd=self.cfg.gripper.open_raw, force=True)

    def _setup_gripper_flange_serial(self) -> None:
        assert self._dr is not None
        if not self.cfg.gripper.enabled:
            return
        retries = max(1, int(self.cfg.gripper.serial_probe_retries))
        for _ in range(retries):
            try:
                self._flange_serial_open()
                probe_ok = self._flange_write_and_ack(
                    self._flange_modbus_fc06(self.cfg.gripper.reg_torque_enable, 1),
                    timeout_s=self.cfg.gripper.ack_timeout_s,
                )
                if probe_ok:
                    break
            except Exception:
                pass
            self._flange_serial_close()
            time.sleep(float(self.cfg.gripper.serial_probe_wait_s))
        if not self._flange_serial_opened:
            raise RuntimeError("flange_serial_open 실패 또는 그리퍼 응답 없음")

        self._flange_write_and_ack(self._flange_modbus_fc06(self.cfg.gripper.reg_torque_enable, 0))
        time.sleep(0.2)
        self._flange_write_and_ack(
            self._flange_modbus_fc06(self.cfg.gripper.reg_operating_mode, int(self.cfg.gripper.operating_mode_value))
        )
        time.sleep(0.2)
        self._flange_write_and_ack(self._flange_modbus_fc06(self.cfg.gripper.reg_torque_enable, 1))
        time.sleep(0.2)
        if int(self.cfg.gripper.init_goal_current) >= 0:
            self._flange_write_and_ack(
                self._flange_modbus_fc06(self.cfg.gripper.reg_goal_current, int(self.cfg.gripper.init_goal_current))
            )
            time.sleep(0.2)
        if int(self.cfg.gripper.init_goal_velocity) >= 0:
            self._flange_write_and_ack(
                self._flange_modbus_fc06(self.cfg.gripper.reg_goal_velocity, int(self.cfg.gripper.init_goal_velocity))
            )
            time.sleep(0.2)
        self._write_gripper_command(raw_cmd=self.cfg.gripper.open_raw, force=True)

    def _setup_gripper(self) -> None:
        if not self.cfg.gripper.enabled:
            return
        protocol = str(self.cfg.gripper.protocol).lower()
        if protocol == "modbus_tcp_signal":
            self._setup_gripper_tcp_signal()
            return
        if protocol == "flange_serial_fc":
            self._setup_gripper_flange_serial()
            return
        raise ValueError(f"지원하지 않는 gripper protocol: {self.cfg.gripper.protocol}")

    def _ratio_to_raw(self, close_ratio: float) -> int:
        close_ratio = float(np.clip(close_ratio, 0.0, 1.0))
        raw = self.cfg.gripper.open_raw + close_ratio * (self.cfg.gripper.close_raw - self.cfg.gripper.open_raw)
        return int(round(raw))

    def _raw_to_ratio(self, raw_val: float) -> float:
        span = float(self.cfg.gripper.close_raw - self.cfg.gripper.open_raw)
        if abs(span) < 1.0e-6:
            return 0.0
        ratio = (float(raw_val) - float(self.cfg.gripper.open_raw)) / span
        return float(np.clip(ratio, 0.0, 1.0))

    def _write_gripper_command(self, raw_cmd: int, force: bool = False) -> None:
        if not self.cfg.gripper.enabled:
            return
        assert self._dr is not None
        now = time.perf_counter()
        deadband = abs(int(self.cfg.gripper.command_deadband_raw))
        if self._last_gripper_raw is not None and not force:
            if abs(int(raw_cmd) - int(self._last_gripper_raw)) <= deadband:
                return
            if (now - self._last_gripper_write_ts) < float(self.cfg.gripper.min_write_interval_s):
                return
        protocol = str(self.cfg.gripper.protocol).lower()
        if protocol == "modbus_tcp_signal":
            self._dr.set_modbus_output(self.cfg.gripper.cmd_signal_name, int(raw_cmd))
        elif protocol == "flange_serial_fc":
            packet = self._flange_modbus_fc16(
                start_addr=self.cfg.gripper.reg_goal_position,
                reg_cnt=2,
                values=[int(raw_cmd), 0],
            )
            if not self._flange_write_and_ack(packet, timeout_s=self.cfg.gripper.ack_timeout_s):
                return
        else:
            return
        self._last_gripper_raw = int(raw_cmd)
        self._last_gripper_write_ts = now
        self._last_gripper_ratio = self._raw_to_ratio(raw_cmd)

    def connect(self) -> None:
        self._dr = self._import_doosan_api()
        assert self._dr is not None
        self._last_speedj_vel_deg_s = np.zeros((6,), dtype=np.float32)
        try:
            self._dr.set_velj(float(self.cfg.servo_vel_deg_s))
            self._dr.set_accj(float(self.cfg.servo_acc_deg_s2))
        except Exception:
            pass
        self._setup_gripper()

    def _read_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        assert self._dr is not None
        posx, _ = self._dr.get_current_posx(self._dr.DR_BASE)
        posx = np.asarray(posx, dtype=np.float32)
        ee_pos_m = (posx[:3] / 1000.0).astype(np.float32)
        if self.cfg.use_posx_orientation:
            ee_quat = euler_xyz_deg_to_quat_wxyz(posx[3:6])
        else:
            ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return ee_pos_m, ee_quat

    def _read_gripper_ratio(self) -> float:
        if not self.cfg.gripper.enabled:
            return float(self._last_gripper_ratio)
        protocol = str(self.cfg.gripper.protocol).lower()
        if protocol == "modbus_tcp_signal":
            if not self.cfg.gripper.pos_signal_name:
                return float(self._last_gripper_ratio)
            assert self._dr is not None
            try:
                raw = self._dr.get_modbus_input(self.cfg.gripper.pos_signal_name)
                return self._raw_to_ratio(float(raw))
            except Exception:
                return float(self._last_gripper_ratio)
        if protocol == "flange_serial_fc":
            try:
                ok, value = self._flange_read_u32(
                    addr=self.cfg.gripper.reg_present_position,
                    word_order=self.cfg.gripper.position_word_order,
                )
                if ok:
                    return self._raw_to_ratio(float(value))
            except Exception:
                return float(self._last_gripper_ratio)
            return float(self._last_gripper_ratio)
        return float(self._last_gripper_ratio)

    def read_robot_state(self) -> RobotState:
        assert self._dr is not None
        q_deg = np.asarray(self._dr.get_current_posj(), dtype=np.float32)[:6]
        v_deg = np.asarray(self._dr.get_current_velj(), dtype=np.float32)[:6]
        ee_pos_m, ee_quat_wxyz = self._read_ee_pose()
        return RobotState(
            joint_pos_rad=np.deg2rad(q_deg).astype(np.float32),
            joint_vel_rad_s=np.deg2rad(v_deg).astype(np.float32),
            gripper_close_ratio=self._read_gripper_ratio(),
            ee_pos_m=ee_pos_m,
            ee_quat_wxyz=ee_quat_wxyz,
        )

    def _read_object_from_json(self) -> ObjectState:
        with open(self.cfg.object_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        pos = np.asarray(payload["position_m"], dtype=np.float32)
        up = np.asarray(payload.get("up_vector_w", [0.0, 0.0, 1.0]), dtype=np.float32)
        class_index = int(payload.get("class_index", self.cfg.object_class_index))
        custom_extra = np.asarray(payload.get("custom_extra", self.cfg.object_fixed_custom_extra), dtype=np.float32).reshape(-1)
        return ObjectState(position_m=pos, up_vector_w=up, class_index=class_index, custom_extra=custom_extra)

    def read_object_state(self) -> ObjectState:
        if self.cfg.object_source == "json":
            try:
                return self._read_object_from_json()
            except Exception:
                pass
        return ObjectState(
            position_m=np.asarray(self.cfg.object_fixed_xyz_m, dtype=np.float32),
            up_vector_w=np.asarray(self.cfg.object_fixed_up_xyz, dtype=np.float32),
            class_index=int(self.cfg.object_class_index),
            custom_extra=np.asarray(self.cfg.object_fixed_custom_extra, dtype=np.float32).reshape(-1),
        )

    def send_joint_targets(
        self,
        joint_targets_rad: np.ndarray,
        gripper_close_ratio: float,
        joint_velocity_cmd_rad_s: np.ndarray | None = None,
    ) -> None:
        assert self._dr is not None
        joint_targets_deg = np.rad2deg(np.asarray(joint_targets_rad, dtype=np.float32)).astype(np.float32)
        command_mode = str(self.cfg.joint_command_mode).lower().strip()

        if command_mode == "speedj":
            if not hasattr(self._dr, "speedj"):
                raise RuntimeError("DSR_ROBOT2에 speedj 함수가 없습니다.")

            n = int(joint_targets_deg.shape[0])
            dt_cmd = max(float(self.cfg.speedj_time_s), 1.0e-6)
            raw_vel_deg_s: np.ndarray | None = None

            # Re-anchor speed command to measured joint state to reduce drift.
            if bool(self.cfg.speedj_reanchor_to_measured):
                try:
                    current_joint_deg = np.asarray(self._dr.get_current_posj(), dtype=np.float32)[:n]
                    raw_vel_deg_s = (joint_targets_deg[: current_joint_deg.shape[0]] - current_joint_deg) / dt_cmd
                except Exception:
                    raw_vel_deg_s = None

            if raw_vel_deg_s is None:
                if joint_velocity_cmd_rad_s is not None:
                    raw_vel_deg_s = np.rad2deg(np.asarray(joint_velocity_cmd_rad_s, dtype=np.float32))[:n]
                else:
                    current_joint_deg = np.asarray(self._dr.get_current_posj(), dtype=np.float32)[:n]
                    raw_vel_deg_s = (joint_targets_deg[: current_joint_deg.shape[0]] - current_joint_deg) / dt_cmd

            if raw_vel_deg_s.shape[0] < n:
                padded_vel = np.zeros((n,), dtype=np.float32)
                padded_vel[: raw_vel_deg_s.shape[0]] = raw_vel_deg_s
                raw_vel_deg_s = padded_vel
            else:
                raw_vel_deg_s = raw_vel_deg_s[:n]

            vel_limit = abs(float(self.cfg.speedj_vel_deg_s))
            raw_vel_deg_s = np.clip(raw_vel_deg_s, -vel_limit, vel_limit)
            acc_limit = abs(float(self.cfg.speedj_acc_deg_s2))
            prev_vel = self._last_speedj_vel_deg_s
            if prev_vel is None or prev_vel.shape[0] != n:
                prev_vel = np.zeros((n,), dtype=np.float32)
            dv_max = acc_limit * dt_cmd
            vel_cmd_deg_s = prev_vel + np.clip(raw_vel_deg_s - prev_vel, -dv_max, dv_max)
            vel_cmd_deg_s = np.clip(vel_cmd_deg_s, -vel_limit, vel_limit).astype(np.float32)

            self._dr.speedj(
                vel_cmd_deg_s.tolist(),
                a=float(self.cfg.speedj_acc_deg_s2),
                t=float(self.cfg.speedj_time_s),
            )
            self._last_speedj_vel_deg_s = vel_cmd_deg_s.copy()
        else:
            self._dr.servoj(
                joint_targets_deg.tolist(),
                v=float(self.cfg.servo_vel_deg_s),
                a=float(self.cfg.servo_acc_deg_s2),
                t=float(self.cfg.servo_time_s),
            )

        if self.cfg.gripper.enabled:
            self._write_gripper_command(self._ratio_to_raw(gripper_close_ratio))

    def shutdown(self) -> None:
        if self._dr is not None and str(self.cfg.joint_command_mode).lower().strip() == "speedj":
            try:
                self._dr.speedj([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], a=float(self.cfg.speedj_acc_deg_s2), t=float(self.cfg.speedj_time_s))
            except Exception:
                pass
        if self.cfg.gripper.enabled:
            try:
                self._write_gripper_command(raw_cmd=self.cfg.gripper.open_raw, force=True)
            except Exception:
                pass
            if str(self.cfg.gripper.protocol).lower() == "flange_serial_fc":
                try:
                    self._flange_serial_close()
                except Exception:
                    pass
