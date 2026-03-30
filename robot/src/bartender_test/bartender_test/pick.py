import cv2
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import rclpy
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

import torch

from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

import message_filters
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
from robotender_msgs.srv import GripperControl
from robotender_msgs.action import PickBottle
from dsr_msgs2.msg import ServojRtStream
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

import DR_init

from .defines import (
    POSJ_PICK_READY,
    GRIPPER_POSITION_OPEN,
    GRIPPER_FORCE_DEFAULT,
    BOTTLE_CONFIG,
    PICK_PLACE_X_OFFSET,
    PICK_PLACE_Y_OFFSET,
    VEL_READY,
    ACC_READY,
    VEL_LIFT,
    ACC_LIFT,
    VEL_RETREAT,
    ACC_RETREAT,
)

ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
RL_READY_POSJ = [90.0, -45.0, 90.0, 0.0, 45.0, -90.0]

ARM_JOINT_MIN = np.array([
    -2.0071287,
    -6.2831855,
    -2.7052603,
    -6.2831855,
    -6.2831855,
    -6.2831855,
], dtype=np.float64)

ARM_JOINT_MAX = np.array([
     2.0071287,
     6.2831855,
     2.7052603,
     6.2831855,
     6.2831855,
     6.2831855,
], dtype=np.float64)

DEFAULT_EXTRA_POS_DIM_4 = np.array([0.02, 0.02, 0.02, 0.02], dtype=np.float64)
DEFAULT_EXTRA_VEL_DIM_4 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def clamp(x: np.ndarray, lo, hi) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def axis_name_to_vector(axis_name: str) -> np.ndarray:
    axis_name = axis_name.lower()
    if axis_name == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if axis_name == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if axis_name == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    raise ValueError(f"Unsupported axis: {axis_name}")


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


@dataclass
class PickRlPolicyConfig:
    policy_path: str = str(Path(__file__).resolve().parents[4] / "robot" / "exported" / "policy_re.pt")
    joint_state_topic: str = "/dsr01/joint_states"
    base_frame: str = "base_link"
    ee_frame: str = "link_6"
    robot_ns: str = "/dsr01"

    policy_hz: float = 30.0
    servo_hz: float = 60.0

    action_scale_rad: float = 0.078
    max_joint_step_rad: float = 0.0028
    max_joint_vel_rad_s: float = 0.115
    max_joint_acc_rad_s2: float = 0.0050

    enable_interpolation: bool = True
    interpolation_alpha: float = 0.11

    enable_jerk_limit: bool = True
    max_joint_jerk_rad_s3: float = 0.14

    global_step_scale: float = 0.82
    global_vel_scale: float = 0.90
    global_acc_scale: float = 0.40

    startup_ramp_sec: float = 2.0
    startup_action_scale: float = 0.08

    goal_y_tolerance: float = 0.15
    pos_tolerance_m: float = 0.012
    stop_vel_tolerance_rad_s: float = 0.03
    settle_count: int = 12

    dof_velocity_scale: float = 0.1
    joint_vel_lpf_alpha: float = 0.20
    joint_vel_clip_rad_s: float = 1.5

    gripper_axis_local: str = "y"
    object_axis_world: str = "z"

    tcp_offset_x: float = 0.0
    tcp_offset_y: float = 0.0
    tcp_offset_z: float = 0.0895

    joint_step_scales: np.ndarray = None
    joint_vel_scales: np.ndarray = None
    joint_acc_scales: np.ndarray = None

    extra_pos_dim_4: np.ndarray = None
    extra_vel_dim_4: np.ndarray = None
    gripper_pos_lower_dim_4: np.ndarray = None
    gripper_pos_upper_dim_4: np.ndarray = None


class TorchPolicy:
    def __init__(self, policy_path: str, device: str = "cpu"):
        self.device = device
        self.model = torch.jit.load(policy_path, map_location=device)
        self.model.eval()

        self.obs_dim = None
        self.act_dim = None
        self._infer_dims()

    def _infer_dims(self):
        try:
            obs_dim = None
            act_dim = None
            for _, mod in self.model.named_modules():
                if hasattr(mod, "weight"):
                    shape = tuple(mod.weight.shape)
                    if len(shape) == 2:
                        if obs_dim is None:
                            obs_dim = shape[1]
                        act_dim = shape[0]
            self.obs_dim = obs_dim
            self.act_dim = act_dim
        except Exception:
            self.obs_dim = 27
            self.act_dim = 6

    @torch.no_grad()
    def infer(self, obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        y = self.model(x)
        return y.squeeze(0).cpu().numpy().astype(np.float64)


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robotender_pick", namespace="/dsr01")
        self._default_callback_group = ReentrantCallbackGroup()

        self.action_cb_group = ReentrantCallbackGroup()
        self.client_cb_group = ReentrantCallbackGroup()

        self.bridge = CvBridge()

        self.get_logger().info("Pick Node Starting (Real Vision)")

        self.intrinsics = None
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None
        self.model = None
        self.state = 'IDLE'

        self.rl_cfg = PickRlPolicyConfig(
            joint_step_scales=np.array([0.40, 0.40, 0.50, 0.82, 0.82, 0.82], dtype=np.float64),
            joint_vel_scales=np.array([0.36, 0.40, 0.52, 0.80, 0.80, 0.80], dtype=np.float64),
            joint_acc_scales=np.array([0.18, 0.22, 0.30, 0.60, 0.60, 0.60], dtype=np.float64),
            extra_pos_dim_4=DEFAULT_EXTRA_POS_DIM_4.copy(),
            extra_vel_dim_4=DEFAULT_EXTRA_VEL_DIM_4.copy(),
            gripper_pos_lower_dim_4=np.array(
                [-1.0000467e-03, -1.0000467e-03, -2.6179934e-01, -2.6179934e-01],
                dtype=np.float64,
            ),
            gripper_pos_upper_dim_4=np.array(
                [1.101, 1.101, 1.3089969, 1.3089969],
                dtype=np.float64,
            ),
        )

        self.policy = None
        self.policy_path = self.rl_cfg.policy_path
        self.q_meas: Optional[np.ndarray] = None
        self.dq_meas: Optional[np.ndarray] = None
        self.dq_meas_filtered: Optional[np.ndarray] = None
        self.last_q_meas_for_dq: Optional[np.ndarray] = None
        self.last_q_meas_time_for_dq: Optional[float] = None
        self.q_obs_10: Optional[np.ndarray] = None
        self.dq_obs_10: Optional[np.ndarray] = None
        self.last_q_cmd: Optional[np.ndarray] = None
        self.last_qd_cmd: Optional[np.ndarray] = None
        self.last_qdd_cmd: Optional[np.ndarray] = None
        self.last_qdd_cmd_filtered: Optional[np.ndarray] = None
        self.last_q_cmd_filtered: Optional[np.ndarray] = None
        self.arm_q_targets: Optional[np.ndarray] = None
        self.arm_dof_speed_scales = np.full(6, 0.45, dtype=np.float64)
        self.last_delta_cmd = np.zeros(6, dtype=np.float64)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.reached_counter = 0
        self.start_time_monotonic = time.monotonic()

        self.policy_dt = 1.0 / self.rl_cfg.policy_hz
        self.servo_dt = 1.0 / self.rl_cfg.servo_hz
        self.latest_action = np.zeros(6, dtype=np.float64)
        self.next_policy_q_cmd: Optional[np.ndarray] = None
        self.next_policy_qd_cmd: Optional[np.ndarray] = None
        self.next_policy_qdd_cmd: Optional[np.ndarray] = None
        self.last_interp_q_cmd: Optional[np.ndarray] = None

        self.last_robot_grasp_pos_w: Optional[np.ndarray] = None
        self.last_object_grasp_pos_w: Optional[np.ndarray] = None
        self.last_axis_align: float = 0.0

        self.per_joint_step_limit = (
            np.full(6, self.rl_cfg.max_joint_step_rad, dtype=np.float64)
            * float(self.rl_cfg.global_step_scale)
            * self.rl_cfg.joint_step_scales.astype(np.float64)
        )
        self.per_joint_vel_limit = (
            np.full(6, self.rl_cfg.max_joint_vel_rad_s, dtype=np.float64)
            * float(self.rl_cfg.global_vel_scale)
            * self.rl_cfg.joint_vel_scales.astype(np.float64)
        )
        self.per_joint_acc_limit = (
            np.full(6, self.rl_cfg.max_joint_acc_rad_s2, dtype=np.float64)
            * float(self.rl_cfg.global_acc_scale)
            * self.rl_cfg.joint_acc_scales.astype(np.float64)
        )

        self.R = np.array([
            [-0.788489317968,  -0.614148198918, -0.0332653756482],
            [-0.0868309265704,  0.0576098432706,  0.994555929121],
            [-0.608888319515,   0.787085189623,  -0.0987518031922],
        ], dtype=np.float64)
        self.t = np.array([521.115058698, 170.946228749, 834.749571453], dtype=np.float64)

        self.gripper_cb_group = MutuallyExclusiveCallbackGroup()
        self.gripper_move_cli = self.create_client(
            GripperControl,
            'robotender_gripper/move',
            callback_group=self.gripper_cb_group
        )

        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera_1/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera_1/aligned_depth_to_color/image_raw')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera_1/aligned_depth_to_color/camera_info')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.info_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synced_callback)

        self.last_pose_pub = self.create_publisher(Float64MultiArray, 'robotender_pick/last_pose', 10)
        self.target_pub = self.create_publisher(PoseStamped, 'robotender_pick/policy_target', 10)
        self.rt_stream_pub = self.create_publisher(ServojRtStream, '/dsr01/servoj_rt_stream', 10)
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.rl_cfg.joint_state_topic,
            self._joint_state_cb,
            10,
        )

        self._action_server = ActionServer(
            self, PickBottle, 'robotender_pick/execute',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_cb_group
        )

        self.load_yolo()
        self._load_policy()

    def _load_policy(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device != "cuda":
                self.get_logger().warn("CUDA is not available. RL policy will run on CPU.")
            self.policy = TorchPolicy(self.policy_path, device=device)
            self.get_logger().info(
                f"Loaded RL policy: {self.policy_path}"
                f"(device={device}, obs_dim={self.policy.obs_dim}, act_dim={self.policy.act_dim})"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load RL policy: {e}")
            self.policy = None

    def _joint_state_cb(self, msg: JointState):
        if len(msg.position) < 6:
            return

        q_now = np.array(msg.position[:6], dtype=np.float64)
        prev_q = None if self.q_meas is None else self.q_meas.copy()
        prev_t = self.last_q_meas_time_for_dq
        now_sec = time.monotonic()

        self.q_meas = q_now
        if self.arm_q_targets is None:
            self.arm_q_targets = self.q_meas.copy()
        elif np.linalg.norm(self.arm_q_targets - self.q_meas) > 0.2:
            self.get_logger().warn("arm_q_targets desynced. resyncing to measured q.")
            self.arm_q_targets = self.q_meas.copy()

        if len(msg.velocity) >= 6:
            arm_dq = np.array(msg.velocity[:6], dtype=np.float64)
        else:
            if prev_q is None or prev_t is None:
                arm_dq = np.zeros(6, dtype=np.float64)
            else:
                dt = max(now_sec - prev_t, 1e-6)
                arm_dq = (q_now - prev_q) / dt

        self.dq_meas = arm_dq.copy()
        arm_dq = np.clip(
            arm_dq,
            -self.rl_cfg.joint_vel_clip_rad_s,
            self.rl_cfg.joint_vel_clip_rad_s,
        )

        alpha_v = float(self.rl_cfg.joint_vel_lpf_alpha)
        if self.dq_meas_filtered is None:
            self.dq_meas_filtered = arm_dq.copy()
        else:
            self.dq_meas_filtered = (
                (1.0 - alpha_v) * self.dq_meas_filtered
                + alpha_v * arm_dq
            )

        self.last_q_meas_for_dq = q_now.copy()
        self.last_q_meas_time_for_dq = now_sec

        grip_q = self.rl_cfg.extra_pos_dim_4.astype(np.float64)
        grip_dq = self.rl_cfg.extra_vel_dim_4.astype(np.float64)
        arm_dq_obs = self.dq_meas_filtered.copy()

        self.q_obs_10 = np.concatenate([self.q_meas.copy(), grip_q], axis=0)
        self.dq_obs_10 = np.concatenate([arm_dq_obs, grip_dq], axis=0)

    def _get_startup_scale(self) -> float:
        elapsed = time.monotonic() - self.start_time_monotonic
        if self.rl_cfg.startup_ramp_sec <= 1e-6:
            return 1.0
        alpha = np.clip(elapsed / self.rl_cfg.startup_ramp_sec, 0.0, 1.0)
        return float(
            self.rl_cfg.startup_action_scale
            + (1.0 - self.rl_cfg.startup_action_scale) * alpha
        )

    def _publish_target(self, target_xyz_m: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.rl_cfg.base_frame
        msg.pose.position.x = float(target_xyz_m[0])
        msg.pose.position.y = float(target_xyz_m[1])
        msg.pose.position.z = float(target_xyz_m[2])
        msg.pose.orientation.w = 1.0
        self.target_pub.publish(msg)

    def _get_ee_pose(self):
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.rl_cfg.base_frame,
                self.rl_cfg.ee_frame,
                rclpy.time.Time(),
            )
            t = tfm.transform.translation
            q = tfm.transform.rotation
            ee_pos = np.array([t.x, t.y, t.z], dtype=np.float64)
            ee_rot = quat_to_rotmat(q.x, q.y, q.z, q.w)
            return ee_pos, ee_rot
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _get_robot_grasp_pos_w(self, ee_pos: np.ndarray, ee_rot: np.ndarray) -> np.ndarray:
        tcp_offset_local = np.array([
            self.rl_cfg.tcp_offset_x,
            self.rl_cfg.tcp_offset_y,
            self.rl_cfg.tcp_offset_z,
        ], dtype=np.float64)
        return ee_pos + ee_rot @ tcp_offset_local

    def _get_gripper_axis_w(self, ee_rot: np.ndarray) -> np.ndarray:
        axis_local = axis_name_to_vector(self.rl_cfg.gripper_axis_local)
        return safe_normalize(ee_rot @ axis_local)

    def _get_tcp_axis_w(self, ee_rot: np.ndarray) -> np.ndarray:
        tcp_axis_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return safe_normalize(ee_rot @ tcp_axis_local)

    def _get_object_axis_w(self) -> np.ndarray:
        return safe_normalize(axis_name_to_vector(self.rl_cfg.object_axis_world))

    def normalize_joint_pos_obs_10(self, q_obs_10: np.ndarray) -> np.ndarray:
        arm_q = q_obs_10[:6]
        grip_q = q_obs_10[6:10]

        arm_center = 0.5 * (ARM_JOINT_MIN + ARM_JOINT_MAX)
        arm_half = 0.5 * (ARM_JOINT_MAX - ARM_JOINT_MIN)
        arm_q_norm = (arm_q - arm_center) / np.maximum(arm_half, 1e-6)
        arm_q_norm = np.clip(arm_q_norm, -1.0, 1.0)

        grip_center = 0.5 * (
            self.rl_cfg.gripper_pos_lower_dim_4 + self.rl_cfg.gripper_pos_upper_dim_4
        )
        grip_half = 0.5 * (
            self.rl_cfg.gripper_pos_upper_dim_4 - self.rl_cfg.gripper_pos_lower_dim_4
        )
        grip_q_norm = (grip_q - grip_center) / np.maximum(grip_half, 1e-6)
        grip_q_norm = np.clip(grip_q_norm, -1.0, 1.0)

        return np.concatenate([arm_q_norm, grip_q_norm], axis=0)

    def build_joint_vel_obs_10(self, dq_obs_10: np.ndarray) -> np.ndarray:
        dq_scaled = dq_obs_10 * self.rl_cfg.dof_velocity_scale
        dq_scaled = np.clip(dq_scaled, -10.0, 10.0)
        return dq_scaled

    def _build_obs_27(
        self,
        q_obs_10: np.ndarray,
        dq_obs_10: np.ndarray,
        robot_grasp_pos_w: np.ndarray,
        object_grasp_pos_w: np.ndarray,
        tcp_axis_w: np.ndarray,
        gripper_axis_w: np.ndarray,
        object_axis_w: np.ndarray,
    ) -> np.ndarray:
        joint_pos_obs = self.normalize_joint_pos_obs_10(q_obs_10)
        joint_vel_obs = self.build_joint_vel_obs_10(dq_obs_10)

        to_object = object_grasp_pos_w - robot_grasp_pos_w
        xz_error = float(np.linalg.norm(to_object[[0, 2]]))
        y_error = float(abs(to_object[1]))

        gripper_axis_w = safe_normalize(gripper_axis_w)
        object_axis_w = safe_normalize(object_axis_w)
        axis_align = float(np.dot(gripper_axis_w, object_axis_w))
        axis_align = float(np.clip(axis_align, -1.0, 1.0))
        axis_align = float(abs(axis_align))

        tcp_axis_w = safe_normalize(tcp_axis_w)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tcp_up_dot = float(np.dot(tcp_axis_w, world_up))
        tcp_up_dot = float(np.clip(tcp_up_dot, -1.0, 1.0))
        tcp_horizontal_align = 1.0 - float(abs(tcp_up_dot))

        obs = np.concatenate([
            joint_pos_obs,
            joint_vel_obs,
            to_object.astype(np.float64),
            np.array([xz_error], dtype=np.float64),
            np.array([y_error], dtype=np.float64),
            np.array([axis_align], dtype=np.float64),
            np.array([tcp_horizontal_align], dtype=np.float64),
        ], axis=0)
        return np.clip(obs, -5.0, 5.0).astype(np.float32)

    def _make_joint_command(self, q_now, dq_now, action, dt):
        if self.arm_q_targets is None:
            self.arm_q_targets = q_now.copy()

        arm_dim = min(6, action.shape[0], q_now.shape[0])
        action = clamp(action[:arm_dim], -1.0, 1.0)
        startup_scale = self._get_startup_scale()

        delta = (
            self.arm_dof_speed_scales[:arm_dim]
            * dt
            * action[:arm_dim]
            * self.rl_cfg.action_scale_rad
            * startup_scale
        )
        delta = np.clip(
            delta,
            -self.per_joint_step_limit[:arm_dim],
            self.per_joint_step_limit[:arm_dim],
        )

        self.last_delta_cmd = np.zeros_like(q_now)
        self.last_delta_cmd[:arm_dim] = delta

        targets = self.arm_q_targets.copy()
        targets[:arm_dim] = self.arm_q_targets[:arm_dim] + delta
        targets[:arm_dim] = np.clip(
            targets[:arm_dim],
            ARM_JOINT_MIN[:arm_dim],
            ARM_JOINT_MAX[:arm_dim],
        )

        qd_cmd = (targets - q_now) / max(dt, 1e-6)
        qd_cmd = clamp(
            qd_cmd,
            -self.per_joint_vel_limit[:arm_dim],
            self.per_joint_vel_limit[:arm_dim],
        )

        if self.last_qd_cmd is None:
            qdd_cmd = np.zeros_like(qd_cmd)
        else:
            qdd_cmd = (qd_cmd - self.last_qd_cmd[:arm_dim]) / max(dt, 1e-6)

        qdd_cmd = clamp(
            qdd_cmd,
            -self.per_joint_acc_limit[:arm_dim],
            self.per_joint_acc_limit[:arm_dim],
        )

        if self.rl_cfg.enable_jerk_limit:
            if self.last_qdd_cmd_filtered is None:
                self.last_qdd_cmd_filtered = np.zeros(6, dtype=np.float64)
                self.last_qdd_cmd_filtered[:arm_dim] = qdd_cmd.copy()
            else:
                jerk = (qdd_cmd - self.last_qdd_cmd_filtered[:arm_dim]) / max(dt, 1e-6)
                jerk = clamp(
                    jerk,
                    -self.rl_cfg.max_joint_jerk_rad_s3,
                    self.rl_cfg.max_joint_jerk_rad_s3,
                )
                qdd_cmd = self.last_qdd_cmd_filtered[:arm_dim] + jerk * dt
                self.last_qdd_cmd_filtered[:arm_dim] = qdd_cmd.copy()

        q_cmd = q_now.copy()
        q_cmd[:arm_dim] = q_now[:arm_dim] + qd_cmd * dt
        q_cmd[:arm_dim] = np.clip(
            q_cmd[:arm_dim],
            ARM_JOINT_MIN[:arm_dim],
            ARM_JOINT_MAX[:arm_dim],
        )

        self.arm_q_targets = q_cmd.copy()

        qd_cmd_full = np.zeros(6, dtype=np.float64)
        qdd_cmd_full = np.zeros(6, dtype=np.float64)
        qd_cmd_full[:arm_dim] = qd_cmd
        qdd_cmd_full[:arm_dim] = qdd_cmd
        return q_cmd, qd_cmd_full, qdd_cmd_full

    def _send_servoj_rt(self, q_cmd, qd_cmd, qdd_cmd, dt):
        msg = ServojRtStream()
        msg.pos = np.rad2deg(q_cmd).astype(np.float64).tolist()
        msg.vel = np.rad2deg(qd_cmd).astype(np.float64).tolist()
        msg.acc = np.rad2deg(qdd_cmd).astype(np.float64).tolist()
        msg.time = float(dt)
        self.rt_stream_pub.publish(msg)

    def _servo_interpolate_and_send(self):
        if self.next_policy_q_cmd is None:
            return

        q_cmd_target = self.next_policy_q_cmd.copy()
        if not self.rl_cfg.enable_interpolation:
            q_cmd_send = q_cmd_target.copy()
        else:
            if self.last_interp_q_cmd is None:
                self.last_interp_q_cmd = (
                    self.q_meas.copy() if self.q_meas is not None else q_cmd_target.copy()
                )
            beta = float(self.rl_cfg.interpolation_alpha)
            q_cmd_send = (1.0 - beta) * self.last_interp_q_cmd + beta * q_cmd_target

        max_servo_step = self.per_joint_vel_limit * self.servo_dt
        if self.last_q_cmd is None:
            q_ref = self.q_meas.copy() if self.q_meas is not None else q_cmd_send.copy()
        else:
            q_ref = self.last_q_cmd.copy()
        q_cmd_send = np.clip(q_cmd_send, q_ref - max_servo_step, q_ref + max_servo_step)

        if self.last_q_cmd is None:
            qd_cmd_send = np.zeros_like(q_cmd_send)
        else:
            qd_cmd_send = (q_cmd_send - self.last_q_cmd) / max(self.servo_dt, 1e-6)
        qd_cmd_send = clamp(qd_cmd_send, -self.per_joint_vel_limit, self.per_joint_vel_limit)

        if self.last_qd_cmd is None:
            qdd_cmd_send = np.zeros_like(qd_cmd_send)
        else:
            qdd_cmd_send = (qd_cmd_send - self.last_qd_cmd) / max(self.servo_dt, 1e-6)
        qdd_cmd_send = clamp(qdd_cmd_send, -self.per_joint_acc_limit, self.per_joint_acc_limit)

        self._send_servoj_rt(q_cmd_send, qd_cmd_send, qdd_cmd_send, self.servo_dt)

        self.last_interp_q_cmd = q_cmd_send.copy()
        self.last_q_cmd = q_cmd_send.copy()
        self.last_qd_cmd = qd_cmd_send.copy()
        self.last_qdd_cmd = qdd_cmd_send.copy()

    def _check_goal(self, robot_grasp_pos_w: np.ndarray, object_grasp_pos_w: np.ndarray) -> bool:
        to_object = object_grasp_pos_w - robot_grasp_pos_w
        xz_error = float(np.linalg.norm(to_object[[0, 2]]))
        y_error = float(abs(to_object[1]))

        if self.last_qd_cmd is None:
            speed = 0.0
        else:
            speed = float(np.linalg.norm(self.last_qd_cmd))

        if (
            xz_error < self.rl_cfg.pos_tolerance_m
            and y_error < self.rl_cfg.goal_y_tolerance
            and speed < self.rl_cfg.stop_vel_tolerance_rad_s
        ):
            self.reached_counter += 1
        else:
            self.reached_counter = 0

        self.get_logger().info(
            f"[RL_GOAL_CHECK] xz_error={xz_error:.4f} "
            f"y_error={y_error:.4f} speed={speed:.4f} "
            f"counter={self.reached_counter}/{self.rl_cfg.settle_count}"
        )
        return self.reached_counter >= self.rl_cfg.settle_count

    def _move_to_rl_ready_pose(self):
        from DSR_ROBOT2 import movej
        self.get_logger().info(f"Moving to RL ready pose before policy loop: {RL_READY_POSJ}")
        movej(RL_READY_POSJ, vel=VEL_READY, acc=ACC_READY)
        self.get_logger().info("movej(RL_READY_POSJ) returned.")

    def _approach_logic_rl(self, p_robot):
        if self.policy is None:
            raise RuntimeError("RL policy is not loaded")

        if self.q_meas is None or self.dq_meas is None:
            self.get_logger().warn("Joint state not ready yet. Waiting before RL loop...")
            t0 = time.time()
            while (self.q_meas is None or self.dq_meas is None) and (time.time() - t0) < 2.0:
                time.sleep(0.02)
            if self.q_meas is None:
                raise RuntimeError("Joint state was not received before RL loop")

        self._move_to_rl_ready_pose()
        time.sleep(0.5)

        target_xyz_m = np.array([
            (p_robot[0] + PICK_PLACE_X_OFFSET) / 1000.0,
            (p_robot[1] + PICK_PLACE_Y_OFFSET) / 1000.0,
            p_robot[2] / 1000.0,
        ], dtype=np.float64)
        self._publish_target(target_xyz_m)

        self.arm_q_targets = None
        self.last_interp_q_cmd = None
        self.last_q_cmd = None
        self.last_qd_cmd = None
        self.last_qdd_cmd = None
        self.last_qdd_cmd_filtered = None
        self.last_q_cmd_filtered = None
        self.next_policy_q_cmd = None
        self.next_policy_qd_cmd = None
        self.next_policy_qdd_cmd = None
        self.reached_counter = 0
        self.start_time_monotonic = time.monotonic()

        dt_policy = self.policy_dt
        dt_servo = self.servo_dt
        servo_per_policy = max(1, int(round(self.rl_cfg.servo_hz / self.rl_cfg.policy_hz)))
        max_policy_steps = int(self.rl_cfg.policy_hz * 20.0)

        self.get_logger().info(
            f"[RL] start target_xyz_m={np.round(target_xyz_m, 4).tolist()}, "
            f"policy_hz={self.rl_cfg.policy_hz:.1f}, servo_hz={self.rl_cfg.servo_hz:.1f}"
        )

        for step in range(max_policy_steps):
            if self.q_meas is None:
                time.sleep(dt_policy)
                continue

            dq_now = self.dq_meas.copy() if self.dq_meas is not None else np.zeros(6, dtype=np.float64)
            ee_state = self._get_ee_pose()
            if ee_state is None:
                self.get_logger().warn("[RL] Waiting for TF ee pose...")
                time.sleep(dt_policy)
                continue

            ee_pos, ee_rot = ee_state
            robot_grasp_pos_w = self._get_robot_grasp_pos_w(ee_pos, ee_rot)
            object_grasp_pos_w = target_xyz_m.copy()
            tcp_axis_w = self._get_tcp_axis_w(ee_rot)
            gripper_axis_w = self._get_gripper_axis_w(ee_rot)
            object_axis_w = self._get_object_axis_w()
            axis_align = float(np.clip(np.dot(gripper_axis_w, object_axis_w), -1.0, 1.0))

            q_obs_10 = (
                self.q_obs_10.copy()
                if self.q_obs_10 is not None
                else np.concatenate([self.q_meas.copy(), self.rl_cfg.extra_pos_dim_4], axis=0)
            )
            dq_obs_10 = (
                self.dq_obs_10.copy()
                if self.dq_obs_10 is not None
                else np.concatenate([dq_now.copy(), self.rl_cfg.extra_vel_dim_4], axis=0)
            )

            obs = self._build_obs_27(
                q_obs_10=q_obs_10,
                dq_obs_10=dq_obs_10,
                robot_grasp_pos_w=robot_grasp_pos_w,
                object_grasp_pos_w=object_grasp_pos_w,
                tcp_axis_w=tcp_axis_w,
                gripper_axis_w=gripper_axis_w,
                object_axis_w=object_axis_w,
            )

            raw_action = self.policy.infer(obs)
            if raw_action.shape[0] != 6:
                raise RuntimeError(f"Policy output dim must be 6, got {raw_action.shape[0]}")

            action = clamp(raw_action, -1.0, 1.0)
            q_cmd, qd_cmd, qdd_cmd = self._make_joint_command(
                q_now=self.q_meas.copy(),
                dq_now=dq_now,
                action=action,
                dt=dt_policy,
            )

            self.latest_action = action.copy()
            self.next_policy_q_cmd = q_cmd.copy()
            self.next_policy_qd_cmd = qd_cmd.copy()
            self.next_policy_qdd_cmd = qdd_cmd.copy()
            self.last_robot_grasp_pos_w = robot_grasp_pos_w.copy()
            self.last_object_grasp_pos_w = object_grasp_pos_w.copy()
            self.last_axis_align = axis_align

            for _ in range(servo_per_policy):
                self._servo_interpolate_and_send()
                time.sleep(dt_servo)

            err = object_grasp_pos_w - robot_grasp_pos_w
            dist = float(np.linalg.norm(err))
            self.get_logger().info(
                f"[RL] step={step:03d} dist={dist:.4f} "
                f"err={np.round(err, 4).tolist()} "
                f"action={np.round(action, 4).tolist()}"
            )

            if self._check_goal(robot_grasp_pos_w, object_grasp_pos_w):
                self.get_logger().info("[RL] target reached and settled.")
                return True

            self._publish_target(object_grasp_pos_w)

        raise RuntimeError("RL approach timeout")

    def load_yolo(self):
        try:
            from ultralytics import YOLO
            model_path = Path(__file__).resolve().parents[4] / "detection" / "weights" / "cam_1.pt"
            self.model = YOLO(str(model_path))
        except Exception as e:
            self.get_logger().error(f"YOLO Fail: {e}")

    def synced_callback(self, color_msg, depth_msg, info_msg):
        if self.state != "DETECTING" and self.latest_cv_color is not None:
            return

        try:
            self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception:
            return

        if self.intrinsics is None and rs is not None:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width, self.intrinsics.height = info_msg.width, info_msg.height
            self.intrinsics.ppx, self.intrinsics.ppy = info_msg.k[2], info_msg.k[5]
            self.intrinsics.fx, self.intrinsics.fy = info_msg.k[0], info_msg.k[4]
            self.intrinsics.coeffs = list(info_msg.d)

    def goal_callback(self, goal_request):
        self.get_logger().info(f"[ACTION] Goal received! State: {self.state}")
        if self.state == "RUNNING" or self.state == "DETECTING":
            self.get_logger().warn(f"[ACTION] Goal REJECTED. Busy in state: {self.state}")
            return GoalResponse.REJECT
        self.get_logger().info("[ACTION] Goal ACCEPTED.")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        from DSR_ROBOT2 import movej, set_robot_mode, wait, ROBOT_MODE_AUTONOMOUS

        self.get_logger().info(f'--- [PICK] EXECUTION STARTED for: {goal_handle.request.bottle_name} ---')
        feedback_msg = PickBottle.Feedback()
        result = PickBottle.Result()
        target_name = goal_handle.request.bottle_name
        self.state = "RUNNING"

        try:
            self.get_logger().info("Step 0: Ensuring Autonomous Mode...")
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            wait(0.2)
            self.get_logger().info("Step 0: Wait completed.")

            self.get_logger().info("Step 1: Preparing (Moving to READY)...")
            feedback_msg.current_state, feedback_msg.progress = "STEP 1: PREPARING", 0.1
            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info(f"Moving to POSJ_PICK_READY: {POSJ_PICK_READY}")
            movej(POSJ_PICK_READY, vel=VEL_READY, acc=ACC_READY)
            self.get_logger().info("movej(READY) returned.")

            if goal_handle.is_cancel_requested:
                self.get_logger().warn("Cancel requested during Step 1.")
                return self._abort(goal_handle, result)

            self.state = "DETECTING"
            self.get_logger().info("Step 2: Detecting bottle. Waiting 1s for camera settle...")
            feedback_msg.current_state, feedback_msg.progress = "STEP 2: DETECTING", 0.3
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1.0)

            p_robot = None
            self.get_logger().info(f"Using REAL vision for {target_name}...")
            for attempt in range(3):
                self.get_logger().info(f"Detection attempt {attempt+1}/3")
                target_data = self.vision_detect(target_name)
                if target_data:
                    p_robot = self.calculate_pose(target_data)
                    self.get_logger().info(f"Detected pose in robot frame: {p_robot}")
                    break
                time.sleep(0.5)

            if p_robot is None:
                self.get_logger().error("Detection failed after 3 attempts.")
                result.success, result.message = False, "Detection failed"
                goal_handle.succeed()
                self.state = "IDLE"
                return result

            cfg = BOTTLE_CONFIG.get(target_name, {})
            release_force = cfg.get('gripper_force', GRIPPER_FORCE_DEFAULT)
            grasp_wait_time = cfg.get('grasp_wait_time', 4.0)

            self.get_logger().info(f"Detection successful. Opening gripper sync for {target_name} (force={release_force})...")
            self._gripper_move_sync(GRIPPER_POSITION_OPEN, release_force)
            time.sleep(grasp_wait_time)

            self.state = "RUNNING"
            self.get_logger().info("Step 3: Moving to bottle approach pose with RL policy...")
            feedback_msg.current_state, feedback_msg.progress = "STEP 3: MOVING_TO_BOTTLE", 0.5
            goal_handle.publish_feedback(feedback_msg)
            self._approach_logic_rl(p_robot)

            if goal_handle.is_cancel_requested:
                self.get_logger().warn("Cancel requested during Step 3.")
                return self._abort(goal_handle, result)

            self.get_logger().info("Step 4: Executing grasp logic...")
            feedback_msg.current_state, feedback_msg.progress = "STEP 4: GRASPING", 0.7
            goal_handle.publish_feedback(feedback_msg)
            self._grasp_logic(target_name)

            self.get_logger().info("Step 5: Pick completed.")
            feedback_msg.current_state, feedback_msg.progress = "STEP 5: COMPLETED", 1.0
            goal_handle.publish_feedback(feedback_msg)

            result.success, result.message = True, "Pick success"
            result.pick_pose = [float(p) for p in p_robot]
            self.get_logger().info("--- [PICK] SUCCESS. Goal Succeeding. ---")
            goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f"--- [PICK] EXECUTION ERROR: {e} ---")
            import traceback
            self.get_logger().error(traceback.format_exc())
            result.success, result.message = False, str(e)
            goal_handle.succeed()
        finally:
            self.state = "IDLE"
        return result

    def _grasp_logic(self, target_name):
        from DSR_ROBOT2 import movel, get_current_posx, fkin
        cfg = BOTTLE_CONFIG.get(target_name, BOTTLE_CONFIG['soju'])
        self.get_logger().info(f"Grasp logic for {target_name}, closing gripper to {cfg['gripper_pos']}")
        self._gripper_move_sync(cfg['gripper_pos'], cfg['gripper_force'])

        grasp_wait_time = cfg['grasp_wait_time']
        self.get_logger().info(f"Wait {grasp_wait_time}s before lifting...")
        time.sleep(grasp_wait_time)

        self.get_logger().info("Step 4.1: Return sequence (Lift then Y-Retreat)...")

        curr = list(get_current_posx()[0])
        self.get_logger().info("Lifting bottle (+30mm Z)")
        target_pos_lift = [curr[0], curr[1], curr[2] + 30.0, curr[3], curr[4], curr[5]]
        movel(target_pos_lift, vel=[VEL_LIFT, VEL_LIFT], acc=[ACC_LIFT, ACC_LIFT])

        ready_posx = list(fkin(POSJ_PICK_READY, ref=0))
        ready_y = ready_posx[1]

        curr_lifted = list(get_current_posx()[0])
        self.get_logger().info(f"Retreating Y to {ready_y:.1f}")
        target_pos_retreat = [curr_lifted[0], ready_y, curr_lifted[2], curr_lifted[3], curr_lifted[4], curr_lifted[5]]
        movel(target_pos_retreat, vel=[VEL_RETREAT, VEL_RETREAT], acc=[ACC_RETREAT, ACC_RETREAT])
        return True

    def _gripper_move_sync(self, pos, force):
        self.get_logger().info(f"Gripper request (Fire & Forget): pos={pos}, force={force}")
        if not self.gripper_move_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Gripper service not available!")
            return False
        req = GripperControl.Request(position=int(pos), force=int(force))

        try:
            self.gripper_move_cli.call_async(req)
            self.get_logger().info("Gripper request sent. Sleeping 2.0s for physical motion...")
            time.sleep(2.0)
            self.get_logger().info("Gripper motion wait complete.")
            return True
        except Exception as e:
            self.get_logger().error(f"Gripper call failed: {e}")
            return False

    def vision_detect(self, target_name):
        self.get_logger().info(f"vision_detect called for {target_name}")
        if self.latest_cv_color is None:
            self.get_logger().warn("vision_detect: latest_cv_color is None!")
            return None
        if self.model is None:
            self.get_logger().error("vision_detect: model is None!")
            return None

        class_id = str(BOTTLE_CONFIG[target_name]['id'])
        self.get_logger().info(f"Running YOLO inference on class_id: {class_id}...")

        img = cv2.flip(self.latest_cv_color.copy(), -1)
        h, w = img.shape[:2]
        results = self.model(img)

        self.get_logger().info(f"YOLO inference completed. Found {len(results)} results.")
        for result in results:
            for box in result.boxes:
                detected_class = str(int(box.cls[0].cpu().numpy()))
                if detected_class == class_id:
                    self.get_logger().info("Target class match found!")
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')
                    u_raw = w - 1 - int((x1 + x2) / 2)
                    v_raw = h - 1 - int((y1 + y2) / 2)
                    roi = self.latest_cv_depth_mm[max(0, v_raw - 5):min(h, v_raw + 6), max(0, u_raw - 5):min(w, u_raw + 6)]
                    valid = roi[roi > 0]
                    if valid.size > 0:
                        depth = np.median(valid)
                        self.get_logger().info(f"Valid depth found: {depth}mm")
                        return {"u_raw": u_raw, "v_raw": v_raw, "depth_mm": depth}

        self.get_logger().warn("Target bottle not found in current frame.")
        return None

    def calculate_pose(self, data):
        depth_m = data["depth_mm"] / 1000.0
        x_cam = (data["u_raw"] - self.intrinsics.ppx) * depth_m / self.intrinsics.fx
        y_cam = (data["v_raw"] - self.intrinsics.ppy) * depth_m / self.intrinsics.fy
        return np.array([x_cam * 1000, y_cam * 1000, depth_m * 1000]) @ self.R + self.t

    def _abort(self, goal_handle, result):
        goal_handle.canceled()
        self.state = "IDLE"
        result.success, result.message = False, "Canceled"
        return result


def main(args=None):
    rclpy.init(args=args)
    from rclpy.executors import MultiThreadedExecutor

    node = RobotControllerNode()

    doosan_node = rclpy.create_node('pick_doosan_internal', namespace='/dsr01')
    doosan_node._default_callback_group = ReentrantCallbackGroup()

    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    DR_init.__dsr__node = doosan_node

    executor = MultiThreadedExecutor(num_threads=20)
    executor.add_node(node)
    executor.add_node(doosan_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        doosan_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
