from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


DEFAULT_OBS_DIM = 29
DEFAULT_ACT_DIM = 7
DEFAULT_OBS_JOINT_DIM = 10
OBJECT_CLASS_COUNT = 3


def compute_obs_extra_dim(
    *,
    include_to_object: bool = True,
    include_lift: bool = True,
    include_gripper_state: bool = True,
    include_object_class: bool = True,
    object_class_dim: int = OBJECT_CLASS_COUNT,
    include_object_up_z: bool = True,
    custom_extra_dim: int = 0,
) -> int:
    dim = 0
    if bool(include_to_object):
        dim += 3
    if bool(include_lift):
        dim += 1
    if bool(include_gripper_state):
        dim += 1
    if bool(include_object_class):
        dim += max(0, int(object_class_dim))
    if bool(include_object_up_z):
        dim += 1
    dim += max(0, int(custom_extra_dim))
    return int(dim)


@dataclass
class PolicyConfig:
    action_scale: float = 2.5
    sim_dt: float = 1.0 / 120.0
    decimation: int = 2
    dof_velocity_scale: float = 0.1
    arm_speed_scale: float = 0.45
    gripper_target_smoothing: float = 0.35
    tcp_offset_open_m: float = 0.107
    tcp_offset_closed_m: float = 0.135
    grasp_center_from_tip_to_ee_m: float = 0.0175
    grasp_offset_z_m: float = 0.085
    joint_1_abs_limit_deg: float = 115.0

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.decimation

    def set_control_dt(self, dt_s: float) -> None:
        dt = float(dt_s)
        if dt <= 0.0:
            raise ValueError(f"control dt must be > 0, got {dt_s}")
        self.sim_dt = dt
        self.decimation = 1

    def set_control_hz(self, hz: float) -> None:
        freq = float(hz)
        if freq <= 0.0:
            raise ValueError(f"control hz must be > 0, got {hz}")
        self.set_control_dt(1.0 / freq)


@dataclass
class RobotState:
    joint_pos_rad: np.ndarray
    joint_vel_rad_s: np.ndarray
    gripper_close_ratio: float
    ee_pos_m: np.ndarray
    ee_quat_wxyz: np.ndarray


@dataclass
class ObjectState:
    position_m: np.ndarray
    up_vector_w: np.ndarray
    class_index: int
    custom_extra: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))


def euler_xyz_deg_to_quat_wxyz(euler_deg_xyz: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(euler_deg_xyz, dtype=np.float32)).tolist()
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    quat = np.asarray([w, x, y, z], dtype=np.float32)
    return quat / (np.linalg.norm(quat) + 1.0e-8)


def quat_apply_np(quat_wxyz: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32)
    v = np.asarray(vec_xyz, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    w, x, y, z = q.tolist()
    q_vec = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)


class ObservationBuilder:
    def __init__(
        self,
        cfg: PolicyConfig,
        joint_lower_rad: np.ndarray,
        joint_upper_rad: np.ndarray,
        obs_joint_dim: int = DEFAULT_OBS_JOINT_DIM,
        target_obs_dim: int = DEFAULT_OBS_DIM,
        include_to_object: bool = True,
        include_lift: bool = True,
        include_gripper_state: bool = True,
        include_object_class: bool = True,
        object_class_dim: int = OBJECT_CLASS_COUNT,
        include_object_up_z: bool = True,
        custom_extra_dim: int = 0,
    ):
        self.cfg = cfg
        self.joint_lower = np.asarray(joint_lower_rad, dtype=np.float32)
        self.joint_upper = np.asarray(joint_upper_rad, dtype=np.float32)
        self.obs_joint_dim = int(obs_joint_dim)
        self.target_obs_dim = int(target_obs_dim)
        self.include_to_object = bool(include_to_object)
        self.include_lift = bool(include_lift)
        self.include_gripper_state = bool(include_gripper_state)
        self.include_object_class = bool(include_object_class)
        self.object_class_dim = max(0, int(object_class_dim))
        self.include_object_up_z = bool(include_object_up_z)
        self.custom_extra_dim = max(0, int(custom_extra_dim))
        self.object_init_height: float | None = None

    def _pad_or_truncate(self, arr: np.ndarray, target_dim: int) -> np.ndarray:
        if arr.shape[0] == target_dim:
            return arr
        if arr.shape[0] > target_dim:
            return arr[:target_dim]
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded

    def _compute_robot_grasp_pos(self, robot: RobotState) -> np.ndarray:
        gripper_state = float(np.clip(robot.gripper_close_ratio, 0.0, 1.0))
        tcp_tip_len = self.cfg.tcp_offset_open_m + gripper_state * (self.cfg.tcp_offset_closed_m - self.cfg.tcp_offset_open_m)
        tcp_len = max(tcp_tip_len - self.cfg.grasp_center_from_tip_to_ee_m, 0.0)
        tcp_local = np.array([0.0, 0.0, tcp_len], dtype=np.float32)
        tcp_world = quat_apply_np(robot.ee_quat_wxyz, tcp_local)
        return robot.ee_pos_m + tcp_world

    def set_object_reference(self, obj: ObjectState) -> None:
        self.object_init_height = float(obj.position_m[2])

    def build(self, robot: RobotState, obj: ObjectState) -> np.ndarray:
        if self.object_init_height is None:
            self.set_object_reference(obj)

        denom = np.maximum(self.joint_upper - self.joint_lower, 1.0e-5)
        dof_pos_scaled = 2.0 * (robot.joint_pos_rad - self.joint_lower) / denom - 1.0
        dof_vel_scaled = robot.joint_vel_rad_s * self.cfg.dof_velocity_scale

        joint_pos_obs = self._pad_or_truncate(dof_pos_scaled.astype(np.float32), self.obs_joint_dim)
        joint_vel_obs = self._pad_or_truncate(dof_vel_scaled.astype(np.float32), self.obs_joint_dim)

        up = np.asarray(obj.up_vector_w, dtype=np.float32)
        up_norm = float(np.linalg.norm(up))
        if up_norm < 1.0e-6:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            up = up / up_norm

        robot_grasp_pos = self._compute_robot_grasp_pos(robot)
        object_grasp_pos = obj.position_m + up * self.cfg.grasp_offset_z_m
        to_object = object_grasp_pos - robot_grasp_pos
        lift_amount = max(float(obj.position_m[2]) - float(self.object_init_height), 0.0)
        gripper_state = float(np.clip(robot.gripper_close_ratio, 0.0, 1.0))

        one_hot = np.zeros((self.object_class_dim,), dtype=np.float32)
        if self.object_class_dim > 0:
            class_idx = int(np.clip(obj.class_index, 0, self.object_class_dim - 1))
            one_hot[class_idx] = 1.0
        object_up_z = float(np.clip(up[2], -1.0, 1.0))
        custom_extra_raw = np.asarray(getattr(obj, "custom_extra", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        custom_extra_obs = self._pad_or_truncate(custom_extra_raw, self.custom_extra_dim)

        obs_parts: list[np.ndarray] = [joint_pos_obs, joint_vel_obs]
        if self.include_to_object:
            obs_parts.append(to_object.astype(np.float32))
        if self.include_lift:
            obs_parts.append(np.array([lift_amount], dtype=np.float32))
        if self.include_gripper_state:
            obs_parts.append(np.array([gripper_state], dtype=np.float32))
        if self.include_object_class and self.object_class_dim > 0:
            obs_parts.append(one_hot)
        if self.include_object_up_z:
            obs_parts.append(np.array([object_up_z], dtype=np.float32))
        if self.custom_extra_dim > 0:
            obs_parts.append(custom_extra_obs.astype(np.float32))

        obs = np.concatenate(obs_parts, axis=0)
        obs = self._pad_or_truncate(obs.astype(np.float32), self.target_obs_dim)
        return np.clip(obs, -5.0, 5.0).astype(np.float32)


if nn is None or torch is None:
    class ActorPolicy:  # type: ignore[no-redef]
        def __init__(
            self,
            obs_dim: int = DEFAULT_OBS_DIM,
            act_dim: int = DEFAULT_ACT_DIM,
            hidden_sizes: tuple[int, ...] = (256, 128, 64),
            activation: str = "elu",
        ):
            raise RuntimeError("ActorPolicy 사용에는 torch가 필요합니다.") from _TORCH_IMPORT_ERROR

        @classmethod
        def from_checkpoint(
            cls,
            checkpoint_path: Path,
            device: object,
            activation: str = "elu",
            obs_dim_override: int | None = None,
            act_dim_override: int | None = None,
            hidden_sizes_override: tuple[int, ...] | None = None,
        ) -> "ActorPolicy":
            raise RuntimeError("체크포인트 로드에는 torch가 필요합니다.") from _TORCH_IMPORT_ERROR

        def summary(self) -> str:
            return "torch unavailable"

        def act(self, obs_np: np.ndarray, device: object) -> np.ndarray:
            raise RuntimeError("정책 추론에는 torch가 필요합니다.") from _TORCH_IMPORT_ERROR

else:
    class ActorPolicy(nn.Module):
        def __init__(
            self,
            obs_dim: int = DEFAULT_OBS_DIM,
            act_dim: int = DEFAULT_ACT_DIM,
            hidden_sizes: tuple[int, ...] = (256, 128, 64),
            activation: str = "elu",
        ):
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.act_dim = int(act_dim)
            self.hidden_sizes = tuple(int(v) for v in hidden_sizes)
            self.activation = str(activation).lower()
            self.actor = self._build_actor(self.obs_dim, self.act_dim, self.hidden_sizes, self.activation)
            self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
            self.register_buffer("obs_std", torch.ones(self.obs_dim))
            self.register_buffer("obs_count", torch.tensor(0.0))

        @staticmethod
        def _activation_factory(name: str) -> nn.Module:
            key = str(name).lower()
            if key == "elu":
                return nn.ELU()
            if key == "relu":
                return nn.ReLU()
            if key == "tanh":
                return nn.Tanh()
            if key == "silu":
                return nn.SiLU()
            raise ValueError(f"지원하지 않는 activation: {name}")

        @classmethod
        def _build_actor(cls, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...], activation: str) -> nn.Sequential:
            layers: list[nn.Module] = []
            in_dim = int(obs_dim)
            for h in hidden_sizes:
                layers.append(nn.Linear(in_dim, int(h)))
                layers.append(cls._activation_factory(activation))
                in_dim = int(h)
            layers.append(nn.Linear(in_dim, int(act_dim)))
            return nn.Sequential(*layers)

        @staticmethod
        def _to_state_dict(raw_ckpt: object) -> dict[str, torch.Tensor]:
            if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
                state_dict = raw_ckpt["model_state_dict"]
            elif isinstance(raw_ckpt, dict):
                state_dict = raw_ckpt
            else:
                raise ValueError("지원하지 않는 체크포인트 형식입니다.")
            if not isinstance(state_dict, dict):
                raise ValueError("state_dict를 찾지 못했습니다.")
            return state_dict

        @staticmethod
        def _extract_actor_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            actor_state = {k: v for k, v in state_dict.items() if isinstance(k, str) and k.startswith("actor.")}
            if len(actor_state) == 0:
                raise RuntimeError("체크포인트에서 actor.* 키를 찾지 못했습니다.")
            return actor_state

        @staticmethod
        def _extract_layer_index(key: str) -> int | None:
            match = re.match(r"actor\.(\d+)\.weight$", key)
            if match is None:
                return None
            return int(match.group(1))

        @classmethod
        def _infer_actor_spec(cls, actor_state: dict[str, torch.Tensor]) -> tuple[int, tuple[int, ...], int]:
            indexed: list[tuple[int, torch.Tensor]] = []
            for key, tensor in actor_state.items():
                idx = cls._extract_layer_index(key)
                if idx is None:
                    continue
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                    continue
                indexed.append((idx, tensor))
            if len(indexed) == 0:
                raise RuntimeError("actor 선형 레이어(weight)를 추론할 수 없습니다.")

            indexed.sort(key=lambda x: x[0])
            first = indexed[0][1]
            last = indexed[-1][1]
            obs_dim = int(first.shape[1])
            act_dim = int(last.shape[0])
            hidden_sizes = tuple(int(t.shape[0]) for _, t in indexed[:-1])
            return obs_dim, hidden_sizes, act_dim

        @classmethod
        def from_checkpoint(
            cls,
            checkpoint_path: Path,
            device: torch.device,
            activation: str = "elu",
            obs_dim_override: int | None = None,
            act_dim_override: int | None = None,
            hidden_sizes_override: tuple[int, ...] | None = None,
        ) -> "ActorPolicy":
            raw = torch.load(str(checkpoint_path), map_location=device)
            state_dict = cls._to_state_dict(raw)
            actor_state = cls._extract_actor_state(state_dict)
            inferred_obs_dim, inferred_hidden_sizes, inferred_act_dim = cls._infer_actor_spec(actor_state)

            obs_dim = int(obs_dim_override) if obs_dim_override is not None else inferred_obs_dim
            act_dim = int(act_dim_override) if act_dim_override is not None else inferred_act_dim
            hidden_sizes = tuple(hidden_sizes_override) if hidden_sizes_override is not None else inferred_hidden_sizes
            model = cls(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes, activation=activation).to(device)

            actor_local_state = {k[len("actor.") :]: v for k, v in actor_state.items() if k.startswith("actor.")}
            try:
                missing, unexpected = model.actor.load_state_dict(actor_local_state, strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"actor 파라미터 shape 불일치: inferred(obs={inferred_obs_dim}, hidden={inferred_hidden_sizes}, act={inferred_act_dim}), "
                    f"applied(obs={obs_dim}, hidden={hidden_sizes}, act={act_dim})"
                ) from exc
            if missing:
                raise RuntimeError(f"actor 파라미터 로드 실패: missing={missing}")
            if unexpected:
                raise RuntimeError(f"actor 파라미터 로드 실패: unexpected={unexpected}")

            with torch.no_grad():
                mean = state_dict.get("actor_obs_normalizer._mean")
                std = state_dict.get("actor_obs_normalizer._std")
                var = state_dict.get("actor_obs_normalizer._var")
                count = state_dict.get("actor_obs_normalizer.count")
                if mean is not None:
                    model.obs_mean.copy_(mean.to(device=device, dtype=torch.float32).view(-1))
                if std is not None:
                    model.obs_std.copy_(std.to(device=device, dtype=torch.float32).view(-1))
                elif var is not None:
                    model.obs_std.copy_(torch.sqrt(torch.clamp(var.to(device=device, dtype=torch.float32).view(-1), min=1.0e-8)))
                if count is not None:
                    model.obs_count.copy_(count.to(device=device, dtype=torch.float32).view(()))
            model.eval()
            return model

        def summary(self) -> str:
            return f"obs_dim={self.obs_dim}, hidden={self.hidden_sizes}, act_dim={self.act_dim}, activation={self.activation}"

        def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
            if float(self.obs_count.item()) <= 0.0:
                return obs
            std = torch.clamp(self.obs_std, min=1.0e-6)
            return (obs - self.obs_mean) / std

        @torch.no_grad()
        def act(self, obs_np: np.ndarray, device: torch.device) -> np.ndarray:
            obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32).unsqueeze(0)
            obs = self._normalize_obs(obs)
            action = self.actor(obs).squeeze(0)
            return torch.clamp(action, -1.0, 1.0).cpu().numpy().astype(np.float32)


class ActionController:
    def __init__(self, cfg: PolicyConfig, joint_lower_rad: np.ndarray, joint_upper_rad: np.ndarray):
        self.cfg = cfg
        self.joint_lower = np.asarray(joint_lower_rad, dtype=np.float32).copy()
        self.joint_upper = np.asarray(joint_upper_rad, dtype=np.float32).copy()
        self.joint_targets: np.ndarray | None = None
        self.gripper_target = 0.0
        joint1_limit = math.radians(self.cfg.joint_1_abs_limit_deg)
        self.joint_lower[0] = max(self.joint_lower[0], -joint1_limit)
        self.joint_upper[0] = min(self.joint_upper[0], joint1_limit)

    def reset(self, current_joint_pos_rad: np.ndarray, current_gripper_close_ratio: float) -> None:
        self.joint_targets = np.asarray(current_joint_pos_rad, dtype=np.float32).copy()
        self.gripper_target = float(np.clip(current_gripper_close_ratio, 0.0, 1.0))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        if self.joint_targets is None:
            raise RuntimeError("ActionController.reset()이 먼저 호출되어야 합니다.")
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        prev_targets = self.joint_targets.copy()
        joint_velocity_cmd = np.zeros_like(self.joint_targets, dtype=np.float32)

        arm_dim = min(6, action.shape[0], self.joint_targets.shape[0])
        if arm_dim > 0:
            delta = self.cfg.arm_speed_scale * self.cfg.control_dt * self.cfg.action_scale * action[:arm_dim]
            next_targets = np.clip(
                self.joint_targets[:arm_dim] + delta,
                self.joint_lower[:arm_dim],
                self.joint_upper[:arm_dim],
            )
            applied_delta = next_targets - prev_targets[:arm_dim]
            self.joint_targets[:arm_dim] = next_targets
            dt = max(float(self.cfg.control_dt), 1.0e-6)
            joint_velocity_cmd[:arm_dim] = applied_delta / dt

        if action.shape[0] > 6:
            desired_close_ratio = 0.5 * (float(action[6]) + 1.0)
            alpha = self.cfg.gripper_target_smoothing
            self.gripper_target = self.gripper_target + alpha * (desired_close_ratio - self.gripper_target)
            self.gripper_target = float(np.clip(self.gripper_target, 0.0, 1.0))

        return self.joint_targets.copy(), float(self.gripper_target), joint_velocity_cmd.astype(np.float32)


def parse_comma_floats(raw: str, expected_len: int) -> np.ndarray:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) != expected_len:
        raise ValueError(f"길이가 {expected_len}여야 합니다. 입력={values}")
    return np.asarray(values, dtype=np.float32)


def parse_vec3(raw: str) -> tuple[float, float, float]:
    vec = parse_comma_floats(raw, expected_len=3)
    return float(vec[0]), float(vec[1]), float(vec[2])


def parse_int_csv(raw: str) -> tuple[int, ...]:
    values = [v.strip() for v in str(raw).split(",") if v.strip()]
    if len(values) == 0:
        return tuple()
    return tuple(int(v) for v in values)
