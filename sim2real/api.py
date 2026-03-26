from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .core import (
    DEFAULT_OBS_JOINT_DIM,
    ActionController,
    ActorPolicy,
    ObjectState,
    ObservationBuilder,
    PolicyConfig,
    RobotState,
    compute_obs_extra_dim,
)


@dataclass
class Sim2RealRuntimeConfig:
    checkpoint: str | Path | None = None
    device: str = "cpu"

    # Optional policy overrides
    policy_activation: str = "elu"
    policy_hidden_sizes: tuple[int, ...] | None = None
    policy_obs_dim: int | None = None
    policy_act_dim: int | None = None

    # Observation composition
    obs_joint_dim: int | None = None
    obs_include_to_object: bool = True
    obs_include_lift: bool = True
    obs_include_gripper_state: bool = True
    obs_include_object_class: bool = True
    obs_object_class_dim: int = 3
    obs_include_object_up_z: bool = True
    obs_custom_extra_dim: int = 0

    # Robot limits (rad)
    joint_lower_rad: tuple[float, float, float, float, float, float] = (
        -2.0071287,
        -3.1415927,
        -3.1415927,
        -3.1415927,
        -3.1415927,
        -3.1415927,
    )
    joint_upper_rad: tuple[float, float, float, float, float, float] = (
        2.0071287,
        3.1415927,
        3.1415927,
        3.1415927,
        3.1415927,
        3.1415927,
    )
    control_hz: float | None = None

    policy_cfg: PolicyConfig = field(default_factory=PolicyConfig)


class Sim2RealRuntime:
    """Embed-friendly inference runtime.

    Usage:
      1) Create once with checkpoint/config
      2) Call reset(initial_robot_state, initial_object_state)
      3) Call step(robot_state, object_state) every control tick
    """

    def __init__(self, cfg: Sim2RealRuntimeConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if cfg.control_hz is not None:
            cfg.policy_cfg.set_control_hz(float(cfg.control_hz))

        checkpoint_raw = None if cfg.checkpoint is None else str(cfg.checkpoint).strip()
        if not checkpoint_raw:
            raise ValueError("Sim2RealRuntimeConfig.checkpoint를 지정하세요. 예: '/path/to/model.pt'")
        checkpoint_path = Path(checkpoint_raw).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")

        self.policy = ActorPolicy.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device,
            activation=cfg.policy_activation,
            obs_dim_override=cfg.policy_obs_dim,
            act_dim_override=cfg.policy_act_dim,
            hidden_sizes_override=cfg.policy_hidden_sizes,
        )

        obs_extra_dim = compute_obs_extra_dim(
            include_to_object=bool(cfg.obs_include_to_object),
            include_lift=bool(cfg.obs_include_lift),
            include_gripper_state=bool(cfg.obs_include_gripper_state),
            include_object_class=bool(cfg.obs_include_object_class),
            object_class_dim=int(cfg.obs_object_class_dim),
            include_object_up_z=bool(cfg.obs_include_object_up_z),
            custom_extra_dim=int(cfg.obs_custom_extra_dim),
        )
        if cfg.obs_joint_dim is None:
            remain = int(self.policy.obs_dim) - int(obs_extra_dim)
            if remain >= 2 and remain % 2 == 0:
                obs_joint_dim = remain // 2
            else:
                obs_joint_dim = DEFAULT_OBS_JOINT_DIM
        else:
            obs_joint_dim = int(cfg.obs_joint_dim)

        self.obs_joint_dim = int(obs_joint_dim)

        joint_lower = np.asarray(cfg.joint_lower_rad, dtype=np.float32)
        joint_upper = np.asarray(cfg.joint_upper_rad, dtype=np.float32)
        if joint_lower.shape[0] != 6 or joint_upper.shape[0] != 6:
            raise ValueError("joint_lower_rad / joint_upper_rad 길이는 6이어야 합니다.")

        self.obs_builder = ObservationBuilder(
            cfg=cfg.policy_cfg,
            joint_lower_rad=joint_lower,
            joint_upper_rad=joint_upper,
            obs_joint_dim=self.obs_joint_dim,
            target_obs_dim=int(self.policy.obs_dim),
            include_to_object=bool(cfg.obs_include_to_object),
            include_lift=bool(cfg.obs_include_lift),
            include_gripper_state=bool(cfg.obs_include_gripper_state),
            include_object_class=bool(cfg.obs_include_object_class),
            object_class_dim=int(cfg.obs_object_class_dim),
            include_object_up_z=bool(cfg.obs_include_object_up_z),
            custom_extra_dim=int(cfg.obs_custom_extra_dim),
        )
        self.controller = ActionController(cfg=cfg.policy_cfg, joint_lower_rad=joint_lower, joint_upper_rad=joint_upper)

        self._initialized = False

    def summary(self) -> str:
        effective_hz = 1.0 / max(float(self.cfg.policy_cfg.control_dt), 1.0e-9)
        return (
            f"policy=({self.policy.summary()}), "
            f"obs_joint_dim={self.obs_joint_dim}, obs_custom_extra_dim={int(self.cfg.obs_custom_extra_dim)}, "
            f"control_hz={effective_hz:.3f}, device={self.device.type}"
        )

    def reset(self, initial_robot_state: RobotState, initial_object_state: ObjectState) -> None:
        self.obs_builder.set_object_reference(initial_object_state)
        self.controller.reset(
            current_joint_pos_rad=initial_robot_state.joint_pos_rad,
            current_gripper_close_ratio=initial_robot_state.gripper_close_ratio,
        )
        self._initialized = True

    def step(self, robot_state: RobotState, object_state: ObjectState) -> dict[str, Any]:
        if not self._initialized:
            self.reset(robot_state, object_state)

        obs = self.obs_builder.build(robot_state, object_state)
        action = self.policy.act(obs, device=self.device)
        joint_targets, gripper_target, joint_velocity_cmd = self.controller.step(action)
        return {
            "obs": obs,
            "action": action,
            "joint_targets": joint_targets,
            "joint_velocity_cmd_rad_s": joint_velocity_cmd,
            "gripper_target": float(gripper_target),
        }
