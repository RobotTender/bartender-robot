# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from ..automate import factory_control as fc
from .grip_bottle_env import GripBottleEnv, GripBottleEnvCfg


@configclass
class MoveBottleEnvCfg(GripBottleEnvCfg):
    # Keep I/O interface aligned with grip-bottle policy wiring.
    action_space = 6
    observation_space = 22
    obs_joint_dim = 6

    # Move-bottle task starts from pre-grasp state above a virtual bottle anchor.
    episode_length_s = 8.3333
    force_open_gripper = False
    success_hold_steps = 2
    use_virtual_bottle = True
    virtual_terminate_on_collision = False
    # Virtual bottle center offset from TCP:
    # center = TCP + tcp_z * (tool_axis_offset + bottle_radius) - world_z * down_offset.
    virtual_bottle_tool_axis_offset_m = 0.055
    virtual_bottle_world_down_offset_m = 0.060

    # Stage-1 move control limits.
    arm_joint_speed_limit_deg_s = 30.0
    arm_joint_acc_limit_deg_s2 = 100.0
    # Freeze arm command for a few steps right after reset to avoid immediate collapse motions.
    reset_action_hold_steps = 6
    # Optional grace period before collision-based termination becomes active.
    collision_terminate_grace_steps = 3
    # Terminate on collision only when threshold violation persists for N consecutive steps.
    collision_terminate_sustained_steps = 1

    # Scripted reset pose (fixed solution-space-2 style start, no reset-time IK).
    start_arm_noise_rad = 0.0
    start_lift_height_m = 0.05
    start_grasp_band_z_range_m = (0.04, 0.08)
    lock_gripper_closed = True
    start_gripper_close_ratio = 1.0
    reset_ik_open_gripper_ratio = 0.0
    reset_ik_max_iters = 8
    reset_ik_pos_tol_m = 0.010
    reset_ik_rot_weight = 0.15
    reset_anchor_max_resample = 4
    reset_start_tcp_tol_m = 0.025
    reset_start_max_collision_force = 1.5
    reset_settle_steps = 3
    # Anchor sampling is computed from inherited spawn ranges at runtime.
    start_anchor_x_margin_m = 0.05
    start_anchor_y_low_margin_m = 0.02
    start_anchor_y_high_margin_m = 0.09
    # Fallback anchor: table-top center with y-offset (local frame).
    reset_fallback_anchor_y_offset_m = -0.05
    # Fixed start joint pose [rad] from platform solution-space setup.
    move_start_joint_pos = (
        math.radians(90.0),
        math.radians(22.24),
        math.radians(50.57),
        math.radians(0.0),
        math.radians(17.19),
        math.radians(-90.0),
    )

    # Stage waypoint and final target arm-joint poses [deg].
    move_standby_joint_pos_deg = (90.0, -45.0, 90.0, 0.0, 45.0, -90.0)
    move_waypoint_joint_pos_deg = move_standby_joint_pos_deg
    use_waypoint_reward = False
    waypoint_proximity_reward_scale = 0.0
    waypoint_proximity_decay = 1.6
    waypoint_reward_decay_steps = 500000
    # Fixed arm-joint target for move task [deg].
    move_goal_joint_pos_deg = (45.0, 0.0, 135.0, 90.0, -90.0, -135.0)
    # Stage-1 default: learn "reach target neighborhood" first.
    goal_joint_tolerance_deg = 20.0

    # Move-bottle reward terms.
    # Keep reward simple and explicit for stable optimization.
    # Stage-1 emphasizes goal-reaching density over strict safety.
    simple_progress_reward_scale = 48.0
    simple_goal_distance_penalty_scale = 0.20
    simple_goal_max_abs_err_penalty_scale = 0.35
    simple_goal_proximity_reward_scale = 2.5
    simple_goal_proximity_decay = 1.2
    # Applied only when success gate is satisfied (not every step in goal region).
    simple_goal_reached_bonus = 16.0
    simple_goal_gate_reward_scale = 6.0
    # Time penalty is annealed from start to end over global environment steps.
    simple_time_penalty_scale_start = 0.0
    simple_time_penalty_scale_end = 0.03
    simple_time_penalty_ramp_steps = 300000
    simple_collision_penalty_scale = 1.0
    # Penalize any arm contact peak (not only over-threshold contact) to discourage resting on shelf.
    simple_collision_peak_penalty_scale = 0.015
    simple_collision_speed_penalty_scale = 0.12
    simple_pinch_penalty_scale = 0.10
    simple_gate_violation_penalty_scale = 0.16
    # If enabled, gate-violation penalty only counts gates that are required for current stage success.
    gate_penalty_respects_success_require = True
    simple_y_parallel_penalty_scale = 0.70
    simple_y_parallel_reward_scale = 0.05
    simple_y_flip_penalty_scale = 0.35
    simple_y_parallel_penalty_power = 2.0
    # Stage-1 helper reward for clearing shelf region while maintaining height.
    simple_shelf_escape_reward_scale = 0.0
    shelf_escape_delta_y_m = 0.305
    shelf_escape_min_tcp_height_from_table_m = 0.08
    # Penalize wrist-dominant joint error to avoid converging to sky-looking local minima.
    simple_weighted_joint_err_penalty_scale = 0.55
    # Joint weighting profile for weighted goal error.
    # - "legacy_wrist": keep wrist-heavy legacy behavior.
    # - "focus_235": emphasize joint-2/3/5 (stage-1).
    # - "auto_delta": weights from abs(goal-start) per joint (stage-2/3).
    simple_joint_weight_mode = "legacy_wrist"
    simple_wrist_joint_error_weight = 1.8
    simple_weighted_joint_err_cap_deg = 180.0
    # Optional additional stage-specific penalty on selected joint errors (deg domain).
    simple_joint_focus_penalty_scale = 0.0
    simple_joint_focus_indices = (2, 3, 5)  # 1-based joint indices.
    simple_joint_focus_tol_deg = (30.0, 24.0, 20.0)
    simple_joint_focus_weights = (1.0, 0.8, 0.45)
    # Optional success gate for the same focused joints.
    enable_joint_focus_success_gate = False
    simple_joint_focus_success_tol_deg = (35.0, 28.0, 24.0)
    # Additional policy observations that are measurable on real robot.
    obs_tcp_height_scale_m = 0.20
    obs_tcp_y_offset_scale_m = 0.35
    # Logging profile for PPO console.
    # True: print compact, decision-critical metrics only.
    # False: print full debug metrics.
    log_compact = True
    # EE/TCP low-height diagnostic threshold (relative to table top) for full logs.
    ee_low_height_from_table_m = 0.0
    # Optional file export for full metrics (independent from compact console logs).
    save_full_log_file = True
    # Write one JSONL record every N environment steps (global env step counter).
    full_log_every_env_steps = 1
    # Flush file handle every N writes to balance safety/performance.
    full_log_flush_every_writes = 50
    # File name under env_cfg.log_dir.
    full_log_file_name = "metrics_full.jsonl"
    # Also save compact-log snapshot in each JSONL record for convenience.
    full_log_include_compact = True
    # Hard tilt safety: immediate reset for near-horizontal pose and sustained reset over moderate tilt.
    tilt_severe_reset_deg = 86.0
    tilt_sustained_reset_deg = 25.0
    tilt_sustained_reset_steps = 90
    tilt_severe_penalty = 7.0
    tilt_sustained_penalty = 1.8
    pinch_joint_margin_deg = 8.0
    # E0509 payload transport: prefer tool +Y anti-parallel to world +Z.
    preferred_tcp_y_world_z_sign = -1.0
    success_min_y_parallel = 0.90
    tilt_proxy_fail_min_y_parallel = 0.86

    # Stage-1 success is goal-centered; strict safety gates are deferred to stage-2.
    success_require_y = False
    success_require_speed = False
    success_require_collision = False
    terminate_on_severe_tilt = False
    terminate_on_sustained_tilt = False

    # Tighten safety constraints for carried-payload motion.
    contact_force_threshold = 4.0
    terminate_contact_force = 12.0
    topple_reset_max_up_z = 0.965
    success_max_ee_speed = 0.12
    # Avoid wrist-up pinched postures by limiting joint_4 range.
    use_joint_2_abs_limit = True
    joint_2_abs_limit_deg = 95.0
    use_joint_4_abs_limit = True
    joint_4_abs_limit_deg = 120.0
    # Real-controller style "inspection point = robot body" safety proxy:
    # keep arm links out of low-height shelf workspace to avoid clamped postures.
    use_robot_body_space_limit = True
    space_limit_body_name_filters = ("link_1", "link_2", "link_3", "link_4", "link_5")
    space_limit_xy_margin_m = 0.010
    space_limit_min_height_from_table_m = 0.020
    space_limit_speed_override_enable = True
    space_limit_speed_scale_when_violating = 0.35
    # 0 disables termination and keeps shaping-only behavior.
    space_limit_terminate_sustained_steps = 0
    simple_space_limit_count_penalty_scale = 0.25
    simple_space_limit_depth_penalty_scale = 5.0
    singularity_penalty_scale = 0.6
    collision_penalty_scale = 1.2
    success_bonus = 100.0


@configclass
class MoveBottleStage1EnvCfg(MoveBottleEnvCfg):
    """Stage-1: fixed start -> standby pose (easy, short-horizon pre-shaping)."""

    episode_length_s = 4.0
    move_goal_joint_pos_deg = (90.0, -45.0, 90.0, 0.0, 45.0, -90.0)
    goal_joint_tolerance_deg = 30.0
    reset_action_hold_steps = 4
    simple_progress_reward_scale = 40.0
    simple_goal_distance_penalty_scale = 0.16
    simple_goal_max_abs_err_penalty_scale = 0.70
    simple_goal_proximity_reward_scale = 3.8
    simple_goal_reached_bonus = 36.0
    simple_goal_gate_reward_scale = 12.0
    simple_time_penalty_scale_end = 0.01
    simple_collision_penalty_scale = 0.15
    simple_collision_peak_penalty_scale = 0.07
    simple_collision_speed_penalty_scale = 0.26
    simple_pinch_penalty_scale = 0.20
    simple_space_limit_count_penalty_scale = 0.22
    simple_space_limit_depth_penalty_scale = 4.0
    simple_gate_violation_penalty_scale = 0.10
    simple_y_parallel_penalty_scale = 0.50
    simple_y_flip_penalty_scale = 0.25
    simple_weighted_joint_err_penalty_scale = 0.30
    # Stage-1(v7): rollback joint-wise shaping to pre-v6 behavior for clean A/B vs v4.
    simple_joint_weight_mode = "legacy_wrist"
    simple_joint_focus_penalty_scale = 0.0
    enable_joint_focus_success_gate = False
    simple_shelf_escape_reward_scale = 1.20
    tilt_severe_penalty = 6.0
    tilt_sustained_penalty = 1.2
    success_min_y_parallel = 0.88
    contact_force_threshold = 5.0
    terminate_contact_force = 18.0
    # Stage-1: learn motion first via dense collision penalties, not hard collision resets.
    virtual_terminate_on_collision = False
    # Require short sustained stay in-goal to avoid "touch-and-reset" behavior.
    success_hold_steps = 4
    success_require_y = False
    success_require_speed = False
    success_require_collision = False
    terminate_on_severe_tilt = False
    terminate_on_sustained_tilt = False
    use_waypoint_reward = False
    log_compact = True


@configclass
class MoveBottleStage2EnvCfg(MoveBottleEnvCfg):
    """Stage-2: fixed start -> final goal with decaying standby-waypoint shaping."""

    use_waypoint_reward = True
    move_waypoint_joint_pos_deg = (90.0, -45.0, 90.0, 0.0, 45.0, -90.0)
    waypoint_proximity_reward_scale = 2.5
    waypoint_proximity_decay = 1.8
    waypoint_reward_decay_steps = 500000
    simple_gate_violation_penalty_scale = 0.22
    simple_y_parallel_penalty_scale = 0.95
    simple_y_flip_penalty_scale = 0.45
    simple_collision_speed_penalty_scale = 0.16
    simple_pinch_penalty_scale = 0.18
    simple_space_limit_count_penalty_scale = 0.30
    simple_space_limit_depth_penalty_scale = 5.5
    simple_goal_max_abs_err_penalty_scale = 0.55
    simple_weighted_joint_err_penalty_scale = 0.70
    simple_joint_weight_mode = "auto_delta"
    simple_joint_focus_penalty_scale = 0.0
    enable_joint_focus_success_gate = False
    success_min_y_parallel = 0.92
    success_require_y = True
    success_require_speed = False
    success_require_collision = False
    terminate_on_severe_tilt = False
    terminate_on_sustained_tilt = False
    virtual_terminate_on_collision = False
    log_compact = True


@configclass
class MoveBottleStage3EnvCfg(MoveBottleStage2EnvCfg):
    """Stage-3 strict config for safety and precise final docking."""

    use_waypoint_reward = False
    success_hold_steps = 3
    arm_joint_speed_limit_deg_s = 30.0
    arm_joint_acc_limit_deg_s2 = 100.0
    goal_joint_tolerance_deg = 5.0
    simple_progress_reward_scale = 24.0
    simple_goal_distance_penalty_scale = 0.35
    simple_goal_proximity_reward_scale = 1.0
    simple_goal_proximity_decay = 1.8
    simple_goal_reached_bonus = 10.0
    simple_goal_gate_reward_scale = 0.5
    simple_collision_penalty_scale = 0.8
    simple_collision_speed_penalty_scale = 0.25
    simple_pinch_penalty_scale = 0.30
    simple_space_limit_count_penalty_scale = 0.45
    simple_space_limit_depth_penalty_scale = 7.0
    space_limit_terminate_sustained_steps = 8
    simple_gate_violation_penalty_scale = 0.35
    simple_y_parallel_penalty_scale = 1.6
    simple_y_parallel_reward_scale = 0.8
    simple_y_flip_penalty_scale = 0.8
    simple_weighted_joint_err_penalty_scale = 0.50
    simple_goal_max_abs_err_penalty_scale = 0.80
    simple_joint_weight_mode = "auto_delta"
    simple_weighted_joint_err_cap_deg = 140.0
    simple_time_penalty_scale_end = 0.08
    success_min_y_parallel = 0.95
    tilt_sustained_reset_deg = 8.0
    tilt_sustained_reset_steps = 10
    tilt_severe_penalty = 12.0
    tilt_sustained_penalty = 3.0
    contact_force_threshold = 1.6
    terminate_contact_force = 4.0
    success_max_ee_speed = 0.06
    success_require_speed = True
    success_require_collision = True
    terminate_on_severe_tilt = True
    terminate_on_sustained_tilt = True
    virtual_terminate_on_collision = True
    log_compact = True


class MoveBottleEnv(GripBottleEnv):
    cfg: MoveBottleEnvCfg

    def __init__(self, cfg: MoveBottleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.arm_dim = min(6, self._robot.num_joints)
        if self.arm_dim > 0:
            max_joint_vel = math.radians(float(self.cfg.arm_joint_speed_limit_deg_s))
            arm_speed_scale = max_joint_vel / max(float(self.cfg.action_scale), 1.0e-6)
            self.robot_dof_speed_scales[: self.arm_dim] = arm_speed_scale
            self.arm_joint_vel_limit_rad_s = max_joint_vel
            self.arm_joint_acc_limit_rad_s2 = math.radians(float(self.cfg.arm_joint_acc_limit_deg_s2))
            self.arm_cmd_vel = torch.zeros((self.num_envs, self.arm_dim), dtype=torch.float, device=self.device)
        else:
            self.arm_joint_vel_limit_rad_s = 0.0
            self.arm_joint_acc_limit_rad_s2 = 0.0
            self.arm_cmd_vel = torch.zeros((self.num_envs, 0), dtype=torch.float, device=self.device)

        self._apply_move_joint_limits()
        self._ensure_move_buffers()
        self._lock_gripper_closed_if_enabled()
        self.space_limit_body_ids = self._resolve_space_limit_body_ids()
        self._init_full_log_writer()

    def _init_full_log_writer(self):
        self._full_log_handle = None
        self._full_log_path = None
        self._full_log_write_count = 0
        self._full_log_last_env_step = -1
        if not bool(getattr(self.cfg, "save_full_log_file", False)):
            return
        log_dir = getattr(self.cfg, "log_dir", None)
        if not log_dir:
            return
        try:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            file_name = str(getattr(self.cfg, "full_log_file_name", "metrics_full.jsonl"))
            self._full_log_path = log_dir_path / file_name
            # line-buffered file for lightweight streaming export.
            self._full_log_handle = self._full_log_path.open("a", encoding="utf-8", buffering=1)
        except Exception:
            self._full_log_handle = None
            self._full_log_path = None

    def _close_full_log_writer(self):
        handle = getattr(self, "_full_log_handle", None)
        if handle is None:
            return
        try:
            handle.flush()
            handle.close()
        except Exception:
            pass
        finally:
            self._full_log_handle = None

    def _to_jsonable_scalar(self, value):
        if isinstance(value, torch.Tensor):
            value_detached = value.detach()
            if value_detached.numel() == 1:
                return float(value_detached.item())
            return [float(v) for v in value_detached.flatten().cpu().tolist()]
        if isinstance(value, (float, int, bool, str)):
            return value
        try:
            return float(value)
        except Exception:
            return str(value)

    def _maybe_write_full_log_record(self, full_log: dict, compact_log: dict):
        handle = getattr(self, "_full_log_handle", None)
        if handle is None:
            return
        every_n = max(int(getattr(self.cfg, "full_log_every_env_steps", 1)), 1)
        env_step = int(self.common_step_counter)
        if env_step == int(getattr(self, "_full_log_last_env_step", -1)):
            return
        if env_step % every_n != 0:
            return
        self._full_log_last_env_step = env_step

        full_payload = {k: self._to_jsonable_scalar(v) for k, v in full_log.items()}
        record = {
            "env_step": env_step,
            "sim_time_s": float(env_step * self.dt),
            "cfg_class": self.cfg.__class__.__name__,
            "full": full_payload,
        }
        if bool(getattr(self.cfg, "full_log_include_compact", True)):
            record["compact"] = {k: self._to_jsonable_scalar(v) for k, v in compact_log.items()}
        try:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            self._full_log_write_count += 1
            flush_n = max(int(getattr(self.cfg, "full_log_flush_every_writes", 50)), 1)
            if self._full_log_write_count % flush_n == 0:
                handle.flush()
        except Exception:
            pass

    def _apply_move_joint_limits(self):
        if bool(getattr(self.cfg, "use_joint_2_abs_limit", False)):
            self._set_named_joint_abs_limit_deg("joint_2", float(self.cfg.joint_2_abs_limit_deg))
        if bool(getattr(self.cfg, "use_joint_4_abs_limit", False)):
            self._set_named_joint_abs_limit_deg("joint_4", float(self.cfg.joint_4_abs_limit_deg))
            if self._robot.num_joints > 0:
                self.home_joint_pos = torch.clamp(self.home_joint_pos, self.policy_dof_lower_limits, self.policy_dof_upper_limits)
                self.robot_dof_targets[:] = torch.clamp(
                    self.robot_dof_targets, self.policy_dof_lower_limits, self.policy_dof_upper_limits
                )

    def _resolve_space_limit_body_ids(self) -> torch.Tensor:
        if not bool(getattr(self.cfg, "use_robot_body_space_limit", False)):
            return torch.zeros((0,), dtype=torch.long, device=self.device)

        filters = tuple(str(name).lower() for name in getattr(self.cfg, "space_limit_body_name_filters", ()))
        if len(filters) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)

        resolved_ids: list[int] = []
        sensor_body_names = getattr(self._contact_sensor, "body_names", [])
        for body_name in sensor_body_names:
            lower_name = str(body_name).lower()
            if not any(key in lower_name for key in filters):
                continue
            for candidate in (str(body_name), str(body_name).split("/")[-1]):
                try:
                    body_ids, _ = self._robot.find_bodies(candidate)
                except ValueError:
                    continue
                if len(body_ids) > 0:
                    resolved_ids.append(int(body_ids[0]))
                    break

        if len(resolved_ids) == 0:
            # Fallback: direct lookup from configured names.
            for body_name in getattr(self.cfg, "space_limit_body_name_filters", ()):
                try:
                    body_ids, _ = self._robot.find_bodies(str(body_name))
                except ValueError:
                    continue
                if len(body_ids) > 0:
                    resolved_ids.append(int(body_ids[0]))

        if len(resolved_ids) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)

        return torch.tensor(sorted(set(resolved_ids)), dtype=torch.long, device=self.device)

    def _ensure_move_buffers(self):
        """Lazy-initialize move-task buffers to avoid init-order attribute errors."""
        arm_dim = min(6, self._robot.num_joints)

        if not hasattr(self, "goal_joint_pos"):
            goal_joint_deg = torch.tensor(self.cfg.move_goal_joint_pos_deg, dtype=torch.float, device=self.device)
            self.goal_joint_pos = (
                torch.deg2rad(goal_joint_deg[:arm_dim]) if arm_dim > 0 else torch.zeros((0,), dtype=torch.float, device=self.device)
            )
            self.goal_joint_tolerance_rad = math.radians(float(self.cfg.goal_joint_tolerance_deg))
        if not hasattr(self, "start_joint_pos"):
            start_joint_pos = torch.tensor(self.cfg.move_start_joint_pos, dtype=torch.float, device=self.device)
            self.start_joint_pos = (
                start_joint_pos[:arm_dim] if arm_dim > 0 else torch.zeros((0,), dtype=torch.float, device=self.device)
            )
        if not hasattr(self, "waypoint_joint_pos"):
            waypoint_joint_deg = torch.tensor(self.cfg.move_waypoint_joint_pos_deg, dtype=torch.float, device=self.device)
            self.waypoint_joint_pos = (
                torch.deg2rad(waypoint_joint_deg[:arm_dim])
                if arm_dim > 0
                else torch.zeros((0,), dtype=torch.float, device=self.device)
            )

        if not hasattr(self, "goal_dist"):
            self.goal_dist = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "waypoint_dist"):
            self.waypoint_dist = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "prev_goal_dist"):
            self.prev_goal_dist = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "goal_joint_abs_err"):
            self.goal_joint_abs_err = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        if not hasattr(self, "weighted_goal_joint_err"):
            self.weighted_goal_joint_err = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "pinch_penalty"):
            self.pinch_penalty = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "ee_jacobi_body_idx"):
            self.ee_jacobi_body_idx = max(int(self.ee_body_idx) - 1, 0)
        if not hasattr(self, "virtual_anchor_pos_w"):
            self.virtual_anchor_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        if not hasattr(self, "reset_tcp_target_height_from_table"):
            self.reset_tcp_target_height_from_table = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "object_init_xy"):
            self.object_init_xy = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        if not hasattr(self, "reset_tcp_pos_y"):
            self.reset_tcp_pos_y = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "xy_drift"):
            self.xy_drift = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "object_upward_speed"):
            self.object_upward_speed = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "tilt_violation_steps"):
            self.tilt_violation_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        if not hasattr(self, "collision_violation_steps"):
            self.collision_violation_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        if not hasattr(self, "arm_cmd_vel"):
            self.arm_cmd_vel = torch.zeros((self.num_envs, self.arm_dim), dtype=torch.float, device=self.device)
        if not hasattr(self, "space_limit_violation_steps"):
            self.space_limit_violation_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        if not hasattr(self, "space_limit_violation_count"):
            self.space_limit_violation_count = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "space_limit_violation_rate"):
            self.space_limit_violation_rate = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "space_limit_min_clearance"):
            self.space_limit_min_clearance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if not hasattr(self, "reset_reason_success_rate"):
            self.reset_reason_success_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_collision_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_severe_tilt_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_sustained_tilt_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_dropped_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_toppled_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_timeout_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)

    def _build_joint_error_weights(self, arm_dim: int, dtype: torch.dtype) -> torch.Tensor:
        mode = str(getattr(self.cfg, "simple_joint_weight_mode", "legacy_wrist")).lower()
        weights = torch.ones((arm_dim,), dtype=dtype, device=self.device)
        if arm_dim <= 0:
            return weights

        if mode == "auto_delta":
            # Use per-stage required motion (abs(goal-start)) to weight joints.
            delta = torch.abs(self.goal_joint_pos[:arm_dim] - self.start_joint_pos[:arm_dim])
            if bool(torch.any(delta > 1.0e-6).item()):
                weights = torch.clamp(delta / torch.max(delta), min=0.2, max=1.0)
                weights = weights / torch.clamp(weights.mean(), min=1.0e-6)
        elif mode == "focus_235":
            # Stage-1 focus profile: prioritize shoulder/elbow/wrist-pitch axes.
            for j_idx, w in ((2, 2.4), (3, 2.0), (5, 1.2)):
                zero_based = j_idx - 1
                if zero_based < arm_dim:
                    weights[zero_based] = float(w)
            for j_idx, w in ((1, 0.6), (4, 0.4), (6, 0.4)):
                zero_based = j_idx - 1
                if zero_based < arm_dim:
                    weights[zero_based] = float(w)
        else:
            # Legacy fallback: keep prior wrist-heavy weighting.
            if arm_dim > 3:
                weights[3:] = float(self.cfg.simple_wrist_joint_error_weight)

        return torch.clamp(weights, min=1.0e-4)

    def _resolve_focus_joint_spec(self, arm_dim: int) -> tuple[list[int], list[float], list[float]]:
        if arm_dim <= 0:
            return [], [], []

        raw_idx = tuple(int(v) for v in getattr(self.cfg, "simple_joint_focus_indices", (2, 3, 5)))
        raw_tol = tuple(float(v) for v in getattr(self.cfg, "simple_joint_focus_tol_deg", (30.0, 24.0, 20.0)))
        raw_w = tuple(float(v) for v in getattr(self.cfg, "simple_joint_focus_weights", (1.0, 0.8, 0.45)))

        n = min(len(raw_idx), len(raw_tol), len(raw_w))
        idx_list: list[int] = []
        tol_list: list[float] = []
        w_list: list[float] = []
        for i in range(n):
            j = raw_idx[i] - 1
            if 0 <= j < arm_dim:
                idx_list.append(j)
                tol_list.append(max(raw_tol[i], 1.0e-3))
                w_list.append(max(raw_w[i], 1.0e-4))
        return idx_list, tol_list, w_list

    def _update_space_limit_metrics(self):
        self._ensure_move_buffers()
        if not bool(getattr(self.cfg, "use_robot_body_space_limit", False)):
            self.space_limit_violation_count[:] = 0.0
            self.space_limit_violation_rate[:] = 0.0
            self.space_limit_min_clearance[:] = 0.0
            return
        if not hasattr(self, "space_limit_body_ids") or self.space_limit_body_ids.numel() == 0:
            self.space_limit_violation_count[:] = 0.0
            self.space_limit_violation_rate[:] = 0.0
            self.space_limit_min_clearance[:] = 0.0
            return

        body_pos_w = self._robot.data.body_pos_w[:, self.space_limit_body_ids]
        body_pos_local = body_pos_w - self.scene.env_origins.unsqueeze(1)

        center_x = float(self.cfg.table_top_center_xy[0])
        center_y = float(self.cfg.table_top_center_xy[1])
        half_x = 0.5 * float(self.cfg.table_top_size_xy[0]) + float(self.cfg.space_limit_xy_margin_m)
        half_y = 0.5 * float(self.cfg.table_top_size_xy[1]) + float(self.cfg.space_limit_xy_margin_m)
        min_z = float(self.cfg.table_top_z) + float(self.cfg.space_limit_min_height_from_table_m)

        inside_x = torch.abs(body_pos_local[:, :, 0] - center_x) <= half_x
        inside_y = torch.abs(body_pos_local[:, :, 1] - center_y) <= half_y
        inside_xy = inside_x & inside_y
        below_min_z = body_pos_local[:, :, 2] < min_z
        violation = inside_xy & below_min_z

        self.space_limit_violation_count[:] = violation.float().sum(dim=1)
        self.space_limit_violation_rate[:] = (self.space_limit_violation_count > 0.0).float()

        clearance = body_pos_local[:, :, 2] - min_z
        large = torch.full_like(clearance, 1.0e6)
        clearance_in_xy = torch.where(inside_xy, clearance, large)
        min_clearance = torch.min(clearance_in_xy, dim=1)[0]
        has_inside_xy = inside_xy.any(dim=1)
        self.space_limit_min_clearance[:] = torch.where(
            has_inside_xy, min_clearance, torch.zeros_like(min_clearance)
        )

    def _lock_gripper_closed_if_enabled(self):
        """Optionally hard-lock gripper joint(s) at closed target to remove finger motion during move stage."""
        if not bool(self.cfg.lock_gripper_closed):
            return
        if self.gripper_joint_ids.numel() == 0:
            return

        close_ratio = min(max(float(self.cfg.start_gripper_close_ratio), 0.0), 1.0)
        desired_close = self.gripper_open_targets + close_ratio * (self.gripper_close_targets - self.gripper_open_targets)
        self.policy_dof_lower_limits[self.gripper_joint_ids] = desired_close
        self.policy_dof_upper_limits[self.gripper_joint_ids] = desired_close
        self.robot_dof_targets[:, self.gripper_joint_ids] = desired_close.unsqueeze(0)

    def _park_all_bottles(self, env_ids: torch.Tensor):
        """Move all physical bottle assets to parked locations outside workspace."""
        for bottle_idx, bottle_obj in enumerate(self._bottle_objects):
            object_state = bottle_obj.data.default_root_state[env_ids].clone()
            object_state[:, 0] = self.scene.env_origins[env_ids, 0] + float(self.cfg.parked_object_x)
            object_state[:, 1] = self.scene.env_origins[env_ids, 1] + float(self.cfg.parked_object_y[bottle_idx])
            object_state[:, 2] = self.scene.env_origins[env_ids, 2] + float(self.cfg.parked_object_z)
            object_state[:, 3] = 1.0
            object_state[:, 4:7] = 0.0
            object_state[:, 7:] = 0.0
            bottle_obj.write_root_pose_to_sim(object_state[:, :7], env_ids=env_ids)
            bottle_obj.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)
            # In virtual-bottle mode, hide physical bottle visuals to reduce GUI rendering cost.
            bottle_obj.set_visibility(False, env_ids=env_ids.tolist())

    def _step_sim_no_action(self):
        """Step one physics step for reset-time IK updates."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.cfg.sim.dt)

    def _build_reset_start_joint_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create deterministic reset start state around fixed solution-space 2 seed."""
        joint_pos = self.home_joint_pos.unsqueeze(0).repeat(len(env_ids), 1)
        arm_dim = min(6, self._robot.num_joints)
        if arm_dim > 0:
            start_pose = torch.tensor(self.cfg.move_start_joint_pos, dtype=joint_pos.dtype, device=self.device)
            joint_pos[:, :arm_dim] = start_pose.unsqueeze(0)
            noise = float(self.cfg.start_arm_noise_rad)
            if noise > 0.0:
                joint_pos[:, :arm_dim] += sample_uniform(-noise, noise, (len(env_ids), arm_dim), self.device)

        if self.gripper_joint_ids.numel() > 0:
            close_ratio = float(self.cfg.start_gripper_close_ratio)
            close_ratio = min(max(close_ratio, 0.0), 1.0)
            desired = self.gripper_open_targets + close_ratio * (self.gripper_close_targets - self.gripper_open_targets)
            joint_pos[:, self.gripper_joint_ids] = desired.unsqueeze(0)

        joint_pos = torch.clamp(joint_pos, self.policy_dof_lower_limits, self.policy_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        return joint_pos, joint_vel

    def _sample_start_anchor_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        x_low = float(self.cfg.object_spawn_x_range[0]) + float(self.cfg.start_anchor_x_margin_m)
        x_high = float(self.cfg.object_spawn_x_range[1]) - float(self.cfg.start_anchor_x_margin_m)
        y_low = float(self.cfg.object_spawn_y_range[0]) + float(self.cfg.start_anchor_y_low_margin_m)
        y_high = float(self.cfg.object_spawn_y_range[1]) - float(self.cfg.start_anchor_y_high_margin_m)
        if x_low > x_high:
            x_low, x_high = x_high, x_low
        if y_low > y_high:
            y_low, y_high = y_high, y_low

        spawn_x = sample_uniform(x_low, x_high, (len(env_ids),), self.device)
        spawn_y = sample_uniform(y_low, y_high, (len(env_ids),), self.device)
        grasp_z_low = float(self.cfg.start_grasp_band_z_range_m[0])
        grasp_z_high = float(self.cfg.start_grasp_band_z_range_m[1])
        if grasp_z_low > grasp_z_high:
            grasp_z_low, grasp_z_high = grasp_z_high, grasp_z_low
        grasp_z_from_table = sample_uniform(grasp_z_low, grasp_z_high, (len(env_ids),), self.device)
        anchor_pos_w = torch.zeros((len(env_ids), 3), dtype=torch.float, device=self.device)
        anchor_pos_w[:, 0] = self.scene.env_origins[env_ids, 0] + spawn_x
        anchor_pos_w[:, 1] = self.scene.env_origins[env_ids, 1] + spawn_y
        anchor_pos_w[:, 2] = self.scene.env_origins[env_ids, 2] + float(self.cfg.table_top_z) + grasp_z_from_table
        return anchor_pos_w

    def _solve_reset_ik_to_tcp_target(
        self, env_ids: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor, tcp_target_w: torch.Tensor
    ):
        arm_dim = min(6, self._robot.num_joints)
        if arm_dim == 0:
            return

        self._compute_intermediate_values(env_ids)
        target_quat_w = self._robot.data.body_quat_w[env_ids, self.ee_body_idx].clone()
        max_iters = int(self.cfg.reset_ik_max_iters)
        pos_tol = float(self.cfg.reset_ik_pos_tol_m)
        rot_weight = float(self.cfg.reset_ik_rot_weight)

        for _ in range(max_iters):
            self._compute_intermediate_values(env_ids)
            ee_quat_w = self._robot.data.body_quat_w[env_ids, self.ee_body_idx]
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.robot_grasp_pos[env_ids],
                fingertip_midpoint_quat=ee_quat_w,
                ctrl_target_fingertip_midpoint_pos=tcp_target_w,
                ctrl_target_fingertip_midpoint_quat=target_quat_w,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )
            pos_err_norm = torch.norm(pos_error, dim=-1)
            active_mask = pos_err_norm >= pos_tol
            if not bool(active_mask.any()):
                break

            delta_pose = torch.cat((pos_error, rot_weight * axis_angle_error), dim=-1)
            jacobians = self._robot.root_physx_view.get_jacobians().clone()
            jac_body_idx = min(self.ee_jacobi_body_idx, jacobians.shape[1] - 1)
            active_env_ids = env_ids[active_mask]
            ee_jacobian = jacobians[active_env_ids, jac_body_idx, 0:6, :arm_dim]
            delta_arm_q = fc._get_delta_dof_pos(
                delta_pose=delta_pose[active_mask],
                ik_method="dls",
                jacobian=ee_jacobian,
                device=self.device,
            )
            joint_pos[active_mask, :arm_dim] += delta_arm_q[:, :arm_dim]
            joint_pos[active_mask, :arm_dim] = torch.clamp(
                joint_pos[active_mask, :arm_dim],
                self.policy_dof_lower_limits[:arm_dim],
                self.policy_dof_upper_limits[:arm_dim],
            )
            joint_vel[:] = 0.0
            self.robot_dof_targets[active_env_ids] = joint_pos[active_mask]
            self._robot.set_joint_position_target(joint_pos[active_mask], env_ids=active_env_ids)
            self._robot.write_joint_state_to_sim(joint_pos[active_mask], joint_vel[active_mask], env_ids=active_env_ids)
            self._step_sim_no_action()

    def _update_goal_metrics(self, env_ids: torch.Tensor | None = None):
        self._ensure_move_buffers()
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        arm_dim = min(6, self._robot.num_joints, int(self.goal_joint_pos.shape[0]))
        if arm_dim == 0:
            self.goal_dist[env_ids] = 0.0
            self.waypoint_dist[env_ids] = 0.0
            self.goal_joint_abs_err[env_ids] = 0.0
            self.weighted_goal_joint_err[env_ids] = 0.0
            return

        arm_q = self._robot.data.joint_pos[env_ids, :arm_dim]
        goal_q = self.goal_joint_pos[:arm_dim].unsqueeze(0)
        # Use direct joint-space error to avoid periodic aliasing around joint limits.
        delta_q = arm_q - goal_q
        self.goal_dist[env_ids] = torch.norm(delta_q, p=2, dim=-1)
        self.goal_joint_abs_err[env_ids] = 0.0
        self.goal_joint_abs_err[env_ids, :arm_dim] = torch.abs(delta_q)
        joint_weights = self._build_joint_error_weights(arm_dim, arm_q.dtype)
        weighted_err = (
            torch.sum(self.goal_joint_abs_err[env_ids, :arm_dim] * joint_weights.unsqueeze(0), dim=-1)
            / torch.clamp(torch.sum(joint_weights), min=1.0e-6)
        )
        joint_err_cap_rad = math.radians(float(self.cfg.simple_weighted_joint_err_cap_deg))
        self.weighted_goal_joint_err[env_ids] = torch.clamp(weighted_err, max=joint_err_cap_rad)
        delta_waypoint_q = arm_q - self.waypoint_joint_pos[:arm_dim].unsqueeze(0)
        self.waypoint_dist[env_ids] = torch.norm(delta_waypoint_q, p=2, dim=-1)

    def _update_pinch_penalty(self):
        self._ensure_move_buffers()
        arm_dim = min(6, self._robot.num_joints)
        if arm_dim == 0:
            self.pinch_penalty[:] = 0.0
            return

        arm_q = self._robot.data.joint_pos[:, :arm_dim]
        lower = self.policy_dof_lower_limits[:arm_dim].unsqueeze(0)
        upper = self.policy_dof_upper_limits[:arm_dim].unsqueeze(0)
        margin = torch.minimum(arm_q - lower, upper - arm_q)

        margin_ref = math.radians(float(self.cfg.pinch_joint_margin_deg))
        margin_ref_safe = max(margin_ref, 1.0e-6)
        joint_pen = torch.clamp((margin_ref - margin) / margin_ref_safe, min=0.0, max=1.0)
        self.pinch_penalty[:] = joint_pen.mean(dim=-1)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        self._ensure_move_buffers()
        super()._compute_intermediate_values(env_ids)
        if not bool(self.cfg.use_virtual_bottle):
            self._update_goal_metrics(env_ids)
            return

        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Virtual bottle follows TCP center (no rigid-body dynamics).
        tool_axis_offset = float(self.cfg.virtual_bottle_tool_axis_offset_m) + self.active_bottle_radius[env_ids]
        self.object_pos_w[env_ids] = self.robot_grasp_pos[env_ids] + self.tcp_z_w[env_ids] * tool_axis_offset.unsqueeze(-1)
        self.object_pos_w[env_ids, 2] -= float(self.cfg.virtual_bottle_world_down_offset_m)
        self.object_up_w[env_ids] = self.object_up_axis_local.unsqueeze(0).expand(len(env_ids), -1)
        self.object_up_z[env_ids] = 1.0
        self.object_height[env_ids] = self.object_pos_w[env_ids, 2] - self.scene.env_origins[env_ids, 2]
        object_xy_local = self.object_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2]
        self.xy_drift[env_ids] = torch.norm(object_xy_local - self.object_init_xy[env_ids], p=2, dim=-1)
        ee_lin_vel_w = self._robot.data.body_lin_vel_w[env_ids, self.ee_body_idx]
        self.object_xy_speed[env_ids] = torch.norm(ee_lin_vel_w[:, :2], p=2, dim=-1)
        self.object_upward_speed[env_ids] = ee_lin_vel_w[:, 2]
        self.ee_center_to_object[env_ids] = torch.norm(self.ee_body_pos[env_ids] - self.object_pos_w[env_ids], p=2, dim=-1)
        self.ee_center_target_dist[env_ids] = self.active_bottle_radius[env_ids] + float(self.cfg.ee_center_clearance_m)
        self.ee_center_dist_error[env_ids] = torch.abs(self.ee_center_to_object[env_ids] - self.ee_center_target_dist[env_ids])
        self._update_goal_metrics(env_ids)

    def _compute_success_now(self) -> torch.Tensor:
        self._ensure_move_buffers()
        y_sign = 1.0 if float(self.cfg.preferred_tcp_y_world_z_sign) >= 0.0 else -1.0
        y_parallel = torch.abs(self.tcp_y_w[:, 2])
        y_signed_align = y_sign * self.tcp_y_w[:, 2]
        goal_ok = self.goal_dist < float(self.goal_joint_tolerance_rad)
        arm_dim = min(6, self._robot.num_joints, int(self.goal_joint_pos.shape[0]))
        joint_focus_ok = torch.ones_like(goal_ok)
        if bool(getattr(self.cfg, "enable_joint_focus_success_gate", False)) and arm_dim > 0:
            idx_list, _, _ = self._resolve_focus_joint_spec(arm_dim)
            gate_tols = tuple(float(v) for v in getattr(self.cfg, "simple_joint_focus_success_tol_deg", ()))
            if len(idx_list) > 0:
                for i, j in enumerate(idx_list):
                    tol_deg = gate_tols[i] if i < len(gate_tols) else float(getattr(self.cfg, "goal_joint_tolerance_deg", 20.0))
                    tol_rad = math.radians(max(tol_deg, 1.0e-3))
                    joint_focus_ok = joint_focus_ok & (self.goal_joint_abs_err[:, j] < tol_rad)
        y_ok = y_signed_align > float(self.cfg.success_min_y_parallel)
        speed_ok = self.ee_body_speed < float(self.cfg.success_max_ee_speed)
        collision_ok = self.arm_collision_peak < float(self.cfg.contact_force_threshold)

        if not bool(self.cfg.success_require_y):
            y_ok = torch.ones_like(y_ok)
        if not bool(self.cfg.success_require_speed):
            speed_ok = torch.ones_like(speed_ok)
        if not bool(self.cfg.success_require_collision):
            collision_ok = torch.ones_like(collision_ok)

        return goal_ok & joint_focus_ok & y_ok & speed_ok & collision_ok

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        hold_steps = max(int(getattr(self.cfg, "reset_action_hold_steps", 0)), 0)
        hold_mask = None
        if hold_steps > 0:
            hold_mask = self.episode_length_buf < hold_steps
            if bool(torch.any(hold_mask).item()):
                hold_dim = min(self.arm_dim, self.actions.shape[1])
                if hold_dim > 0:
                    self.actions[hold_mask, :hold_dim] = 0.0
                if self.arm_dim > 0:
                    self.arm_cmd_vel[hold_mask, : self.arm_dim] = 0.0
        targets = self.robot_dof_targets.clone()

        arm_dim = min(self.arm_dim, self.actions.shape[1], self._robot.num_joints)
        if arm_dim > 0:
            desired_vel = self.arm_joint_vel_limit_rad_s * self.actions[:, :arm_dim]
            max_delta_vel = float(self.arm_joint_acc_limit_rad_s2) * self.dt
            vel_delta = torch.clamp(
                desired_vel - self.arm_cmd_vel[:, :arm_dim], min=-max_delta_vel, max=max_delta_vel
            )
            cmd_vel = torch.clamp(
                self.arm_cmd_vel[:, :arm_dim] + vel_delta,
                min=-float(self.arm_joint_vel_limit_rad_s),
                max=float(self.arm_joint_vel_limit_rad_s),
            )
            if bool(getattr(self.cfg, "space_limit_speed_override_enable", False)):
                self._update_space_limit_metrics()
                slow_mask = self.space_limit_violation_rate > 0.0
                if bool(torch.any(slow_mask).item()):
                    cmd_vel[slow_mask] *= float(self.cfg.space_limit_speed_scale_when_violating)
            self.arm_cmd_vel[:, :arm_dim] = cmd_vel
            targets[:, :arm_dim] = self.robot_dof_targets[:, :arm_dim] + cmd_vel * self.dt

        if self.gripper_joint_ids.numel() > 0:
            close_ratio = float(self.cfg.start_gripper_close_ratio)
            close_ratio = min(max(close_ratio, 0.0), 1.0)
            desired = self.gripper_open_targets + close_ratio * (self.gripper_close_targets - self.gripper_open_targets)
            targets[:, self.gripper_joint_ids] = desired.unsqueeze(0)

        # During early reset-hold steps, explicitly keep arm at reset start pose.
        if hold_mask is not None and bool(torch.any(hold_mask).item()) and self.arm_dim > 0:
            start_pose = self.start_joint_pos[: self.arm_dim].unsqueeze(0).expand(self.num_envs, -1)
            targets[hold_mask, : self.arm_dim] = start_pose[hold_mask]
            self.arm_cmd_vel[hold_mask, : self.arm_dim] = 0.0

        self.robot_dof_targets[:] = torch.clamp(targets, self.policy_dof_lower_limits, self.policy_dof_upper_limits)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        self._ensure_move_buffers()
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if bool(self.cfg.use_virtual_bottle):
            # Disable physical bottle dynamics in move-bottle training.
            self._park_all_bottles(env_ids)
            self.active_bottle_idx[env_ids] = 0
            self.active_bottle_one_hot[env_ids] = 0.0
            self.active_bottle_one_hot[env_ids, 0] = 1.0
            self.active_bottle_radius[env_ids] = self.bottle_radius_by_type[0]
            self._step_sim_no_action()

        # Fixed reset: directly place robot at configured start joints (no reset-time IK/random anchors).
        joint_pos, joint_vel = self._build_reset_start_joint_state(env_ids)
        self.robot_dof_targets[env_ids] = joint_pos
        if self.arm_dim > 0:
            self.arm_cmd_vel[env_ids, : self.arm_dim] = 0.0
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        for _ in range(max(int(self.cfg.reset_settle_steps), 1)):
            self._step_sim_no_action()
        # Re-apply exact reset pose after settle to avoid starting from a contact-deflected posture.
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # Start with virtual payload centered at TCP.
        self._compute_intermediate_values(env_ids)
        self.reset_tcp_pos_y[env_ids] = self.robot_grasp_pos[env_ids, 1]
        self.virtual_anchor_pos_w[env_ids] = self.robot_grasp_pos[env_ids]
        self.virtual_anchor_pos_w[env_ids, 2] -= float(self.cfg.start_lift_height_m)
        self.reset_tcp_target_height_from_table[env_ids] = self.robot_grasp_pos[env_ids, 2] - (
            self.scene.env_origins[env_ids, 2] + float(self.cfg.table_top_z)
        )
        if bool(self.cfg.use_virtual_bottle):
            self.object_init_xy[env_ids] = self.object_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2]
        else:
            self.object_init_xy[env_ids, 0] = self.virtual_anchor_pos_w[env_ids, 0] - self.scene.env_origins[env_ids, 0]
            self.object_init_xy[env_ids, 1] = self.virtual_anchor_pos_w[env_ids, 1] - self.scene.env_origins[env_ids, 1]

        self.success_hold_count[env_ids] = 0
        self.success_active[env_ids] = False
        self.success_last_step[env_ids] = -1
        self.pinch_penalty[env_ids] = 0.0
        self.tilt_violation_steps[env_ids] = 0
        self.collision_violation_steps[env_ids] = 0
        self.space_limit_violation_steps[env_ids] = 0
        self.space_limit_violation_count[env_ids] = 0.0
        self.space_limit_violation_rate[env_ids] = 0.0
        self.space_limit_min_clearance[env_ids] = 0.0

        self._compute_intermediate_values(env_ids)
        self.prev_ee_center_dist_error[env_ids] = self.ee_center_dist_error[env_ids]
        self.prev_goal_dist[env_ids] = self.goal_dist[env_ids]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_move_buffers()
        self._compute_intermediate_values()
        self._update_collision_metrics()
        self._update_singularity_metrics()
        self._update_pinch_penalty()
        self._update_space_limit_metrics()

        success = self._compute_success()
        y_parallel = torch.abs(self.tcp_y_w[:, 2])
        severe_tilt_thresh = math.cos(math.radians(float(self.cfg.tilt_severe_reset_deg)))
        sustained_tilt_thresh = math.cos(math.radians(float(self.cfg.tilt_sustained_reset_deg)))
        severe_tilt_raw = y_parallel < severe_tilt_thresh
        sustained_tilt_raw = y_parallel < sustained_tilt_thresh
        dropped = self.object_height < (float(self.cfg.table_top_z) - float(self.cfg.drop_below_table_margin))
        toppled = self.object_up_z < float(self.cfg.topple_reset_max_up_z)
        collided_raw = self.arm_collision_peak > float(self.cfg.terminate_contact_force)
        settle_done_mask = self.episode_length_buf >= int(self.cfg.reset_settle_steps)
        collision_done_mask = self.episode_length_buf >= max(
            int(self.cfg.reset_settle_steps), int(getattr(self.cfg, "collision_terminate_grace_steps", 0))
        )
        toppled = toppled & settle_done_mask
        collision_active = collided_raw & collision_done_mask
        self.collision_violation_steps[:] = torch.where(
            collision_active, self.collision_violation_steps + 1, torch.zeros_like(self.collision_violation_steps)
        )
        collided = self.collision_violation_steps >= int(getattr(self.cfg, "collision_terminate_sustained_steps", 1))
        space_limit_active = (self.space_limit_violation_rate > 0.0) & settle_done_mask
        self.space_limit_violation_steps[:] = torch.where(
            space_limit_active, self.space_limit_violation_steps + 1, torch.zeros_like(self.space_limit_violation_steps)
        )
        space_limit_terminated = (
            int(getattr(self.cfg, "space_limit_terminate_sustained_steps", 0)) > 0
        ) & (
            self.space_limit_violation_steps
            >= int(getattr(self.cfg, "space_limit_terminate_sustained_steps", 0))
        )
        sustained_tilt_active = sustained_tilt_raw & settle_done_mask
        self.tilt_violation_steps[:] = torch.where(
            sustained_tilt_active, self.tilt_violation_steps + 1, torch.zeros_like(self.tilt_violation_steps)
        )
        sustained_tilt = self.tilt_violation_steps >= int(self.cfg.tilt_sustained_reset_steps)
        severe_tilt = severe_tilt_raw & settle_done_mask
        if bool(self.cfg.use_virtual_bottle):
            # In virtual mode, avoid reset loops from proxy physics states.
            terminated = success.clone()
            if bool(self.cfg.terminate_on_severe_tilt):
                terminated = terminated | severe_tilt
            if bool(self.cfg.terminate_on_sustained_tilt):
                terminated = terminated | sustained_tilt
            if bool(self.cfg.virtual_terminate_on_collision):
                terminated = terminated | collided
            terminated = terminated | space_limit_terminated
        else:
            terminated = success | dropped | toppled | severe_tilt | sustained_tilt | collided | space_limit_terminated
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # Reset-reason breakdown (mutually exclusive priority) for practical debugging.
        done_mask = terminated | truncated
        if bool(torch.any(done_mask).item()):
            reason = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
            reason = torch.where(done_mask & truncated, 7, reason)  # timeout
            reason = torch.where(done_mask & toppled, 6, reason)  # toppled
            reason = torch.where(done_mask & dropped, 5, reason)  # dropped
            reason = torch.where(done_mask & sustained_tilt, 4, reason)  # sustained tilt
            reason = torch.where(done_mask & severe_tilt, 3, reason)  # severe tilt
            reason = torch.where(done_mask & collided, 2, reason)  # collision
            reason = torch.where(done_mask & success, 1, reason)  # success (highest priority)
            done_count = torch.clamp(done_mask.float().sum(), min=1.0)
            self.reset_reason_success_rate = (reason == 1).float().sum() / done_count
            self.reset_reason_collision_rate = (reason == 2).float().sum() / done_count
            self.reset_reason_severe_tilt_rate = (reason == 3).float().sum() / done_count
            self.reset_reason_sustained_tilt_rate = (reason == 4).float().sum() / done_count
            self.reset_reason_dropped_rate = (reason == 5).float().sum() / done_count
            self.reset_reason_toppled_rate = (reason == 6).float().sum() / done_count
            self.reset_reason_timeout_rate = (reason == 7).float().sum() / done_count
        else:
            self.reset_reason_success_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_collision_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_severe_tilt_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_sustained_tilt_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_dropped_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_toppled_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
            self.reset_reason_timeout_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._ensure_move_buffers()
        self._compute_intermediate_values()
        self._update_collision_metrics()
        self._update_pinch_penalty()
        self._update_space_limit_metrics()

        goal_progress = torch.clamp(self.prev_goal_dist - self.goal_dist, min=-0.10, max=0.10)
        goal_reached = self.goal_dist < float(self.goal_joint_tolerance_rad)
        goal_proximity = torch.exp(-float(self.cfg.simple_goal_proximity_decay) * self.goal_dist)
        goal_dist_p50_rad = torch.quantile(self.goal_dist, 0.5)
        goal_dist_p90_rad = torch.quantile(self.goal_dist, 0.9)
        near_goal_20deg_rate = (self.goal_dist < math.radians(20.0)).float().mean()
        arm_dim = min(6, self._robot.num_joints, int(self.goal_joint_pos.shape[0]))
        if arm_dim > 0:
            goal_joint_max_abs_err = torch.max(self.goal_joint_abs_err[:, :arm_dim], dim=1)[0]
        else:
            goal_joint_max_abs_err = torch.zeros_like(self.goal_dist)
        joint_focus_penalty = torch.zeros_like(self.goal_dist)
        joint_focus_ok = torch.ones_like(self.goal_dist, dtype=torch.bool)
        if arm_dim > 0:
            focus_idx, focus_tol_deg, focus_w = self._resolve_focus_joint_spec(arm_dim)
            if len(focus_idx) > 0:
                arm_abs_err_deg = torch.rad2deg(self.goal_joint_abs_err[:, :arm_dim])
                focus_acc = torch.zeros_like(self.goal_dist)
                focus_w_sum = 0.0
                for j, tol_deg, w in zip(focus_idx, focus_tol_deg, focus_w):
                    tol_safe = max(float(tol_deg), 1.0e-3)
                    focus_acc = focus_acc + float(w) * torch.clamp(
                        (arm_abs_err_deg[:, j] - tol_safe) / tol_safe, min=0.0, max=6.0
                    )
                    focus_w_sum += float(w)
                if focus_w_sum > 0.0:
                    joint_focus_penalty = focus_acc / focus_w_sum

                if bool(getattr(self.cfg, "enable_joint_focus_success_gate", False)):
                    gate_tols = tuple(float(v) for v in getattr(self.cfg, "simple_joint_focus_success_tol_deg", ()))
                    for i, j in enumerate(focus_idx):
                        tol_deg = gate_tols[i] if i < len(gate_tols) else float(getattr(self.cfg, "goal_joint_tolerance_deg", 20.0))
                        tol_rad = math.radians(max(tol_deg, 1.0e-3))
                        joint_focus_ok = joint_focus_ok & (self.goal_joint_abs_err[:, j] < tol_rad)
        y_sign = 1.0 if float(self.cfg.preferred_tcp_y_world_z_sign) >= 0.0 else -1.0
        y_parallel = torch.abs(self.tcp_y_w[:, 2])
        y_signed_align = torch.clamp(y_sign * self.tcp_y_w[:, 2], min=-1.0, max=1.0)
        y_ok = y_signed_align > float(self.cfg.success_min_y_parallel)
        speed_ok = self.ee_body_speed < float(self.cfg.success_max_ee_speed)
        collision_ok = self.arm_collision_peak < float(self.cfg.contact_force_threshold)
        joint_focus_gate_required = bool(getattr(self.cfg, "enable_joint_focus_success_gate", False)) and arm_dim > 0
        if bool(getattr(self.cfg, "gate_penalty_respects_success_require", True)):
            y_gate_bad = (~y_ok).float() if bool(self.cfg.success_require_y) else torch.zeros_like(y_parallel)
            speed_gate_bad = (~speed_ok).float() if bool(self.cfg.success_require_speed) else torch.zeros_like(y_parallel)
            collision_gate_bad = (
                (~collision_ok).float() if bool(self.cfg.success_require_collision) else torch.zeros_like(y_parallel)
            )
            focus_gate_bad = (~joint_focus_ok).float() if joint_focus_gate_required else torch.zeros_like(y_parallel)
            gate_violation = y_gate_bad + speed_gate_bad + collision_gate_bad + focus_gate_bad
        else:
            focus_gate_bad = (~joint_focus_ok).float() if joint_focus_gate_required else torch.zeros_like(y_parallel)
            gate_violation = (~y_ok).float() + (~speed_ok).float() + (~collision_ok).float() + focus_gate_bad
        y_aligned = torch.clamp(y_signed_align, min=0.0, max=1.0)
        y_align_error = torch.clamp(1.0 - y_aligned, min=0.0, max=1.0)
        y_align_penalty = y_align_error.pow(float(self.cfg.simple_y_parallel_penalty_power))
        y_parallel_reward = y_aligned
        y_flip_penalty = torch.clamp(-y_signed_align, min=0.0, max=1.0)
        severe_tilt_thresh = math.cos(math.radians(float(self.cfg.tilt_severe_reset_deg)))
        sustained_tilt_thresh = math.cos(math.radians(float(self.cfg.tilt_sustained_reset_deg)))
        severe_tilt = y_parallel < severe_tilt_thresh
        # Predict sustained-tilt trigger using current step signal for stable penalty timing.
        next_tilt_steps = torch.where(
            y_parallel < sustained_tilt_thresh,
            self.tilt_violation_steps + 1,
            torch.zeros_like(self.tilt_violation_steps),
        )
        sustained_tilt = next_tilt_steps >= int(self.cfg.tilt_sustained_reset_steps)
        tilt_penalty = float(self.cfg.tilt_severe_penalty) * severe_tilt.float() + float(
            self.cfg.tilt_sustained_penalty
        ) * sustained_tilt.float()

        ramp_steps = max(int(self.cfg.simple_time_penalty_ramp_steps), 1)
        ramp_alpha = min(max(float(self.common_step_counter) / float(ramp_steps), 0.0), 1.0)
        time_penalty_scale = float(self.cfg.simple_time_penalty_scale_start) + ramp_alpha * (
            float(self.cfg.simple_time_penalty_scale_end) - float(self.cfg.simple_time_penalty_scale_start)
        )
        time_ratio = self.episode_length_buf.float() / max(float(self.max_episode_length), 1.0)
        collision_peak_penalty = torch.clamp(self.arm_collision_peak, max=float(self.cfg.terminate_contact_force))
        collision_over_penalty = torch.clamp(self.arm_collision_over, max=float(self.cfg.terminate_contact_force))
        collision_speed_penalty = collision_over_penalty * self.ee_body_speed
        space_limit_depth_penalty = torch.clamp(-self.space_limit_min_clearance, min=0.0, max=0.25)
        table_top_world_z = self.scene.env_origins[:, 2] + float(self.cfg.table_top_z)
        tcp_height_from_table = self.robot_grasp_pos[:, 2] - table_top_world_z
        ee_world_z = self.robot_grasp_pos[:, 2]
        ee_world_z_p10 = torch.quantile(ee_world_z, 0.1)
        ee_height_from_table_p10 = torch.quantile(tcp_height_from_table, 0.1)
        ee_below_table_top_rate = (tcp_height_from_table < 0.0).float().mean()
        ee_low_height_rate = (tcp_height_from_table < float(self.cfg.ee_low_height_from_table_m)).float().mean()
        shelf_escape_target = max(float(self.cfg.shelf_escape_delta_y_m), 1.0e-6)
        shelf_escape_delta_y = torch.clamp(self.reset_tcp_pos_y - self.robot_grasp_pos[:, 1], min=0.0)
        shelf_escape_ratio = torch.clamp(shelf_escape_delta_y / shelf_escape_target, min=0.0, max=1.0)
        shelf_escape_height_ok = (
            tcp_height_from_table >= float(self.cfg.shelf_escape_min_tcp_height_from_table_m)
        ).float()
        shelf_escape_reward = shelf_escape_ratio * shelf_escape_height_ok
        waypoint_weight_now = 0.0
        waypoint_reward = torch.zeros_like(goal_proximity)
        if bool(self.cfg.use_waypoint_reward):
            waypoint_proximity = torch.exp(-float(self.cfg.waypoint_proximity_decay) * self.waypoint_dist)
            waypoint_decay_steps = max(int(self.cfg.waypoint_reward_decay_steps), 1)
            waypoint_weight_now = 1.0 - min(max(float(self.common_step_counter) / float(waypoint_decay_steps), 0.0), 1.0)
            waypoint_reward = (
                float(self.cfg.waypoint_proximity_reward_scale) * waypoint_weight_now * waypoint_proximity
            )

        rewards = (
            self.cfg.simple_progress_reward_scale * goal_progress
            + self.cfg.simple_goal_proximity_reward_scale * goal_proximity
            + waypoint_reward
            + self.cfg.simple_goal_gate_reward_scale * goal_reached.float()
            + self.cfg.simple_y_parallel_reward_scale * y_parallel_reward
            + self.cfg.simple_shelf_escape_reward_scale * shelf_escape_reward
            - self.cfg.simple_goal_distance_penalty_scale * self.goal_dist
            - self.cfg.simple_goal_max_abs_err_penalty_scale * goal_joint_max_abs_err
            - self.cfg.simple_weighted_joint_err_penalty_scale * self.weighted_goal_joint_err
            - self.cfg.simple_joint_focus_penalty_scale * joint_focus_penalty
            - time_penalty_scale * time_ratio
            - self.cfg.simple_collision_peak_penalty_scale * collision_peak_penalty
            - self.cfg.simple_collision_penalty_scale * collision_over_penalty
            - self.cfg.simple_collision_speed_penalty_scale * collision_speed_penalty
            - self.cfg.simple_gate_violation_penalty_scale * gate_violation
            - self.cfg.simple_y_parallel_penalty_scale * y_align_penalty
            - self.cfg.simple_y_flip_penalty_scale * y_flip_penalty
            - self.cfg.simple_pinch_penalty_scale * self.pinch_penalty
            - self.cfg.simple_space_limit_count_penalty_scale * self.space_limit_violation_count
            - self.cfg.simple_space_limit_depth_penalty_scale * space_limit_depth_penalty
            - tilt_penalty
        )

        success = self._compute_success()
        rewards = torch.where(
            success,
            rewards + float(self.cfg.success_bonus) + float(self.cfg.simple_goal_reached_bonus),
            rewards,
        )

        # Per-joint diagnostics for offline analysis (saved in full log only).
        joint_pos_mean_deg = torch.zeros((6,), dtype=torch.float, device=self.device)
        joint_abs_err_mean_deg = torch.zeros((6,), dtype=torch.float, device=self.device)
        joint_abs_err_p50_deg = torch.zeros((6,), dtype=torch.float, device=self.device)
        joint_abs_err_p90_deg = torch.zeros((6,), dtype=torch.float, device=self.device)
        if arm_dim > 0:
            arm_q_deg = torch.rad2deg(self._robot.data.joint_pos[:, :arm_dim])
            arm_abs_err_deg = torch.rad2deg(self.goal_joint_abs_err[:, :arm_dim])
            joint_pos_mean_deg[:arm_dim] = arm_q_deg.mean(dim=0)
            joint_abs_err_mean_deg[:arm_dim] = arm_abs_err_deg.mean(dim=0)
            joint_abs_err_p50_deg[:arm_dim] = torch.quantile(arm_abs_err_deg, 0.5, dim=0)
            joint_abs_err_p90_deg[:arm_dim] = torch.quantile(arm_abs_err_deg, 0.9, dim=0)

        full_log = {
            "success_rate": success.float().mean(),
            "mean_goal_joint_dist_rad": self.goal_dist.mean(),
            "mean_goal_joint_dist_deg": torch.rad2deg(self.goal_dist).mean(),
            "mean_goal_joint_dist_p50_deg": torch.rad2deg(goal_dist_p50_rad),
            "mean_goal_joint_dist_p90_deg": torch.rad2deg(goal_dist_p90_rad),
            "mean_goal_joint_abs_err_deg": torch.rad2deg(self.goal_joint_abs_err).mean(),
            "mean_goal_joint_max_abs_err_deg": torch.rad2deg(goal_joint_max_abs_err).mean(),
            "mean_weighted_goal_joint_err_deg": torch.rad2deg(self.weighted_goal_joint_err).mean(),
            "mean_joint_focus_penalty": joint_focus_penalty.mean(),
            "mean_waypoint_dist_deg": torch.rad2deg(self.waypoint_dist).mean(),
            "waypoint_weight_now": torch.tensor(waypoint_weight_now, dtype=torch.float, device=self.device),
            "mean_waypoint_reward": waypoint_reward.mean(),
            "mean_shelf_escape_ratio": shelf_escape_ratio.mean(),
            "mean_shelf_escape_reward": shelf_escape_reward.mean(),
            "mean_goal_progress": goal_progress.mean(),
            "goal_reached_rate": goal_reached.float().mean(),
            "near_goal_20deg_rate": near_goal_20deg_rate,
            "mean_goal_proximity": goal_proximity.mean(),
            "mean_y_parallel": y_parallel.mean(),
            "mean_y_signed_align": y_signed_align.mean(),
            "mean_y_parallel_reward": y_parallel_reward.mean(),
            "mean_y_align_penalty": y_align_penalty.mean(),
            "mean_y_flip_penalty": y_flip_penalty.mean(),
            "flip_rate": (y_signed_align < 0.0).float().mean(),
            "mean_tilt_penalty": tilt_penalty.mean(),
            "severe_tilt_rate": severe_tilt.float().mean(),
            "sustained_tilt_rate": sustained_tilt.float().mean(),
            "mean_tilt_violation_steps": self.tilt_violation_steps.float().mean(),
            "mean_time_ratio": time_ratio.mean(),
            "time_penalty_scale_now": torch.tensor(time_penalty_scale, dtype=torch.float, device=self.device),
            "mean_time_penalty": (time_penalty_scale * time_ratio).mean(),
            "virtual_mode": torch.tensor(1.0 if bool(self.cfg.use_virtual_bottle) else 0.0, device=self.device),
            "mean_arm_collision_peak": self.arm_collision_peak.mean(),
            "mean_collision_peak_penalty": collision_peak_penalty.mean(),
            "mean_arm_collision_over": collision_over_penalty.mean(),
            "mean_collision_speed_penalty": collision_speed_penalty.mean(),
            "mean_pinch_penalty": self.pinch_penalty.mean(),
            "space_limit_violation_rate": self.space_limit_violation_rate.mean(),
            "mean_space_limit_violation_count": self.space_limit_violation_count.mean(),
            "mean_space_limit_min_clearance_m": self.space_limit_min_clearance.mean(),
            "mean_space_limit_depth_penalty": space_limit_depth_penalty.mean(),
            "mean_ee_world_z_m": ee_world_z.mean(),
            "p10_ee_world_z_m": ee_world_z_p10,
            "mean_tcp_height_from_table": tcp_height_from_table.mean(),
            "p10_tcp_height_from_table_m": ee_height_from_table_p10,
            "ee_below_table_top_rate": ee_below_table_top_rate,
            "ee_low_height_rate": ee_low_height_rate,
            "mean_gate_violation": gate_violation.mean(),
            "mean_reset_tcp_target_height_from_table": self.reset_tcp_target_height_from_table.mean(),
            "goal_gate_rate": (self.goal_dist < float(self.goal_joint_tolerance_rad)).float().mean(),
            "joint_focus_gate_rate": joint_focus_ok.float().mean(),
            "y_gate_rate": y_ok.float().mean(),
            "speed_gate_rate": speed_ok.float().mean(),
            "collision_gate_rate": collision_ok.float().mean(),
            "reset_reason_success_rate": self.reset_reason_success_rate,
            "reset_reason_collision_rate": self.reset_reason_collision_rate,
            "reset_reason_severe_tilt_rate": self.reset_reason_severe_tilt_rate,
            "reset_reason_sustained_tilt_rate": self.reset_reason_sustained_tilt_rate,
            "reset_reason_dropped_rate": self.reset_reason_dropped_rate,
            "reset_reason_toppled_rate": self.reset_reason_toppled_rate,
            "reset_reason_timeout_rate": self.reset_reason_timeout_rate,
        }
        for j in range(6):
            key_idx = j + 1
            full_log[f"mean_joint_pos_deg_j{key_idx}"] = joint_pos_mean_deg[j]
            full_log[f"mean_goal_abs_err_deg_j{key_idx}"] = joint_abs_err_mean_deg[j]
            full_log[f"p50_goal_abs_err_deg_j{key_idx}"] = joint_abs_err_p50_deg[j]
            full_log[f"p90_goal_abs_err_deg_j{key_idx}"] = joint_abs_err_p90_deg[j]

        compact_log = {
            "success_rate": full_log["success_rate"],
            "mean_goal_joint_dist_deg": full_log["mean_goal_joint_dist_deg"],
            "mean_goal_joint_dist_p50_deg": full_log["mean_goal_joint_dist_p50_deg"],
            "mean_goal_joint_dist_p90_deg": full_log["mean_goal_joint_dist_p90_deg"],
            "mean_goal_progress": full_log["mean_goal_progress"],
            "goal_reached_rate": full_log["goal_reached_rate"],
            "near_goal_20deg_rate": full_log["near_goal_20deg_rate"],
            "mean_y_signed_align": full_log["mean_y_signed_align"],
            "mean_time_ratio": full_log["mean_time_ratio"],
            "time_penalty_scale_now": full_log["time_penalty_scale_now"],
            "mean_arm_collision_peak": full_log["mean_arm_collision_peak"],
            "mean_arm_collision_over": full_log["mean_arm_collision_over"],
            "space_limit_violation_rate": full_log["space_limit_violation_rate"],
            "mean_space_limit_violation_count": full_log["mean_space_limit_violation_count"],
            "mean_space_limit_min_clearance_m": full_log["mean_space_limit_min_clearance_m"],
            "goal_gate_rate": full_log["goal_gate_rate"],
            "y_gate_rate": full_log["y_gate_rate"],
            "speed_gate_rate": full_log["speed_gate_rate"],
            "collision_gate_rate": full_log["collision_gate_rate"],
            "reset_reason_success_rate": full_log["reset_reason_success_rate"],
            "reset_reason_collision_rate": full_log["reset_reason_collision_rate"],
            "reset_reason_timeout_rate": full_log["reset_reason_timeout_rate"],
        }
        if bool(getattr(self.cfg, "log_compact", True)):
            self.extras["log"] = compact_log
            # Keep full diagnostics available for optional downstream readers.
            self.extras["log_full"] = full_log
        else:
            self.extras["log"] = full_log
        self._maybe_write_full_log_record(full_log=full_log, compact_log=compact_log)

        self.prev_ee_center_dist_error[:] = self.ee_center_dist_error
        self.prev_goal_dist[:] = self.goal_dist
        return rewards

    def close(self):
        self._close_full_log_writer()
        super().close()

    def _get_observations(self) -> dict:
        self._ensure_move_buffers()
        self._compute_intermediate_values()

        denom = torch.clamp(self.policy_dof_upper_limits - self.policy_dof_lower_limits, min=1.0e-5)
        dof_pos_scaled = 2.0 * (self._robot.data.joint_pos - self.policy_dof_lower_limits) / denom - 1.0
        dof_vel_scaled = self._robot.data.joint_vel * float(self.cfg.dof_velocity_scale)

        joint_pos_obs = self._pad_or_truncate(dof_pos_scaled[:, self.obs_joint_ids], int(self.cfg.obs_joint_dim))
        joint_vel_obs = self._pad_or_truncate(dof_vel_scaled[:, self.obs_joint_ids], int(self.cfg.obs_joint_dim))

        goal_err_obs = torch.zeros((self.num_envs, int(self.cfg.obs_joint_dim)), dtype=torch.float, device=self.device)
        arm_dim = min(6, self._robot.num_joints, int(self.goal_joint_pos.shape[0]))
        if arm_dim > 0:
            arm_q = self._robot.data.joint_pos[:, :arm_dim]
            goal_q = self.goal_joint_pos[:arm_dim].unsqueeze(0)
            delta_q = arm_q - goal_q
            arm_span = torch.clamp(
                self.policy_dof_upper_limits[:arm_dim] - self.policy_dof_lower_limits[:arm_dim], min=1.0e-5
            )
            goal_err_obs[:, :arm_dim] = torch.clamp(delta_q / arm_span.unsqueeze(0), min=-1.0, max=1.0)

        y_sign = 1.0 if float(self.cfg.preferred_tcp_y_world_z_sign) >= 0.0 else -1.0
        y_parallel_obs = torch.clamp(y_sign * self.tcp_y_w[:, 2], min=-1.0, max=1.0).unsqueeze(-1)
        speed_scale = max(float(self.cfg.success_max_ee_speed), 1.0e-6)
        ee_speed_obs = torch.clamp(self.ee_body_speed / speed_scale, min=0.0, max=5.0).unsqueeze(-1)
        table_top_world_z = self.scene.env_origins[:, 2] + float(self.cfg.table_top_z)
        tcp_height_from_table = self.robot_grasp_pos[:, 2] - table_top_world_z
        tcp_height_obs = torch.clamp(
            tcp_height_from_table / max(float(self.cfg.obs_tcp_height_scale_m), 1.0e-6),
            min=-5.0,
            max=5.0,
        ).unsqueeze(-1)
        table_center_y_w = self.scene.env_origins[:, 1] + float(self.cfg.table_top_center_xy[1])
        tcp_y_offset_obs = torch.clamp(
            (self.robot_grasp_pos[:, 1] - table_center_y_w) / max(float(self.cfg.obs_tcp_y_offset_scale_m), 1.0e-6),
            min=-5.0,
            max=5.0,
        ).unsqueeze(-1)

        obs = torch.cat(
            (
                joint_pos_obs,
                joint_vel_obs,
                goal_err_obs,
                y_parallel_obs,
                ee_speed_obs,
                tcp_height_obs,
                tcp_y_offset_obs,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}
