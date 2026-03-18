# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_conjugate, quat_mul, sample_uniform

_ASSET_ROOT = Path(__file__).resolve().parent / "USD"
_ROBOT_USD_PATH = str(_ASSET_ROOT / "e0509" / "e0509.usd")
_TABLE_USD_PATH = str(_ASSET_ROOT / "table_e0509.usd")


@configclass
class E0509PickPlaceKEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 steps with dt=1/120 and decimation=2
    decimation = 2
    action_space = 7  # 6 arm joints + 1 synchronized gripper command
    observation_space = 29
    state_space = 0
    obs_joint_dim = 10

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=2.2,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # placement (workspace-aligned)
    # Robot base mounting block top: 730 mm above world ground.
    robot_base_z_offset = 0.730

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ROBOT_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 1.57079632679,
                "joint_4": 0.0,
                "joint_5": 1.57079632679,
                "joint_6": 0.0,
                "rh_l1": 0.02,
                "rh_r1": 0.02,
                "rh_l2": 0.02,
                "rh_r2": 0.02,
            },
            pos=(0.0, 0.0, robot_base_z_offset),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "robot": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=250.0,
                stiffness=90.0,
                damping=8.0,
            ),
        },
    )

    # static table from custom USD (workspace reference)
    table_usd_path = _TABLE_USD_PATH
    workspace_contains_robot = True
    workspace_robot_prim_path = "/World/envs/env_.*/Table/e0509"
    # Workspace tabletop world Z: 700 mm. (previous temporary values: 0.713, 0.693)
    table_top_z = 0.700
    # Hidden collision slab used to guarantee stable table contacts.
    use_table_collision_proxy = True
    table_collision_size = (1.70, 0.568, 0.04)
    table_collision_center_z = table_top_z - 0.5 * table_collision_size[2]
    table_collision = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TableCollision",
        spawn=sim_utils.CuboidCfg(
            size=table_collision_size,
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, table_collision_center_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    # target object to grasp and lift
    object_size = (0.04, 0.04, 0.04)
    object_spawn_z = table_top_z + 0.5 * object_size[2] + 0.01
    # Fixed narrow spawn range (no curriculum): easier early-stage pickup learning.
    object_spawn_x_range = (0.495, 0.505)
    object_spawn_y_range = (-0.01, 0.01)

    target_object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TargetObject",
        spawn=sim_utils.CuboidCfg(
            size=object_size,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.6,
                dynamic_friction=1.6,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.20),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.20, 0.20)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.50, 0.0, object_spawn_z),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # control
    action_scale = 2.5
    dof_velocity_scale = 0.1

    gripper_joint_names = ("rh_l1", "rh_r1", "rh_l2", "rh_r2")
    gripper_open_joint_pos = 0.02
    gripper_close_joint_pos = 1.0
    gripper_target_smoothing = 0.35
    # TCP length from ee body origin to fingertip center (meters)
    tcp_offset_open_m = 0.107
    tcp_offset_closed_m = 0.135
    tcp_axis_local = (0.0, 0.0, 1.0)

    # task
    target_lift_height = 0.05
    success_lift_height = 0.04
    success_ee_to_object = 0.055
    success_min_gripper_state = 0.50
    drop_below_table_margin = 0.03

    # rewards
    dist_reward_scale = 0.5
    grasp_reward_scale = 3.5
    lift_reward_scale = 40.0
    xy_drift_penalty_scale = 1.0
    xy_drift_free_margin = 0.03
    xy_penalty_start_lift = 0.04
    false_grasp_penalty_scale = 1.0
    false_grasp_close_distance = 0.06
    false_grasp_min_gripper_state = 0.55
    false_grasp_min_lift = 0.01
    lift_handover_start = 0.03
    lift_handover_span = 0.04
    post_lift_dist_keep_ratio = 0.30
    post_lift_grasp_keep_ratio = 0.50
    upright_reward_scale = 2.0
    tilt_penalty_scale = 2.5
    ang_vel_penalty_scale = 1.0
    ang_vel_ref = 8.0
    ee_pose_ref_lift_start = 0.01
    ee_pose_ref_min_gripper_state = 0.60
    ee_pose_rp_penalty_scale = 0.9
    ee_pose_yaw_penalty_scale = 0.05
    return_home_start_lift = 0.045
    return_home_min_gripper_state = 0.55
    return_home_hold_lift = 0.035
    return_home_command_mode = False
    return_home_command_arm_alpha = 0.20
    return_home_command_gripper_alpha = 0.35
    return_home_command_close_ratio = 1.0
    return_home_joint_tolerance = 0.22
    return_home_dist_keep_ratio = 0.40
    return_home_grasp_keep_ratio = 0.60
    return_home_reward_scale = 0.0
    return_home_error_penalty_scale = 0.0
    return_home_progress_reward_scale = 0.0
    return_home_near_success_reward_scale = 0.0
    return_home_near_success_band = 0.08
    return_home_lift_hold_scale = 0.0
    return_home_gripper_hold_scale = 0.0
    return_home_bonus = 0.0
    # Stage-1 default: learn stable lift first, then enable return-home fine-tuning.
    enable_return_home_stage = False
    action_penalty_scale = 0.005
    success_bonus = 16.0
    upright_success_min_up_z = 0.95
    upright_success_bonus = 6.0
    diag_lift_threshold = 0.04
    grasp_dist_decay = 32.0
    # stage progression monitor (logging only)
    success_rate_target = 0.70
    success_rate_ema_alpha = 0.02
    success_rate_hold_min_steps = 500


class E0509PickPlaceKEnv(DirectRLEnv):
    """Minimal pick-and-lift environment for Doosan E0509.

    Design goal: keep the same simple structure as Franka-Cabinet DirectRL env,
    but replace cabinet interaction with grasp-and-lift of a single cuboid.
    """

    cfg: E0509PickPlaceKEnvCfg

    def __init__(self, cfg: E0509PickPlaceKEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        arm_dof_count = min(6, self._robot.num_joints)
        self.robot_dof_speed_scales[:arm_dof_count] = 0.45
        self._set_named_joint_speed_scale("joint_2", 0.38)
        self._set_named_joint_speed_scale("joint_4", 0.38)

        self.gripper_joint_ids = self._collect_exact_joint_ids(list(self.cfg.gripper_joint_names))
        if self.gripper_joint_ids.numel() > 0:
            self.robot_dof_speed_scales[self.gripper_joint_ids] = 0.12

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.home_joint_pos = self._robot.data.default_joint_pos[0].clone()
        self._set_named_joint_home("joint_2", -0.25)
        self._set_named_joint_home("joint_3", 0.5 * math.pi)
        self._set_named_joint_home("joint_4", 0.30)
        self._set_named_joint_home("joint_5", 0.5 * math.pi)

        if self.gripper_joint_ids.numel() > 0:
            low = self.robot_dof_lower_limits[self.gripper_joint_ids]
            high = self.robot_dof_upper_limits[self.gripper_joint_ids]
            self.gripper_open_targets = torch.clamp(torch.full_like(low, self.cfg.gripper_open_joint_pos), low, high)
            close_from_cfg = torch.clamp(torch.full_like(low, self.cfg.gripper_close_joint_pos), low, high)
            # If configured close target collapses to open target due to limit clamping, push to the farthest bound.
            collapsed = torch.abs(close_from_cfg - self.gripper_open_targets) < 1.0e-4
            farthest = torch.where(
                torch.abs(high - self.gripper_open_targets) >= torch.abs(low - self.gripper_open_targets), high, low
            )
            self.gripper_close_targets = torch.where(collapsed, farthest, close_from_cfg)
            self.home_joint_pos[self.gripper_joint_ids] = self.gripper_open_targets
        else:
            self.gripper_open_targets = torch.zeros((0,), dtype=torch.float, device=self.device)
            self.gripper_close_targets = torch.zeros((0,), dtype=torch.float, device=self.device)

        self.ee_body_idx = self._find_first_body_idx(["gripper", "rh_p12_rn_E", "link_6"])

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)
        self.object_init_height = torch.full((self.num_envs,), self.cfg.object_spawn_z, dtype=torch.float, device=self.device)
        self.object_init_xy = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ee_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.ee_quat_w[:, 0] = 1.0
        self.ee_pose_ref_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.ee_pose_ref_quat[:, 0] = 1.0
        self.ee_pose_ref_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.object_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_height = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_tilt_deg = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_up_z = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_ang_vel_norm = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.lift_amount = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.xy_drift = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ee_to_object = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.tcp_offset_axis_local = torch.tensor(self.cfg.tcp_axis_local, dtype=torch.float, device=self.device)
        self.object_up_axis_local = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float, device=self.device)
        self.tcp_offset_len = torch.full(
            (self.num_envs,), float(self.cfg.tcp_offset_open_m), dtype=torch.float, device=self.device
        )
        self.success_rate_ema = torch.zeros((), dtype=torch.float, device=self.device)
        self.success_rate_stable_steps = torch.zeros((), dtype=torch.long, device=self.device)
        self.return_home_active = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.return_home_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.return_home_entry_error = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

    def _setup_scene(self):
        # Keep user-authored table USD as visual/workspace asset.
        table_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.table_usd_path)
        table_cfg.func("/World/envs/env_.*/Table", table_cfg)

        if self.cfg.workspace_contains_robot:
            # Reuse robot already contained in table/workspace USD to avoid duplicate robot spawn.
            robot_cfg = self.cfg.robot.replace(prim_path=self.cfg.workspace_robot_prim_path, spawn=None)
        else:
            robot_cfg = self.cfg.robot

        self._robot = Articulation(robot_cfg)
        self._object = RigidObject(self.cfg.target_object)
        if self.cfg.use_table_collision_proxy:
            self._table_collision = RigidObject(self.cfg.table_collision)
        else:
            self._table_collision = None

        self.scene.articulations["robot"] = self._robot
        if self._table_collision is not None:
            self.scene.rigid_objects["table_collision"] = self._table_collision
        self.scene.rigid_objects["target_object"] = self._object

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _find_first_body_idx(self, candidates: list[str]) -> int:
        for name in candidates:
            try:
                body_ids, _ = self._robot.find_bodies(name)
                if len(body_ids) > 0:
                    return int(body_ids[0])
            except ValueError:
                continue
        return self._robot.num_bodies - 1

    def _set_named_joint_home(self, joint_name: str, joint_value: float):
        try:
            joint_ids, _ = self._robot.find_joints(joint_name)
            if len(joint_ids) > 0:
                self.home_joint_pos[int(joint_ids[0])] = joint_value
        except ValueError:
            return

    def _set_named_joint_speed_scale(self, joint_name: str, speed_scale: float):
        try:
            joint_ids, _ = self._robot.find_joints(joint_name)
            if len(joint_ids) > 0:
                self.robot_dof_speed_scales[int(joint_ids[0])] = speed_scale
        except ValueError:
            return

    def _collect_exact_joint_ids(self, joint_names: list[str]) -> torch.Tensor:
        joint_ids = []
        for name in joint_names:
            try:
                ids, _ = self._robot.find_joints(name)
                if len(ids) > 0:
                    joint_ids.append(int(ids[0]))
            except ValueError:
                continue
        if len(joint_ids) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        return torch.tensor(sorted(set(joint_ids)), dtype=torch.long, device=self.device)

    def _pad_or_truncate(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        if tensor.shape[-1] == target_dim:
            return tensor
        if tensor.shape[-1] > target_dim:
            return tensor[..., :target_dim]
        pad = torch.zeros((tensor.shape[0], target_dim - tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, pad), dim=-1)

    def _compute_gripper_state(self) -> torch.Tensor:
        if self.gripper_joint_ids.numel() == 0:
            return torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

        joint_pos = self._robot.data.joint_pos[:, self.gripper_joint_ids]
        open_targets = self.gripper_open_targets.unsqueeze(0)
        close_targets = self.gripper_close_targets.unsqueeze(0)
        span = close_targets - open_targets
        span_safe = torch.where(torch.abs(span) > 1.0e-5, span, torch.ones_like(span))
        per_joint_state = torch.clamp((joint_pos - open_targets) / span_safe, 0.0, 1.0)
        return per_joint_state.mean(dim=1)

    def _compute_success(self, gripper_state: torch.Tensor | None = None) -> torch.Tensor:
        if gripper_state is None:
            gripper_state = self._compute_gripper_state()
        lifted = self.lift_amount > float(self.cfg.success_lift_height)
        near_object = self.ee_to_object < float(self.cfg.success_ee_to_object)
        closed = gripper_state > float(self.cfg.success_min_gripper_state)
        return lifted & near_object & closed

    def _compute_home_joint_error(self) -> torch.Tensor:
        arm_dof_count = min(6, self._robot.num_joints)
        if arm_dof_count <= 0:
            return torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        arm_joint_pos = self._robot.data.joint_pos[:, :arm_dof_count]
        arm_home = self.home_joint_pos[:arm_dof_count].unsqueeze(0)
        return torch.mean(torch.abs(arm_joint_pos - arm_home), dim=-1)

    def _update_return_home_stage(self, gripper_state: torch.Tensor) -> torch.Tensor:
        activate = (
            (~self.return_home_active)
            & (self.lift_amount > float(self.cfg.return_home_start_lift))
            & (gripper_state > float(self.cfg.return_home_min_gripper_state))
        )
        self.return_home_active = self.return_home_active | activate
        return activate

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        ee_pos_w = self._robot.data.body_pos_w[env_ids, self.ee_body_idx]
        ee_quat_w = self._robot.data.body_quat_w[env_ids, self.ee_body_idx]
        self.ee_quat_w[env_ids] = ee_quat_w
        gripper_state = self._compute_gripper_state()[env_ids]
        tcp_len = (
            float(self.cfg.tcp_offset_open_m)
            + gripper_state * (float(self.cfg.tcp_offset_closed_m) - float(self.cfg.tcp_offset_open_m))
        )
        self.tcp_offset_len[env_ids] = tcp_len
        tcp_local = self.tcp_offset_axis_local.unsqueeze(0) * tcp_len.unsqueeze(-1)
        tcp_world = quat_apply(ee_quat_w, tcp_local)
        self.robot_grasp_pos[env_ids] = ee_pos_w + tcp_world
        self.object_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]
        object_quat_w = self._object.data.root_quat_w[env_ids]
        object_up_w = quat_apply(object_quat_w, self.object_up_axis_local.unsqueeze(0).expand(len(env_ids), -1))
        object_up_z = torch.clamp(object_up_w[:, 2], -1.0, 1.0)
        self.object_up_z[env_ids] = object_up_z
        self.object_tilt_deg[env_ids] = torch.rad2deg(torch.acos(object_up_z))
        self.object_ang_vel_norm[env_ids] = torch.norm(self._object.data.root_ang_vel_w[env_ids], dim=-1)

        object_height = self.object_pos_w[env_ids, 2] - self.scene.env_origins[env_ids, 2]
        self.object_height[env_ids] = object_height
        self.lift_amount[env_ids] = torch.clamp(object_height - self.object_init_height[env_ids], min=0.0)
        object_xy_local = self.object_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2]
        self.xy_drift[env_ids] = torch.norm(object_xy_local - self.object_init_xy[env_ids], p=2, dim=-1)

        to_object = self.robot_grasp_pos[env_ids] - self.object_pos_w[env_ids]
        self.ee_to_object[env_ids] = torch.norm(to_object, p=2, dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets.clone()

        arm_action_dim = min(6, self.actions.shape[1], self._robot.num_joints)
        if arm_action_dim > 0:
            targets[:, :arm_action_dim] = (
                self.robot_dof_targets[:, :arm_action_dim]
                + self.robot_dof_speed_scales[:arm_action_dim]
                * self.dt
                * self.actions[:, :arm_action_dim]
                * self.cfg.action_scale
            )

        if self.actions.shape[1] > 6 and self.gripper_joint_ids.numel() > 0:
            close_ratio = 0.5 * (self.actions[:, 6] + 1.0)
            desired = self.gripper_open_targets.unsqueeze(0) + close_ratio.unsqueeze(-1) * (
                self.gripper_close_targets - self.gripper_open_targets
            ).unsqueeze(0)
            alpha = float(self.cfg.gripper_target_smoothing)
            current = targets[:, self.gripper_joint_ids]
            targets[:, self.gripper_joint_ids] = current + alpha * (desired - current)

        # Commanded return-home stage: once lift threshold is reached, override policy actions with home targets.
        if bool(self.cfg.return_home_command_mode):
            gripper_state = self._compute_gripper_state()
            self._update_return_home_stage(gripper_state)
            active = self.return_home_active
            if torch.any(active):
                arm_dof_count = min(6, self._robot.num_joints)
                if arm_dof_count > 0:
                    arm_alpha = float(self.cfg.return_home_command_arm_alpha)
                    arm_home = self.home_joint_pos[:arm_dof_count].unsqueeze(0)
                    current_arm = targets[active, :arm_dof_count]
                    targets[active, :arm_dof_count] = current_arm + arm_alpha * (arm_home - current_arm)

                if self.gripper_joint_ids.numel() > 0:
                    rows = torch.nonzero(active, as_tuple=False).squeeze(-1)
                    if rows.numel() > 0:
                        close_ratio_cmd = float(self.cfg.return_home_command_close_ratio)
                        close_ratio_cmd = max(0.0, min(1.0, close_ratio_cmd))
                        desired = self.gripper_open_targets.unsqueeze(0) + close_ratio_cmd * (
                            self.gripper_close_targets - self.gripper_open_targets
                        ).unsqueeze(0)
                        desired = desired.expand(rows.numel(), -1)
                        current = targets[rows][:, self.gripper_joint_ids]
                        alpha = float(self.cfg.return_home_command_gripper_alpha)
                        updated = current + alpha * (desired - current)
                        targets[rows[:, None], self.gripper_joint_ids[None, :]] = updated

        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        gripper_state = self._compute_gripper_state()
        success = self._compute_success(gripper_state=gripper_state)
        return_home_mode = bool(self.cfg.enable_return_home_stage) or bool(self.cfg.return_home_command_mode)
        if return_home_mode:
            activate = self._update_return_home_stage(gripper_state)
            home_joint_error = self._compute_home_joint_error()
            self.return_home_entry_error[activate] = home_joint_error[activate]
            self.return_home_success = (
                self.return_home_active
                & (home_joint_error < float(self.cfg.return_home_joint_tolerance))
                & (self.lift_amount > float(self.cfg.return_home_hold_lift))
                & (gripper_state > float(self.cfg.return_home_min_gripper_state))
            )
        else:
            self.return_home_active[:] = False
            self.return_home_success[:] = False
        dropped = self.object_height < (float(self.cfg.table_top_z) - float(self.cfg.drop_below_table_margin))

        terminated = (self.return_home_success if return_home_mode else success) | dropped
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        gripper_state = self._compute_gripper_state()
        dist_reward = torch.exp(-8.0 * self.ee_to_object)
        grasp_reward = torch.exp(-float(self.cfg.grasp_dist_decay) * self.ee_to_object) * gripper_state
        handover = torch.clamp(
            (self.lift_amount - float(self.cfg.lift_handover_start)) / max(float(self.cfg.lift_handover_span), 1.0e-6),
            min=0.0,
            max=1.0,
        )
        dist_scale_handover = 1.0 - (1.0 - float(self.cfg.post_lift_dist_keep_ratio)) * handover
        grasp_scale_handover = 1.0 - (1.0 - float(self.cfg.post_lift_grasp_keep_ratio)) * handover
        dist_reward_eff = dist_reward * dist_scale_handover
        grasp_reward_eff = grasp_reward * grasp_scale_handover
        lift_progress = torch.clamp(self.lift_amount / float(self.cfg.target_lift_height), min=0.0, max=1.0)
        lift_reward = lift_progress * (0.4 + 0.6 * grasp_reward_eff)

        upright_reward = handover * self.object_up_z
        tilt_penalty = handover * torch.clamp(1.0 - self.object_up_z, min=0.0)
        ang_vel_penalty = handover * torch.clamp(
            self.object_ang_vel_norm / max(float(self.cfg.ang_vel_ref), 1.0e-6), min=0.0, max=1.0
        )

        # Latch reference EE orientation once after initial lift+grasp, then penalize orientation drift.
        ref_capture = (
            (~self.ee_pose_ref_valid)
            & (self.lift_amount > float(self.cfg.ee_pose_ref_lift_start))
            & (gripper_state > float(self.cfg.ee_pose_ref_min_gripper_state))
        )
        self.ee_pose_ref_quat[ref_capture] = self.ee_quat_w[ref_capture]
        self.ee_pose_ref_valid[ref_capture] = True
        ee_pose_ref_valid_f = self.ee_pose_ref_valid.float()
        q_err = quat_mul(quat_conjugate(self.ee_pose_ref_quat), self.ee_quat_w)
        roll_err, pitch_err, yaw_err = euler_xyz_from_quat(q_err)
        ee_rp_error = torch.abs(roll_err) + torch.abs(pitch_err)
        ee_yaw_error = torch.abs(yaw_err)
        ee_pose_hold_active = handover * ee_pose_ref_valid_f
        ee_pose_rp_penalty = ee_pose_hold_active * ee_rp_error
        ee_pose_yaw_penalty = ee_pose_hold_active * ee_yaw_error

        home_joint_error = self._compute_home_joint_error()
        return_home_mode = bool(self.cfg.enable_return_home_stage) or bool(self.cfg.return_home_command_mode)
        if return_home_mode:
            activate = self._update_return_home_stage(gripper_state)
            return_home_active_f = self.return_home_active.float()
            self.return_home_entry_error[activate] = home_joint_error[activate]
            if bool(self.cfg.enable_return_home_stage):
                return_dist_keep = 1.0 - (1.0 - float(self.cfg.return_home_dist_keep_ratio)) * return_home_active_f
                return_grasp_keep = 1.0 - (1.0 - float(self.cfg.return_home_grasp_keep_ratio)) * return_home_active_f
                dist_reward_eff = dist_reward_eff * return_dist_keep
                grasp_reward_eff = grasp_reward_eff * return_grasp_keep
                lift_keep = 1.0 - 0.8 * return_home_active_f
                lift_reward_eff = lift_reward * lift_keep
                return_home_reward = return_home_active_f * torch.exp(-10.0 * home_joint_error)
                return_home_error_penalty = return_home_active_f * home_joint_error
                entry_error = torch.clamp(self.return_home_entry_error, min=1.0e-4)
                home_error_reduction = torch.clamp(entry_error - home_joint_error, min=0.0)
                return_home_progress_reward = return_home_active_f * torch.clamp(
                    home_error_reduction / entry_error, 0.0, 1.0
                )
                near_band = max(float(self.cfg.return_home_near_success_band), 1.0e-6)
                near_success_ratio = torch.clamp(
                    (float(self.cfg.return_home_joint_tolerance) + near_band - home_joint_error) / near_band,
                    min=0.0,
                    max=1.0,
                )
                return_home_near_success_reward = return_home_active_f * near_success_ratio
                return_home_lift_hold = return_home_active_f * torch.clamp(
                    self.lift_amount / max(float(self.cfg.return_home_hold_lift), 1.0e-6), min=0.0, max=1.0
                )
                return_home_gripper_hold = return_home_active_f * gripper_state
            else:
                lift_reward_eff = lift_reward
                return_home_reward = torch.zeros_like(self.lift_amount)
                return_home_error_penalty = torch.zeros_like(self.lift_amount)
                return_home_progress_reward = torch.zeros_like(self.lift_amount)
                return_home_near_success_reward = torch.zeros_like(self.lift_amount)
                return_home_lift_hold = torch.zeros_like(self.lift_amount)
                return_home_gripper_hold = torch.zeros_like(self.lift_amount)
        else:
            self.return_home_active[:] = False
            return_home_active_f = torch.zeros_like(self.lift_amount)
            lift_reward_eff = lift_reward
            return_home_reward = torch.zeros_like(self.lift_amount)
            return_home_error_penalty = torch.zeros_like(self.lift_amount)
            return_home_progress_reward = torch.zeros_like(self.lift_amount)
            return_home_near_success_reward = torch.zeros_like(self.lift_amount)
            return_home_lift_hold = torch.zeros_like(self.lift_amount)
            return_home_gripper_hold = torch.zeros_like(self.lift_amount)

        xy_over = torch.clamp(self.xy_drift - float(self.cfg.xy_drift_free_margin), min=0.0)
        xy_penalty_active = (self.lift_amount > float(self.cfg.xy_penalty_start_lift)).float()
        xy_drift_penalty = xy_penalty_active * xy_over

        action_penalty = torch.sum(self.actions**2, dim=-1)
        false_grasp = (
            (self.ee_to_object > float(self.cfg.false_grasp_close_distance))
            & (gripper_state > float(self.cfg.false_grasp_min_gripper_state))
            & (self.lift_amount < float(self.cfg.false_grasp_min_lift))
        ).float()
        success = self._compute_success(gripper_state=gripper_state)
        if return_home_mode:
            self.return_home_success = (
                self.return_home_active
                & (home_joint_error < float(self.cfg.return_home_joint_tolerance))
                & (self.lift_amount > float(self.cfg.return_home_hold_lift))
                & (gripper_state > float(self.cfg.return_home_min_gripper_state))
            )
        else:
            self.return_home_success[:] = False
        upright_success = success & (self.object_up_z > float(self.cfg.upright_success_min_up_z))
        dropped = self.object_height < (float(self.cfg.table_top_z) - float(self.cfg.drop_below_table_margin))
        current_success_rate = success.float().mean()
        alpha = float(self.cfg.success_rate_ema_alpha)
        self.success_rate_ema = (1.0 - alpha) * self.success_rate_ema + alpha * current_success_rate
        is_stable = self.success_rate_ema >= float(self.cfg.success_rate_target)
        self.success_rate_stable_steps = torch.where(
            is_stable,
            self.success_rate_stable_steps + 1,
            torch.zeros_like(self.success_rate_stable_steps),
        )
        stage_ready = self.success_rate_stable_steps >= int(self.cfg.success_rate_hold_min_steps)

        rewards = (
            self.cfg.dist_reward_scale * dist_reward_eff
            + self.cfg.grasp_reward_scale * grasp_reward_eff
            + self.cfg.lift_reward_scale * lift_reward_eff
            + self.cfg.upright_reward_scale * upright_reward
            + self.cfg.return_home_reward_scale * return_home_reward
            + self.cfg.return_home_progress_reward_scale * return_home_progress_reward
            + self.cfg.return_home_near_success_reward_scale * return_home_near_success_reward
            + self.cfg.return_home_lift_hold_scale * return_home_lift_hold
            + self.cfg.return_home_gripper_hold_scale * return_home_gripper_hold
            - self.cfg.return_home_error_penalty_scale * return_home_error_penalty
            - self.cfg.tilt_penalty_scale * tilt_penalty
            - self.cfg.ang_vel_penalty_scale * ang_vel_penalty
            - self.cfg.ee_pose_rp_penalty_scale * ee_pose_rp_penalty
            - self.cfg.ee_pose_yaw_penalty_scale * ee_pose_yaw_penalty
            - self.cfg.xy_drift_penalty_scale * xy_drift_penalty
            - self.cfg.false_grasp_penalty_scale * false_grasp
            - self.cfg.action_penalty_scale * action_penalty
        )
        pre_home = ~self.return_home_active if return_home_mode else torch.ones_like(success, dtype=torch.bool)
        rewards = torch.where(success & pre_home, rewards + float(self.cfg.success_bonus), rewards)
        rewards = torch.where(upright_success & pre_home, rewards + float(self.cfg.upright_success_bonus), rewards)
        rewards = torch.where(self.return_home_success, rewards + float(self.cfg.return_home_bonus), rewards)

        diag_mask = self.lift_amount > float(self.cfg.diag_lift_threshold)
        diag_mask_f = diag_mask.float()
        diag_count = torch.clamp(diag_mask_f.sum(), min=1.0)
        diag_rate = diag_mask_f.mean()
        diag_ee_to_object = (self.ee_to_object * diag_mask_f).sum() / diag_count
        diag_gripper_state = (gripper_state * diag_mask_f).sum() / diag_count
        diag_object_ang_vel = (self.object_ang_vel_norm * diag_mask_f).sum() / diag_count
        diag_drop_rate = ((dropped.float() * diag_mask_f).sum() / diag_count)

        self.extras["log"] = {
            "dist_reward": (self.cfg.dist_reward_scale * dist_reward_eff).mean(),
            "grasp_reward": (self.cfg.grasp_reward_scale * grasp_reward_eff).mean(),
            "lift_reward": (self.cfg.lift_reward_scale * lift_reward_eff).mean(),
            "upright_reward": (self.cfg.upright_reward_scale * upright_reward).mean(),
            "return_home_reward": (self.cfg.return_home_reward_scale * return_home_reward).mean(),
            "return_home_progress_reward": (
                self.cfg.return_home_progress_reward_scale * return_home_progress_reward
            ).mean(),
            "return_home_near_success_reward": (
                self.cfg.return_home_near_success_reward_scale * return_home_near_success_reward
            ).mean(),
            "return_home_lift_hold_reward": (self.cfg.return_home_lift_hold_scale * return_home_lift_hold).mean(),
            "return_home_gripper_hold_reward": (
                self.cfg.return_home_gripper_hold_scale * return_home_gripper_hold
            ).mean(),
            "return_home_error_penalty": (
                -self.cfg.return_home_error_penalty_scale * return_home_error_penalty
            ).mean(),
            "tilt_penalty": (-self.cfg.tilt_penalty_scale * tilt_penalty).mean(),
            "ang_vel_penalty": (-self.cfg.ang_vel_penalty_scale * ang_vel_penalty).mean(),
            "ee_pose_rp_penalty": (-self.cfg.ee_pose_rp_penalty_scale * ee_pose_rp_penalty).mean(),
            "ee_pose_yaw_penalty": (-self.cfg.ee_pose_yaw_penalty_scale * ee_pose_yaw_penalty).mean(),
            "xy_drift_penalty": (-self.cfg.xy_drift_penalty_scale * xy_drift_penalty).mean(),
            "false_grasp_penalty": (-self.cfg.false_grasp_penalty_scale * false_grasp).mean(),
            "action_penalty": (-self.cfg.action_penalty_scale * action_penalty).mean(),
            "success_rate": success.float().mean(),
            "upright_success_rate": upright_success.float().mean(),
            "return_home_success_rate": self.return_home_success.float().mean(),
            "success_rate_ema": self.success_rate_ema,
            "success_stage_stable_steps": self.success_rate_stable_steps.float(),
            "success_stage_ready": stage_ready.float(),
            "handover_gate": handover.mean(),
            "return_home_active_rate": return_home_active_f.mean(),
            "mean_home_joint_error_rad": home_joint_error.mean(),
            "ee_pose_ref_valid_rate": ee_pose_ref_valid_f.mean(),
            "mean_ee_rp_error_deg": torch.rad2deg(ee_rp_error).mean(),
            "mean_ee_yaw_error_deg": torch.rad2deg(ee_yaw_error).mean(),
            "xy_penalty_active_rate": xy_penalty_active.mean(),
            "mean_lift_amount": self.lift_amount.mean(),
            "mean_object_tilt_deg": self.object_tilt_deg.mean(),
            "mean_xy_drift": self.xy_drift.mean(),
            "mean_ee_to_object": self.ee_to_object.mean(),
            "mean_tcp_offset_m": self.tcp_offset_len.mean(),
            "mean_gripper_state": gripper_state.mean(),
            "diag_lift_mask_rate": diag_rate,
            "diag_lift_mask_mean_ee_to_object": diag_ee_to_object,
            "diag_lift_mask_mean_gripper_state": diag_gripper_state,
            "diag_lift_mask_mean_object_ang_vel": diag_object_ang_vel,
            "diag_lift_mask_drop_rate": diag_drop_rate,
        }

        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # robot state
        joint_pos = self.home_joint_pos.unsqueeze(0).repeat(len(env_ids), 1)
        arm_dof_count = min(6, self._robot.num_joints)
        if arm_dof_count > 0:
            joint_pos[:, :arm_dof_count] += sample_uniform(-0.05, 0.05, (len(env_ids), arm_dof_count), self.device)
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # target object state
        object_state = self._object.data.default_root_state[env_ids].clone()
        spawn_x = sample_uniform(
            self.cfg.object_spawn_x_range[0], self.cfg.object_spawn_x_range[1], (len(env_ids),), self.device
        )
        spawn_y = sample_uniform(
            self.cfg.object_spawn_y_range[0], self.cfg.object_spawn_y_range[1], (len(env_ids),), self.device
        )
        object_state[:, 0] = self.scene.env_origins[env_ids, 0] + spawn_x
        object_state[:, 1] = self.scene.env_origins[env_ids, 1] + spawn_y
        object_state[:, 2] = self.scene.env_origins[env_ids, 2] + float(self.cfg.object_spawn_z)
        object_state[:, 7:] = 0.0

        self._object.write_root_pose_to_sim(object_state[:, :7], env_ids=env_ids)
        self._object.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)

        self.object_init_height[env_ids] = float(self.cfg.object_spawn_z)
        self.object_init_xy[env_ids, 0] = spawn_x
        self.object_init_xy[env_ids, 1] = spawn_y
        self.ee_pose_ref_quat[env_ids] = 0.0
        self.ee_pose_ref_quat[env_ids, 0] = 1.0
        self.ee_pose_ref_valid[env_ids] = False
        self.return_home_active[env_ids] = False
        self.return_home_success[env_ids] = False
        self.return_home_entry_error[env_ids] = 0.0

        self._compute_intermediate_values(env_ids)
        self.return_home_entry_error[env_ids] = self._compute_home_joint_error()[env_ids]

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        denom = torch.clamp(self.robot_dof_upper_limits - self.robot_dof_lower_limits, min=1.0e-5)
        dof_pos_scaled = 2.0 * (self._robot.data.joint_pos - self.robot_dof_lower_limits) / denom - 1.0
        dof_vel_scaled = self._robot.data.joint_vel * float(self.cfg.dof_velocity_scale)

        joint_pos_obs = self._pad_or_truncate(dof_pos_scaled, int(self.cfg.obs_joint_dim))
        joint_vel_obs = self._pad_or_truncate(dof_vel_scaled, int(self.cfg.obs_joint_dim))

        to_object = self.object_pos_w - self.robot_grasp_pos
        gripper_state = self._compute_gripper_state().unsqueeze(-1)
        if bool(self.cfg.enable_return_home_stage) or bool(self.cfg.return_home_command_mode):
            return_home_active = self.return_home_active.float().unsqueeze(-1)
            home_joint_error = self._compute_home_joint_error().unsqueeze(-1)
        else:
            return_home_active = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
            home_joint_error = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)

        obs = torch.cat(
            (
                joint_pos_obs,
                joint_vel_obs,
                to_object,
                self.lift_amount.unsqueeze(-1),
                gripper_state,
                self.object_up_z.unsqueeze(-1),
                self.object_ang_vel_norm.unsqueeze(-1),
                return_home_active,
                home_joint_error,
            ),
            dim=-1,
        )

        return {"policy": torch.clamp(obs, -5.0, 5.0)}
