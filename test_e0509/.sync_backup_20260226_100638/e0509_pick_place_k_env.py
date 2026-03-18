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
from isaaclab.utils.math import quat_apply, sample_uniform

_ASSET_ROOT = Path(__file__).resolve().parent / "USD"
_ROBOT_USD_PATH = str(_ASSET_ROOT / "e0509" / "e0509.usd")
_TABLE_USD_PATH = str(_ASSET_ROOT / "table_693.usd")


@configclass
class E0509PickPlaceKEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 steps with dt=1/120 and decimation=2
    decimation = 2
    action_space = 7  # 6 arm joints + 1 synchronized gripper command
    observation_space = 28
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
        clone_in_fabric=True,
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
    # Workspace tabletop world Z: 700 mm. (previous temporary values: 0.713, 0.693)
    table_top_z = 0.700
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=table_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Keep the custom USD at world origin as workspace reference.
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # target object to grasp and lift
    object_size = (0.05, 0.04, 0.03)
    object_spawn_z = table_top_z + 0.5 * object_size[2] + 0.01
    object_spawn_x_range = (0.42, 0.56)
    object_spawn_y_range = (-0.14, 0.14)

    target_object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TargetObject",
        spawn=sim_utils.CuboidCfg(
            size=object_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
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
    target_lift_height = 0.08
    drop_below_table_margin = 0.03

    # rewards
    dist_reward_scale = 2.0
    grasp_reward_scale = 1.0
    lift_reward_scale = 12.0
    false_grasp_penalty_scale = 2.0
    false_grasp_close_distance = 0.06
    false_grasp_min_gripper_state = 0.55
    false_grasp_min_lift = 0.01
    action_penalty_scale = 0.02
    success_bonus = 5.0


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

        self.gripper_joint_ids = self._collect_exact_joint_ids(list(self.cfg.gripper_joint_names))
        if self.gripper_joint_ids.numel() > 0:
            self.robot_dof_speed_scales[self.gripper_joint_ids] = 0.12

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.home_joint_pos = self._robot.data.default_joint_pos[0].clone()
        self._set_named_joint_home("joint_3", 0.5 * math.pi)
        self._set_named_joint_home("joint_5", 0.5 * math.pi)

        if self.gripper_joint_ids.numel() > 0:
            low = self.robot_dof_lower_limits[self.gripper_joint_ids]
            high = self.robot_dof_upper_limits[self.gripper_joint_ids]
            self.gripper_open_targets = torch.clamp(torch.full_like(low, self.cfg.gripper_open_joint_pos), low, high)
            self.gripper_close_targets = torch.clamp(torch.full_like(low, self.cfg.gripper_close_joint_pos), low, high)
            self.home_joint_pos[self.gripper_joint_ids] = self.gripper_open_targets
        else:
            self.gripper_open_targets = torch.zeros((0,), dtype=torch.float, device=self.device)
            self.gripper_close_targets = torch.zeros((0,), dtype=torch.float, device=self.device)

        self.ee_body_idx = self._find_first_body_idx(["gripper", "rh_p12_rn_E", "link_6"])

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)
        self.object_init_height = torch.full((self.num_envs,), self.cfg.object_spawn_z, dtype=torch.float, device=self.device)

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_height = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.lift_amount = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ee_to_object = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.tcp_offset_axis_local = torch.tensor(self.cfg.tcp_axis_local, dtype=torch.float, device=self.device)
        self.tcp_offset_len = torch.full(
            (self.num_envs,), float(self.cfg.tcp_offset_open_m), dtype=torch.float, device=self.device
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._table = RigidObject(self.cfg.table)
        self._object = RigidObject(self.cfg.target_object)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
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

        joint_mean = self._robot.data.joint_pos[:, self.gripper_joint_ids].mean(dim=1)
        open_mean = float(self.gripper_open_targets.mean().item())
        close_mean = float(self.gripper_close_targets.mean().item())
        denom = max(abs(close_mean - open_mean), 1.0e-5)
        return torch.clamp((joint_mean - open_mean) / denom, 0.0, 1.0)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        ee_pos_w = self._robot.data.body_pos_w[env_ids, self.ee_body_idx]
        ee_quat_w = self._robot.data.body_quat_w[env_ids, self.ee_body_idx]
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

        object_height = self.object_pos_w[env_ids, 2] - self.scene.env_origins[env_ids, 2]
        self.object_height[env_ids] = object_height
        self.lift_amount[env_ids] = torch.clamp(object_height - self.object_init_height[env_ids], min=0.0)

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

        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        success = self.lift_amount > float(self.cfg.target_lift_height)
        dropped = self.object_height < (float(self.cfg.table_top_z) - float(self.cfg.drop_below_table_margin))

        terminated = success | dropped
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        gripper_state = self._compute_gripper_state()

        dist_reward = torch.exp(-8.0 * self.ee_to_object)
        grasp_reward = gripper_state * torch.exp(-20.0 * self.ee_to_object)

        lift_progress = torch.clamp(self.lift_amount / float(self.cfg.target_lift_height), min=0.0, max=1.0)
        lift_gate = (gripper_state > 0.40).float()
        lift_reward = lift_progress * lift_gate
        false_grasp = (
            (gripper_state > float(self.cfg.false_grasp_min_gripper_state))
            & (self.ee_to_object < float(self.cfg.false_grasp_close_distance))
            & (self.lift_amount < float(self.cfg.false_grasp_min_lift))
        )
        false_grasp_penalty = false_grasp.float() * torch.exp(-12.0 * self.ee_to_object)

        action_penalty = torch.sum(self.actions**2, dim=-1)
        success = self.lift_amount > float(self.cfg.target_lift_height)

        rewards = (
            self.cfg.dist_reward_scale * dist_reward
            + self.cfg.grasp_reward_scale * grasp_reward
            + self.cfg.lift_reward_scale * lift_reward
            - self.cfg.false_grasp_penalty_scale * false_grasp_penalty
            - self.cfg.action_penalty_scale * action_penalty
        )
        rewards = torch.where(success, rewards + float(self.cfg.success_bonus), rewards)

        self.extras["log"] = {
            "dist_reward": (self.cfg.dist_reward_scale * dist_reward).mean(),
            "grasp_reward": (self.cfg.grasp_reward_scale * grasp_reward).mean(),
            "lift_reward": (self.cfg.lift_reward_scale * lift_reward).mean(),
            "false_grasp_penalty": (-self.cfg.false_grasp_penalty_scale * false_grasp_penalty).mean(),
            "false_grasp_rate": false_grasp.float().mean(),
            "action_penalty": (-self.cfg.action_penalty_scale * action_penalty).mean(),
            "success_rate": success.float().mean(),
            "mean_lift_amount": self.lift_amount.mean(),
            "mean_ee_to_object": self.ee_to_object.mean(),
            "mean_tcp_offset_m": self.tcp_offset_len.mean(),
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

        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        denom = torch.clamp(self.robot_dof_upper_limits - self.robot_dof_lower_limits, min=1.0e-5)
        dof_pos_scaled = 2.0 * (self._robot.data.joint_pos - self.robot_dof_lower_limits) / denom - 1.0
        dof_vel_scaled = self._robot.data.joint_vel * float(self.cfg.dof_velocity_scale)

        joint_pos_obs = self._pad_or_truncate(dof_pos_scaled, int(self.cfg.obs_joint_dim))
        joint_vel_obs = self._pad_or_truncate(dof_vel_scaled, int(self.cfg.obs_joint_dim))

        to_object = self.object_pos_w - self.robot_grasp_pos
        object_pos_local = self.object_pos_w - self.scene.env_origins
        gripper_state = self._compute_gripper_state().unsqueeze(-1)

        obs = torch.cat(
            (
                joint_pos_obs,
                joint_vel_obs,
                to_object,
                self.lift_amount.unsqueeze(-1),
                gripper_state,
                object_pos_local,
            ),
            dim=-1,
        )

        return {"policy": torch.clamp(obs, -5.0, 5.0)}
