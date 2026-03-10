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
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, sample_uniform

_ASSET_ROOT = Path(__file__).resolve().parent / "USD"
_ROBOT_USD_PATH = str(_ASSET_ROOT / "e0509" / "e0509_model.usd")


def _first_existing_usd(*file_names: str) -> str:
    for file_name in file_names:
        candidate = _ASSET_ROOT / file_name
        if candidate.exists():
            return str(candidate)
    return str(_ASSET_ROOT / file_names[-1])


_TABLE_PRIMARY_USD_PATH = str(_ASSET_ROOT / "tables_3.usd")
_TABLE_SECONDARY_USD_PATH = str(_ASSET_ROOT / "table_hole.usd")
_SOJU_USD_PATH = _first_existing_usd("soju.usd")
_ORANGE_USD_PATH = _first_existing_usd("orange.usd")
_BEER_USD_PATH = _first_existing_usd("beer.usd")


@configclass
class GripBottleEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333
    decimation = 2
    action_space = 7
    observation_space = 21
    state_space = 0
    obs_joint_dim = 6

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
        num_envs=64,
        env_spacing=2.2,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # robot
    robot_base_z_offset = 0.730
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ROBOT_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                # e0509_model.usd already contains a fixed joint to base; keep this off.
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint_1": 1.57079632679,
                "joint_2": -0.78539816339,
                "joint_3": 1.57079632679,
                "joint_4": 0.0,
                "joint_5": 0.78539816339,
                "joint_6": -1.57079632679,
                "rh_l1": 0.02,
                "rh_r1_joint": 0.02,
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

    # tables
    table_usd_paths = (_TABLE_PRIMARY_USD_PATH, _TABLE_SECONDARY_USD_PATH)
    workspace_contains_robot = False
    workspace_robot_prim_path = "/World/envs/env_.*/Table/e0509"

    # black table top (user-provided)
    table_top_center_xy = (0.0, 0.650)
    table_top_size_xy = (0.724, 0.300)
    table_top_z = 1.300

    # collision proxy
    use_table_collision_proxy = True
    table_collision_size = (table_top_size_xy[0], table_top_size_xy[1], 0.04)
    table_collision_center_xy = table_top_center_xy
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
            pos=(table_collision_center_xy[0], table_collision_center_xy[1], table_collision_center_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # spawn area
    table_spawn_edge_margin = 0.03
    object_spawn_z = table_top_z + 0.002
    object_spawn_x_range = (
        table_top_center_xy[0] - 0.5 * table_top_size_xy[0] + table_spawn_edge_margin,
        table_top_center_xy[0] + 0.5 * table_top_size_xy[0] - table_spawn_edge_margin,
    )
    object_spawn_y_range = (
        table_top_center_xy[1] - 0.5 * table_top_size_xy[1] + table_spawn_edge_margin,
        table_top_center_xy[1] + 0.5 * table_top_size_xy[1] - table_spawn_edge_margin,
    )
    object_spawn_yaw_range = (-math.pi, math.pi)

    # inactive bottle parking
    parked_object_x = -0.45
    parked_object_y = (-0.28, 0.0, 0.28)
    parked_object_z = -1.0

    # bottle assets
    bottle_soju = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SojuBottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_SOJU_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.6585),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, object_spawn_z), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    bottle_orange = RigidObjectCfg(
        prim_path="/World/envs/env_.*/OrangeBottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ORANGE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3933),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, object_spawn_z), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    bottle_beer = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BeerBottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_BEER_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5177),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, object_spawn_z), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # contact sensor for collision-avoidance shaping
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/e0509/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
    )

    # ground
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
    # Observation joints for sim2real consistency: 6 arm joints only (gripper uses gripper_state scalar).
    obs_joint_names = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
    # Real hardware command channel: use only rh_r1_joint for open/close.
    gripper_joint_names = ("rh_r1_joint",)
    gripper_open_joint_pos = 0.02
    gripper_close_joint_pos = 1.0
    gripper_target_smoothing = 0.35
    arm_reset_noise_rad = 0.0

    # manual-informed arm limits (Doosan E0509 user manual defaults)
    use_manual_joint_limits = True
    joint_1_abs_limit_deg = 115.0

    # tcp
    # Distance from EE to gripper tip along local tool +Z.
    tcp_offset_open_m = 0.107
    tcp_offset_closed_m = 0.135
    # Grasp center is 17.5 mm from tip toward EE (not outward from tip).
    grasp_center_from_tip_to_ee_m = 0.0175
    tcp_axis_local = (0.0, 0.0, 1.0)

    # task + rewards
    target_lift_height = 0.05
    success_lift_height = 0.045
    success_ee_to_object = 0.055
    object_grasp_offset_z_range = (0.04, 0.08)
    success_min_gripper_state = 0.50
    stable_grasp_min_lift = 0.01
    upright_success_min_up_z = 0.96
    topple_reset_max_up_z = 0.90
    drop_below_table_margin = 0.03
    contact_force_threshold = 1.0
    terminate_contact_force = 6.0
    dist_reward_scale = 0.4
    grasp_reward_scale = 5.0
    lift_reward_scale = 20.0
    upright_reward_scale = 6.0
    lift_upright_gate_min_up_z = 0.90
    lift_upright_gate_span = 0.06
    tilt_penalty_start_lift = 0.01
    tilt_penalty_scale = 3.0
    post_grasp_arm_slow_min_gripper_state = 0.60
    post_grasp_arm_action_scale_ratio = 0.60
    post_grasp_xy_penalty_min_gripper_state = 0.60
    post_grasp_xy_penalty_min_lift = 0.01
    post_grasp_xy_speed_penalty_scale = 1.5
    post_grasp_z_bonus_min_gripper_state = 0.60
    post_grasp_z_bonus_min_lift = 0.005
    post_grasp_z_vel_bonus_scale = 2.0
    post_grasp_z_vel_clip = 0.20
    approach_pose_max_dist = 0.12
    approach_pose_max_lift = 0.005
    approach_pose_max_gripper_state = 0.60
    approach_pose_dist_decay = 6.0
    approach_pose_reward_scale = 1.5
    collision_penalty_scale = 2.0
    xy_drift_penalty_scale = 1.0
    xy_drift_free_margin = 0.03
    action_penalty_scale = 0.005
    singularity_joint_3_threshold_deg = 8.0
    singularity_joint_5_threshold_deg = 8.0
    singularity_penalty_scale = 0.2
    success_bonus = 12.0
    grasp_dist_decay = 24.0


class GripBottleEnv(DirectRLEnv):
    cfg: GripBottleEnvCfg

    def __init__(self, cfg: GripBottleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)
        self.policy_dof_lower_limits = self.robot_dof_lower_limits.clone()
        self.policy_dof_upper_limits = self.robot_dof_upper_limits.clone()
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        arm_dof_count = min(6, self._robot.num_joints)
        self.robot_dof_speed_scales[:arm_dof_count] = 0.45

        self.gripper_joint_ids = self._collect_exact_joint_ids(list(self.cfg.gripper_joint_names))
        if self.gripper_joint_ids.numel() > 0:
            self.robot_dof_speed_scales[self.gripper_joint_ids] = 0.12
        self.obs_joint_ids = self._collect_exact_joint_ids(list(self.cfg.obs_joint_names))

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.home_joint_pos = self._robot.data.default_joint_pos[0].clone()
        self._set_named_joint_home("joint_1", 0.5 * math.pi)
        self._set_named_joint_home("joint_2", -0.25 * math.pi)
        self._set_named_joint_home("joint_3", 0.5 * math.pi)
        self._set_named_joint_home("joint_4", 0.0)
        self._set_named_joint_home("joint_5", 0.25 * math.pi)
        self._set_named_joint_home("joint_6", -0.5 * math.pi)
        self._apply_manual_arm_limits()

        if self.gripper_joint_ids.numel() > 0:
            low = self.robot_dof_lower_limits[self.gripper_joint_ids]
            high = self.robot_dof_upper_limits[self.gripper_joint_ids]
            self.gripper_open_targets = torch.clamp(torch.full_like(low, self.cfg.gripper_open_joint_pos), low, high)
            close_from_cfg = torch.clamp(torch.full_like(low, self.cfg.gripper_close_joint_pos), low, high)
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

        self.active_bottle_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.active_bottle_one_hot = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_grasp_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_height = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_up_z = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.lift_amount = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.xy_drift = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ee_to_object = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.tcp_height = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_xy_speed = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_upward_speed = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.post_grasp_arm_scale = torch.ones((self.num_envs,), dtype=torch.float, device=self.device)
        self.tcp_x_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_up_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.tcp_offset_axis_local = torch.tensor(self.cfg.tcp_axis_local, dtype=torch.float, device=self.device)
        self.tcp_x_axis_local = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float, device=self.device)
        self.object_up_axis_local = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float, device=self.device)
        self.object_grasp_offset_z = sample_uniform(
            float(self.cfg.object_grasp_offset_z_range[0]),
            float(self.cfg.object_grasp_offset_z_range[1]),
            (self.num_envs,),
            self.device,
        )
        self.arm_collision_peak = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.arm_collision_over = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.joint_3_abs_rad = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.joint_5_abs_rad = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.singularity_penalty = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.joint_3_id = self._find_joint_id("joint_3")
        self.joint_5_id = self._find_joint_id("joint_5")
        self._arm_contact_body_ids = self._resolve_arm_contact_body_ids()

    def _setup_scene(self):
        for table_idx, table_usd_path in enumerate(self.cfg.table_usd_paths):
            table_cfg = sim_utils.UsdFileCfg(usd_path=table_usd_path)
            table_cfg.func(f"/World/envs/env_.*/Table_{table_idx}", table_cfg)

        if self.cfg.workspace_contains_robot:
            robot_cfg = self.cfg.robot.replace(prim_path=self.cfg.workspace_robot_prim_path, spawn=None)
        else:
            robot_cfg = self.cfg.robot

        self._robot = Articulation(robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._bottle_soju = RigidObject(self.cfg.bottle_soju)
        self._bottle_orange = RigidObject(self.cfg.bottle_orange)
        self._bottle_beer = RigidObject(self.cfg.bottle_beer)
        self._bottle_objects = [self._bottle_soju, self._bottle_orange, self._bottle_beer]

        self._table_collision = RigidObject(self.cfg.table_collision) if self.cfg.use_table_collision_proxy else None

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if self._table_collision is not None:
            self.scene.rigid_objects["table_collision"] = self._table_collision
        self.scene.rigid_objects["bottle_soju"] = self._bottle_soju
        self.scene.rigid_objects["bottle_orange"] = self._bottle_orange
        self.scene.rigid_objects["bottle_beer"] = self._bottle_beer

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

    def _collect_exact_joint_ids(self, joint_names: list[str]) -> torch.Tensor:
        joint_ids = []
        seen = set()
        for name in joint_names:
            try:
                ids, _ = self._robot.find_joints(name)
                if len(ids) > 0:
                    joint_id = int(ids[0])
                    if joint_id not in seen:
                        joint_ids.append(joint_id)
                        seen.add(joint_id)
            except ValueError:
                continue
        if len(joint_ids) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        return torch.tensor(joint_ids, dtype=torch.long, device=self.device)

    def _find_joint_id(self, joint_name: str) -> int:
        try:
            joint_ids, _ = self._robot.find_joints(joint_name)
            if len(joint_ids) > 0:
                return int(joint_ids[0])
        except ValueError:
            pass
        return -1

    def _set_named_joint_home(self, joint_name: str, joint_value: float):
        joint_id = self._find_joint_id(joint_name)
        if joint_id >= 0:
            self.home_joint_pos[joint_id] = float(joint_value)

    def _set_named_joint_abs_limit_deg(self, joint_name: str, abs_limit_deg: float):
        joint_id = self._find_joint_id(joint_name)
        if joint_id < 0:
            return
        limit = float(abs_limit_deg) * math.pi / 180.0
        limit_tensor = self.policy_dof_lower_limits.new_tensor(limit)
        self.policy_dof_lower_limits[joint_id] = torch.maximum(
            self.policy_dof_lower_limits[joint_id], -limit_tensor
        )
        self.policy_dof_upper_limits[joint_id] = torch.minimum(
            self.policy_dof_upper_limits[joint_id], limit_tensor
        )

    def _apply_manual_arm_limits(self):
        if not bool(self.cfg.use_manual_joint_limits):
            return
        self._set_named_joint_abs_limit_deg("joint_1", float(self.cfg.joint_1_abs_limit_deg))

    def _resolve_arm_contact_body_ids(self) -> torch.Tensor:
        names = self._contact_sensor.body_names
        exclude_keys = ("rh_", "gripper", "finger", "hand")
        ids = [idx for idx, name in enumerate(names) if not any(key in name.lower() for key in exclude_keys)]
        if len(ids) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _update_collision_metrics(self):
        if self._arm_contact_body_ids.numel() == 0:
            self.arm_collision_peak[:] = 0.0
            self.arm_collision_over[:] = 0.0
            return

        net_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._arm_contact_body_ids]
        peak_per_body = torch.max(torch.norm(net_forces, dim=-1), dim=1)[0]
        peak_force = torch.max(peak_per_body, dim=1)[0]
        self.arm_collision_peak[:] = peak_force
        self.arm_collision_over[:] = torch.clamp(peak_force - float(self.cfg.contact_force_threshold), min=0.0)

    def _update_singularity_metrics(self):
        joint_pos = self._robot.data.joint_pos
        if self.joint_3_id >= 0:
            self.joint_3_abs_rad[:] = torch.abs(joint_pos[:, self.joint_3_id])
            threshold_3 = float(self.cfg.singularity_joint_3_threshold_deg) * math.pi / 180.0
            q3_margin = torch.clamp((threshold_3 - self.joint_3_abs_rad) / max(threshold_3, 1.0e-6), min=0.0, max=1.0)
        else:
            self.joint_3_abs_rad[:] = 0.0
            q3_margin = torch.zeros_like(self.joint_3_abs_rad)

        if self.joint_5_id >= 0:
            self.joint_5_abs_rad[:] = torch.abs(joint_pos[:, self.joint_5_id])
            threshold_5 = float(self.cfg.singularity_joint_5_threshold_deg) * math.pi / 180.0
            q5_margin = torch.clamp((threshold_5 - self.joint_5_abs_rad) / max(threshold_5, 1.0e-6), min=0.0, max=1.0)
        else:
            self.joint_5_abs_rad[:] = 0.0
            q5_margin = torch.zeros_like(self.joint_5_abs_rad)

        self.singularity_penalty[:] = torch.maximum(q3_margin, q5_margin)

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
        return torch.clamp((joint_pos - open_targets) / span_safe, 0.0, 1.0).mean(dim=1)

    def _gather_active_bottle_tensor(self, env_ids: torch.Tensor, attr_name: str) -> torch.Tensor:
        stacked = torch.stack([getattr(obj.data, attr_name)[env_ids] for obj in self._bottle_objects], dim=1)
        rows = torch.arange(stacked.shape[0], device=self.device)
        return stacked[rows, self.active_bottle_idx[env_ids]]

    def _compute_success(self, gripper_state: torch.Tensor) -> torch.Tensor:
        lifted = self.lift_amount > float(self.cfg.success_lift_height)
        near = self.ee_to_object < float(self.cfg.success_ee_to_object)
        closed = gripper_state > float(self.cfg.success_min_gripper_state)
        upright = self.object_up_z > float(self.cfg.upright_success_min_up_z)
        return lifted & near & closed & upright

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        ee_pos_w = self._robot.data.body_pos_w[env_ids, self.ee_body_idx]
        ee_quat_w = self._robot.data.body_quat_w[env_ids, self.ee_body_idx]

        gripper_state = self._compute_gripper_state()[env_ids]
        tcp_tip_len = (
            float(self.cfg.tcp_offset_open_m)
            + gripper_state * (float(self.cfg.tcp_offset_closed_m) - float(self.cfg.tcp_offset_open_m))
        )
        tcp_len = torch.clamp(tcp_tip_len - float(self.cfg.grasp_center_from_tip_to_ee_m), min=0.0)
        tcp_local = self.tcp_offset_axis_local.unsqueeze(0) * tcp_len.unsqueeze(-1)
        self.robot_grasp_pos[env_ids] = ee_pos_w + quat_apply(ee_quat_w, tcp_local)
        self.tcp_x_w[env_ids] = quat_apply(ee_quat_w, self.tcp_x_axis_local.unsqueeze(0).expand(len(env_ids), -1))

        self.object_pos_w[env_ids] = self._gather_active_bottle_tensor(env_ids, "root_pos_w")
        object_quat_w = self._gather_active_bottle_tensor(env_ids, "root_quat_w")
        object_up_w = quat_apply(object_quat_w, self.object_up_axis_local.unsqueeze(0).expand(len(env_ids), -1))
        self.object_up_w[env_ids] = object_up_w
        self.object_up_z[env_ids] = torch.clamp(object_up_w[:, 2], -1.0, 1.0)
        self.object_grasp_pos_w[env_ids] = self.object_pos_w[env_ids] + object_up_w * self.object_grasp_offset_z[env_ids].unsqueeze(-1)

        object_height = self.object_pos_w[env_ids, 2] - self.scene.env_origins[env_ids, 2]
        self.object_height[env_ids] = object_height
        self.lift_amount[env_ids] = torch.clamp(object_height - self.object_init_height[env_ids], min=0.0)

        object_xy_local = self.object_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2]
        self.xy_drift[env_ids] = torch.norm(object_xy_local - self.object_init_xy[env_ids], p=2, dim=-1)
        self.ee_to_object[env_ids] = torch.norm(self.robot_grasp_pos[env_ids] - self.object_grasp_pos_w[env_ids], p=2, dim=-1)
        self.tcp_height[env_ids] = self.robot_grasp_pos[env_ids, 2] - self.scene.env_origins[env_ids, 2]
        object_lin_vel_w = self._gather_active_bottle_tensor(env_ids, "root_lin_vel_w")
        self.object_xy_speed[env_ids] = torch.norm(object_lin_vel_w[:, :2], p=2, dim=-1)
        self.object_upward_speed[env_ids] = object_lin_vel_w[:, 2]

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets.clone()

        arm_dim = min(6, self.actions.shape[1], self._robot.num_joints)
        if arm_dim > 0:
            gripper_state = self._compute_gripper_state()
            post_grasp_arm_scale = torch.where(
                gripper_state > float(self.cfg.post_grasp_arm_slow_min_gripper_state),
                torch.full_like(gripper_state, float(self.cfg.post_grasp_arm_action_scale_ratio)),
                torch.ones_like(gripper_state),
            )
            self.post_grasp_arm_scale[:] = post_grasp_arm_scale
            targets[:, :arm_dim] = (
                self.robot_dof_targets[:, :arm_dim]
                + self.robot_dof_speed_scales[:arm_dim]
                * self.dt
                * self.actions[:, :arm_dim]
                * self.cfg.action_scale
                * post_grasp_arm_scale.unsqueeze(-1)
            )

        if self.actions.shape[1] > 6 and self.gripper_joint_ids.numel() > 0:
            close_ratio = 0.5 * (self.actions[:, 6] + 1.0)
            desired = self.gripper_open_targets.unsqueeze(0) + close_ratio.unsqueeze(-1) * (
                self.gripper_close_targets - self.gripper_open_targets
            ).unsqueeze(0)
            alpha = float(self.cfg.gripper_target_smoothing)
            current = targets[:, self.gripper_joint_ids]
            targets[:, self.gripper_joint_ids] = current + alpha * (desired - current)

        self.robot_dof_targets[:] = torch.clamp(targets, self.policy_dof_lower_limits, self.policy_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        self._update_collision_metrics()
        self._update_singularity_metrics()
        gripper_state = self._compute_gripper_state()
        success = self._compute_success(gripper_state)
        dropped = self.object_height < (float(self.cfg.table_top_z) - float(self.cfg.drop_below_table_margin))
        toppled = self.object_up_z < float(self.cfg.topple_reset_max_up_z)
        collided = self.arm_collision_peak > float(self.cfg.terminate_contact_force)
        terminated = success | dropped | toppled | collided
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        self._update_collision_metrics()
        self._update_singularity_metrics()

        gripper_state = self._compute_gripper_state()
        dist_reward = torch.exp(-8.0 * self.ee_to_object)
        grasp_reward = torch.exp(-float(self.cfg.grasp_dist_decay) * self.ee_to_object) * gripper_state
        lift_reward_raw = torch.clamp(self.lift_amount / float(self.cfg.target_lift_height), min=0.0, max=1.0)

        near_mask = self.ee_to_object < float(self.cfg.success_ee_to_object)
        closed_mask = gripper_state > float(self.cfg.success_min_gripper_state)
        grasp_candidate_mask = near_mask & closed_mask

        near_soft = torch.clamp(
            1.0 - self.ee_to_object / max(float(self.cfg.success_ee_to_object), 1.0e-6),
            min=0.0,
            max=1.0,
        )
        closed_soft = torch.clamp(
            (gripper_state - float(self.cfg.success_min_gripper_state))
            / max(1.0 - float(self.cfg.success_min_gripper_state), 1.0e-6),
            min=0.0,
            max=1.0,
        )
        stable_grasp_gate = near_soft * closed_soft

        lift_upright_gate = torch.clamp(
            (self.object_up_z - float(self.cfg.lift_upright_gate_min_up_z)) / max(float(self.cfg.lift_upright_gate_span), 1.0e-6),
            min=0.0,
            max=1.0,
        )
        lift_reward = lift_reward_raw * lift_upright_gate * stable_grasp_gate
        upright_reward = lift_reward_raw * torch.clamp(self.object_up_z, min=0.0, max=1.0) * stable_grasp_gate
        tilt_penalty = torch.clamp(float(self.cfg.upright_success_min_up_z) - self.object_up_z, min=0.0)
        tilt_penalty = torch.where(
            self.lift_amount > float(self.cfg.tilt_penalty_start_lift),
            tilt_penalty,
            torch.zeros_like(tilt_penalty),
        )
        post_grasp_xy_mask = (
            (gripper_state > float(self.cfg.post_grasp_xy_penalty_min_gripper_state))
            & (self.lift_amount > float(self.cfg.post_grasp_xy_penalty_min_lift))
        )
        post_grasp_xy_penalty = self.object_xy_speed * post_grasp_xy_mask.float()

        post_grasp_z_mask = (
            (gripper_state > float(self.cfg.post_grasp_z_bonus_min_gripper_state))
            & (self.lift_amount > float(self.cfg.post_grasp_z_bonus_min_lift))
        )
        post_grasp_z_up = torch.clamp(self.object_upward_speed, min=0.0, max=float(self.cfg.post_grasp_z_vel_clip))
        post_grasp_z_bonus = post_grasp_z_up * post_grasp_z_mask.float() * torch.clamp(1.0 - lift_reward_raw, min=0.0, max=1.0)

        # Encourage side-approach posture before grasp: TCP x-axis parallel to bottle up-axis.
        x_parallel = torch.abs(torch.sum(self.tcp_x_w * self.object_up_w, dim=-1))
        approach_pose_mask = (
            (self.ee_to_object < float(self.cfg.approach_pose_max_dist))
            & (self.lift_amount < float(self.cfg.approach_pose_max_lift))
            & (gripper_state < float(self.cfg.approach_pose_max_gripper_state))
        )
        approach_pose_dist_weight = torch.exp(-float(self.cfg.approach_pose_dist_decay) * self.ee_to_object)
        approach_pose_reward = x_parallel * approach_pose_dist_weight * approach_pose_mask.float()

        xy_penalty = torch.clamp(self.xy_drift - float(self.cfg.xy_drift_free_margin), min=0.0)
        action_penalty = torch.sum(self.actions**2, dim=-1)

        rewards = (
            self.cfg.dist_reward_scale * dist_reward
            + self.cfg.grasp_reward_scale * grasp_reward
            + self.cfg.lift_reward_scale * lift_reward
            + self.cfg.upright_reward_scale * upright_reward
            + self.cfg.post_grasp_z_vel_bonus_scale * post_grasp_z_bonus
            + self.cfg.approach_pose_reward_scale * approach_pose_reward
            - self.cfg.collision_penalty_scale * self.arm_collision_over
            - self.cfg.singularity_penalty_scale * self.singularity_penalty
            - self.cfg.tilt_penalty_scale * tilt_penalty
            - self.cfg.post_grasp_xy_speed_penalty_scale * post_grasp_xy_penalty
            - self.cfg.xy_drift_penalty_scale * xy_penalty
            - self.cfg.action_penalty_scale * action_penalty
        )

        success = self._compute_success(gripper_state)
        rewards = torch.where(success, rewards + float(self.cfg.success_bonus), rewards)

        self.extras["log"] = {
            "success_rate": success.float().mean(),
            "mean_lift": self.lift_amount.mean(),
            "mean_dist": self.ee_to_object.mean(),
            "mean_up_z": self.object_up_z.mean(),
            "mean_lift_upright_gate": lift_upright_gate.mean(),
            "mean_stable_grasp_gate": stable_grasp_gate.mean(),
            "near_rate": near_mask.float().mean(),
            "closed_rate": closed_mask.float().mean(),
            "grasp_candidate_rate": grasp_candidate_mask.float().mean(),
            "stable_grasp_rate": (grasp_candidate_mask & (self.lift_amount > float(self.cfg.stable_grasp_min_lift))).float().mean(),
            "mean_tilt_penalty": tilt_penalty.mean(),
            "mean_post_grasp_xy_penalty": post_grasp_xy_penalty.mean(),
            "mean_post_grasp_z_bonus": post_grasp_z_bonus.mean(),
            "mean_object_xy_speed": self.object_xy_speed.mean(),
            "mean_object_upward_speed": self.object_upward_speed.mean(),
            "mean_post_grasp_arm_scale": self.post_grasp_arm_scale.mean(),
            "mean_x_parallel": x_parallel.mean(),
            "mean_approach_pose_reward": approach_pose_reward.mean(),
            "approach_pose_mask_rate": approach_pose_mask.float().mean(),
            "mean_tcp_height": self.tcp_height.mean(),
            "mean_arm_collision_peak": self.arm_collision_peak.mean(),
            "collision_over_rate": (self.arm_collision_over > 0.0).float().mean(),
            "toppled_rate": (self.object_up_z < float(self.cfg.topple_reset_max_up_z)).float().mean(),
            "mean_singularity_penalty": self.singularity_penalty.mean(),
            "mean_abs_q3_deg": torch.rad2deg(self.joint_3_abs_rad).mean(),
            "mean_abs_q5_deg": torch.rad2deg(self.joint_5_abs_rad).mean(),
            "active_soju": self.active_bottle_one_hot[:, 0].mean(),
            "active_orange": self.active_bottle_one_hot[:, 1].mean(),
            "active_beer": self.active_bottle_one_hot[:, 2].mean(),
        }
        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # robot reset
        joint_pos = self.home_joint_pos.unsqueeze(0).repeat(len(env_ids), 1)
        arm_dof_count = min(6, self._robot.num_joints)
        if arm_dof_count > 0:
            reset_noise = float(self.cfg.arm_reset_noise_rad)
            joint_pos[:, :arm_dof_count] += sample_uniform(-reset_noise, reset_noise, (len(env_ids), arm_dof_count), self.device)
        joint_pos = torch.clamp(joint_pos, self.policy_dof_lower_limits, self.policy_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # bottle reset (only one active)
        spawn_x = sample_uniform(self.cfg.object_spawn_x_range[0], self.cfg.object_spawn_x_range[1], (len(env_ids),), self.device)
        spawn_y = sample_uniform(self.cfg.object_spawn_y_range[0], self.cfg.object_spawn_y_range[1], (len(env_ids),), self.device)
        spawn_yaw = sample_uniform(self.cfg.object_spawn_yaw_range[0], self.cfg.object_spawn_yaw_range[1], (len(env_ids),), self.device)

        active_idx = torch.randint(0, len(self._bottle_objects), (len(env_ids),), device=self.device)
        self.active_bottle_idx[env_ids] = active_idx
        self.active_bottle_one_hot[env_ids] = 0.0
        self.active_bottle_one_hot[env_ids, active_idx] = 1.0

        for bottle_idx, bottle_obj in enumerate(self._bottle_objects):
            object_state = bottle_obj.data.default_root_state[env_ids].clone()
            is_active = active_idx == bottle_idx

            local_x = torch.where(is_active, spawn_x, torch.full_like(spawn_x, float(self.cfg.parked_object_x)))
            local_y = torch.where(
                is_active,
                spawn_y,
                torch.full_like(spawn_y, float(self.cfg.parked_object_y[bottle_idx])),
            )
            local_z = torch.where(
                is_active,
                torch.full_like(spawn_x, float(self.cfg.object_spawn_z)),
                torch.full_like(spawn_x, float(self.cfg.parked_object_z)),
            )
            yaw = torch.where(is_active, spawn_yaw, torch.zeros_like(spawn_yaw))

            object_state[:, 0] = self.scene.env_origins[env_ids, 0] + local_x
            object_state[:, 1] = self.scene.env_origins[env_ids, 1] + local_y
            object_state[:, 2] = self.scene.env_origins[env_ids, 2] + local_z
            object_state[:, 3] = torch.cos(0.5 * yaw)
            object_state[:, 4] = 0.0
            object_state[:, 5] = 0.0
            object_state[:, 6] = torch.sin(0.5 * yaw)
            object_state[:, 7:] = 0.0

            bottle_obj.write_root_pose_to_sim(object_state[:, :7], env_ids=env_ids)
            bottle_obj.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)

        self.object_init_height[env_ids] = float(self.cfg.object_spawn_z)
        self.object_init_xy[env_ids, 0] = spawn_x
        self.object_init_xy[env_ids, 1] = spawn_y
        self.arm_collision_peak[env_ids] = 0.0
        self.arm_collision_over[env_ids] = 0.0
        self.joint_3_abs_rad[env_ids] = 0.0
        self.joint_5_abs_rad[env_ids] = 0.0
        self.singularity_penalty[env_ids] = 0.0
        self.tcp_height[env_ids] = 0.0
        self.object_xy_speed[env_ids] = 0.0
        self.object_upward_speed[env_ids] = 0.0
        self.post_grasp_arm_scale[env_ids] = 1.0
        self.tcp_x_w[env_ids] = 0.0
        self.object_up_w[env_ids] = 0.0
        self.object_grasp_offset_z[env_ids] = sample_uniform(
            float(self.cfg.object_grasp_offset_z_range[0]),
            float(self.cfg.object_grasp_offset_z_range[1]),
            (len(env_ids),),
            self.device,
        )

        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        denom = torch.clamp(self.policy_dof_upper_limits - self.policy_dof_lower_limits, min=1.0e-5)
        dof_pos_scaled = 2.0 * (self._robot.data.joint_pos - self.policy_dof_lower_limits) / denom - 1.0
        dof_vel_scaled = self._robot.data.joint_vel * float(self.cfg.dof_velocity_scale)

        joint_pos_obs = self._pad_or_truncate(dof_pos_scaled[:, self.obs_joint_ids], int(self.cfg.obs_joint_dim))
        joint_vel_obs = self._pad_or_truncate(dof_vel_scaled[:, self.obs_joint_ids], int(self.cfg.obs_joint_dim))

        to_object = self.object_grasp_pos_w - self.robot_grasp_pos
        gripper_state = self._compute_gripper_state().unsqueeze(-1)

        obs = torch.cat(
            (
                joint_pos_obs,
                joint_vel_obs,
                gripper_state,
                to_object,
                self.lift_amount.unsqueeze(-1),
                self.object_up_z.unsqueeze(-1),
                self.active_bottle_one_hot,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}
