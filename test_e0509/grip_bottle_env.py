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
    action_space = 6
    observation_space = 20
    state_space = 0
    obs_joint_dim = 6

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            # Prevent PxGpuDynamicsMemoryConfig::collisionStackSize overflow
            # in high-parallel, contact-rich training runs.
            gpu_collision_stack_size=2**27,
        ),
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
    table_top_center_xy = (0.0, 0.670)
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
    # Bottle radii (meters) for (soju, orange, beer). Update with measured real values if available.
    bottle_radius_m = (0.033, 0.032, 0.032)
    ee_center_clearance_m = 0.065
    ee_center_dist_tolerance_m = 0.010
    ee_center_lateral_tolerance_m = 0.015
    success_min_y_parallel = 0.95
    force_open_gripper = True
    upright_success_min_up_z = 0.985
    topple_reset_max_up_z = 0.98
    drop_below_table_margin = 0.03
    contact_force_threshold = 0.4
    terminate_contact_force = 1.2
    dist_reward_scale = 1.8
    dist_progress_reward_scale = 4.0
    dist_error_penalty_scale = 3.0
    dist_error_reward_decay = 8.0
    lateral_error_reward_decay = 8.0
    lateral_reward_scale = 1.8
    y_parallel_reward_scale = 0.2
    alignment_focus_dist_decay = 12.0
    lateral_error_penalty_scale = 2.0
    lateral_progress_reward_scale = 2.0
    tilt_penalty_scale = 3.0
    pre_grasp_arm_slow_start_dist = 0.40
    pre_grasp_arm_slow_min_scale = 0.45
    success_hold_steps = 2
    log_extended_metrics = True
    collision_penalty_scale = 4.0
    action_penalty_scale = 0.005
    singularity_joint_3_threshold_deg = 8.0
    singularity_joint_5_threshold_deg = 8.0
    singularity_penalty_scale = 0.0
    stall_timeout_steps = 320
    stall_dist_error_m = 0.03
    stall_lateral_error_m = 0.025
    success_bonus = 500.0


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

        self.active_bottle_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.active_bottle_one_hot = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.bottle_radius_by_type = torch.tensor(self.cfg.bottle_radius_m, dtype=torch.float, device=self.device)
        self.active_bottle_radius = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

        self.ee_body_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ee_body_speed = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_height = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_up_z = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ee_center_to_object = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ee_center_signed_axial_dist = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ee_center_axial_error = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ee_center_lateral_error = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ee_center_target_dist = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ee_center_dist_error = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.prev_ee_center_dist_error = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.prev_ee_center_lateral_error = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.object_xy_speed = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.pre_grasp_arm_scale = torch.ones((self.num_envs,), dtype=torch.float, device=self.device)
        self.success_hold_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.success_active = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.success_last_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.tcp_x_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.tcp_y_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.tcp_z_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_up_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.tcp_offset_axis_local = torch.tensor(self.cfg.tcp_axis_local, dtype=torch.float, device=self.device)
        self.tcp_x_axis_local = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float, device=self.device)
        self.tcp_y_axis_local = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float, device=self.device)
        self.object_up_axis_local = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float, device=self.device)
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

    def _compute_success_now(self) -> torch.Tensor:
        # Keep alignment reference fixed to world +Z to avoid reward hacking by tilting the bottle.
        y_parallel = torch.abs(self.tcp_y_w[:, 2])
        dist_error_mask = self.ee_center_dist_error < float(self.cfg.ee_center_dist_tolerance_m)
        lateral_error_mask = self.ee_center_lateral_error < float(self.cfg.ee_center_lateral_tolerance_m)
        y_parallel_mask = y_parallel > float(self.cfg.success_min_y_parallel)
        upright_mask = self.object_up_z > float(self.cfg.upright_success_min_up_z)
        # Strict success condition at current step.
        return dist_error_mask & lateral_error_mask & y_parallel_mask & upright_mask

    def _update_success_state(self):
        success_now = self._compute_success_now()
        current_step = self.episode_length_buf.to(torch.long)
        step_changed = current_step != self.success_last_step
        success_hold_steps = int(self.cfg.success_hold_steps)

        next_count = torch.where(success_now, self.success_hold_count + 1, torch.zeros_like(self.success_hold_count))
        self.success_hold_count[:] = torch.where(step_changed, next_count, self.success_hold_count)
        self.success_last_step[:] = torch.where(step_changed, current_step, self.success_last_step)
        self.success_active[:] = self.success_hold_count >= success_hold_steps

    def _compute_success(self) -> torch.Tensor:
        self._update_success_state()
        return self.success_active

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
        self.ee_body_pos[env_ids] = ee_pos_w
        ee_vel_w = self._robot.data.body_lin_vel_w[env_ids, self.ee_body_idx]
        self.ee_body_speed[env_ids] = torch.norm(ee_vel_w, p=2, dim=-1)

        self.robot_grasp_pos[env_ids] = ee_pos_w + quat_apply(ee_quat_w, tcp_local)
        self.tcp_x_w[env_ids] = quat_apply(ee_quat_w, self.tcp_x_axis_local.unsqueeze(0).expand(len(env_ids), -1))
        self.tcp_y_w[env_ids] = quat_apply(ee_quat_w, self.tcp_y_axis_local.unsqueeze(0).expand(len(env_ids), -1))
        self.tcp_z_w[env_ids] = quat_apply(ee_quat_w, self.tcp_offset_axis_local.unsqueeze(0).expand(len(env_ids), -1))

        self.object_pos_w[env_ids] = self._gather_active_bottle_tensor(env_ids, "root_pos_w")
        object_quat_w = self._gather_active_bottle_tensor(env_ids, "root_quat_w")
        object_up_w = quat_apply(object_quat_w, self.object_up_axis_local.unsqueeze(0).expand(len(env_ids), -1))
        self.object_up_w[env_ids] = object_up_w
        self.object_up_z[env_ids] = torch.clamp(object_up_w[:, 2], -1.0, 1.0)

        object_height = self.object_pos_w[env_ids, 2] - self.scene.env_origins[env_ids, 2]
        self.object_height[env_ids] = object_height

        object_lin_vel_w = self._gather_active_bottle_tensor(env_ids, "root_lin_vel_w")
        self.object_xy_speed[env_ids] = torch.norm(object_lin_vel_w[:, :2], p=2, dim=-1)
        ee_to_object_vec = self.object_pos_w[env_ids] - ee_pos_w
        ee_to_object_axial = torch.sum(ee_to_object_vec * self.tcp_z_w[env_ids], dim=-1)
        ee_to_object_lateral_vec = ee_to_object_vec - ee_to_object_axial.unsqueeze(-1) * self.tcp_z_w[env_ids]
        ee_to_object_lateral = torch.norm(ee_to_object_lateral_vec, p=2, dim=-1)
        self.ee_center_to_object[env_ids] = torch.norm(ee_to_object_vec, p=2, dim=-1)
        self.ee_center_signed_axial_dist[env_ids] = ee_to_object_axial
        self.ee_center_lateral_error[env_ids] = ee_to_object_lateral
        self.ee_center_target_dist[env_ids] = self.active_bottle_radius[env_ids] + float(self.cfg.ee_center_clearance_m)
        axial_error = torch.abs(ee_to_object_axial - self.ee_center_target_dist[env_ids])
        self.ee_center_axial_error[env_ids] = axial_error
        # Geometric error to the target point on tool-z line: combines axial + lateral mismatch.
        self.ee_center_dist_error[env_ids] = torch.sqrt(axial_error * axial_error + ee_to_object_lateral * ee_to_object_lateral)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets.clone()

        arm_dim = min(6, self.actions.shape[1], self._robot.num_joints)
        if arm_dim > 0:
            pre_grasp_start_err = max(float(self.cfg.pre_grasp_arm_slow_start_dist), 1.0e-6)
            pre_grasp_min_scale = float(self.cfg.pre_grasp_arm_slow_min_scale)
            pre_grasp_error = torch.sqrt(self.ee_center_dist_error**2 + self.ee_center_lateral_error**2)
            pre_grasp_err_ratio = torch.clamp(pre_grasp_error / pre_grasp_start_err, min=0.0, max=1.0)
            pre_grasp_arm_scale = pre_grasp_min_scale + (1.0 - pre_grasp_min_scale) * pre_grasp_err_ratio
            arm_action_scale = pre_grasp_arm_scale
            self.pre_grasp_arm_scale[:] = pre_grasp_arm_scale
            targets[:, :arm_dim] = (
                self.robot_dof_targets[:, :arm_dim]
                + self.robot_dof_speed_scales[:arm_dim]
                * self.dt
                * self.actions[:, :arm_dim]
                * self.cfg.action_scale
                * arm_action_scale.unsqueeze(-1)
            )

        if self.gripper_joint_ids.numel() > 0:
            if bool(self.cfg.force_open_gripper):
                targets[:, self.gripper_joint_ids] = self.gripper_open_targets.unsqueeze(0)
            elif self.actions.shape[1] > 6:
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
        success = self._compute_success()
        dropped = self.object_height < (float(self.cfg.table_top_z) - float(self.cfg.drop_below_table_margin))
        toppled = self.object_up_z < float(self.cfg.topple_reset_max_up_z)
        collided = self.arm_collision_peak > float(self.cfg.terminate_contact_force)
        stalled = (self.episode_length_buf >= int(self.cfg.stall_timeout_steps)) & (
            (self.ee_center_dist_error > float(self.cfg.stall_dist_error_m))
            | (self.ee_center_lateral_error > float(self.cfg.stall_lateral_error_m))
        )
        terminated = success | dropped | toppled | collided | stalled
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        self._update_collision_metrics()
        self._update_singularity_metrics()

        dist_progress = torch.clamp(self.prev_ee_center_dist_error - self.ee_center_dist_error, min=-0.05, max=0.05)
        lateral_progress = torch.clamp(
            self.prev_ee_center_lateral_error - self.ee_center_lateral_error, min=-0.05, max=0.05
        )
        dist_reward = torch.exp(-float(self.cfg.dist_error_reward_decay) * self.ee_center_dist_error)
        lateral_reward = torch.exp(-float(self.cfg.lateral_error_reward_decay) * self.ee_center_lateral_error)
        dist_focus = torch.exp(-float(self.cfg.alignment_focus_dist_decay) * self.ee_center_dist_error)
        dist_gate = self.ee_center_dist_error < float(self.cfg.ee_center_dist_tolerance_m)
        lateral_gate = self.ee_center_lateral_error < float(self.cfg.ee_center_lateral_tolerance_m)
        near_target_mask = dist_gate & lateral_gate
        # Encourage side-approach posture: TCP y-axis parallel to world +Z.
        y_parallel = torch.abs(self.tcp_y_w[:, 2])
        y_gate = y_parallel > float(self.cfg.success_min_y_parallel)
        upright_gate = self.object_up_z > float(self.cfg.upright_success_min_up_z)
        success_now = dist_gate & lateral_gate & y_gate & upright_gate
        y_align_soft = torch.clamp(
            (y_parallel - float(self.cfg.success_min_y_parallel))
            / max(1.0 - float(self.cfg.success_min_y_parallel), 1.0e-6),
            min=0.0,
            max=1.0,
        )
        tilt_penalty = torch.clamp(float(self.cfg.upright_success_min_up_z) - self.object_up_z, min=0.0)
        tilt_deg = torch.rad2deg(torch.acos(torch.clamp(self.object_up_z, -1.0, 1.0)))
        action_penalty = torch.sum(self.actions**2, dim=-1)

        rewards = (
            self.cfg.dist_reward_scale * dist_reward
            + float(self.cfg.lateral_reward_scale) * (lateral_reward * dist_focus)
            + self.cfg.dist_progress_reward_scale * dist_progress
            + float(self.cfg.lateral_progress_reward_scale) * lateral_progress
            + float(self.cfg.y_parallel_reward_scale) * (y_align_soft * dist_focus)
            - self.cfg.dist_error_penalty_scale * self.ee_center_dist_error
            - float(self.cfg.lateral_error_penalty_scale) * self.ee_center_lateral_error
            - self.cfg.collision_penalty_scale * self.arm_collision_over
            - self.cfg.singularity_penalty_scale * self.singularity_penalty
            - self.cfg.tilt_penalty_scale * tilt_penalty
            - self.cfg.action_penalty_scale * action_penalty
        )

        success = self._compute_success()
        rewards = torch.where(success, rewards + float(self.cfg.success_bonus), rewards)
        mean_success_dist = torch.where(
            success.any(),
            self.ee_center_to_object[success].mean(),
            torch.zeros((), dtype=self.ee_center_to_object.dtype, device=self.device),
        )

        # Keep terminal log concise by default.
        log_data = {
            "success_rate": success.float().mean(),
            "target_zone_rate": near_target_mask.float().mean(),
            "collision_over_rate": (self.arm_collision_over > 0.0).float().mean(),
            "toppled_rate": (self.object_up_z < float(self.cfg.topple_reset_max_up_z)).float().mean(),
            "mean_dist": self.ee_center_to_object.mean(),
            "mean_target_dist": self.ee_center_target_dist.mean(),
            "mean_dist_error": self.ee_center_dist_error.mean(),
            "mean_axial_error": self.ee_center_axial_error.mean(),
            "mean_lateral_error": self.ee_center_lateral_error.mean(),
            "mean_y_parallel": y_parallel.mean(),
            "mean_ee_speed": self.ee_body_speed.mean(),
            "dist_gate_rate": dist_gate.float().mean(),
            "lateral_gate_rate": lateral_gate.float().mean(),
            "y_gate_rate": y_gate.float().mean(),
            "dist_y_gate_rate": (dist_gate & lateral_gate & y_gate).float().mean(),
            "upright_gate_rate": upright_gate.float().mean(),
            "success_now_rate": success_now.float().mean(),
            "mean_arm_collision_over": self.arm_collision_over.mean(),
        }
        if bool(self.cfg.log_extended_metrics):
            log_data.update(
                {
                    # detailed episode outcome
                    "dropped_rate": (
                        self.object_height < (float(self.cfg.table_top_z) - float(self.cfg.drop_below_table_margin))
                    ).float().mean(),
                    # reach and progress
                    "mean_signed_axial_dist": self.ee_center_signed_axial_dist.mean(),
                    "mean_axial_error": self.ee_center_axial_error.mean(),
                    "mean_success_dist": mean_success_dist,
                    "mean_dist_progress": dist_progress.mean(),
                    "mean_lateral_progress": lateral_progress.mean(),
                    "mean_dist_focus": dist_focus.mean(),
                    # object posture
                    "mean_up_z": self.object_up_z.mean(),
                    "min_up_z": self.object_up_z.min(),
                    "mean_tilt_deg": tilt_deg.mean(),
                    # collision force magnitude
                    "mean_arm_collision_peak": self.arm_collision_peak.mean(),
                    # secondary penalties and motion
                    "mean_tilt_penalty": tilt_penalty.mean(),
                    "mean_object_xy_speed": self.object_xy_speed.mean(),
                    # arm control scaling
                    "mean_pre_grasp_arm_scale": self.pre_grasp_arm_scale.mean(),
                    # singularity diagnostics
                    "mean_singularity_penalty": self.singularity_penalty.mean(),
                    "mean_abs_q3_deg": torch.rad2deg(self.joint_3_abs_rad).mean(),
                    "mean_abs_q5_deg": torch.rad2deg(self.joint_5_abs_rad).mean(),
                    # active bottle mix
                    "active_soju": self.active_bottle_one_hot[:, 0].mean(),
                    "active_orange": self.active_bottle_one_hot[:, 1].mean(),
                    "active_beer": self.active_bottle_one_hot[:, 2].mean(),
                }
            )
        self.extras["log"] = log_data
        self.prev_ee_center_dist_error[:] = self.ee_center_dist_error
        self.prev_ee_center_lateral_error[:] = self.ee_center_lateral_error
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
        self.active_bottle_radius[env_ids] = self.bottle_radius_by_type[active_idx]

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

        self.arm_collision_peak[env_ids] = 0.0
        self.arm_collision_over[env_ids] = 0.0
        self.joint_3_abs_rad[env_ids] = 0.0
        self.joint_5_abs_rad[env_ids] = 0.0
        self.singularity_penalty[env_ids] = 0.0
        self.object_xy_speed[env_ids] = 0.0
        self.pre_grasp_arm_scale[env_ids] = 1.0
        self.success_hold_count[env_ids] = 0
        self.success_active[env_ids] = False
        self.success_last_step[env_ids] = -1
        self.tcp_x_w[env_ids] = 0.0
        self.tcp_y_w[env_ids] = 0.0
        self.tcp_z_w[env_ids] = 0.0
        self.object_up_w[env_ids] = 0.0

        self._compute_intermediate_values(env_ids)
        self.prev_ee_center_dist_error[env_ids] = self.ee_center_dist_error[env_ids]
        self.prev_ee_center_lateral_error[env_ids] = self.ee_center_lateral_error[env_ids]

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        denom = torch.clamp(self.policy_dof_upper_limits - self.policy_dof_lower_limits, min=1.0e-5)
        dof_pos_scaled = 2.0 * (self._robot.data.joint_pos - self.policy_dof_lower_limits) / denom - 1.0
        dof_vel_scaled = self._robot.data.joint_vel * float(self.cfg.dof_velocity_scale)

        joint_pos_obs = self._pad_or_truncate(dof_pos_scaled[:, self.obs_joint_ids], int(self.cfg.obs_joint_dim))
        joint_vel_obs = self._pad_or_truncate(dof_vel_scaled[:, self.obs_joint_ids], int(self.cfg.obs_joint_dim))

        to_object = self.object_pos_w - self.ee_body_pos
        gripper_state = self._compute_gripper_state().unsqueeze(-1)

        obs = torch.cat(
            (
                joint_pos_obs,
                joint_vel_obs,
                gripper_state,
                to_object,
                self.object_up_z.unsqueeze(-1),
                self.active_bottle_one_hot,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}
