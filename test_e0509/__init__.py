# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
E0509 Pick-Place-K environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-E0509-Pick-Place-K-Direct-v0",
    entry_point=f"{__name__}.e0509_pick_place_k_env:E0509PickPlaceKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.e0509_pick_place_k_env:E0509PickPlaceKEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_e0509_pick_place_k_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:E0509PickPlaceKPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_e0509_pick_place_k_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-E0509-Grip-Bottle-Direct-v0",
    entry_point=f"{__name__}.grip_bottle_env:GripBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grip_bottle_env:GripBottleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_grip_bottle_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GripBottlePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_grip_bottle_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-E0509-Move-Bottle-Direct-v0",
    entry_point=f"{__name__}.move_bottle_env:MoveBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.move_bottle_env:MoveBottleStage2EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_move_bottle_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveBottleStage2PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_move_bottle_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-E0509-Move-Bottle-Stage1-Direct-v0",
    entry_point=f"{__name__}.move_bottle_env:MoveBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.move_bottle_env:MoveBottleStage1EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_move_bottle_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveBottleStage1PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_move_bottle_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-E0509-Move-Bottle-Stage2-Direct-v0",
    entry_point=f"{__name__}.move_bottle_env:MoveBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.move_bottle_env:MoveBottleStage2EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_move_bottle_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveBottleStage2PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_move_bottle_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-E0509-Move-Bottle-Stage3-Direct-v0",
    entry_point=f"{__name__}.move_bottle_env:MoveBottleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.move_bottle_env:MoveBottleStage3EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_move_bottle_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveBottleStage3PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_move_bottle_ppo_cfg.yaml",
    },
)
