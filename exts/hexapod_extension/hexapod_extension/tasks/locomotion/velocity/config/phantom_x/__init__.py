#TODO: add SB3 to all envs

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Phantom-X-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PhantomXFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Phantom-X-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PhantomXFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXRoughPPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-No-Curriculum-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughNoCurriculumEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXRoughPPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Height-Scan-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHeightScanRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHeightScanRoughPPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Test-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughEnvCfg_TEST,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Height-Scan-Test-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHeightScanRoughEnvCfg_TEST,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHeightScanRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Test-One-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughEnvCfg_TEST_ONE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Height-Scan-Test-One-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHeightScanRoughEnvCfg_TEST_ONE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHeightScanRoughPPORunnerCfg,
    },
)
