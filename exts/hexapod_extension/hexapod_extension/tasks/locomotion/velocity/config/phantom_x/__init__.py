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
