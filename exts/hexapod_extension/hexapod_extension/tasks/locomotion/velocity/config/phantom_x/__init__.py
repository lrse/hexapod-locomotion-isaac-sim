#TODO: add SB3 to all envs

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Phantom-X-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PhantomXFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Phantom-X-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PhantomXFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXMLPRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-No-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughNoCurriculumEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Height-Scan-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHeightScanRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHeightScanRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-HIMLocomotion",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHIMLocomotionRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-DreamWaQ",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg, # The config for HIM Locomotion is the same as needed here
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXDreamWaQRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Ours",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg, # The config for HIM Locomotion is the same as needed here
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOursRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Ours2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg, # The config for HIM Locomotion is the same as needed here
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs2RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Ours3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg, # The config for HIM Locomotion is the same as needed here
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs3RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Ours4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg, # The config for HIM Locomotion is the same as needed here
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs4RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Ours5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg, # The config for HIM Locomotion is the same as needed here
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs5RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Ours6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg, # The config for HIM Locomotion is the same as needed here
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs6RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Oracle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXOracleRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOracleRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Oracle-Normalized",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXOracleRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOracleNormalizedRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Oracle2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXOracle2RoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOracle2RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Oracle3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXOracle3RoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOracle3RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Play-HIM",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHIMLocomotionRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Play-Ours4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs4RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughEnvCfg_TEST,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Height-Scan-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHeightScanRoughEnvCfg_TEST,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHeightScanRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-HIMLocomotion-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg_TEST,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHIMLocomotionRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Phantom-X-HIMLocomotion-Test-Inv-Slope",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionEnvCfg_TEST_INV_SLOPE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHIMLocomotionRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Phantom-X-HIMLocomotion-Test-Slope",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionEnvCfg_TEST_SLOPE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHIMLocomotionRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Phantom-X-HIMLocomotion-Test-Boxes",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionEnvCfg_TEST_BOXES,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHIMLocomotionRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Phantom-X-HIMLocomotion-Test-Inv-Stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionEnvCfg_TEST_INV_STAIRS,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHIMLocomotionRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Phantom-X-HIMLocomotion-Test-Stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionEnvCfg_TEST_STAIRS,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHIMLocomotionRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-DreamWaQ-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg_TEST, # The config for HIM Locomotion is the same as needed here
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXDreamWaQRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Ours-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg_TEST, # The config for HIM Locomotion is the same as needed here
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOursRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Ours4-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg_TEST, # The config for HIM Locomotion is the same as needed here
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs4RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Phantom-X-Ours4-Test-Inv-Slope",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionEnvCfg_TEST_INV_SLOPE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs4RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Phantom-X-Ours4-Test-Slope",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionEnvCfg_TEST_SLOPE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs4RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Phantom-X-Ours4-Test-Boxes",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionEnvCfg_TEST_BOXES,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs4RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Phantom-X-Ours4-Test-Inv-Stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionEnvCfg_TEST_INV_STAIRS,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs4RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Phantom-X-Ours4-Test-Stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionEnvCfg_TEST_STAIRS,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs4RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Test-One-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXRoughEnvCfg_TEST_ONE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Test-One-HIMLocomotion",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg_TEST_ONE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHIMLocomotionRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Test-One-Ours4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHIMLocomotionRoughEnvCfg_TEST_ONE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXOurs4RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Phantom-X-Height-Scan-Test-One-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PhantomXHeightScanRoughEnvCfg_TEST_ONE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PhantomXHeightScanRoughPPORunnerCfg,
    },
)
