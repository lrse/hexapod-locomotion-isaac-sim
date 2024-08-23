from hexapod_extension.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from .phantomx import PHANTOM_X

@configclass
class PhantomXRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to phantom-x
        self.scene.robot = PHANTOM_X.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class PhantomXRoughEnvCfg_PLAY(PhantomXRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # # remove random pushing
        # self.randomization.base_external_force_torque = None
        # self.randomization.push_robot = None

        #IMPORTANTE: esto lo tuve que modificar porque me tiraba error, porque randomization era None. No sé por qué saltó este error
        # remove random pushing
        if (self.randomization is not None):
            self.randomization.base_external_force_torque = None
            self.randomization.push_robot = None
