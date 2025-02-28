from hexapod_extension.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import hexapod_extension.tasks.locomotion.velocity.mdp as mdp
import torch

from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from .phantomx import PHANTOM_X
from hexapod_extension.tasks.locomotion.velocity.rough import TEST_TERRAINS_CFG, TEST_ONE_TERRAINS_CFG 

@configclass
class PhantomXRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to phantom-x
        self.scene.robot = PHANTOM_X.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class PhantomXRoughNoCurriculumEnvCfg(PhantomXRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.curriculum = {}
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

@configclass
class PhantomXHeightScanRoughEnvCfg(PhantomXRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Add height scan observation term
        self.observations.policy.height_scan = ObsTerm( # Height scan from the given sensor w.r.t. the sensor's frame.
                                                            # The provided offset (Defaults to 0.5) is subtracted from the returned values.
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )

@configclass
class PhantomXHIMLocomotionRoughEnvCfg(PhantomXRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Remove the attribute 'base_lin_vel' from self.observations.policy
        if hasattr(self.observations.policy, 'base_lin_vel'):
            del self.observations.policy.base_lin_vel
        self.observations.policy.history_length = 5 #Number of past observation to store in the observation buffers 
                                                    #for all observation terms in group.
        self.observations.policy.flatten_history_dim = True #Flag to flatten history-based observation terms to a 2D 
                                                            #(num_env, D) tensor for all observation terms in group.
        

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


# def heading_error(
#         env: ManagerBasedRLEnv, limit_heading_error: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
#         ) -> torch.Tensor:
#     '''Verify the heading error is within the accepted threshold'''
#     asset: Articulation = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command("base_velocity")[:, :2]
#     velocity = asset.data.root_lin_vel_b[:, :2]
#     # compute magnitude of the vectors
#     velocity_norm = torch.norm(velocity, dim=1)
#     command_norm = torch.norm(command, dim=1)
#     # Compute the dot product for each pair of vectors (element-wise for each row)
#     dot_product = torch.sum(command * velocity, dim=1)
#     # Compute the cosine of the angle between the vectors
#     heading_error = torch.where(
#         velocity_norm > 0.001, #velocity_threshold
#         torch.acos(torch.clamp(dot_product / (command_norm * velocity_norm), -1.0, 1.0)),
#         0
#     )
#     # Terminate episode if episode length is bigger than 10 and the heading error is too big
#     terminate = (abs(heading_error) > limit_heading_error) & (env.episode_length_buf > 10)
#     if terminate.any():
#         print(velocity[terminate,:])
#         print(command[terminate,:])
#     return  terminate
    
def add_TEST_parameters(cfg_obj):
    # Override command ranges and porcentage of standing envs
    cfg_obj.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(1.0, 1.0), lin_vel_y=(-0.1, 0.1), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        )
    cfg_obj.commands.base_velocity.rel_standing_envs=0.0
    # Override resampling_time_range so that the command is not changed during the episode
    cfg_obj.commands.base_velocity.resampling_time_range=(cfg_obj.episode_length_s, cfg_obj.episode_length_s)

    # self.terminations.heading_error = DoneTerm(
    #     func=heading_error, #Terminate when the heading error is over a threshold
    #     params={"limit_heading_error": 0.05, "asset_cfg": SceneEntityCfg("robot")},
    # )

    # Override the terrain_generator to use TEST_TERRAINS_CFG
    cfg_obj.scene.terrain = TerrainImporterCfg(
        prim_path=cfg_obj.scene.terrain.prim_path,
        terrain_type=cfg_obj.scene.terrain.terrain_type,
        terrain_generator=TEST_TERRAINS_CFG,
        max_init_terrain_level=None, # If this is None it takes the maximum possible level
        collision_group=cfg_obj.scene.terrain.collision_group,
        physics_material=cfg_obj.scene.terrain.physics_material,
        visual_material=cfg_obj.scene.terrain.visual_material,
        debug_vis=cfg_obj.scene.terrain.debug_vis,
    )
    
    # Remove the attribute 'terrain_levels' from self.curriculum
    if hasattr(cfg_obj.curriculum, 'terrain_levels'):
        del cfg_obj.curriculum.terrain_levels
    cfg_obj.curriculum.success_analysis = CurrTerm(func=mdp.success_rate)
    # # remove random pushing
    # self.randomization.base_external_force_torque = None
    # self.randomization.push_robot = None

    #IMPORTANTE: esto lo tuve que modificar porque me tiraba error, porque randomization era None. No sé por qué saltó este error
    # remove random pushing
    # if (self.randomization is not None):
    #     self.randomization.base_external_force_torque = None
    #     self.randomization.push_robot = None
    # TODO: ver cómo queda esto

@configclass
class PhantomXRoughEnvCfg_TEST(PhantomXRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        add_TEST_parameters(self)

@configclass
class PhantomXHIMLocomotionRoughEnvCfg_TEST(PhantomXRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        add_TEST_parameters(self)

        # Remove the attribute 'base_lin_vel' from self.observations.policy
        if hasattr(self.observations.policy, 'base_lin_vel'):
            del self.observations.policy.base_lin_vel
        self.observations.policy.history_length = 5 #Number of past observation to store in the observation buffers 
                                                    #for all observation terms in group.
        self.observations.policy.flatten_history_dim = True #Flag to flatten history-based observation terms to a 2D 
                                                            #(num_env, D) tensor for all observation terms in group.

@configclass
class PhantomXHeightScanRoughEnvCfg_TEST(PhantomXRoughEnvCfg_TEST):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Add height scan observation term
        self.observations.policy.height_scan = ObsTerm( # Height scan from the given sensor w.r.t. the sensor's frame.
                                                            # The provided offset (Defaults to 0.5) is subtracted from the returned values.
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )


    
@configclass
class PhantomXRoughEnvCfg_TEST_ONE(PhantomXRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Override command ranges and porcentage of standing envs
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5), lin_vel_y=(-0.05, 0.05), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
         )
        self.commands.base_velocity.rel_standing_envs=0.0
        # Override resampling_time_range so that the command is not changed during the episode
        self.commands.base_velocity.resampling_time_range=(self.episode_length_s, self.episode_length_s)

        # Override Base Reset so that it always faces and walks forward from the center
        self.events.reset_base.params['pose_range']['x'] = (0.0, 0.0)
        self.events.reset_base.params['pose_range']['y'] = (0.0, 0.0)
        self.events.reset_base.params['pose_range']['yaw'] = (0.0, 0.0)

        # Override the terrain_generator to use STAIR_TERRAINS_CFG
        self.scene.terrain = TerrainImporterCfg(
            prim_path=self.scene.terrain.prim_path,
            terrain_type=self.scene.terrain.terrain_type,
            terrain_generator=TEST_ONE_TERRAINS_CFG,
            max_init_terrain_level=None, # If this is None it takes the maximum possible level
            collision_group=self.scene.terrain.collision_group,
            physics_material=self.scene.terrain.physics_material,
            visual_material=self.scene.terrain.visual_material,
            debug_vis=self.scene.terrain.debug_vis,
        )
        
        # Remove the attribute 'terrain_levels' from self.curriculum
        if hasattr(self.curriculum, 'terrain_levels'):
            del self.curriculum.terrain_levels
        self.curriculum.success_analysis = CurrTerm(func=mdp.success_rate)
        # # remove random pushing
        # self.randomization.base_external_force_torque = None
        # self.randomization.push_robot = None

        #IMPORTANTE: esto lo tuve que modificar porque me tiraba error, porque randomization era None. No sé por qué saltó este error
        # remove random pushing
        if (self.randomization is not None):
            self.randomization.base_external_force_torque = None
            self.randomization.push_robot = None
        # TODO: ver cómo queda esto


@configclass
class PhantomXHeightScanRoughEnvCfg_TEST_ONE(PhantomXRoughEnvCfg_TEST_ONE):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Add height scan observation term
        self.observations.policy.height_scan = ObsTerm( # Height scan from the given sensor w.r.t. the sensor's frame.
                                                            # The provided offset (Defaults to 0.5) is subtracted from the returned values.
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )