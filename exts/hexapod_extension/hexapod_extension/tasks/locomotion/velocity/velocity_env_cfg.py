from __future__ import annotations

import math
from dataclasses import MISSING

import hexapod_extension.tasks.locomotion.velocity.mdp as mdp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from .rough import ROUGH_TERRAINS_CFG, EXTRA_ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=EXTRA_ROUGH_TERRAINS_CFG,#ROUGH_TERRAINS_CFG,#
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", #This parameter defines the friction value to use when objects collide
            restitution_combine_mode="multiply", #This parameter defines the restitution value to use when objects collide
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/MP_BODY",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True, # We only set yaw to true because we only care about height information
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), # For a uniform grid pattern, we specify the pattern using GridPatternCfg
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.75, 0.75]), # For a uniform grid pattern, we specify the pattern using GridPatternCfg
        debug_vis=False, # If True, lets you visualize where the rays hit the mesh
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), #Time before commands are changed [s]
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True, # If True, the angular velocity command is computed from the heading error, 
                              # where the target heading is sampled uniformly from provided range.
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            # lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0, 0), heading=(-math.pi, math.pi)
            # lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0, 0), heading=(-0, 0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True # Adds the specified noises if True
            self.concatenate_terms = True
            # self.history_length = 5 #Number of past observation to store in the observation buffers 
            #                         #for all observation terms in group.
            # self.flatten_history_dim = True #Flag to flatten history-based observation terms to a 2D 
            #                                 #(num_env, D) tensor for all observation terms in group.

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (contains privileged information for the critic network).
        
        Here me repeat the observation terms in policy, but without noise, and add the height scan 
        (absent in the blind robot).
        
        """

        # observation terms (order preserved)
        # base linear velocity (needed here in this order for HIM Locomotion)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # other propioceptive meassurements without error (needed here in this order for HIM Locomotion)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        
        # Privileged Information
        height_scan = ObsTerm( # Height scan from the given sensor w.r.t. the sensor's frame.
                               # The provided offset (Defaults to 0.5) is subtracted from the returned values.
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            # noise=Unoise(n_min=-0.05, n_max=0.05),
            # # noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )

        # TODO: expand critic observations.
        # forces = ObsTerm(func=mdp.net_forces,
        #                         params={"sensor_cfg": SceneEntityCfg("contact_forces",
        #                                                             body_names=["MP_BODY", "c1_.*", "thigh_.*", "tibia_.*"]),},
        #                 )
        # foot_forces = ObsTerm(func=mdp.net_forces,
        #                     params={"sensor_cfg": SceneEntityCfg("contact_forces",
        #                                                         body_names=["tibia_.*"])},
        #                     )
        # base_forces = ObsTerm(func=mdp.net_forces,
        #                         params={"sensor_cfg": SceneEntityCfg("contact_forces",
        #                                                             body_names=["MP_BODY"]),},
        #                         )
        # thigh_contact_states = ObsTerm(func=mdp.contact_states,
        #                         params={"threshold": 1.0,
        #                                 "sensor_cfg": SceneEntityCfg("contact_forces",
        #                                                             body_names=["thigh_.*"]),},
        #                         )
        # tibia_contact_states = ObsTerm(func=mdp.contact_states,
        #                         params={"threshold": 1.0,
        #                                 "sensor_cfg": SceneEntityCfg("contact_forces",
        #                                                             body_names=["tibia_.*"]),},
        #                         )

        def __post_init__(self):
            self.enable_corruption = False # Adds the specified noises if True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg() # The name has to be critic for rsl_rl to correctly identify it


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),#(0.8*0.9, 0.8*1.1),
            "dynamic_friction_range": (0.6, 0.6),#(0.6*0.9, 0.6*1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="MP_BODY"),#"base"),
            "mass_distribution_params": (0.8, 1.2),#$(-5.0, 5.0),
            "operation": "scale",#"add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="MP_BODY"),#"base"),
            "force_range": (0.0, 0.0),#(-1.0, 1.0),#
            "torque_range": (-0.0, 0.0), #TODO: agregar?
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset, #TODO: since default positions are 0, change this to reset_joints_by_offset
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),#(-0.5, 0.5),#(0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0), # The range of time in seconds at which the term is applied (sampled uniformly 
                                       # between the specified range for each environment instance)
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        # TODO: ver si se me fue la mano con estos valores (lo dejo entrenando vie a la noche)
        # params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)#-1.0e-2)#-1.0e-3)#-1.0e-4)#
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0) # Penalizes the xy-components of the projected gravity vector using L2 norm
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0) # Penalizes joint positions if they cross the soft limits.
    # tripod_gate = RewTerm(
    #     func=mdp.tripod_gate,
    #     weight=0.01,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", 
    #                                      body_names=["foottip_rf",
    #                                                  "foottip_rm",
    #                                                  "foottip_rr",
    #                                                  "foottip_lf",
    #                                                  "foottip_lm",
    #                                                  "foottip_lr"]),
    #         "command_name": "base_velocity",
    #         "threshold": 1.0,#0.5,
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact, #Terminate when the contact force on the sensor exceeds the force threshold.
    #     # params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="MP_BODY"), "threshold": 1.0},
    #     # TODO: verify if the threshold for the contact force is appropriate
    # )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation, #Terminate when the asset's orientation is too far from the desired orientation limits.
        params={"limit_angle": math.pi/2, "asset_cfg": SceneEntityCfg("robot")},
    )
    # joint_effort_out_of_limit = DoneTerm(
    #     func=mdp.joint_effort_out_of_limit, #Terminate when effort applied on the asset's joints are outside of the soft joint limits.
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    # TODO: add this?


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4 # Apply control action every decimation number of simulation steps
        self.episode_length_s = 20.0 # Duration of an episode (in seconds)
        # simulation settings
        self.sim.dt = 0.005 # Physics simulation time-step (in seconds)
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True # This flag allows disabling the contact processing 
                                                   # and querying the contacts manually by the user over
                                                   # a limited set of primitives in the scene.
        self.sim.physics_material = self.scene.terrain.physics_material # Defaults to this physics material for 
                                                                        # all the rigid body prims that do not 
                                                                        # have any physics material specified on them.
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # if self.scene.contact_forces_base is not None:
        #     self.scene.contact_forces_base.update_period = self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
