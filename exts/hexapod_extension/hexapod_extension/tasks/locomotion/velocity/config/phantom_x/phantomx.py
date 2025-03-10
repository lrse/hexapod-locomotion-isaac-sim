# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Phantom X robot.

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration - Actuators.
##

# TODO: verificar que los motores estén bien calculados.
# Del urdf del PhantomX: todos los joints tienen la misma configuración:
# <limit effort="8.8" lower="-2.6179939" upper="2.6179939" velocity="5.6548668"/>
SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=["j_c1_.*", "j_thigh_.*", "j_tibia_.*"],
    saturation_effort= 8.8, # Este directamente lo inventé (TODO)      #120.0,
    effort_limit= 8.8, #80.0,
    velocity_limit= 5.6548668, #7.5,
    stiffness={".*": 10.0},#TODO: no sé si esto está bien, lo saqué de los mensajes de ROS cuando cargo el URDF en Gazebo      #40.0},
    damping={".*": 0.3055},#TODO: no sé si esto está bien, lo saqué de los mensajes de ROS cuando cargo el URDF en Gazebo      #5.0},
)

##
# Configuration - Articulation.
##

PHANTOM_X = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/tania/Codigos/hexapod-locomotion-isaac-sim/exts/hexapod_extension/hexapod_extension/tasks/locomotion/velocity/config/phantom_x/phantomx.usd",
        activate_contact_sensors=True,
        # TODO: ver que las propiedades estén bien definidas
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,#1000.0,
            max_angular_velocity=1000.0,#1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
        # collision_props can be used to set a margin for contact generation, meaning contact generation can potentially start before the shapes actually penetrate.
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.193),
    ),
    actuators={"legs": SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
