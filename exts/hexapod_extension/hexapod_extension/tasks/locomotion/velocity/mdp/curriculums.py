"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2

    # # compute the distance the robot walked
    # distance = torch.norm(asset.data.root_pos_w[env_ids, :] - env.scene.env_origins[env_ids, :], dim=1)
    # # robots that walked more than 90% of their required distance go to harder terrains
    # move_up = distance > torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.9

    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # # return the mean terrain level (this is done for logging purposes)
    # return torch.mean(terrain.terrain_levels.float())
    # return the mean and max terrain level (this is done for logging purposes)
    return {"mean": torch.mean(terrain.terrain_levels.float()), 
            "max": torch.max(terrain.terrain_levels)}


def success_rate(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None: #torch.Tensor:
    """Curriculum to see if I can save the final robot position to later calculate success rate.

    CODING IN PROGRESS

    Returns:
        Nothing.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    if env.common_step_counter == 0:
        env.test_result = [None] * env.num_envs
        # env.initial_position = asset.data.root_pos_w
        # env.initial_orientation = asset.data.root_quat_w
        return # To prevent the rest from executing on the initial reset
    # for idx, env_id in enumerate(env_ids.view(-1)):
    #     if env.test_result[env_id.item()] is None: # We only change the test result if I have no previous result
    #         # Add a third zero component to the command to make it 3D
    #         command = env.command_manager.get_command("base_velocity")[env_id, :2]
    #         # command_3d = torch.cat((
    #         #     env.command_manager.get_command("base_velocity")[env_id, :2], 
    #         #     torch.zeros(1, 1)), 
    #         #     dim=1)
    #         # # calculate target position in b
    #         # target_position_b = command_3d * env.max_episode_length_s
    #         # # Calculate rotation matrix from quaternion
    #         # q0 = env.initial_orientation[env_id,0]
    #         # q1 = env.initial_orientation[env_id,1]
    #         # q2 = env.initial_orientation[env_id,2]
    #         # q3 = env.initial_orientation[env_id,3]
    #         # R = torch.tensor([
    #         #     [2*(q0 * q0 + q1 * q1) - 1, 2*(q1 * q2 - q0 * q3)    , 2*(q1 * q3 + q0 * q2)],
    #         #     [2*(q1 * q2 + q0 * q3)    , 2*(q0 * q0 + q2 * q2) - 1, 2*(q2 * q3 - q0 * q1)],
    #         #     [2*(q1 * q3 - q0 * q2)    , 2*(q2 * q3 + q0 * q1)    , 2*(q0 * q0 + q3 * q3) - 1]
    #         #     ], device=env.device)
    #         # # transform to world coordinate using inital position and rotation matrix
    #         # target_position_w = env.initial_position[env_id,:] + torch.matmul(R,target_position_b)
    #         # # save the final position reached
    #         # final_position_w = asset.data.root_pos_w[env_id, :]
    #         # # calculate error in position
    #         # error = torch.norm(final_position_w - target_position_w, dim=1)
    #         # # robots that where close enough to the target position have succeded,
    #         # # where "close enough" is defined as 5% of distance to the target position
    #         # # (here we assume this distance is never too small)
    #         # success = error < torch.norm(command_3d) * env.max_episode_length_s * 0.05
    #         distance = torch.norm(final_position - env.scene.env_origins[env_ids, :], dim=1)
    #         # success is defined as having traverse a distance of at least 95% of the distance that
    #         # would be traversed if the robot was moving at the target velocity the whole episode
    #         success = (distance > torch.norm(command) * env.max_episode_length_s * 0.95)
    #         env.test_result[env_id.item()] = success
    # final_velocity = asset.data.root_lin_vel_b[env_ids, :]
    # command = env.command_manager.get_command('base_velocity')
    # heading_error = torch.acos(torch.dot(final_velocity, command)/(torch.norm(final_velocity, dim=1)*torch.norm(command, dim=1)))
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough are considered to have succeded in traversing the terrain
    success = distance > terrain.cfg.terrain_generator.size[0] / 2
    # # use the command to calculate the target position:
    # # the target position is the one reached if the robot moved at the 
    # # target velocity for the whole episode duration (without early termination)
    # target_position = env.scene.env_origins[env_ids, :2] + command * env.max_episode_length_s
    # # TODO: change command to world coordinates
    # # calculate error in position
    # error = torch.norm(final_position - target_position, dim=1)
    
    # success is defined as having traverse a distance of at least 95% of the distance that
    # would be traversed if the robot was moving at the target velocity the whole episode
    # while having a final heading error lower than a threshold
    # success = (distance > torch.norm(command) * env.max_episode_length_s * 0.95) and (abs(heading_error) < 0.08)
    # update successfull robots
    # success_rate(env_ids, success)
    for idx, env_id in enumerate(env_ids.view(-1)):
        # if str(env_id.item()) not in env.cfg.test_result:
        if env.test_result[env_id.item()] is None:
            env.test_result[env_id.item()] = success[idx].item()
