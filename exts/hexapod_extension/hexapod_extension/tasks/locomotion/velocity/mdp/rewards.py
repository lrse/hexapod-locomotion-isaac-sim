from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def tripod_gate(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward stances compatible with tripod gate.

    This function rewards the agent if it's standing using front and back legs of one side and center 
    leg of the other, or both.

    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    in_contact = {}
    for idx, leg_idx in enumerate(sensor_cfg.body_ids):
        force = contact_sensor.data.net_forces_w[:,leg_idx, :]
        in_contact[sensor_cfg.body_names[idx]] = torch.norm(force, dim=1) > threshold
    tripod_stance = (in_contact["foottip_rf"] & in_contact["foottip_lm"] & in_contact["foottip_rr"]) \
                    | (in_contact["foottip_lf"] & in_contact["foottip_rm"] & in_contact["foottip_lr"])
    return tripod_stance.int()
