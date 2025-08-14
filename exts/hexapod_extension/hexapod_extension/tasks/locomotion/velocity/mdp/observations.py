from __future__ import annotations

import torch
import pytorch3d.transforms
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def net_forces(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reshape contact force information available in the contact sensor 
    to be used as observation.

    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # sensor_pos = contact_sensor.data.pos_w
    sensor_quat = contact_sensor.data.quat_w
    # net_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids,:]
    net_forces_w = contact_sensor.data.net_forces_w
    # Change the coordinate system of the forces
    net_forces_w_relative = pytorch3d.transforms.quaternion_apply(sensor_quat, net_forces_w)
    reshaped_tensor = net_forces_w_relative.contiguous().view(net_forces_w_relative.shape[0], -1)
    return reshaped_tensor

def contact_states(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reshape contact force information available in the contact sensor 
    and determine contact states to be used as observation.

    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # contact_states = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    contact_states = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold
    # TODO: verify this function works as intended    
    return contact_states