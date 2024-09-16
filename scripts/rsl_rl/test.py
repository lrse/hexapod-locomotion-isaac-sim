"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import numpy as np
import os
import torch

from unittest.mock import patch
# from omni.isaac.lab.terrains import _generate_curriculum_terrains
from omni.isaac.lab.terrains.trimesh.mesh_terrains import pyramid_stairs_terrain


from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import hexapod_extension.tasks  # noqa: F401

from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx

# # Global variable to store the captured step heights
# terrain_parameter = None

# # Store a reference to the original pyramid_stairs_terrain function
# original_pyramid_stairs_terrain = pyramid_stairs_terrain

# # Define a function to capture step_height during terrain generation
# def capture_pyramid_stairs_terrain(difficulty, cfg):
#     global terrain_parameter
#     print("Using modified version of pyramid_stairs_terrain")
#     step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
#     terrain_parameter = step_height
#     # Call the original terrain generation function
#     return original_pyramid_stairs_terrain(difficulty, cfg)

# # Patch the _generate_curriculum_terrains method
# def custom_generate_curriculum_terrains(self):
#     """Copied from _generate_curriculum_terrains. Adds terrains based on the difficulty parameter."""
#     global terrain_parameter
#     # normalize the proportions of the sub-terrains
#     proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
#     proportions /= np.sum(proportions)

#     # find the sub-terrain index for each column
#     # we generate the terrains based on their proportion (not randomly sampled)
#     sub_indices = []
#     for index in range(self.cfg.num_cols):
#         sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
#         sub_indices.append(sub_index)
#     sub_indices = np.array(sub_indices, dtype=np.int32)
#     # create a list of all terrain configs
#     sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

#     # curriculum-based sub-terrains
#     for sub_col in range(self.cfg.num_cols):
#         for sub_row in range(self.cfg.num_rows):
#             # vary the difficulty parameter linearly over the number of rows
#             lower, upper = self.cfg.difficulty_range
#             difficulty = (sub_row + np.random.uniform()) / self.cfg.num_rows
#             difficulty = lower + (upper - lower) * difficulty
#             # generate terrain
#             mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_indices[sub_col]])
#             # add to sub-terrains
#             self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_indices[sub_col]])
#     # This method is executed inside the terrain generation loop
#     self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols)) # Agregado
#     for sub_row in range(self._sub_terrain_rows):
#         for sub_col in range(self._sub_terrain_cols):
#             sub_indices = self._sub_terrain_indices
#             sub_terrains_cfgs = self._sub_terrains_cfgs

#             # Call the patched pyramid_stairs_terrain function
#             mesh, origin = pyramid_stairs_terrain(self._difficulty, sub_terrains_cfgs[sub_indices[sub_col]])

#             # Use the original behavior to add the terrain to the simulation
#             self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_indices[sub_col]])
#             self.terrain_parameter[sub_row, sub_col] = terrain_parameter# Agregado
#             terrain_parameter = None # Agregado

#     # After the loop, print captured step heights
#     print(f"Captured step heights: {terrain_parameter}")

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # # Patch the pyramid_stairs_terrain function to capture step_height
    # with patch('omni.isaac.lab.terrains.trimesh.mesh_terrains.pyramid_stairs_terrain', new=capture_pyramid_stairs_terrain):
    # Create Isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    print(env.env.unwrapped.scene.terrain.terrain_parameter)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name+"_blind")
    # log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name+"_height_scan")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()
    # Keep a record of episodes finished by each environment
    num_episodes = torch.zeros(env.num_envs, device=env.unwrapped.device)  # Track completed episodes
    truncated = torch.zeros(env.num_envs, device=env.unwrapped.device)  # Track completed episodes
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, extras = env.step(actions) # Step the environment

            truncated = extras["time_outs"].long()
            # Count finished episodes
            num_episodes += dones + truncated
            
            # Check if all elements in num_episodes are >= 1 (all environments have finished one episode)
            if all(episode >= 1 for episode in num_episodes):
                lista = [[] for _ in range(len(env.unwrapped.scene._terrain.terrain_parameter))]
                test_result = env.unwrapped.test_result
                for env_id in range(env.num_envs):
                    if test_result[env_id] is None:
                        raise ValueError(f"Something went wrong, test result not found for environment {env_id}.")
                    level = env.unwrapped.scene._terrain.terrain_levels[env_id].item()
                    lista[level].append(test_result[env_id])
                # Calculate the success rate for each level
                successful = [sublist.count(True) if sublist else 0 for sublist in lista]
                success_rate = [sublist.count(True) / len(sublist) if sublist else 0 for sublist in lista]
                total_robots = [len(sublist) if sublist else 0 for sublist in lista]
                # parameters = list(env.unwrapped.scene._terrain.terrain_parameter.flatten())
                parameters = env.unwrapped.scene._terrain.terrain_parameter
                print('# of successful robots: ', successful)
                print('success_rate: ', success_rate)
                print('parameters: ', parameters)
                print('parameters: ', list(parameters[:,0]))
                print('total_robots: ', total_robots)
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()