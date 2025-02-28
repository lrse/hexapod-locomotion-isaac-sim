"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
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
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_modified.runners import HIMOnPolicyRunner, DreamWaQOnPolicyRunner, OursOnPolicyRunner

# Import extensions to set up environment tasks
import hexapod_extension.tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True # Whether TensorFloat-32 tensor cores may be used in 
                                             # matrix multiplications on Ampere or newer GPUs.
                                             # TF32 tensor cores are designed to achieve better 
                                             # performance on matmul and convolutions on torch.float32 tensors 
torch.backends.cudnn.allow_tf32 = True # Controls where TensorFloat-32 tensor cores may be used in 
                                       # cuDNN convolutions on Ampere or newer GPUs. 
torch.backends.cudnn.deterministic = False # Causes cuDNN to only use deterministic convolution algorithms
torch.backends.cudnn.benchmark = False # Causes cuDNN to benchmark multiple convolution algorithms and select the fastest.


def main():
    """Train with RSL-RL agent."""
    # parse configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(args_cli.task, device=agent_cfg.device, num_envs=args_cli.num_envs)
    

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # if args_cli.video:
    #     # adjust camera resolution and pose
    #     # env_cfg.viewer.resolution = (640, 480)
    #     env_cfg.viewer.eye = (1.0, 1.0, 1.0)
    #     env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        # Configure camera view through ViewerCfg
        env_cfg.viewer.origin_type = "asset_root"
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    terrain_parameters = env.unwrapped.scene.terrain.terrain_parameter
    # for lvl in range(terrain_parameters.shape[0]):
    #     print("Terrain Parameters in lvl ", lvl, ": ", terrain_parameters[lvl,:])
    from contextlib import redirect_stdout

    with open('logs/terrain_parameters.txt', 'w') as f:
        with redirect_stdout(f):
            print("stairs down: ", terrain_parameters[:,0])
            print("stairs down: ", terrain_parameters[:,1])
            print("stairs up: ", terrain_parameters[:,2])
            print("stairs up: ", terrain_parameters[:,3])
            print("boxes: ", terrain_parameters[:,4])
            print("boxes: ", terrain_parameters[:,5])
            # print("random rough: ", terrain_parameters[:,6])
            # print("random rough: ", terrain_parameters[:,7])
            print("slope down: ", terrain_parameters[:,8])
            print("slope up: ", terrain_parameters[:,9])


    # create runner from rsl-rl
    if agent_cfg.experiment_name == "phantom_x_rough_him_locomotion":
        # For experiments using the strategies in HIM Lomotion we need to use the rsl_rl_modified library
        agent_cfg.algorithm.class_name = 'HIMPPO' # TODO: see why it gets overwritten if I don't add this
        env.num_one_step_obs = int(env.unwrapped.observation_manager._group_obs_dim['policy'][0] / env_cfg.observations.policy.history_length)
        runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.experiment_name == "phantom_x_rough_dreamwaq":
        # For experiments using the strategies in DreamWaQ we need to use the rsl_rl_modified library
        agent_cfg.algorithm.class_name = 'DreamWaQPPO' # TODO: see why it gets overwritten if I don't add this
        env.num_one_step_obs = int(env.unwrapped.observation_manager._group_obs_dim['policy'][0] / env_cfg.observations.policy.history_length)
        runner = DreamWaQOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.experiment_name == "phantom_x_rough_ours":
        agent_cfg.algorithm.class_name = 'OursPPO' # TODO: see why it gets overwritten if I don't add this
        env.num_one_step_obs = int(env.unwrapped.observation_manager._group_obs_dim['policy'][0] / env_cfg.observations.policy.history_length)
        runner = OursOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        # write git state to logs
        runner.add_git_repo_to_log(__file__) #TODO: add this to HIMOnPolicyRunner?
        
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
