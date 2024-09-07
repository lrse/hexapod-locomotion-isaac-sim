"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

# TODO: Revisar los comentarios y sacar todo lo que está en español cuando entienda bien el código


import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Calibrate hyperparameters of an RL agent with RSL-RL.")
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

# Hiperparameters for Optuna #TODO: acomodar esto
N_TRIALS = 100  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 1  # Number of evaluations during the training
# N_TIMESTEPS = int(2e4)  # Training budget
EVAL_FREQ = 50#int(N_TIMESTEPS / N_EVALUATIONS)
# N_EVAL_ENVS = 5
N_EVAL_EPISODES = 1#
TIMEOUT = int(3*24*60*60)  # Time out en segundos (N * 24 * 60 * 60 daría N días)


import gymnasium as gym
import os
import torch
from datetime import datetime
from hexapod_extension.tasks.locomotion.velocity.config.phantom_x import agents, rough_env_cfg

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import hexapod_extension.tasks  # noqa: F401

from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx

# import omni.isaac.lab.terrains as terrains
# print(dir(terrains))

# from omni.isaac.lab.terrains import configure_env_origins

torch.backends.cuda.matmul.allow_tf32 = True # Whether TensorFloat-32 tensor cores may be used in 
                                             # matrix multiplications on Ampere or newer GPUs.
                                             # TF32 tensor cores are designed to achieve better 
                                             # performance on matmul and convolutions on torch.float32 tensors 
torch.backends.cudnn.allow_tf32 = True # Controls where TensorFloat-32 tensor cores may be used in 
                                       # cuDNN convolutions on Ampere or newer GPUs. 
torch.backends.cudnn.deterministic = False # Causes cuDNN to only use deterministic convolution algorithms
torch.backends.cudnn.benchmark = False # Causes cuDNN to benchmark multiple convolution algorithms and select the fastest.


from stable_baselines3.common.callbacks import EvalCallback
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances


from typing import Any, Dict

def sample_ppo_params(trial: optuna.Trial, agent_cfg: RslRlOnPolicyRunnerCfg):
    """
    Sampler for PPO hyperparameters.

    :param trial: Optuna trial object
    :return: A dictionary with the sampled hyperparameters for the given trial.
    """
    # agent_cfg_dict = {'seed': 42, 
    #                   'device': 'cuda:0', #TODO
    #                   'num_steps_per_env': 24, 
    #                   'max_iterations': 40,#6000, 
    #                   'empirical_normalization': False, 
    #                   'policy': {'class_name': 'ActorCritic',#'ActorCriticRecurrent', 
    #                              'init_noise_std': 1.0, 
    #                              'actor_hidden_dims': [512, 256, 128],
    #                              'critic_hidden_dims': [512, 256, 128],
    #                              'activation': 'elu'}, 
    #                   'algorithm': {'class_name': 'PPO', 
    #                                 'value_loss_coef': 1.0, 
    #                                 'use_clipped_value_loss': True, 
    #                                 'clip_param': 0.2, 
    #                                 'entropy_coef': 0.005, 
    #                                 'num_learning_epochs': 5, 
    #                                 'num_mini_batches': 4, 
    #                                 'learning_rate': trial.suggest_float("lr", 1e-5, 1, log=True),#0.001, 
    #                                 'schedule': 'adaptive', 
    #                                 'gamma': 0.99, #1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True), 
    #                                 'lam': 0.95, 
    #                                 'desired_kl': 0.01, 
    #                                 # 'max_grad_norm': trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)}, 
    #                                 'max_grad_norm': 1.0}, 
    #                   'save_interval': 5,#50, 
    #                   'experiment_name': 'phantom_x_rough', 
    #                   'run_name': '', 
    #                 #   'logger': 'tensorboard', 
    #                 #   'neptune_project': 'isaaclab', 
    #                 #   'wandb_project': 'isaaclab', 
    #                   'resume': False, 
    #                   'load_run': None, 
    #                   'load_checkpoint': None}
    clip_param = trial.suggest_float("clip_param", 0.05, 0.5)
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    lam = 1.0 - trial.suggest_float("lam", 0.001, 0.2, log=True) # lam es el lambda del GAE
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    num_steps_per_env = trial.suggest_int("num_steps_per_env", 24, 100)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 0.00000001, 0.1, log=True)
    num_learning_epochs = trial.suggest_int("num_learning_epochs", 4, 20)
    num_mini_batches = 2 ** trial.suggest_int("exponent_num_mini_batches", 2, 6)
    # ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    # net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "elu", "relu"])
    
    agent_cfg.algorithm.clip_param = clip_param
    agent_cfg.algorithm.gamma = gamma
    agent_cfg.algorithm.lam = lam
    agent_cfg.algorithm.max_grad_norm = max_grad_norm
    agent_cfg.num_steps_per_env = num_steps_per_env
    agent_cfg.algorithm.learning_rate = learning_rate
    agent_cfg.algorithm.entropy_coef = entropy_coef
    agent_cfg.algorithm.num_learning_epochs = num_learning_epochs
    agent_cfg.algorithm.num_mini_batches = num_mini_batches
    agent_cfg.policy.activation_fn = activation_fn


def evaluate_policy(agent_cfg, eval_env, resume_path, n_eval_episodes=5):
    """Evaluate the policy in a given environment."""

    # load previously trained model
    ppo_runner = OnPolicyRunner(eval_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=eval_env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    total_rewards = torch.zeros(eval_env.num_envs, device=eval_env.unwrapped.device)  # Track rewards for each environment
    num_episodes = torch.zeros(eval_env.num_envs, device=eval_env.unwrapped.device)  # Track completed episodes
    # reset environmen
    with torch.inference_mode():
        eval_env.reset()
    obs, _ = eval_env.get_observations() #TODO: esto hace falta?
    # simulate environment
    while num_episodes.sum() < n_eval_episodes * eval_env.num_envs:
        with torch.inference_mode():
            actions = policy(obs)  # Get actions for each environment
            obs, rewards, dones, _ = eval_env.step(actions)  # Step the environment

            # print(f"Actions: {actions}")
            # print(f"Rewards: {rewards}")
            # print(f"Dones: {dones}")

            # Accumulate rewards where episodes are not done
            total_rewards += rewards * (num_episodes < n_eval_episodes)

            # Count finished episodes
            num_episodes += dones
            aux = num_episodes.sum()

            # # # Reset environments where episodes are done TODO: necesito hacer esto?
            # # if dones.any():
            # #     obs[dones], _ = eval_env.reset_done(dones)
            # # If the environment resets automatically, just continue
            # if dones.any():
            #     obs, _ = eval_env.get_observations()

    # Calculate mean reward across all episodes and environments
    mean_reward = total_rewards.sum() / (n_eval_episodes * eval_env.num_envs)
    return mean_reward.item()


# parse configuration
env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

# Create envs used for training and evaluation
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)#"rgb_array" if args_cli.video else None)
# wrap around environment for rsl-rl
env: ManagerBasedRLEnv = RslRlVecEnvWrapper(env)


def objective(trial: optuna.Trial) -> float:
    """
    Objective function used by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """
    # parse configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name+"_calibration")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Sample hyperparameters
    sample_ppo_params(trial, agent_cfg)
    
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # set seed of the environment
    env.seed(agent_cfg.seed)

    nan_encountered = False
    eval_freq = EVAL_FREQ
    n_eval_episodes = N_EVAL_EPISODES
    
    total_timesteps = 0
    last_mean_reward = -float("inf")

    try:
        # reset levels
        env.unwrapped.unwrapped.scene.terrain.configure_env_origins(env.unwrapped.unwrapped.scene.terrain.terrain_origins)
        terrain_levels = env.unwrapped.unwrapped.scene.terrain.terrain_levels
        # run training:
        while total_timesteps < agent_cfg.max_iterations:
            if agent_cfg.resume:
                # get path to previous checkpoint
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
                print(f"[INFO]: Loading model checkpoint from: {resume_path}")
                # load previously trained model
                runner.load(resume_path)
            # restore levels
            env.unwrapped.unwrapped.scene.terrain.terrain_levels = terrain_levels
            with torch.inference_mode():
                env.reset()
            # Train the model for eval_freq timesteps
            runner.learn(num_learning_iterations=eval_freq+1,
                         init_at_random_ep_len=True)
            # save the reached terrain levels
            terrain_levels = env.unwrapped.unwrapped.scene.terrain.terrain_levels
            # To resume training if not done:
            agent_cfg.resume = True
            # TODO: falta chequea que esté bien que en cada iteración del while retome el aprendizaje de donde quedó antes, no que empiece de nuevo
            # TODO: Hay algo raro con los puntos en los que se guarda.
            # TODO: 
            total_timesteps += eval_freq

            # specify directory for logging experiments
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            # Periodically evaluate the model
            # reset levels
            env.unwrapped.unwrapped.scene.terrain.configure_env_origins(env.unwrapped.unwrapped.scene.terrain.terrain_origins)
            # evaluate policy
            mean_reward = evaluate_policy(agent_cfg, env, resume_path, n_eval_episodes)
            trial.report(mean_reward, total_timesteps)

            # Prune the trial if the mean reward is not promising
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            last_mean_reward = mean_reward
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    return last_mean_reward
# TODO: che, quiero maximizar esto o el nivel alcanzado?


def main():
    """Calibrate Hyperparameters of the RSL-RL agent with Optuna."""

    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:") #TODO: qué es esto?
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name+"_calibration")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging calibration in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, "study_results_ppo.csv")
    study.trials_dataframe().to_csv(log_dir)
    # TODO: ver que esté bien cómo guardo los resultados. 
    # TODO: Ver si estoy guardando bien la configuración de parámetros usada

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
