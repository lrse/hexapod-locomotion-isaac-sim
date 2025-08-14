"""Script to analyze and compare the results of different policies."""

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
import pickle
import subprocess

def run_test_script(args):
    try:
        # Run train.py with the provided arguments
        result = subprocess.run(['python', 'scripts/rsl_rl/test.py'] + args, check=True)
        print(f"Completed test with args: {args}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False
    return True

def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

# Helper function for subsampling evenly
def subsample_indices(length, num_points):
    if length <= num_points:
        return list(range(length))
    return np.round(np.linspace(0, length - 1, num_points)).astype(int)

def read_tensorboard():
    compare = ['him_locomotion', 'ours4']
    # compare = ['ours4']
    name = {'him_locomotion': 'HIM Locomotion', 'ours4': 'Ours'}
    colores = {'him_locomotion': 'blue', 'ours4': 'red'}
    ax1 = [None] * 4
    ax2 = [None] * 4
    size = (4.5,2.3)
    for height_scanner_number in [1]:#, 2, 3]:#, 4]:
        _, ax1[height_scanner_number-1] = plt.subplots(figsize=size)
        _, ax2[height_scanner_number-1] = plt.subplots(figsize=size)
        ax1[height_scanner_number-1].set_title('Average Terrain Level')# (height scanner '+str(height_scanner_number)+')')
        ax1[height_scanner_number-1].set_xlabel('Iteration')# (height scanner '+str(height_scanner_number)+')')
        ax1[height_scanner_number-1].set_ylabel('Terrain Level')# (height scanner '+str(height_scanner_number)+')')
        ax2[height_scanner_number-1].set_title('Mean Reward Through Training')# (height scanner '+str(height_scanner_number)+')')
        ax2[height_scanner_number-1].set_xlabel('Iteration')# (height scanner '+str(height_scanner_number)+')')
        ax2[height_scanner_number-1].set_ylabel('Mean Reward')# (height scanner '+str(height_scanner_number)+')')

        for method in compare:
            # specify directory for logging experiments
            log_root_path = os.path.join("logs", "rsl_rl", "height_scanner"+str(height_scanner_number)+'_'+method)
            log_root_path = os.path.abspath(log_root_path)
            # print(f"[INFO] Loading experiments from directory: {log_root_path}")
            
            runs = os.listdir(log_root_path)
            steps = [[]] * len(runs)
            terrain_mean_lvls = [[]] * len(runs)
            mean_rewards = [[]] * len(runs)
            for idx, run in enumerate(runs):
                file_path = os.path.join(log_root_path, run, 'tensorboard_data.pkl')
                if os.path.exists(file_path):
                    # File exists: load it
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    print(f"[INFO] Loading existing data: {file_path}")
                    terrain_mean_lvls[idx] = data['terrain_mean_lvls']
                    mean_rewards[idx] = data['mean_rewards']
                    steps[idx] = data['steps']
                else:
                    print(f"[INFO] No saved data found in {log_root_path}. Generating new data...")
                    scalars = ["Curriculum/terrain_levels/mean", "Train/mean_reward"]
                    path = os.path.join(log_root_path,run)
                    print(f"[INFO] Loading tensorboard information number {idx} from directory: {path}")
                    result = parse_tensorboard(path, scalars)
                    steps[idx] = [x[1] for x in result['Curriculum/terrain_levels/mean'].values]
                    terrain_mean_lvls[idx] = [x[2] for x in result['Curriculum/terrain_levels/mean'].values]
                    assert(steps[idx] == steps[0])
                    assert([x[1] for x in result['Train/mean_reward'].values] == steps[idx])
                    mean_rewards[idx] = [x[2] for x in result['Train/mean_reward'].values]
                    # Save variables
                    data = {'terrain_mean_lvls': terrain_mean_lvls[idx], 'mean_rewards': mean_rewards[idx], 'steps': steps[idx]}
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f"[INFO] Saved data as: {file_path}")
                ax1[height_scanner_number-1].plot(steps[0], terrain_mean_lvls[idx], color=colores[method], alpha=0.1, rasterized=True)
                ax2[height_scanner_number-1].plot(steps[0], mean_rewards[idx], color=colores[method], alpha=0.1, rasterized=True)
                
                step = steps[0]

            terrain_mean_lvls_mean = []
            terrain_mean_lvls_std = []
            mean_rewards_mean = []
            mean_rewards_std = []
            for idx, s in enumerate(step):
                terrain_mean_lvls_mean.append(mean([terrain_mean_lvls[run][idx] for run in range(10)]))
                terrain_mean_lvls_std.append(stdev([terrain_mean_lvls[run][idx] for run in range(10)]))
                mean_rewards_mean.append(mean([mean_rewards[run][idx] for run in range(10)]))
                mean_rewards_std.append(stdev([mean_rewards[run][idx] for run in range(10)]))
            ax1[height_scanner_number-1].errorbar(step, terrain_mean_lvls_mean, yerr=terrain_mean_lvls_std, errorevery=200, color=colores[method], label=name[method])
            ax2[height_scanner_number-1].errorbar(step, mean_rewards_mean, yerr=mean_rewards_std, errorevery=200, color=colores[method], label=name[method])
        ax1[height_scanner_number-1].grid(True)
        ax2[height_scanner_number-1].grid(True)
        ax1[height_scanner_number-1].legend()
        ax2[height_scanner_number-1].legend()

def compare_performance():
    # compare = ['him_locomotion', 'ours4']
    compare = ['ours4']
    task_name = {'him_locomotion': 'HIMLocomotion', 'ours4': 'Ours4'}
    terrains = ['Inv-Slope', 'Slope', 'Boxes', 'Inv-Stairs', 'Stairs']
    colores = {'him_locomotion': 'blue', 'ours4': 'red',
               'HIM_hs1': '#1f77b4',  # blue
               'HIM_hs2': '#4a90e2',  # lighter blue
               'HIM_hs3': '#a6c8ff',  # even lighter blue
               'Ours_hs1': '#d62728', # red
               'Ours_hs2': '#ff6b6b', # lighter red
               'Ours_hs3': '#ffb3b3',  # even lighter red
               'Ours_hs4': 'orange'  # even lighter red
               }
    name = {'him_locomotion': 'HIM', 'ours4': 'Ours'}
    ax = [None for i in range(len(terrains))]
    all_results = {}
    parameter_values = {}

    for idx_terrain, terrain in enumerate(terrains):
        all_results[terrain] = {}
        parameter_values[terrain] = None

        _, ax[idx_terrain] = plt.subplots()
        ax[idx_terrain].set_title('Success Rate for ' + terrain)
        
        for height_scanner_number in [1, 2, 3, 4]:
            print('[INFO] Warning: (TODO) falta mejorar el código, en este momento hay que modificar a mano el sensor de alturas para correr test, sino salta error.')
            for method in compare:
                # specify directory for logging experiments
                log_root_path = os.path.join("logs", "rsl_rl", "height_scanner"+str(height_scanner_number)+'_'+method)
                log_root_path = os.path.abspath(log_root_path)
                
                runs = os.listdir(log_root_path)
                parameters = [[]] * len(runs)
                success_rate = [[]] * len(runs)

                for idx, run in enumerate(runs):
                    file_path = os.path.join(log_root_path, run, terrain+'_data.pkl')
                    if not os.path.exists(file_path):
                        # If the file doesn't exist, perform the test to create it
                        print(f"[INFO] No saved data found in {os.path.join(log_root_path, run)}. Generating new data...")
                        task = "Isaac-Velocity-Phantom-X-"+task_name[method]+"-Test-"+terrain
                        args = ["--task", task, "--load_run", run, 
                                "--changed_experiment_name", "height_scanner"+str(height_scanner_number)+'_'+method, 
                                "--terrain", terrain,
                                "--num_envs", "1024", "--headless"]
                        success = run_test_script(args)
                        if not success:
                            print(f"Test for {file_path} failed. Stopping.")
                            break

                    # Load results from test
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    print(f"[INFO] Loading data from: {file_path}")
                    if idx>0: #Verify that every run after the first has the same parameter values
                        assert(parameters[0] == data['parameters'])

                    parameters[idx] = data['parameters']
                    success_rate[idx] = data['success_rate']

                    ax[idx_terrain].plot(parameters[idx], success_rate[idx], alpha=0.1, color=colores[method])
                    
                success_rate_mean = []
                success_rate_std = []
                for idx, p in enumerate(parameters[0]):
                    success_rate_mean.append(mean([success_rate[run][idx] for run in range(len(runs))]))
                    success_rate_std.append(stdev([success_rate[run][idx] for run in range(len(runs))]))
                
                # Save parameters only once
                if parameter_values[terrain] is None:
                    parameter_values[terrain] = parameters[0]

                nombre = f"{name[method]}_hs{height_scanner_number}"
                ax[idx_terrain].errorbar(parameters[0], success_rate_mean, 
                                         yerr=success_rate_std, errorevery=1, 
                                         color=colores[nombre], label=nombre, linewidth=2.0)

                # Subsample and add to table data
                all_results[terrain][(method, height_scanner_number)] = list(zip(success_rate_mean, success_rate_std))
                
        ax[idx_terrain].grid(True)
        ax[idx_terrain].legend()

    for idx_terrain, terrain in enumerate(terrains):
        # === Construct table ===
        max_rows = 10
        max_digits = 3
        indices = subsample_indices(len(parameter_values[terrain]), max_rows)

        table_dict = {'parameter': [parameter_values[terrain][i] for i in indices]}
        for (method, hs_num), stats in all_results[terrain].items():
            means = [stats[i][0] for i in indices]
            stds = [stats[i][1] for i in indices]
            # table_dict[f"{name[method]}_hs{hs_num}_mean"] = means
            # table_dict[f"{name[method]}_hs{hs_num}_std"] = stds
            combined = [
                f"{mean_val:.{max_digits}f} ± {std_val:.{max_digits}f}"
                for mean_val, std_val in zip(means, stds)
            ]
            table_dict[f"{name[method]}_hs{hs_num}"] = combined
            
        df = pd.DataFrame(table_dict)

        # Show full precision
        pd.set_option("display.precision", max_digits)
        pd.set_option("display.float_format", f"{{:.{max_digits}f}}".format)

        print(f"\n=== Table for terrain '{terrain}' ===")
        print(df.to_string(index=False))

        # === Optional: save to LaTeX ===
        latex_filename = f"success_rates_table_{terrain}.tex"
        float_fmt = f"%.{max_digits}f"  # Dynamically set precision
        with open(latex_filename, "w") as f:
            f.write(df.to_latex(index=False, float_format=float_fmt))
        print(f"[INFO] LaTeX table saved to {latex_filename}")

def main():
    read_tensorboard()
    # compare_performance()
    plt.show()

if __name__ == "__main__":
    # run the main execution
    main()
