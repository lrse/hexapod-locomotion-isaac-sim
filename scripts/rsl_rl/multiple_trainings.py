import subprocess

def run_train_script(args):
    try:
        # Run train.py with the provided arguments
        result = subprocess.run(['python', 'scripts/rsl_rl/train.py'] + args, check=True)
        print(f"Completed training with args: {args}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False
    return True

if __name__ == "__main__":
    n = 10  # Number of training curves for each system
    
    for i in range(n):
        # args = ["--task", "Isaac-Velocity-Rough-Phantom-X-HIMLocomotion", "--num_envs", "1024", "--headless"]
        # success = run_train_script(args)
        # if not success:
        #    print(f"Training {i + 1} for HIM Locomotion failed. Stopping.")
        #    break
        args = ["--task", "Isaac-Velocity-Rough-Phantom-X-Ours4", "--num_envs", "1024", "--headless"]
        success = run_train_script(args)
        if not success:
            print(f"Training {i + 1} for Ours failed. Stopping.")
            break
        #args = ["--task", "Isaac-Velocity-Rough-Phantom-X-DreamWaQ", "--num_envs", "1024", "--headless"]
        #success = run_train_script(args)
        #if not success:
        #    print(f"Training {i + 1} for DreamWaQ failed. Stopping.")
        #    break

        # args = ["--task", "Isaac-Velocity-Rough-Phantom-X-Oracle", "--num_envs", "1024", "--headless"]
        # success = run_train_script(args)
        # if not success:
        #     print(f"Training {i + 1} for Ours failed. Stopping.")
        #     break
        # args = ["--task", "Isaac-Velocity-Rough-Phantom-X-Oracle2", "--num_envs", "1024", "--headless"]
        # success = run_train_script(args)
        # if not success:
        #     print(f"Training {i + 1} for Ours failed. Stopping.")
        #     break
        # args = ["--task", "Isaac-Velocity-Rough-Phantom-X-Oracle-Normalized", "--num_envs", "1024", "--headless"]
        # success = run_train_script(args)
        # if not success:
        #     print(f"Training {i + 1} for Ours failed. Stopping.")
        #     break
        # args = ["--task", "Isaac-Velocity-Rough-Phantom-X-Oracle3", "--num_envs", "1024", "--headless"]
        # success = run_train_script(args)
        # if not success:
        #     print(f"Training {i + 1} for Ours failed. Stopping.")
        #     break
