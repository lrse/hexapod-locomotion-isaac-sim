# Blind Hexapod Locomotion Over Rough Terrains using Parallel DRL


## Overview ##
This repository provides the codes used to train a phantomx robot to walk on rough terrain using NVIDIA's Isaac Gym. It includes the scripts used for training as well as those used for evaluation of the obtained policies.


**Maintainer**: Tania Sofía Ferreyra  
**Affiliation**: Laboratorio de Robótica y Sistemas Embebidos, Instituto de Ciencias de la Computación, UBA-CONICET
**Contact**: tferreyra@icc.fcen.uba.ar

---

## Installation ##
1. Create a new python virtual env with python 3.10.
2. Install Isaac Gym and Isaac Lab following the [documentation's instructions](https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html). Make sure to include the installation of rsl_rl.
3. Clone this repository.
4. Install python extension hexapod_extension using pip. You can do this by executing from the repository's root:
python -m pip install -e  exts/hexapod_extension/

## Usage ##
1. Train:  
  ```python hexapod-locomotion-isaac-sim/scripts/train.py --task=Isaac-Velocity-Rough-Phantom-X-v0```
    - To run headless add `--headless`.
    - To save videos of the training add `--video`.
    - The trained policy is saved in `hexapod-locomotion-isaac-sim/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    - You can override values set int he configuration files by adding any of the following line arguments:
     - --task TASK_NAME: changes the task.
     - --resume: resumes training from a given checkpoint.
     - --experiment_name EXPERIMENT_NAME: name of the experiment to run or load.
     - --num_envs NUM_ENVS:  number of environments to use.
     - --seed SEED:  random seed.
     - --max_iterations MAX_ITERATIONS:  maximum number of training iterations.
2. Play a trained policy:  
```python hexapod-locomotion-isaac-sim/scripts/play.py --task=Isaac-Velocity-Rough-Phantom-X-Play-v0```
    - By default, the loaded policy is the last model of the last run of the experiment folder.
