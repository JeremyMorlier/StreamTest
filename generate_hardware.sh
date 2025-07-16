#!/bin/bash
#SBATCH --partition=BrainA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=168:00:00
#SBATCH --output=log/%x/%j/logs.out
#SBATCH --error=log/%x/%j/errors.err
#SBATCH --qos=highbrain

export XDG_CONFIG_HOME=/Brain/private/j20morli/.config

source /Brain/private/j20morli/StreamTest/.venv/bin/activate
srun python3 resnet18_hardware_search.py