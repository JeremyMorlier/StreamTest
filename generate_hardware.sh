#!/bin/bash
#SBATCH --partition=BrainA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=168:00:00
#SBATCH --output=log/%x/%j/logs.out
#SBATCH --error=log/%x/%j/errors.err
#SBATCH --qos=highbrain

export XDG_CONFIG_HOME=/Brain/private/j20morli/.config
export UV_CACHE_DIR=/Brain/private/j20morli/.cache/uv
export UV_PYTHON_INSTALL_DIR=/Brain/private/j20morli/.cache/uv/python

source /Brain/private/j20morli/StreamTest2/.venv/bin/activate

srun python3 resnet18_hardware_search.py