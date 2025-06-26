#!/bin/bash
#SBATCH --partition=Brain3090
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=168:00:00
#SBATCH --output=log/%x/%j/logs.out
#SBATCH --error=log/%x/%j/errors.err
#SBATCH --qos=highbrain

source .venv/bin/activate
srun python3 resnet18_hardware_search.py