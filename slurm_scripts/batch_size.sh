#!/bin/bash
#SBATCH --job-name=BS # nom du job
#SBATCH --output=log/BS/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/BS/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 #reserver 10 taches (ou processus)
#SBATCH --cpus-per-task=64
#SBATCH --mem=0 # reserve toute la mémoire
#SBATCH --time=168:00:00 # temps d'allocation

source .venvbatch/bin/activate
srun python3 evaluate_batch_size.py