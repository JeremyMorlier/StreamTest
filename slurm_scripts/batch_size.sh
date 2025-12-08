#!/bin/bash
#SBATCH --job-name=BS # nom du job
#SBATCH --output=log/BS/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/BS/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --ntasks=1 #reserver 10 taches (ou processus)
#SBATCH --time=168:00:00 # temps d'allocation
source .venvbatch/bin/activate
srun python3 evaluate_batch_size.py