#!/bin/bash
#SBATCH --job-name=SA # nom du job
#SBATCH --output=log/SA/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/SA/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 #reserver 10 taches (ou processus)
#SBATCH --cpus-per-task=64
#SBATCH --mem=0 # reserve toute la mémoire
#SBATCH --time=168:00:00 # temps d'allocation
#SBATCH --nodelist=sl-mee-br-119
#SBATCH -p AAI

source .venvbatch/bin/activate
srun python3 comparatif.py --processes 8 --output_path results/full_resnet/
