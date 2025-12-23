#!/bin/bash
#SBATCH --job-name=CA # nom du job
#SBATCH --output=log/CA/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/CA/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 #reserver 10 taches (ou processus)
#SBATCH --time=168:00:00 # temps d'allocation
#SBATCH --nodelist=sl-mee-br-119
#SBATCH -p AAI

export GRB_LICENSE_FILE=../jobsSlurm/gurobi.llc

source .venvbatch/bin/activate
srun python3 comparatif_gpt2.py --processes 4 --output_path results/ca_gpt_sa/ --batch_size 1
