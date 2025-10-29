#SBATCH --job-name=TWG # nom du job
#SBATCH --output=log/TWG/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/TWG/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 #reserver 4 taches (ou processus)
#SBATCH --cpus-per-task=24 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=sxq@a100 # comptabilite 1100
#SBATCH --signal=USR1@40

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load anaconda-py3/2023.09
source .venv/bin/activate

set -x

srun python3 generate_tiled_workload.py