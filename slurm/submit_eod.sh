#!/bin/bash
#SBATCH --job-name=eod
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=logs/eod_%A_%a.out
#SBATCH --array=0-4
mkdir -p logs
module purge
module load 2025
PYBIN=/gpfs/home6/yelkacemi/.conda/envs/thesis/bin/python
SEEDS=(42 13 7 21 100)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "EOD: legal-bert pairs=scm seed=$S"
srun $PYBIN -u fairness_ci_eod_multiseed.py --pairs scm --seed $S
