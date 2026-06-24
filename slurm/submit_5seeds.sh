#!/bin/bash
#SBATCH --job-name=scm_5seeds
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_%A_%a.out
#SBATCH --array=0-9

mkdir -p logs

module purge
module load 2025

PYBIN=/gpfs/home6/yelkacemi/.conda/envs/thesis/bin/python

PAIRS=(scm scm scm scm scm shuffled shuffled shuffled shuffled shuffled)
SEEDS=(42 13 7 21 100 42 13 7 21 100)

P=${PAIRS[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "FULL with SHAP: legal-bert pairs=$P seed=$S"

srun $PYBIN run_contrastive_v2.py --encoder legal-bert --pairs $P --seed $S
