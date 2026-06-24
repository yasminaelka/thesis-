#!/bin/bash
#SBATCH --job-name=ig_multiseed
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=ig_multiseed_%A_%a.log

SEEDS=(42 13 7 21 100)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

PY=/gpfs/home6/yelkacemi/.conda/envs/thesis/bin/python
cd /gpfs/home6/yelkacemi

echo "=== IG multiseed: array task $SLURM_ARRAY_TASK_ID -> seed $SEED ==="
$PY run_ig_multiseed.py \
    --encoder legal-bert \
    --pairs scm \
    --seed $SEED \
    --root /gpfs/home6/yelkacemi/output \
    --n_docs 200 \
    --ig_steps 50

echo "=== done seed $SEED ==="
