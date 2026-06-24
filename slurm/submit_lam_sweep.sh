#!/bin/bash
#SBATCH --job-name=lam_sweep_fair
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/gpfs/home6/yelkacemi/logs/lam_sweep_%A_%a.out
#SBATCH --error=/gpfs/home6/yelkacemi/logs/lam_sweep_%A_%a.err

set -euo pipefail

SEEDS=(42 13 7 21 100)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

PY=/gpfs/home6/yelkacemi/.conda/envs/thesis/bin/python

mkdir -p /gpfs/home6/yelkacemi/logs

echo "=== seed $SEED  ($(date)) ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"

$PY /gpfs/home6/yelkacemi/lambda_sweep_fairness.py \
    --encoder legal-bert \
    --pairs   scm \
    --seed    "$SEED" \
    --root    /gpfs/home6/yelkacemi/output

echo "=== done seed $SEED  ($(date)) ==="
