#!/bin/bash
#SBATCH --job-name=lam_sweep_faith
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home6/yelkacemi/logs/lam_faith_%j.out
#SBATCH --error=/gpfs/home6/yelkacemi/logs/lam_faith_%j.err

# Single-seed SHAP faithfulness lambda sweep: baseline + 4 lambda checkpoints,
# 200 docs each via KernelSHAP. This is the expensive run (~5x a single-model
# faithfulness job), hence the 12h walltime ceiling.

set -euo pipefail

SEED=42
PY=/gpfs/home6/yelkacemi/.conda/envs/thesis/bin/python
mkdir -p /gpfs/home6/yelkacemi/logs

echo "=== faithfulness lambda sweep, seed $SEED  ($(date)) ==="
$PY /gpfs/home6/yelkacemi/lambda_sweep_faithfulness.py \
    --encoder legal-bert \
    --pairs   scm \
    --seed    "$SEED" \
    --root    /gpfs/home6/yelkacemi/output
echo "=== done  ($(date)) ==="
