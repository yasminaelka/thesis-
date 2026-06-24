#!/bin/bash
#SBATCH --job-name=wordpair
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=wordpair_%A_%a.log

SEEDS=(42 13 7 21 100)
MODES=(wordpair_gender wordpair_eth)

MODE_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
MODE=${MODES[$MODE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

PY=/gpfs/home6/yelkacemi/.conda/envs/thesis/bin/python
cd /gpfs/home6/yelkacemi

echo "=== wordpair baseline: task $SLURM_ARRAY_TASK_ID -> mode=$MODE seed=$SEED ==="
$PY run_contrastive_wordpair.py \
    --encoder legal-bert \
    --pairs $MODE \
    --keyword_set targeted \
    --seed $SEED

echo "=== done mode=$MODE seed=$SEED ==="
