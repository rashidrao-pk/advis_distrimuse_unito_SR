#!/usr/bin/env bash

#SBATCH --job-name=R2AD-SafetyZone
#SBATCH --partition=epito
#SBATCH --reservation=ADVIS-Training_R2AD
#SBATCH --nodelist=epito01
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r2ad_%j.out
#SBATCH --error=logs/r2ad_%j.err


set -euo pipefail

echo "=========================================="
echo "R2AD training job"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "=========================================="

# Project
cd /beegfs/home/mrashid/repos/DistriMuSe/advis_distrimuse_unito_SR

# Environment
source /beegfs/home/mrashid/pt_312/bin/activate

export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages/
export PYTHONUNBUFFERED=1

echo
echo "Python:"
which python
python --version

echo
echo "GPU:"
nvidia-smi

echo
echo "Starting training..."
echo

python -u scripts/train_r2ad.py \
    --config configs/cf_dataset_epito.yaml \
    --safety-area SafetyZone \
    --train-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv \
    --val-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/val.csv \
    --output results/r2ad/SafetyZone_fast_b32/model_r2ad.pt \
    --patch-size 224 \
    --samples-per-frame 6 \
    --epochs 200 \
    --batch-size 32 \
    --workers 8 \
    --device cuda \
    --precision bf16 \
    --augmentation-mode crop \
    --persistent-workers \
    --prefetch-factor 2 \
    --monitor-every 1000 \
    --save-cp-k 1


echo
echo "=========================================="
echo "Training finished: $(date)"
echo "=========================================="
