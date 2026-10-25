#!/bin/bash
#SBATCH --job-name=dmg-features
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Step 4: tile embeddings.  Usage: sbatch slurm/04_extract_features.sh <model>
#   <model> = retccl | ctranspath | remedis | uni | virchow | provgigapath
# UNI, Virchow and Prov-GigaPath need `export HF_TOKEN=...` (gated Hugging Face models).
source slurm/env.sh
MODEL="${1:?usage: sbatch slurm/04_extract_features.sh <model>}"
dmg-he extract-features -c "$CONFIG" -m "$MODEL" --shard slurm
