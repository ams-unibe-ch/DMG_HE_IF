#!/bin/bash
#SBATCH --job-name=dmg-collect
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
# Step 5: merge per-core feature files.  Usage: sbatch slurm/05_collect_features.sh <model>
source slurm/env.sh
MODEL="${1:?usage: sbatch slurm/05_collect_features.sh <model>}"
dmg-he collect-features -c "$CONFIG" -m "$MODEL"
