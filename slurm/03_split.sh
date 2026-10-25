#!/bin/bash
#SBATCH --job-name=dmg-split
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.out
# Step 3: merge tile labels and define the cross-validation folds.
source slurm/env.sh
dmg-he split -c "$CONFIG"
