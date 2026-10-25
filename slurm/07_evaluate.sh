#!/bin/bash
#SBATCH --job-name=dmg-evaluate
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
# Step 7: summary tables and plots.  Usage: sbatch slurm/07_evaluate.sh <model> [<model> ...]
source slurm/env.sh
dmg-he evaluate -c "$CONFIG" -m "${@:?usage: sbatch slurm/07_evaluate.sh <model> [...]}"
