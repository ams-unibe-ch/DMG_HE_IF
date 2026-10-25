#!/bin/bash
#SBATCH --job-name=dmg-train
#SBATCH --array=0-109
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Step 6: lightGBM per marker and CV fold.  Usage: sbatch slurm/06_train.sh <model> [extra args]
# 22 markers x 5 folds = 110 combinations -> one per array task (0-109). Use fewer
# tasks to run several combinations per job. Extra arguments are passed on, e.g.
#   sbatch --array=0-21 slurm/06_train.sh uni --fold full     # model trained on the whole block
# Set classifier.n_jobs in the config to --cpus-per-task.
source slurm/env.sh
MODEL="${1:?usage: sbatch slurm/06_train.sh <model> [extra args]}"
shift
dmg-he train -c "$CONFIG" -m "$MODEL" --shard slurm "$@"
