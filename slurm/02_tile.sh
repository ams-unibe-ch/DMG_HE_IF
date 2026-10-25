#!/bin/bash
#SBATCH --job-name=dmg-tile
#SBATCH --array=0-19
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Step 2: tile the cores. The cores are distributed over the array tasks;
# change --array freely (e.g. 0-0 for a single job).
source slurm/env.sh
dmg-he tile -c "$CONFIG" --shard slurm
