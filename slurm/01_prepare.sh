#!/bin/bash
#SBATCH --job-name=dmg-prepare
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
# Step 1: split the cell-by-cell tables into per-core tables (both blocks).
source slurm/env.sh
dmg-he prepare -c "$CONFIG"
