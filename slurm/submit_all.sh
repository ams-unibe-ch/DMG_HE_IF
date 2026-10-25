#!/bin/bash
# Submit the complete pipeline with job dependencies, from the repository root:
#   bash slurm/submit_all.sh uni virchow provgigapath ...
# Steps that are already complete are skipped automatically by dmg-he.
set -euo pipefail
MODELS=("${@:-uni}")
mkdir -p logs
sub() { sbatch --parsable "$@"; }

j1=$(sub slurm/01_prepare.sh)
j2=$(sub --dependency=afterok:$j1 slurm/02_tile.sh)
j3=$(sub --dependency=afterok:$j2 slurm/03_split.sh)
train_jobs=()
for m in "${MODELS[@]}"; do
    j4=$(sub --dependency=afterok:$j2 slurm/04_extract_features.sh "$m")
    j5=$(sub --dependency=afterok:$j3:$j4 slurm/05_collect_features.sh "$m")
    j6=$(sub --dependency=afterok:$j5 slurm/06_train.sh "$m")
    train_jobs+=("$j6")
    echo "$m: features $j4, collect $j5, train $j6"
done
dep=$(IFS=:; echo "${train_jobs[*]}")
j7=$(sub --dependency=afterok:$dep slurm/07_evaluate.sh "${MODELS[@]}")
echo "prepare $j1, tile $j2, split $j3, evaluate $j7"
