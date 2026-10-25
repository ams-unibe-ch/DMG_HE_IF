# Settings shared by all SLURM scripts - edit these two lines.
# Submit all jobs from the repository root (e.g. `sbatch slurm/02_tile.sh`).
export CONFIG="${CONFIG:-config.yaml}"                    # your config file
export VENV="${VENV:-$HOME/venvs/dmg-he}"                 # virtual environment with dmg-he installed

# Cluster-specific setup (modules, proxies, ...). Example for ETH Euler:
# module load stack/2024-06 gcc/12.2.0 python/3.11.6
# module load eth_proxy          # internet access for Hugging Face downloads

source "${VENV}/bin/activate"
mkdir -p logs
set -euo pipefail
