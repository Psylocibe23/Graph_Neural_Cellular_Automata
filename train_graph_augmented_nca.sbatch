#!/bin/bash
#SBATCH --job-name=graph_nca_train
#SBATCH --partition=GPU
#SBATCH --account=dssc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_graph_%j.out
#SBATCH --error=logs/train_graph_%j.err

# --- Cluster info (useful for debugging) ---
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# --- Project root on the cluster ---
WORKDIR="/orfeo/scratch/dssc/fspreafichi/Graph_Neural_Cellular_Automata"
cd "$WORKDIR" || { echo "Failed to cd to $WORKDIR"; exit 1; }

# --- Modules / environment ---
module load cuda/11.8

# Activate your Python 3.10 venv
source "$WORKDIR/graph_nca_py310_env/bin/activate"

# --- Housekeeping ---
mkdir -p logs
mkdir -p outputs

# Perf env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

# (optional) show GPU visibility
nvidia-smi -L || true

# --- Train: auto-resume from outputs/graphaug_nca/.../checkpoints ---
python --version
PYTHONPATH=src python src/training/train_graph_augmented_nca.py

echo "Job finished at: $(date)"
