#!/bin/bash
#SBATCH --job-name=graph_nca
#SBATCH --output=logs/train_graph_%j.out
#SBATCH --error=logs/train_graph_%j.err
#SBATCH --partition=GPU
#SBATCH --account=dssc
#SBATCH --gres=gpu:1            
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# --- (1) Setup paths ---
# Your project root on the cluster:
WORKDIR="$HOME/Graph_Neural_Cellular_Automata"    # <-- change if needed
cd "$WORKDIR" || exit 1

mkdir -p logs

# --- (2) Environment ---
# If you use modules, uncomment and adapt:
# module purge
# module load cuda/12.1
# module load anaconda/2024.02

# Activate your venv/conda env
source ~/.bashrc
# conda activate agnn_env           # if conda
source agnn_env/bin/activate        # if venv; adjust to your env

# Safety / performance knobs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

# Optional: pin to one GPU if the node has many
# export CUDA_VISIBLE_DEVICES=0

# --- (3) Run training ---
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
python --version

# Train graph-augmented NCA (auto-resume from outputs/graphaug_nca/.../checkpoints)
python src/training/train_graph_ncagraph.py

echo "Job finished at: $(date)"
