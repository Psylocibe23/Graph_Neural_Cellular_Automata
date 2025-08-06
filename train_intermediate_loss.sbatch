#!/bin/bash
#SBATCH --job-name=nca_train_intermediate
#SBATCH --partition=GPU
#SBATCH --account=dssc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_intermediate_%j.out

# Print some cluster info for debugging
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# Load CUDA module
module load cuda/11.8

# Activate your Python 3.10 venv
source /orfeo/scratch/dssc/fspreafichi/Graph_Neural_Cellular_Automata/graph_nca_py310_env/bin/activate

# Create logs dir if not exists
mkdir -p logs
mkdir -p outputs

# Run training script
PYTHONPATH=src python src/training/train_intermediate_loss.py

# Print completion info
echo "Job finished at: $(date)"
