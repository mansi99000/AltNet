#!/bin/bash

# Setup script for RDE Sample Efficient project
# This script helps reproduce the exact environment used in this project

set -e  # Exit on any error

echo "🚀 Setting up RDE Sample Efficient project environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment from environment.yml
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate the environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RDE_atari_dmc_3_16

# Install additional pip packages if needed
echo "📋 Installing additional pip packages..."
pip install -r RDE_atari_dmc/requirements.txt

# Install AutoROM for Atari games
echo "🎮 Setting up AutoROM for Atari games..."
python -c "import ale_py; ale_py.importROM()" || echo "⚠️  AutoROM setup may require manual intervention"

# Verify installation
echo "✅ Verifying installation..."
python -c "
import torch
import gymnasium
import stable_baselines3
import dm_control
import ale_py
print('✅ All core packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'Gymnasium version: {gymnasium.__version__}')
print(f'Stable Baselines3 version: {stable_baselines3.__version__}')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate RDE_atari_dmc_3_16"
echo ""
echo "To run the project:"
echo "  cd RDE_atari_dmc && python train_atari.py"
echo "  cd RDE_minigrid && python train.py"
