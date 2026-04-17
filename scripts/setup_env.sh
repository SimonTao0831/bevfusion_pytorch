#!/bin/bash

set -e
ENV_NAME="bevfusion_pytorch"

echo "Starting to configure the $ENV_NAME environment..."

export PYTHONPATH=""
export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook)"

# Check if the conda environment already exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Activating and continuing installation..."
else
    echo "Creating conda environment: $ENV_NAME (Python 3.10)..."
    # -y automatically answers 'yes' to the installation prompts
    conda create -y -n $ENV_NAME python=3.10
fi

# Activate the environment
echo "Activating environment..."
conda activate $ENV_NAME

# Install PyTorch
echo "Installing PyTorch (CUDA 12.6), please refer PyTorch official website for specific GPUs..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other core dependencies
echo "Installing other required libraries..."
python -m pip install nuscenes-devkit==1.2.0 \
            matplotlib==3.9.4 \
            spconv-cu126==2.3.8 \
            ipykernel==7.2.0 \
            ipywidgets==8.1.8 \

echo "Environment setup is complete!"
echo "Please run the following command in your terminal to activate and use it:"
echo "conda activate $ENV_NAME"
