#!/bin/bash

# Display header
echo "====================================="
echo "FICC Spread Analysis - Setup Script"
echo "====================================="

# Set virtual environment name
VENV_NAME="ficc_env"

# Create directory structure if it doesn't exist
mkdir -p ficc_ai/data
mkdir -p ficc_ai/models
mkdir -p ficc_ai/visualizations

# Create virtual environment
echo "Creating virtual environment: $VENV_NAME"
python3 -m venv $VENV_NAME

# Activate virtual environment
echo "Activating virtual environment"
source $VENV_NAME/bin/activate

# Install dependencies
echo "Installing dependencies from requirements.txt"
pip install -r ficc_ai/requirements.txt

echo "====================================="
echo "Setup complete!"
echo ""
echo "To activate the virtual environment manually:"
echo "source $VENV_NAME/bin/activate"
echo ""
echo "To run the FICC analysis application:"
echo "python ficc_ai/ficc_ai.py"
echo "====================================="
