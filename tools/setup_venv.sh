#!/bin/bash

# Setup Virtual Environment for Java Dependency Analyzer
# This script creates a Python virtual environment and installs all required dependencies

# Configuration
VENV_DIR="venv"
REQUIREMENTS_FILE="../requirements.txt"

# Display banner
echo "================================================"
echo "Java Dependency Analyzer - Virtual Environment Setup"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Using Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
else
    echo "Virtual environment already exists in $VENV_DIR"
fi

# Determine activation script based on OS
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    # macOS or Linux
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
else
    echo "Error: Unsupported operating system: $OSTYPE"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$ACTIVATE_SCRIPT"

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

# Install requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies."
        exit 1
    fi
else
    echo "Error: Requirements file not found at $REQUIREMENTS_FILE"
    exit 1
fi

# Additional installs if needed
echo "Installing optional visualization dependencies..."
pip install graphviz networkx matplotlib

echo "================================================"
echo "Setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source $ACTIVATE_SCRIPT  # On macOS/Linux"
echo "  .\\$VENV_DIR\\Scripts\\activate  # On Windows"
echo ""
echo "To run the dependency analyzer:"
echo "  python java_dependency_analyzer.py /path/to/java/project"
echo ""
echo "To deactivate the virtual environment when finished:"
echo "  deactivate"
echo "================================================"
