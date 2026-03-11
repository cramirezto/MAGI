#!/bin/bash
# Setup script for MAGI protein visualization environment

set -e

echo "Setting up MAGI protein visualization environment..."

# Create and activate a virtual environment (optional)
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

echo ""
echo "Installation complete! Activate the environment with:"
echo "  source venv/bin/activate"
