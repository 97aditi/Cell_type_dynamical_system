#!/bin/bash

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda first."
    echo "Visit https://docs.conda.io/en/latest/miniconda.html for installation instructions."
    echo "After installing Conda, restart your terminal and run this script again."
    exit 1
fi

# Create and activate the Conda environment
conda env create -f ctds.yml
conda activate ctds

# Install SSM package
cd ssm
pip install -e . --no-build-isolation
cd ..

# Remind about MOSEK license
echo "================================================================"
echo "Don't forget to set up your MOSEK license!"
echo "Get a free academic license from https://docs.mosek.com/10.2/licensing/quickstart.html#i-don-t-have-a-license-file-yet"
echo "Save the license file as mosek.lic in your home directory:"
echo "- macOS: /Users/[username]/mosek/mosek.lic"
echo "- Linux: /home/[username]/mosek/mosek.lic"
echo "- Windows: C:\Users\[username]\mosek\mosek.lic"
echo "================================================================"
echo "Setup complete! You can now use the project."
