# DiffuScent Installation Guide

This guide provides detailed instructions for setting up the DiffuScent gas diffusion simulator.

## System Requirements

- Python 3.7 or higher
- 4GB RAM minimum (8GB recommended for larger simulations)
- Any modern operating system (Windows, macOS, or Linux)

## Step 1: Set up a Python Environment

It's recommended to use a virtual environment to avoid conflicts with other Python packages.

### Using venv (recommended)

```bash
# Create a virtual environment
python -m venv diffuscent-env

# Activate the environment
# On Windows:
diffuscent-env\Scripts\activate
# On macOS/Linux:
source diffuscent-env/bin/activate
```

### Using conda

```bash
# Create a conda environment
conda create -n diffuscent python=3.9

# Activate the environment
conda activate diffuscent
```

## Step 2: Install FiPy and Dependencies

FiPy is the core PDE solver used by DiffuScent. Install it along with other required packages:

```bash
# Install NumPy and other dependencies first
pip install numpy matplotlib pyyaml

# Install FiPy
pip install fipy
```

### Troubleshooting FiPy Installation

If you encounter issues installing FiPy, you may need to install additional dependencies:

#### On Ubuntu/Debian:

```bash
sudo apt-get install python3-dev gcc gmsh
```

#### On macOS (with Homebrew):

```bash
brew install gcc gmsh
```

#### On Windows:

Ensure you have a C++ compiler installed, such as the one provided with Visual Studio Build Tools.

For more detailed information, refer to the [official FiPy installation guide](https://www.ctcms.nist.gov/fipy/INSTALLATION.html).

## Step 3: Download DiffuScent

Clone or download the DiffuScent repository:

```bash
git clone https://github.com/your-username/diffuscent.git
cd diffuscent
```

## Step 4: Verify Installation

Run a simple test to verify that everything is working correctly:

```bash
# Run the basic diffusion model with default settings
python basic_diffusion_model.py

# You should see output indicating the simulation is running,
# and eventually some plots should be displayed
```

If the script runs without errors and shows visualizations, your installation is successful.

## Common Issues

### ImportError: No module named 'fipy'

Make sure you have activated your virtual environment and installed FiPy:

```bash
pip install fipy
```

### Solver Convergence Warnings

FiPy may display warnings about solver convergence. These are usually not critical and the simulation will continue. If you encounter persistent solver problems, try:

- Reducing the time step size in `config.yaml`
- Decreasing the mesh resolution
- Ensuring your diffusion coefficient is appropriate for your mesh size

### Visualization Problems

If plots don't appear or show unexpected results:

- Make sure Matplotlib is installed: `pip install matplotlib`
- Try using a different Matplotlib backend if necessary
- Check if your terminal supports GUI display

## Running with Custom Configuration

After verifying the installation, you can run simulations with custom parameters:

```bash
# Create a copy of the example config
cp config.yaml my_config.yaml

# Edit the configuration file
# nano my_config.yaml (or use any text editor)

# Run with your custom configuration
python basic_diffusion_model.py my_config.yaml
```

## Getting Help

If you encounter any issues not covered in this guide, please:

1. Check the README.md file for more information
2. Look for similar issues in the project repository
3. Contact the project maintainers
