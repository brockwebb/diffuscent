# DiffuScent Setup Guide

This guide walks you through setting up and running the DiffuScent gas diffusion simulator.

## Directory Structure

Ensure your project has the following structure:

```
DiffuScent/
├── requirements.txt
├── SETUP_GUIDE.md
├── README.md
└── src/
    ├── __init__.py
    ├── app.py
    ├── configs/
    │   ├── __init__.py
    │   └── default.yaml
    ├── physics/
    │   ├── __init__.py
    │   ├── advection_diffusion.py
    │   └── gas_profiles.py
    └── visualization/
        ├── __init__.py
        └── visualization.py
```

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FiPy dependencies (if not automatically installed):
   ```bash
   pip install gmsh
   pip install pyamg
   ```

## Usage

### Running the Simulation

To run the simulation with default settings:

```bash
python src/app.py
```

### Command Line Options

```bash
# Run the simulation from the src directory
cd src
python app.py

# Run with a specific configuration file
python app.py --config configs/custom.yaml

# Specify an output directory
python app.py --output ../results/test1

# Only create visualizations from existing results
python app.py --visualize-only --results-file ../results/simulation_20250420_123456.pkl
```

### Working with Gas Profiles

The simulation includes several predefined gas profiles. To list them:

```bash
# From the src directory
python physics/gas_profiles.py --list
```

To generate a configuration file with a specific profile:

```bash
python physics/gas_profiles.py --profile taco_bell --output configs/taco.yaml
```

And then run the simulation with that configuration:

```bash
python app.py --config configs/taco.yaml
```

## Output

The simulation creates two types of output:

1. **Simulation results** (saved as pickle files in the `results` directory)
2. **Visualizations** (saved in the `results/visualizations/...` directory)

The visualizations include:
- 2D slice plots showing gas concentration at different planes
- 3D interactive visualizations (HTML files)
- Detection summary charts

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'fipy'**
   - Ensure you've installed all dependencies: `pip install -r requirements.txt`

2. **Missing modules within the project**
   - Make sure your directory structure matches the one above
   - Verify all `__init__.py` files are in place

3. **FiPy solver errors**
   - Try installing additional solvers: `pip install pyamg gmsh`
   - If using Windows, you might need the Visual C++ Build Tools

### Performance Tuning

If the simulation is running too slowly:

1. Reduce the mesh resolution in `config.yaml`
2. Increase the time step (ensure it's stable)
3. Reduce the total simulation time
4. Set `parallel_processing: true` in the config file

## Next Steps

After successfully running the simulation, you can:

1. Experiment with different gas profiles
2. Modify the room dimensions and detector positions
3. Adjust physics parameters like initial velocity and buoyancy
4. Create custom visualizations using the saved simulation results
