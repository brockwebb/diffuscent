# DiffuScent - Gas Diffusion Simulator

DiffuScent is a scientifically accurate gas diffusion simulator that models how flatulence spreads in a room environment. The project provides an engaging way to understand fluid dynamics and gas diffusion through a relatable scenario.

## Project Overview

This simulator uses the finite volume method to model gas diffusion in a room, allowing users to:

- Simulate gas dispersion in a 3D space
- Visualize concentration changes over time
- Determine detection times at different locations
- Experiment with different gas properties and room configurations

## Scientific Background

### Diffusion Physics

The simulation is based on Fick's Second Law of diffusion:

$$\frac{\partial C}{\partial t} = D \nabla^2 C$$

Where:
- $C$ is the concentration of gas (in parts per million, ppm)
- $t$ is time (in seconds)
- $D$ is the diffusion coefficient (in m²/s)
- $\nabla^2$ is the Laplacian operator (second spatial derivative)

This partial differential equation describes how concentration changes over time due to diffusion. The diffusion coefficient $D$ is a property of the specific gas and determines how quickly it spreads.

### Gas Properties

Different gas profiles in the simulator are based on real properties:

| Gas Component | Diffusion Coefficient (m²/s) | Detection Threshold (ppm) |
|---------------|------------------------------|---------------------------|
| Methane       | 1.76 × 10⁻⁵                  | ~5000 (odorless)         |
| Hydrogen Sulfide | 1.76 × 10⁻⁵              | ~0.5 (rotten egg smell)  |
| Carbon Dioxide | 1.42 × 10⁻⁵                | ~5000 (odorless)         |

Flatulence typically contains a mixture of these gases, with the sulfur-containing compounds (like hydrogen sulfide) responsible for the characteristic odor despite being present in small amounts.

## Installation

### Prerequisites

- Python 3.7 or higher
- FiPy (the finite volume PDE solver)
- NumPy
- Matplotlib
- PyYAML

### Install Dependencies

```bash
pip install fipy numpy matplotlib pyyaml
```

Note: FiPy may require additional dependencies depending on your system. See the [FiPy documentation](https://www.ctcms.nist.gov/fipy/INSTALLATION.html) for more details.

## Usage

### Basic Usage

Run the model with default settings:

```bash
python basic_diffusion_model.py
```

Run with a custom configuration file:

```bash
python basic_diffusion_model.py config.yaml
```

### Configuration

All simulation parameters can be customized through a YAML configuration file. See `config.yaml` for a complete example with comments explaining each parameter.

Key parameters include:
- Room dimensions
- Gas properties (diffusion coefficient, initial volume)
- Simulation time and time step
- Visualization settings

## Understanding the Output

The simulation produces several outputs:

1. **Console Output**: Shows simulation progress and detection events
2. **2D Slice Visualizations**: Shows gas concentration at nose height (1.6m)
3. **Concentration vs. Time Plot**: Shows how maximum concentration changes over time

The gas concentration is measured in parts per million (ppm), which represents the number of gas molecules per million air molecules.

## Model Limitations

This simplified model has several limitations:

1. It only considers diffusion, not advection (air currents)
2. It assumes constant temperature and pressure
3. It doesn't account for reactions between gases
4. It uses a simple detection threshold rather than a more complex olfactory model

Future versions may address these limitations for more realistic simulations.

## Contributing

Contributions to improve the model are welcome! Key areas for enhancement include:

- Adding advection to model air currents
- Implementing temperature effects on diffusion
- Adding more realistic gas profiles
- Improving visualization options

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The FiPy team at NIST for their excellent finite volume PDE solver
- The FluidDyn project for inspiration and reference implementations
