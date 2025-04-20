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

# DiffuScent Phase 1 Implementation

This document describes the implementation of Phase 1 improvements to the DiffuScent simulation as outlined in the project roadmap.

## Performance Optimizations

The following performance optimizations have been implemented:

1. **Reduced Mesh Resolution**
   - Changed from 50x40x25 to 25x20x12 cells
   - Provides approximately 8x speedup while maintaining adequate spatial resolution
   - Grid spacing is now 20cm in each dimension, sufficient for educational demonstration

2. **Increased Time Step**
   - Changed from 0.1s to 1.0s timestep
   - Verified against diffusion stability criteria (Δt ≤ Δx²/2D)
   - Yields 10x speedup in time integration

3. **Deferred Visualization**
   - Now only generates visualization at end of computation
   - Stores raw data during simulation to avoid visualization overhead
   - Separation of computation and visualization improves overall performance

4. **Parallel Processing**
   - Added optional parallel processing support
   - Implemented simple fallback to sequential processing when not available
   - Future implementation will include domain decomposition for better parallelization

## Physics Enhancements

The following physics enhancements have been implemented:

1. **Initial Momentum**
   - Added initial velocity vector (3 m/s at 30° downward angle)
   - Creates a realistic "jet" effect at release point
   - Gaussian spatial profile for smooth transition of velocity field

2. **Buoyancy Model**
   - Implemented temperature-dependent buoyancy based on gas density
   - Calculates buoyancy force from density difference between gas mixture and surrounding air
   - Updates vertical velocity component based on local gas concentration

3. **Room Air Circulation**
   - Added basic circular air current pattern in room
   - Configurable strength parameter (default 0.2 m/s)
   - Support for different pattern types (basic, HVAC)

## Visualization Improvements

The following visualization improvements have been implemented:

1. **Logarithmic Concentration Scale**
   - Added logarithmic scale option for concentration visualization
   - Provides much better visibility of gas movement at low concentrations
   - Configurable via settings

2. **Detection Threshold Markers**
   - Added visual indicators for detection threshold boundaries
   - Red markers show where concentration equals detection threshold
   - Makes it easier to understand when detection will occur

3. **Enhanced Color Mapping**
   - Improved color scale with configurable colormap
   - Better contrast at low concentrations
   - Added detection markers with appropriate labels

## Gas Profile System

A comprehensive gas profile system has been implemented:

1. **Multiple Gas Profiles**
   - Standard: Balanced profile with moderate volume
   - Veggie Burger: Higher in hydrogen, lower in sulfur compounds
   - Taco Bell Banger: High volume with extra methane
   - Egg's Revenge: Sulfur-rich profile
   - Silent But Deadly: Small volume but high sulfur content

2. **Realistic Gas Properties**
   - Proper diffusion coefficients for each gas component
   - Temperature-dependent properties
   - Density calculations based on composition and ideal gas law

3. **Detection System**
   - Uses realistic odor detection thresholds
   - Multiple detector positions with named identifiers
   - Time-to-detection calculation and reporting

## Usage

The improved simulation can be run as follows:

```bash
# Run simulation with default settings
python app.py

# Run with specific gas profile
python gas_profiles.py --profile taco_bell
python app.py

# List available gas profiles
python gas_profiles.py --list

# Create visualizations from existing results
python app.py --visualize-only --results-file results/simulation_20250419_123456.pkl
```

## Next Steps

With Phase 1 complete, we will move on to Phase 2 enhancements:

1. Improve Transport Model
   - Implement more sophisticated advection terms
   - Add temperature-dependent buoyancy effects
   - Model impact of breathing on local air currents

2. Add Environmental Factors
   - HVAC effects (air vents, return registers)
   - Door/window effects on airflow
   - Temperature gradients in the room

3. Enhance Gas Properties
   - Multi-component gas diffusion
   - Temperature effects on diffusion coefficients
   - Concentration-dependent detection thresholds
