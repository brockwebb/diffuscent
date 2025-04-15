# DiffuScent

A scientifically accurate gas diffusion simulator with an engaging, humorous interface that teaches fluid dynamics through the relatable scenario of fart detection. This simulator provides both entertainment and education about gas movement, diffusion rates, and detection thresholds.

## Project Overview

Ever wondered if it's safe to break wind in a confined space? Science has the answer! This simulator models gas diffusion in a room with realistic physics, letting you determine if your gaseous emission will be detected by others. Educational content is delivered through our friendly mascot, Farty.

## Core Technologies

This project leverages several powerful open-source libraries:

- **FluidDyn/FluidSim**: Core physics simulation engine, providing high-performance fluid dynamics calculations with Python interfaces and C++/Cython optimizations
- **Streamlit**: User interface framework for building interactive web applications
- **Plotly**: 3D visualization library for rendering gas concentration

We selected FluidDyn/FluidSim after thorough evaluation of several potential libraries (FiPy, FEniCS, OpenFOAM). FluidDyn offers the optimal balance of performance, ease of development, and educational focus for our specific needs. See `docs/library_selection.md` for details.

## Features

- **Multiple Gas Profiles**: Choose between different preset flatulence compositions (Veggie Burger, Taco Bell Banger, Egg's Revenge, Silent But Deadly)
- **Physics Models**: Two simulation options - quick diffusion model and advanced diffusion-advection model with buoyancy
- **Temperature Effects**: Adjust room temperature to see how it affects gas diffusion rates
- **3D Visualization**: Watch the gas cloud spread in an interactive 3D view with complementary 2D cross-sections
- **Detection Metrics**: Get a "SAFE" or "BUSTED" verdict with countdown to detection
- **Educational Content**: Learn real gas diffusion physics through humorous scenarios

## Installation

```bash
# Clone the repository
git clone https://github.com/username/DiffuScent.git
cd DiffuScent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/app.py
```

## Project Structure

```
DiffuScent/
│
├── README.md                 # Project description and setup instructions
├── LICENSE                   # Open source license (MIT)
├── .gitignore                # Standard Python/JS gitignore
├── requirements.txt          # Project dependencies
│
├── docs/
│   ├── physics.md            # Documentation of physics models and assumptions
│   ├── library_selection.md  # Library selection rationale
│   ├── user-guide.md         # How to use the simulator
│   └── future-ideas.md       # Potential future enhancements
│
├── src/
│   ├── physics/              # Physics simulation code using FluidDyn/FluidSim
│   │   ├── diffusion.py      # Basic diffusion model
│   │   ├── advection.py      # Advanced diffusion-advection model
│   │   ├── gas_profiles.py   # Gas composition definitions
│   │   └── detection.py      # Detection threshold calculations
│   │
│   ├── visualization/        # Visualization components
│   │   ├── renderer.py       # 3D/2D rendering logic
│   │   ├── colors.py         # Color mapping for concentrations
│   │   └── camera.py         # View management
│   │
│   ├── ui/                   # Streamlit-based user interface
│   │   ├── main_screen.py    # Main interface
│   │   ├── profile_selector.py # Fart profile selector
│   │   ├── room_setup.py     # Room dimension and position setup
│   │   └── mascot.py         # Farty mascot animations and dialog
│   │
│   └── app.py                # Main Streamlit application entry point
│
└── tests/
    ├── test_diffusion.py     # Tests for diffusion models
    ├── test_visualization.py # Tests for visualization
    └── test_detection.py     # Tests for detection calculations
```

## Development Roadmap

1. **Phase 1: Core Physics (2-3 weeks)**
   - Implement basic diffusion model with FluidDyn/FluidSim
   - Create gas profile definitions
   - Add temperature effects
   - Implement detection threshold logic

2. **Phase 2: Visualization (2-3 weeks)**
   - Develop 3D isometric view
   - Implement 2D cross-sections
   - Create concentration colormap
   - Setup camera controls

3. **Phase 3: User Interface (2-3 weeks)**
   - Develop profile selection screen
   - Create room setup interface
   - Design and animate Farty mascot
   - Implement simulation controls

4. **Phase 4: Integration and Testing (1-2 weeks)**
   - Connect physics with visualization
   - Implement time stepping
   - Add result calculations and display
   - Perform testing and optimization

## Future Directions

1. **Advanced Fluid Dynamics**
   - Full Navier-Stokes implementation
   - Turbulence modeling approaches
   - Computational requirements and optimization techniques

2. **Environmental Factors**
   - HVAC effects on gas distribution
   - Furniture and obstacle interactions
   - Room geometry beyond simple rectangles

3. **Enhanced User Experience**
   - Multiple detector positions
   - Custom gas profile creation
   - Video export and sharing capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The FluidDyn project for their excellent fluid dynamics libraries
- The Streamlit team for their interactive app framework
- Everyone who's ever been in an awkward social situation involving flatulence
