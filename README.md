# DiffuScent 💨

**A scientifically grounded gas diffusion simulator that teaches fluid dynamics through the universally relatable scenario of fart detection.**

DiffuScent is an educational demonstration tool that makes real physics accessible and entertaining. It models how gas spreads through a room using analytical solutions to the diffusion equation, with an interface designed to be intuitive enough for kids and engaging enough that they'll accidentally learn Fick's Second Law while laughing.

## Why This Exists

Fluid dynamics is one of the most important and least accessible branches of physics. The math is intimidating, the simulations are complex, and the textbooks are dry. But every kid already has an intuitive understanding of gas diffusion — they just don't know it yet.

DiffuScent bridges that gap. By framing diffusion physics around a scenario kids find hilarious, it creates a context where they *want* to explore the science. What happens if the room is warmer? What if you open a window? Why does distance matter? These are real physics questions, and kids will ask them voluntarily when the answer determines whether a cartoon character gets busted.

## What It Teaches

- **Diffusion** — Gas molecules spread through random motion (Brownian motion / Fick's Law)
- **Temperature dependence** — Hotter molecules move faster, diffusion accelerates
- **Buoyancy** — Warm gas rises, dense gas sinks, mixture composition matters
- **Concentration gradients** — The further from the source, the lower the concentration
- **Detection thresholds** — Human noses can detect H₂S at 0.00047 ppm (one molecule per two billion)
- **Ventilation** — Air exchange dilutes gas concentration exponentially
- **Reading scientific visualizations** — 3D volume renders, 2D contour maps, log-scale time series

## Installation

```bash
# Clone the repository
git clone https://github.com/brockwebb/diffuscent.git
cd diffuscent

# Create a conda environment (recommended)
conda create -n diffuscent python=3.12 -y
conda activate diffuscent

# Install dependencies
pip install -r requirements.txt
```

Alternatively, without conda:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
streamlit run src/diffuscent_app.py
```

Opens at http://localhost:8501. Pick a gas profile, place your source and detector on the room grid, and hit **💨 Let It Rip!**

## How It Works

### The Interface

- **Room grid** — Click to place the gas source (💨) and detector (👃) in a top-down room view
- **Gas profiles** — Five presets with different volumes, compositions, and H₂S concentrations
- **Room settings** — Adjust dimensions, temperature, and ventilation
- **Results** — Verdict banner (SAFE or BUSTED), 3D gas cloud, 2D stink map, detection timeline

### The Physics

Gas concentration is modeled with the 3D Gaussian analytical solution to the diffusion equation:

```
C(x,y,z,t) = M / (4πDt)^(3/2) × exp(-r² / 4Dt)
```

The diffusion coefficient is temperature-adjusted: `D(T) = D_base × (T/T_ref)^1.75`

The model uses a turbulent effective diffusion coefficient (~1.76×10⁻³ m²/s) rather than the molecular value (~1.76×10⁻⁵ m²/s), because real indoor gas transport is dominated by turbulent mixing from convection currents, not pure molecular diffusion. This produces realistic room-scale spreading behavior.

Additional physics:
- **Buoyancy** — Vertical drift based on gas density vs. air density at body temperature (37°C) vs. room temperature
- **Ventilation decay** — Exponential dilution when a window is open, modeling air exchange
- **Detection** — H₂S concentration evaluated at the detector position against the human olfactory threshold

### Gas Profiles

| Profile | Volume | H₂S | Density | Character |
|---------|--------|------|---------|-----------|
| The Veggie Burger 🥦 | 0.3 L | 0.0005% | Light | Plant-powered, mostly methane from fiber fermentation |
| Taco Bell Banger 🌮 | 0.5 L | 0.003% | Neutral | Beans + cheese = sulfur from protein breakdown |
| Egg's Revenge 🥚 | 0.4 L | 0.01% | Heavy | Sulfur amino acids → the rotten egg chemical |
| Silent But Deadly 🤫 | 0.15 L | 0.03% | Heavy | Low volume, maximum H₂S concentration |
| The Master Blaster 💨 | 0.8 L | 0.005% | Light | High volume but mostly odorless methane |

### What It Doesn't Do

This is an educational demonstration, not a research-grade CFD simulation. The analytical Gaussian model doesn't account for wall reflections, furniture obstacles, turbulent eddies, or multi-component diffusion with different rates per gas species. For teaching the *concepts* of diffusion, buoyancy, and detection thresholds, it doesn't need to.

## Architecture

Single-file Streamlit application. No external configs, no databases, no build steps.

```
diffuscent/
├── README.md
├── LICENSE
├── requirements.txt          # streamlit, numpy, plotly
├── docs/                     # Background documentation
└── src/
    └── diffuscent_app.py     # The entire application
```

Dependencies: `streamlit`, `numpy`, `plotly`. That's it.

## Background

DiffuScent was originally built with research-grade tools (FiPy PDE solver, FluidDyn libraries, YAML configurations, CLI interfaces) that made installation painful and the codebase unnecessarily complex. The current version strips all of that away in favor of an analytical physics model and a single interactive web app — proving that educational tools should prioritize accessibility over technical sophistication.

## License

MIT — see [LICENSE](LICENSE) for details.
