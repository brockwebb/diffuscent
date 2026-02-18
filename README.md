# DiffuScent 💨

An interactive gas diffusion simulator that teaches kids about the science of fart physics. Powered by Farty, your friendly science mascot.

## What It Does

DiffuScent lets you simulate how a gas cloud spreads through a room using real physics:

- **Gaussian diffusion** — 3D analytical model based on Fick's Second Law
- **5 gas profiles** — Veggie Burger, Taco Bell Banger, Egg's Revenge, Silent But Deadly, The Master Blaster
- **Buoyancy effects** — warm gas rises, dense gas sinks
- **Temperature-dependent diffusion** — warmer rooms = faster spreading
- **Detection system** — will your colleague's nose detect the H₂S?

## Quick Start

```bash
pip install streamlit numpy plotly
streamlit run src/diffuscent_app.py
```

Then open http://localhost:8501 in your browser.

## Gas Profiles

| Profile | Volume | H₂S Fraction | Character |
|---------|--------|--------------|-----------|
| The Veggie Burger 🥦 | 0.3 L | 0.0005% | Plant-powered and proud |
| Taco Bell Banger 🌮 | 0.5 L | 0.003% | South-of-the-border symphony |
| Egg's Revenge 🥚 | 0.4 L | 0.01% | Sulfurous and unforgiving |
| Silent But Deadly 🤫 | 0.15 L | 0.03% | Small volume, maximum impact |
| The Master Blaster 💨 | 0.8 L | 0.005% | Maximum volume, crowd clearer |

## The Physics

Gas concentration is modeled with the 3D Gaussian analytical solution to the diffusion equation:

```
C(x,y,z,t) = M / (4πDt)^(3/2) × exp(-r² / 4Dt)
```

Where D is temperature-adjusted: `D(T) = D_ref × (T/T_ref)^1.75`

Detection occurs when H₂S concentration exceeds 0.00047 ppm — your nose is *extremely* sensitive!

## License

MIT
