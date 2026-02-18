# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run the App

```bash
streamlit run src/diffuscent_app.py
```

Requires: `pip install streamlit numpy plotly`

## Architecture

Single-file Streamlit app at `src/diffuscent_app.py` (~750 lines). No config files, no multi-module structure.

**Physics model** — 3D Gaussian analytical solution (no PDE solver):
```
C(x,y,z,t) = M / (4πDt)^(3/2) × exp(-r² / 4Dt)
```
- `D_BASE = 1.76e-3 m²/s` (turbulent indoor mixing, not molecular diffusion)
- Temperature scaling: `D(T) = D_BASE × (T/T_ref)^1.75`
- Buoyancy: cloud center drifts vertically based on density difference
- `M_SCALE = 1e7` — tuning factor so detection feels realistic

**Detection** — `H2S_THRESHOLD = 4.7e-7 kg/m³` (≈ 0.00047 ppm). Tuned so "Silent But Deadly" at 2m detects in ~30-60s.

**App structure** (all in `main()`):
1. Sidebar: gas profile radio, room sliders, position sliders, "Let It Rip!" button
2. Simulation stored in `st.session_state["results"]` — survives widget reruns
3. Main area: verdict banner → time slider → 3D `go.Volume` + 2D `go.Heatmap` → detection timeline → science corner

**Gas profiles** — 5 hardcoded dicts in `GAS_PROFILES`. Each has: `volume_liters`, `h2s_fraction`, `density_factor`, `farty_says`.

## Key Constraints

- **No FiPy, no SciPy, no matplotlib, no YAML** — numpy + plotly only
- Single file — do not split into modules
- Simulation must run in <2 seconds
- If you change `M_SCALE` or `D_BASE`, re-validate: SBD at 2m ≈ 30-60s detection

## Local-Only Directories (gitignored)

- `cc_tasks/` — task tracking files
- `handoffs/` — session handoff docs
- `archive/` — old FiPy-based code (moved here during v2 rebuild)
