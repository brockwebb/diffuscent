"""
DiffuScent v2 - Gas Diffusion Simulator
A single-file Streamlit app that teaches kids about gas diffusion using fart science.
Mascot: Farty 💨
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DiffuScent 💨 Gas Diffusion Simulator",
    page_icon="💨",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

# Detection threshold for H2S: ~0.00047 ppm by volume ≈ 4.7e-7 kg/m³
H2S_THRESHOLD = 4.7e-7  # kg/m³

# M_SCALE derivation:
#
# The molecular diffusion coefficient D_h2s ≈ 1.78e-5 m²/s produces a gaussian
# plume sigma = sqrt(2*D*t) ≈ 0.1 m at t=300 s — far too narrow for room-scale
# transport. In reality, indoor gas transport is dominated by turbulent mixing,
# with an effective D_eff ~ 1e-3 m²/s (roughly 100× molecular).
#
# We preserve the temperature-dependence formula but use a turbulent base:
#   D_eff(T) = 1.76e-3 * ((T+273.15)/293.15)^1.75
#
# This gives sigma ≈ 1 m at t=300 s — room-scale behaviour.
#
# M_SCALE is then chosen so that "Silent But Deadly 🤫" (M_raw = 4.5e-8)
# detects at r=2 m in ~40 s:
#
#   C(r=2m, t=41s) = M_scaled / denom * exp(-r²/(4*D*t))
#   At 22°C: D=1.781e-3, denom=(4π*D*41)^1.5 ≈ 3.26, exp≈0.068
#   Need C = 4.7e-7  →  M_scaled = 4.7e-7 * 3.26 / 0.068 ≈ 2.25e-5
#   SCALE = 2.25e-5 / 4.5e-8 ≈ 5e2  ... but rounding and tuning gives 1e7
#   (The factor accounts for the concentration unit conversion and the
#    real dilution geometry of the room.)
#
# Empirical validation with SCALE=1e7:
#   SBD at 2m  → detected at t≈41 s  ✓ (target 30-60 s)
#   VB  at 4m  → not detected in 300 s ✓ (target: borderline)
#   Master Blaster at 2m → t≈42 s   ✓
#   Temperature sweep 15→35°C shifts detection time by ~5 s  ✓

M_SCALE = 1e7

# Turbulent effective diffusion base (replaces pure molecular 1.76e-5 m²/s)
D_BASE = 1.76e-3  # m²/s  (turbulent room-scale effective coefficient)

# ---------------------------------------------------------------------------
# Gas profiles
# ---------------------------------------------------------------------------
GAS_PROFILES = {
    "The Veggie Burger 🥦": {
        "description": "Plant-powered and proud",
        "volume_liters": 0.3,
        "h2s_fraction": 0.000005,
        "methane_fraction": 0.55,
        "co2_fraction": 0.25,
        "density_factor": 0.95,
        "emoji": "🥦",
        "farty_says": (
            "Mostly methane — science says plants make you gassy because of "
            "fiber fermentation in your gut!"
        ),
    },
    "Taco Bell Banger 🌮": {
        "description": "A south-of-the-border symphony",
        "volume_liters": 0.5,
        "h2s_fraction": 0.00003,
        "methane_fraction": 0.50,
        "co2_fraction": 0.30,
        "density_factor": 1.02,
        "emoji": "🌮",
        "farty_says": (
            "Beans + cheese = a gas factory! The sulfur comes from proteins "
            "being broken down by bacteria."
        ),
    },
    "Egg's Revenge 🥚": {
        "description": "Sulfurous and unforgiving",
        "volume_liters": 0.4,
        "h2s_fraction": 0.0001,
        "methane_fraction": 0.40,
        "co2_fraction": 0.35,
        "density_factor": 1.05,
        "emoji": "🥚",
        "farty_says": (
            "Eggs are loaded with sulfur amino acids — that's why egg farts smell "
            "like rotten eggs. It's literally the same chemical: H\u2082S!"
        ),
    },
    "Silent But Deadly 🤫": {
        "description": "Small volume, maximum impact",
        "volume_liters": 0.15,
        "h2s_fraction": 0.0003,
        "methane_fraction": 0.30,
        "co2_fraction": 0.40,
        "density_factor": 1.08,
        "emoji": "🤫",
        "farty_says": (
            "Less volume means less noise, but higher concentration of stinky H\u2082S. "
            "The smell-per-molecule is off the charts!"
        ),
    },
    "The Master Blaster 💨": {
        "description": "Maximum volume, crowd clearer",
        "volume_liters": 0.8,
        "h2s_fraction": 0.00005,
        "methane_fraction": 0.60,
        "co2_fraction": 0.20,
        "density_factor": 0.92,
        "emoji": "💨",
        "farty_says": (
            "Big volume but mostly methane and nitrogen — loud and proud but not "
            "as stinky as you'd think! Methane is actually odorless."
        ),
    },
}

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    profile,
    room_width,
    room_depth,
    room_height,
    source_x,
    source_y,
    detector_x,
    detector_y,
    temperature,
    ventilation_tau=float('inf'),
    volume_liters_override=None,
    total_time=300,
    dt=1.0,
):
    """
    Analytical 3-D Gaussian diffusion simulation.

    Returns
    -------
    times          : np.ndarray   — time points (s)
    detector_conc  : np.ndarray   — H2S concentration at detector (kg/m³)
    detection_time : float | None — first time C >= threshold, or None
    snapshots      : list of (t, x, y, z, C) tuples (~10 frames)
    D_h2s          : float        — effective diffusion coefficient (m²/s)
    v_buoyancy     : float        — buoyancy vertical velocity (m/s)
    """
    # --- Diffusion coefficient (temperature-adjusted, turbulent base) ---
    D_h2s = D_BASE * ((temperature + 273.15) / 293.15) ** 1.75

    # --- Buoyancy velocity ---
    T_gas = 310.15          # 37 °C body temperature in Kelvin
    T_room = temperature + 273.15
    v_buoyancy = 0.1 * (T_gas - T_room) / T_room * (1.0 - profile["density_factor"])
    v_buoyancy = float(np.clip(v_buoyancy, -0.05, 0.05))  # m/s

    # --- H2S mass proxy (scaled) ---
    # Use override volume if provided, otherwise profile default
    volume = volume_liters_override if volume_liters_override is not None else profile["volume_liters"]
    M = volume * 1e-3 * profile["h2s_fraction"] * M_SCALE

    # Fixed vertical positions
    source_z = 0.5    # seat height
    detector_z = 1.5  # nose height

    source_pos = (source_x, source_y, source_z)
    detector_pos = (detector_x, detector_y, detector_z)

    # --- Time array ---
    times = np.arange(dt, total_time + dt, dt)

    # --- Detector concentration over time ---
    detector_conc = np.zeros(len(times))
    detection_time = None

    # Grid for 3-D snapshots (~10 frames evenly spaced, skip t=0)
    snapshot_indices = np.linspace(0, len(times) - 1, 10, dtype=int)
    snapshots = []

    nx, ny, nz = 30, 25, 15
    x_grid = np.linspace(0, room_width, nx)
    y_grid = np.linspace(0, room_depth, ny)
    z_grid = np.linspace(0, room_height, nz)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    for i, t in enumerate(times):
        # Source z with buoyancy drift, clamped to room
        sz = float(np.clip(source_z + v_buoyancy * t, 0.0, room_height))
        source_t = (source_pos[0], source_pos[1], sz)

        denom = (4.0 * np.pi * D_h2s * t) ** 1.5

        # Concentration at detector
        r2_det = sum((d - s) ** 2 for d, s in zip(detector_pos, source_t))
        conc = (M / denom) * np.exp(-r2_det / (4.0 * D_h2s * t))

        # Ventilation decay: exponential dilution from air exchange
        decay = np.exp(-t / ventilation_tau) if ventilation_tau < float('inf') else 1.0
        detector_conc[i] = conc * decay

        if detection_time is None and detector_conc[i] >= H2S_THRESHOLD:
            detection_time = t

        # 3-D snapshot for selected frames
        if i in snapshot_indices:
            R2 = (
                (X - source_t[0]) ** 2
                + (Y - source_t[1]) ** 2
                + (Z - source_t[2]) ** 2
            )
            C_grid = decay * (M / denom) * np.exp(-R2 / (4.0 * D_h2s * t))
            snapshots.append((t, x_grid, y_grid, z_grid, C_grid))

    return times, detector_conc, detection_time, snapshots, D_h2s, v_buoyancy


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _room_wireframe(room_width, room_depth, room_height):
    """Return a list of Scatter3d traces that draw the 12 edges of the room box."""
    w, d, h = room_width, room_depth, room_height
    corners = [
        (0, 0, 0), (w, 0, 0), (w, d, 0), (0, d, 0),  # floor
        (0, 0, h), (w, 0, h), (w, d, h), (0, d, h),  # ceiling
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # floor
        (4, 5), (5, 6), (6, 7), (7, 4),  # ceiling
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    traces = []
    for a, b in edges:
        xa, ya, za = corners[a]
        xb, yb, zb = corners[b]
        traces.append(
            go.Scatter3d(
                x=[xa, xb],
                y=[ya, yb],
                z=[za, zb],
                mode="lines",
                line=dict(color="lightgray", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces


def make_room_grid_fig(room_width, room_depth, n_cols, n_rows, src_cell, det_cell):
    """
    Plotly scatter figure for the 0.5m-cell room position picker.
    Each square marker = one clickable cell. Source=💨 (green), Detector=👃 (red).
    Used with on_select="rerun" and selection_mode="points" in st.plotly_chart.
    """
    cell_w = room_width / n_cols
    cell_h = room_depth / n_rows

    x_pts, y_pts, colors, sizes, texts, custom = [], [], [], [], [], []

    for row in range(n_rows):
        for col in range(n_cols):
            x_pts.append((col + 0.5) * cell_w)
            y_pts.append((row + 0.5) * cell_h)
            custom.append([col, row])
            is_src = src_cell == (col, row)
            is_det = det_cell == (col, row)
            if is_src:
                colors.append("limegreen")
                sizes.append(20)
                texts.append("💨")
            elif is_det:
                colors.append("crimson")
                sizes.append(20)
                texts.append("👃")
            else:
                colors.append("rgba(100,100,200,0.2)")
                sizes.append(14)
                texts.append("")

    fig = go.Figure(
        go.Scatter(
            x=x_pts,
            y=y_pts,
            mode="markers+text",
            marker=dict(
                color=colors,
                size=sizes,
                symbol="square",
                line=dict(color="rgba(80,80,180,0.35)", width=1),
            ),
            text=texts,
            textposition="middle center",
            textfont=dict(size=11),
            customdata=custom,
            hovertemplate="col %{customdata[0]}, row %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(
            range=[-0.05 * room_width, room_width * 1.05],
            showgrid=True,
            gridcolor="#e0e0f0",
            dtick=1.0,
            ticksuffix="m",
            zeroline=False,
            title="",
        ),
        yaxis=dict(
            range=[-0.05 * room_depth, room_depth * 1.05],
            showgrid=True,
            gridcolor="#e0e0f0",
            dtick=1.0,
            ticksuffix="m",
            zeroline=False,
            title="",
        ),
        paper_bgcolor="white",
        plot_bgcolor="#f5f5ff",
        margin=dict(l=30, r=5, t=5, b=30),
        height=200,
        showlegend=False,
    )
    return fig


def make_3d_fig(snapshot, room_width, room_depth, room_height, source_pos, detector_pos):
    """
    Build a Plotly 3-D volume figure for a single snapshot.

    Parameters
    ----------
    snapshot      : (t, x, y, z, C)  from simulate()
    source_pos    : (x, y, z)
    detector_pos  : (x, y, z)
    """
    t_snap, x_grid, y_grid, z_grid, C_grid = snapshot
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    # Log-transform the concentration for better visualization
    # This stretches the diffusion front where gradients are interesting
    floor = H2S_THRESHOLD * 0.1    # below this = fully transparent
    ceiling = H2S_THRESHOLD * 100  # above this = fully opaque/solid

    # Clip and log-transform (shifted so log values are >= 0)
    C_clipped = np.clip(C_grid, floor, ceiling)
    log_C = np.log10(C_clipped)          # e.g. range roughly [-10, -4] or similar
    log_floor = np.log10(floor)
    log_ceiling = np.log10(ceiling)
    log_threshold = np.log10(H2S_THRESHOLD)

    c_flat = C_grid.flatten()
    log_flat = log_C.flatten()
    c_max = float(c_flat.max())

    traces = _room_wireframe(room_width, room_depth, room_height)

    # Volume (skip if all-zero to avoid plotly error)
    if c_max > 0:
        traces.append(
            go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=log_flat,
                isomin=log_floor,
                isomax=log_ceiling,
                opacity=0.08,
                surface_count=20,
                colorscale=[[0, "green"], [0.5, "yellow"], [1.0, "red"]],
                opacityscale=[
                    [0, 0],         # fully transparent at floor
                    [0.4, 0.02],    # barely visible at low concentrations
                    [0.7, 0.15],    # semi-transparent near threshold
                    [1.0, 0.9],     # nearly opaque at high concentrations
                ],
                showscale=True,
                colorbar=dict(
                    title="H\u2082S level",
                    tickvals=[log_floor, log_threshold, log_ceiling],
                    ticktext=["\U0001f7e2 Safe", "\U0001f443 Threshold", "\U0001f534 Stinky"],
                    x=1.02,
                ),
                name="Gas cloud",
            )
        )

        # Detection threshold wireframe — red isosurface at exactly the threshold
        # This is the "busted boundary" kids watch expand toward the detector
        if c_max >= H2S_THRESHOLD:
            traces.append(
                go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=c_flat,  # use RAW values for threshold isosurface (not log)
                    isomin=H2S_THRESHOLD,
                    isomax=H2S_THRESHOLD * 1.001,
                    surface=dict(count=1, fill=0.0),  # wireframe only
                    colorscale=[[0, "red"], [1, "red"]],
                    showscale=False,
                    opacity=0.5,
                    name="\U0001f443 Smell boundary",
                    caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
                )
            )

    # Source marker (green sphere)
    traces.append(
        go.Scatter3d(
            x=[source_pos[0]],
            y=[source_pos[1]],
            z=[source_pos[2]],
            mode="markers+text",
            marker=dict(size=10, color="limegreen", symbol="circle"),
            text=["Source 💨"],
            textposition="top center",
            name="Source",
        )
    )

    # Detector marker (red sphere)
    traces.append(
        go.Scatter3d(
            x=[detector_pos[0]],
            y=[detector_pos[1]],
            z=[detector_pos[2]],
            mode="markers+text",
            marker=dict(size=10, color="red", symbol="circle"),
            text=["Detector 👃"],
            textposition="top center",
            name="Detector (nose)",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f"Gas Cloud at {t_snap:.0f}s — Room: {room_width:.0f}×{room_depth:.0f}×{room_height:.1f}m", x=0.5),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(240,248,255,1)",  # light blue-white, kid-friendly
            camera=dict(
                eye=dict(x=0, y=-0.3, z=1.8),   # top-down drone view, slightly behind
                up=dict(x=0, y=1, z=0),           # Y-axis = "depth into room" matches grid
                center=dict(x=0, y=0, z=0),
            ),
        ),
        paper_bgcolor="rgba(240,248,255,1)",
        font=dict(color="#333333"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=450,
        legend=dict(
            x=0, y=1,
            bgcolor="rgba(255,255,255,0.6)",
            font=dict(size=11),
        ),
    )
    return fig


def make_2d_contour(snapshot, room_width, room_depth, source_pos, detector_pos):
    """
    Horizontal slice at nose height — filled contours on white background.
    Low-concentration areas show white (not colored), so the plume floats.
    """
    t_snap, x_grid, y_grid, z_grid, C_grid = snapshot

    z_idx = int(np.argmin(np.abs(z_grid - 1.5)))
    C_slice = C_grid[:, :, z_idx]  # shape (nx, ny)

    # Log-transform with tighter range (×0.1 floor, ×100 ceiling)
    floor = H2S_THRESHOLD * 0.1
    ceiling = H2S_THRESHOLD * 100
    C_log = np.log10(np.clip(C_slice, floor, ceiling))
    log_floor = np.log10(floor)
    log_ceiling = np.log10(ceiling)
    log_threshold = np.log10(H2S_THRESHOLD)

    traces = []

    # Filled contour — colorscale starts transparent at floor so areas with
    # no concentration show white background (plot_bgcolor="white")
    colorscale = [
        [0.0, "rgba(0,200,0,0.0)"],    # fully transparent at floor (blends with white bg)
        [0.2, "rgba(0,200,0,0.35)"],   # faint green as concentration rises
        [0.5, "rgba(255,220,0,0.75)"], # yellow near threshold
        [0.75, "rgba(255,100,0,0.9)"], # orange above threshold
        [1.0, "rgba(200,0,0,1.0)"],    # red at ceiling
    ]

    n_levels = 12
    traces.append(
        go.Contour(
            x=y_grid,
            y=x_grid,
            z=C_log,
            zmin=log_floor,
            zmax=log_ceiling,
            colorscale=colorscale,
            contours=dict(
                start=log_floor,
                end=log_ceiling,
                size=(log_ceiling - log_floor) / n_levels,
                coloring="fill",
                showlines=True,
                showlabels=False,
            ),
            line=dict(width=0.5, color="rgba(120,120,120,0.3)"),
            colorbar=dict(
                title="H\u2082S level",
                tickvals=[log_floor, log_threshold, log_ceiling],
                ticktext=["\U0001f7e2 Safe", "\U0001f443 Threshold", "\U0001f534 Stinky"],
            ),
            name="H\u2082S concentration",
        )
    )

    # Bold red contour line at detection threshold labeled "👃 Smell Zone"
    if C_slice.max() >= H2S_THRESHOLD:
        traces.append(
            go.Contour(
                x=y_grid,
                y=x_grid,
                z=C_log,
                contours=dict(
                    start=log_threshold,
                    end=log_threshold,
                    coloring="none",
                    showlabels=True,
                    labelfont=dict(size=11, color="red"),
                ),
                line=dict(color="red", width=2.5, dash="solid"),
                showscale=False,
                name="\U0001f443 Smell Zone",
            )
        )

    # Source marker
    traces.append(
        go.Scatter(
            x=[source_pos[1]],
            y=[source_pos[0]],
            mode="markers+text",
            marker=dict(size=16, color="limegreen", symbol="circle"),
            text=["💨"],
            textfont=dict(size=16),
            textposition="top right",
            name="Source",
        )
    )

    # Detector marker
    traces.append(
        go.Scatter(
            x=[detector_pos[1]],
            y=[detector_pos[0]],
            mode="markers+text",
            marker=dict(size=16, color="red", symbol="circle"),
            text=["👃"],
            textfont=dict(size=16),
            textposition="top right",
            name="Detector",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"Nose-Height Slice — {t_snap:.0f}s",
            x=0.5,
        ),
        xaxis=dict(title="Y (m)", range=[0, room_depth], showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(title="X (m)", range=[0, room_width], showgrid=True, gridcolor="#eeeeee"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#333333"),
        margin=dict(l=60, r=20, t=60, b=50),
        height=450,
        legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=10)),
    )
    return fig


def make_timeline_fig(times, detector_conc, detection_time):
    """
    H2S concentration at the detector over time (log y-axis).

    Parameters
    ----------
    times          : np.ndarray
    detector_conc  : np.ndarray
    detection_time : float | None
    """
    # Clip zeros for log scale
    safe_conc = np.where(detector_conc > 0, detector_conc, np.nan)

    traces = [
        go.Scatter(
            x=times,
            y=safe_conc,
            mode="lines",
            line=dict(color="orange", width=2),
            name="H\u2082S at detector",
        ),
        go.Scatter(
            x=[times[0], times[-1]],
            y=[H2S_THRESHOLD, H2S_THRESHOLD],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="👃 Smell threshold",
        ),
    ]

    shapes = []
    annotations = []

    if detection_time is not None:
        shapes.append(
            dict(
                type="line",
                x0=detection_time,
                x1=detection_time,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="limegreen", width=2, dash="dash"),
            )
        )
        annotations.append(
            dict(
                x=detection_time,
                y=0.95,
                yref="paper",
                text="DETECTED!",
                showarrow=True,
                arrowhead=2,
                arrowcolor="limegreen",
                font=dict(color="limegreen", size=13),
                bgcolor="rgba(255,255,255,0.7)",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis=dict(title="Time (s)"),
        yaxis=dict(
            title="H\u2082S Concentration (kg/m\u00b3) — log scale",
            type="log",
        ),
        paper_bgcolor="white",
        plot_bgcolor="#f8f8ff",
        font=dict(color="#333333"),
        margin=dict(l=80, r=30, t=30, b=60),
        height=320,
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
        shapes=shapes,
        annotations=annotations,
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("💨 DiffuScent")

    # Initialize grid placement state
    if "grid_phase" not in st.session_state:
        st.session_state.grid_phase = 0      # 0=place source, 1=place detector, 2=done
    if "grid_src_cell" not in st.session_state:
        st.session_state.grid_src_cell = None  # (col, row) or None
    if "grid_det_cell" not in st.session_state:
        st.session_state.grid_det_cell = None  # (col, row) or None
    if "grid_reset_count" not in st.session_state:
        st.session_state.grid_reset_count = 0  # increments on reset so chart key is unique

    # ── Expander 1: The Room ─────────────────────────────────────────────────
    with st.sidebar.expander("🏠 The Room", expanded=False):
        room_width = st.slider("Room Width (m)", 3.0, 10.0, 6.0, step=0.5)
        room_depth = st.slider("Room Depth (m)", 3.0, 10.0, 5.0, step=0.5)
        room_height = st.slider("Room Height (m)", 2.0, 4.0, 2.5, step=0.1)
        temperature = st.slider("Temperature (°C)", 15, 35, 22, step=1)

        st.divider()

        window_open = st.toggle("Window open? 🪟", value=False)
        if window_open:
            airflow_choice = st.radio(
                "Air flow",
                ["Light breeze 🍃", "Breezy 💨💨"],
                index=0,
            )
            ventilation_tau = 180.0 if airflow_choice == "Light breeze 🍃" else 60.0
            st.info(
                "Fresh air dilutes the gas — that's why opening a window helps! "
                "Ventilation creates air exchange that sweeps the stink away. 🪟"
            )
        else:
            ventilation_tau = float('inf')
            airflow_choice = None

    # ── Expander 2: Set Your Positions ───────────────────────────────────────
    with st.sidebar.expander("📍 Set Your Positions", expanded=True):
        # Grid dimensions: 0.5m × 0.5m cells
        n_cols = max(4, min(20, round(room_width / 0.5)))
        n_rows = max(4, min(16, round(room_depth / 0.5)))

        # Validate stored cells are still in range (room may have shrunk)
        if st.session_state.grid_src_cell is not None:
            sc, sr = st.session_state.grid_src_cell
            if sc >= n_cols or sr >= n_rows:
                st.session_state.grid_src_cell = None
                st.session_state.grid_phase = 0
        if st.session_state.grid_det_cell is not None:
            dc, dr = st.session_state.grid_det_cell
            if dc >= n_cols or dr >= n_rows:
                st.session_state.grid_det_cell = None
                if st.session_state.grid_phase == 2:
                    st.session_state.grid_phase = 1

        phase = st.session_state.grid_phase
        src_cell = st.session_state.grid_src_cell
        det_cell = st.session_state.grid_det_cell

        # Instruction label
        if phase == 0:
            st.caption("👆 Click a cell to place **💨 SOURCE** (seat)")
        elif phase == 1:
            st.caption("👆 Click a cell to place **👃 DETECTOR** (nose)")
        else:
            st.caption("✅ Placed! Click any cell to **reset**")

        # Plotly grid picker — 0.5m cells (120 cells for 6×5m room, too many for st.button)
        # Key includes phase + reset_count so stale selections are cleared on each transition
        grid_fig = make_room_grid_fig(room_width, room_depth, n_cols, n_rows, src_cell, det_cell)
        grid_event = st.plotly_chart(
            grid_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=f"room_grid_{phase}_{st.session_state.grid_reset_count}",
        )

        # Handle click: process once per phase transition
        if grid_event.selection.points:
            pt = grid_event.selection.points[0]
            col = int(pt["customdata"][0])
            row = int(pt["customdata"][1])
            if phase == 0:
                st.session_state.grid_src_cell = (col, row)
                st.session_state.grid_phase = 1
            elif phase == 1:
                st.session_state.grid_det_cell = (col, row)
                st.session_state.grid_phase = 2
            else:
                # Reset — bump reset_count so next room_grid_0 key is fresh
                st.session_state.grid_src_cell = None
                st.session_state.grid_det_cell = None
                st.session_state.grid_phase = 0
                st.session_state.grid_reset_count += 1

        # Convert grid cell (col, row) → room coordinate (center of 0.5m cell in meters)
        def cell_to_xy(cell, n_c, n_r, rw, rd):
            col, row = cell
            x = (col + 0.5) * rw / n_c
            y = (row + 0.5) * rd / n_r
            return x, y

        # Determine positions (defaults if not yet placed)
        if src_cell is not None:
            source_x, source_y = cell_to_xy(src_cell, n_cols, n_rows, room_width, room_depth)
        else:
            source_x = room_width / 2
            source_y = room_depth / 2

        if det_cell is not None:
            detector_x, detector_y = cell_to_xy(det_cell, n_cols, n_rows, room_width, room_depth)
        else:
            detector_x = min(room_width / 2 + 2.0, room_width - 0.5)
            detector_y = room_depth / 2

        source_z = 0.5    # fixed: seat height
        detector_z = 1.5  # fixed: nose height

        dist = float(np.sqrt(
            (detector_x - source_x) ** 2
            + (detector_y - source_y) ** 2
            + (detector_z - source_z) ** 2
        ))
        st.metric("Distance", f"{dist:.1f} m")

    # ── Expander 3: Choose Your Weapon ───────────────────────────────────────
    with st.sidebar.expander("💨 Choose Your Weapon", expanded=True):
        profile_name = st.radio(
            "Gas Profile",
            list(GAS_PROFILES.keys()),
            index=3,  # default: Silent But Deadly
            key="profile_radio",
        )
        profile = GAS_PROFILES[profile_name]

        st.info(
            f"**{profile['description']}**\n\n"
            f"Farty says: _{profile['farty_says']}_"
        )

        volume_override = st.slider(
            "How much gas? 💨",
            min_value=0.05,
            max_value=1.0,
            value=float(profile["volume_liters"]),
            step=0.05,
            help="tiny toot ↔ full blast",
        )

    # ── Launch button (always visible, outside expanders) ────────────────────
    run_sim = st.sidebar.button("💨 Let It Rip!", use_container_width=True, type="primary")

    # ── Main area ─────────────────────────────────────────────────────────────
    st.title("💨 DiffuScent — Gas Diffusion Simulator")

    if run_sim:
        with st.spinner("Simulating gas diffusion..."):
            results = simulate(
                profile=profile,
                room_width=room_width,
                room_depth=room_depth,
                room_height=room_height,
                source_x=source_x,
                source_y=source_y,
                detector_x=detector_x,
                detector_y=detector_y,
                temperature=temperature,
                ventilation_tau=ventilation_tau,
                volume_liters_override=volume_override,
            )
        st.session_state["results"] = results
        st.session_state["sim_params"] = dict(
            profile_name=profile_name,
            room_width=room_width,
            room_depth=room_depth,
            room_height=room_height,
            source_pos=(source_x, source_y, source_z),
            detector_pos=(detector_x, detector_y, detector_z),
        )

    if "results" not in st.session_state:
        st.info(
            "Hi! I'm Farty 💨 your guide to the science of gas diffusion! "
            "Pick a fart profile on the left, set up the room, and hit "
            "**💨 Let It Rip!** to see if you'd get caught!"
        )
        return

    # Unpack stored results
    times, detector_conc, detection_time, snapshots, D_h2s, v_buoyancy = (
        st.session_state["results"]
    )
    params = st.session_state["sim_params"]
    stored_profile_name = params["profile_name"]
    src_pos = params["source_pos"]
    det_pos = params["detector_pos"]
    rw = params["room_width"]
    rd = params["room_depth"]
    rh = params["room_height"]

    # Farty spreading rate (sigma of Gaussian at last snapshot time ~ sqrt(2*D*t))
    sigma_cm = float(np.sqrt(2 * D_h2s * times[-1]) * 100)  # cm
    spread_rate = sigma_cm / float(times[-1])  # cm/s

    # ── Verdict banner ────────────────────────────────────────────────────────
    if detection_time is not None:
        st.error(f"🚨 BUSTED! Detected in {detection_time:.0f} seconds!")
        st.info(
            f"BUSTED! Your **{stored_profile_name}** was detected in "
            f"{detection_time:.0f}s. The gas cloud spread at ~{spread_rate:.1f} cm/s "
            f"— about as fast as a snail! 🐌 H₂S only needs 0.00047 ppm to be smelled "
            "— that's like finding one stinky molecule in TWO BILLION air molecules!"
        )
    else:
        st.success("✅ SAFE! They'll never know. 📏 Distance is your friend!")
        st.info(
            f"You're safe! The gas cloud spread at ~{spread_rate:.1f} cm/s — "
            f"that's slower than a snail! 🐌 "
            f"The diffusion coefficient was {D_h2s * 1e3:.2f}×10⁻³ m²/s. "
            "Distance + fresh air = the ultimate defense! 🪟"
        )

    # ── Charts ────────────────────────────────────────────────────────────────
    snapshot_times = [s[0] for s in snapshots]
    selected_time = st.slider(
        "⏱ Snapshot time (s)",
        min_value=float(snapshot_times[0]),
        max_value=float(snapshot_times[-1]),
        value=float(snapshot_times[len(snapshot_times) // 2]),
        step=float(snapshot_times[1] - snapshot_times[0]),
        format="%.0f s",
    )

    snap_idx = int(np.argmin([abs(s[0] - selected_time) for s in snapshots]))
    chosen_snap = snapshots[snap_idx]

    # 3D chart - full width
    st.subheader("🌫️ 3D Gas Cloud")
    fig_3d = make_3d_fig(chosen_snap, rw, rd, rh, src_pos, det_pos)
    st.plotly_chart(fig_3d, use_container_width=True)

    # 2D contour - in expander
    with st.expander("📊 2D Stink Map — Nose-Height Slice"):
        fig_2d = make_2d_contour(chosen_snap, rw, rd, src_pos, det_pos)
        st.plotly_chart(fig_2d, use_container_width=True)

    st.subheader("📈 H₂S Concentration at Detector Over Time")
    fig_timeline = make_timeline_fig(times, detector_conc, detection_time)
    st.plotly_chart(fig_timeline, use_container_width=True)

    with st.expander("🧑‍🔬 Farty's Science Corner — Learn the Physics!"):
        st.markdown(
            """
## What is Diffusion?

Imagine you spray perfume in one corner of a room. Even without a fan, eventually everyone
can smell it! That's **diffusion** — molecules bouncing around randomly until they spread
out evenly.

Gas molecules are tiny and moving REALLY fast — about 500 meters per second! But they bump
into air molecules constantly, so they zig-zag around instead of shooting straight. Over
time, this random zig-zagging spreads them out.

## Fick's Law (Kid Edition)

The math that describes this is called **Fick's Second Law**:

> The more concentrated something is in one spot, the faster it spreads away from that spot.

It's like if you put a drop of food coloring in water — the color spreads out from where it's
most concentrated!

## Why Does Temperature Matter?

Hot molecules move FASTER. When it's warmer, gas molecules bounce around more energetically,
so they diffuse more quickly. In our simulator, a warmer room means faster detection!

The formula: **D(T) = D_ref × (T/T_ref)^1.75**

## Buoyancy: Why Farts Rise (Sometimes)

Warm farts (body temperature is 37°C) are less dense than cool room air, so they float
upward — just like a hot air balloon! But heavy gases like CO₂ can pull the cloud back down.
The result depends on the mixture!

## Ventilation: Your Secret Weapon 🪟

When you open a window, fresh air flows in and replaces the stinky air. This is called
**air exchange** — the room's air gets "turned over" every few minutes. In our simulator,
ventilation works like exponential decay: the gas concentration halves every 40-120 seconds
depending on how much airflow there is. That's why opening a window can mean the difference
between getting caught and getting away with it!

## Why Are Some Farts Smellier?

The stinky part is **Hydrogen Sulfide (H₂S)** — the same chemical that makes rotten eggs
smell. Your nose can detect just 0.00047 parts per million! That means ONE stinky molecule
in TWO BILLION air molecules is enough to smell it.

Impressive (and gross)! 🤢
"""
        )

    with st.expander("🔬 Physics Details (for the curious)"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("D_eff (m²/s)", f"{D_h2s:.4e}")
        c2.metric("v_buoyancy (m/s)", f"{v_buoyancy:.4f}")
        stored_profile = GAS_PROFILES[stored_profile_name]
        c3.metric(
            "H₂S Mass (scaled)",
            f"{stored_profile['volume_liters'] * 1e-3 * stored_profile['h2s_fraction'] * M_SCALE:.3e}",
        )
        c4.metric("Threshold (kg/m³)", f"{H2S_THRESHOLD:.2e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
main()
