"""
DiffuScent - A scientifically accurate gas diffusion simulator

This is the main entry point for the DiffuScent Streamlit application.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

# Import our modules
from src.physics.diffusion import DiffusionModel
from src.physics.advection import AdvectionDiffusionModel
from src.physics.gas_profiles import get_gas_profile, GAS_PROFILES
from src.physics.detection import GasDetector
from src.visualization.renderer import MatplotlibRenderer, PlotlyRenderer


# Set page configuration
st.set_page_config(
    page_title="DiffuScent - Gas Diffusion Simulator",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_profile_image(profile_name):
    """
    Load an image for a gas profile.
    This is a placeholder function that would load actual images in the final version.
    """
    # In a real implementation, we would load actual images
    # For now, we'll just return different colors for different profiles
    colors = {
        "taco_bell_banger": "#FF9933",  # Orange
        "veggie_burger": "#00CC66",     # Green
        "eggs_revenge": "#CCCC00",      # Yellow
        "silent_but_deadly": "#990033"   # Dark red
    }
    
    profile_key = profile_name.lower().replace(" ", "_")
    color = colors.get(profile_key, "#0099CC")  # Default blue
    
    # Create a colored SVG
    return f"""
    <svg width="80" height="80" xmlns="http://www.w3.org/2000/svg">
        <rect width="80" height="80" rx="10" ry="10" fill="{color}" />
        <text x="40" y="45" font-family="Arial" font-size="40" text-anchor="middle" fill="white">💨</text>
    </svg>
    """


def create_farty_mascot():
    """
    Create a cartoon image of Farty, the friendly mascot.
    This is a placeholder function that would display an actual mascot in the final version.
    """
    # In a real implementation, we would load an actual image
    # For now, we'll create a simple SVG
    return """
    <svg width="150" height="150" xmlns="http://www.w3.org/2000/svg">
        <circle cx="75" cy="75" r="70" fill="#99CCFF" />
        <circle cx="50" cy="60" r="10" fill="white" />
        <circle cx="100" cy="60" r="10" fill="white" />
        <circle cx="50" cy="60" r="5" fill="black" />
        <circle cx="100" cy="60" r="5" fill="black" />
        <path d="M 40,90 C 55,110 95,110 110,90" stroke="black" stroke-width="3" fill="none" />
        <text x="75" y="40" font-family="Arial" font-size="16" text-anchor="middle">Farty</text>
    </svg>
    """


def run_simulation(model_type, gas_profile, room_dims, position, temperature, simulation_time):
    """
    Run the simulation and return results.
    
    Args:
        model_type (str): The type of model to use ('diffusion' or 'advection').
        gas_profile (GasProfile): The gas profile to use.
        room_dims (tuple): The dimensions of the room (x, y, z) in meters.
        position (tuple): The position of the gas source (x, y, z) in meters.
        temperature (float): The ambient temperature in Celsius.
        simulation_time (float): The total simulation time in seconds.
        
    Returns:
        tuple: (concentration, detection_results, verdict)
    """
    # Set grid resolution based on room dimensions
    grid_res = (
        int(room_dims[0] * 10),
        int(room_dims[1] * 10),
        int(room_dims[2] * 10)
    )
    
    # Create model based on type
    if model_type == 'diffusion':
        model = DiffusionModel(
            room_dimensions=room_dims,
            temperature=temperature,
            diffusion_coefficient=gas_profile.diffusion_coefficient,
            time_step=0.1,
            max_time=simulation_time,
            grid_resolution=grid_res
        )
    else:  # advection
        model = AdvectionDiffusionModel(
            room_dimensions=room_dims,
            temperature=temperature,
            diffusion_coefficient=gas_profile.diffusion_coefficient,
            buoyancy_factor=gas_profile.buoyancy_factor,
            time_step=0.1,
            max_time=simulation_time,
            grid_resolution=grid_res
        )
    
    # Add gas source
    model.add_gas_source(position, amount=1.0, radius=0.2)
    
    # Set up detector positions (default to far corners at nose height)
    nose_height = 1.6  # meters
    people_positions = [
        (room_dims[0] - 0.5, room_dims[1] - 0.5, nose_height),  # Far corner
        (room_dims[0] / 2, room_dims[1] / 2, nose_height)       # Center
    ]
    
    # Create gas detector
    detector = GasDetector(
        detector_positions=people_positions,
        room_dimensions=room_dims,
        temperature=temperature
    )
    
    # Run the simulation (use a progress bar in Streamlit)
    progress_bar = st.progress(0)
    steps = 20  # Fixed number of steps
    step_time = simulation_time / steps
    
    for i in range(steps):
        # Update progress
        progress = (i + 1) / steps
        progress_bar.progress(progress)
        
        # Step the model
        steps_per_update = max(1, int(step_time / model.time_step))
        concentration = model.step(num_steps=steps_per_update)
        
        # Sleep a bit to show progress
        time.sleep(0.05)
    
    # Get detection results
    detection_results = detector.check_detection(
        concentration, gas_profile, grid_res
    )
    
    # Get overall verdict
    verdict = detector.get_overall_detection_verdict(detection_results)
    
    return concentration, detection_results, verdict, people_positions, model


def show_sidebar():
    """
    Display the sidebar with configuration options.
    
    Returns:
        tuple: (selected_profile, model_type, room_dims, position, temperature, simulation_time)
    """
    st.sidebar.title("DiffuScent Settings")
    
    # Gas profile selection
    st.sidebar.header("Gas Profile")
    profile_options = list(GAS_PROFILES.keys())
    profile_names = [name.replace("_", " ").title() for name in profile_options]
    
    selected_profile_name = st.sidebar.selectbox(
        "Select a gas profile:",
        options=profile_names
    )
    
    # Convert back to the format used in our code
    selected_profile_key = selected_profile_name.lower().replace(" ", "_")
    selected_profile = get_gas_profile(selected_profile_name)
    
    # Show profile details
    st.sidebar.markdown(f"**Description:** {selected_profile.description}")
    st.sidebar.markdown(f"**Composition:** {selected_profile.get_composition_string()}")
    st.sidebar.markdown(f"**Fun Fact:** {selected_profile.fun_fact}")
    
    # Physics model selection
    st.sidebar.header("Physics Model")
    model_type = st.sidebar.radio(
        "Select simulation model:",
        options=["diffusion", "advection"],
        format_func=lambda x: "Basic Diffusion" if x == "diffusion" else "Advanced Diffusion-Advection"
    )
    
    # Room configuration
    st.sidebar.header("Room Configuration")
    room_length = st.sidebar.slider("Room Length (m)", 2.0, 10.0, 5.0, 0.1)
    room_width = st.sidebar.slider("Room Width (m)", 2.0, 10.0, 4.0, 0.1)
    room_height = st.sidebar.slider("Room Height (m)", 2.0, 5.0, 2.5, 0.1)
    room_dims = (room_length, room_width, room_height)
    
    # Source position
    st.sidebar.header("Gas Source Position")
    pos_x = st.sidebar.slider("X Position (m)", 0.1, room_length - 0.1, 0.5, 0.1)
    pos_y = st.sidebar.slider("Y Position (m)", 0.1, room_width - 0.1, 0.5, 0.1)
    pos_z = st.sidebar.slider("Z Position (m)", 0.1, 1.0, 0.3, 0.1, 
                             help="Height from floor (usually low for flatulence)")
    position = (pos_x, pos_y, pos_z)
    
    # Environmental conditions
    st.sidebar.header("Environmental Conditions")
    temperature = st.sidebar.slider("Room Temperature (°C)", 15.0, 30.0, 20.0, 0.5)
    
    # Simulation parameters
    st.sidebar.header("Simulation Parameters")
    simulation_time = st.sidebar.slider("Simulation Time (s)", 10.0, 300.0, 60.0, 10.0,
                                      help="Total time to simulate gas diffusion")
    
    return selected_profile, model_type, room_dims, position, temperature, simulation_time


def show_results(concentration, detection_results, verdict, people_positions, model, gas_profile, room_dims):
    """
    Display the simulation results in the main area.
    
    Args:
        concentration (numpy.ndarray): The final gas concentration field.
        detection_results (dict): The detection results from the detector.
        verdict (dict): The overall detection verdict.
        people_positions (list): List of (x, y, z) positions of people in the room.
        model: The simulation model used.
        gas_profile (GasProfile): The gas profile used.
        room_dims (tuple): The dimensions of the room (x, y, z) in meters.
    """
    # Set up renderers
    matplotlib_renderer = MatplotlibRenderer(room_dims, model.grid_resolution)
    plotly_renderer = PlotlyRenderer(room_dims, model.grid_resolution)
    
    # Display verdict with appropriate styling
    st.header("Detection Verdict")
    if verdict["status"] == "BUSTED":
        st.error(f"### {verdict['status']}: {verdict['message']}")
    elif verdict["status"] == "RISKY":
        st.warning(f"### {verdict['status']}: {verdict['message']}")
    else:  # SAFE
        st.success(f"### {verdict['status']}: {verdict['message']}")
    
    st.info(verdict["time_info"])
    
    # Display 2D and 3D visualizations
    st.header("Gas Distribution Visualization")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["3D View", "Horizontal Slices", "Vertical Slices"])
    
    with viz_tab1:
        # Create 3D visualization
        fig_3d = plotly_renderer.create_3d_volume(
            concentration,
            threshold=0.01,
            opacity=0.5,
            people_positions=people_positions,
            source_position=(model.room_dimensions[0]*0.1, model.room_dimensions[1]*0.1, model.room_dimensions[2]*0.1)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with viz_tab2:
        # Create horizontal slices (z-slices)
        fig_z = plt.figure(figsize=(12, 4))
        
        # Calculate slice indices for low, middle, and high positions
        z_low = int(0.1 * model.grid_resolution[2])  # Near floor
        z_mid = int(0.5 * model.grid_resolution[2])  # Middle of room
        z_high = int(0.7 * model.grid_resolution[2])  # Head height
        
        # Create subplots
        axs = [plt.subplot(1, 3, i+1) for i in range(3)]
        
        # Plot each slice
        matplotlib_renderer.plot_slice(concentration, 'z', z_low, axs[0], 
                                     title="Near Floor", 
                                     people_positions=people_positions, 
                                     source_position=(model.room_dimensions[0]*0.1, model.room_dimensions[1]*0.1, model.room_dimensions[2]*0.1))
        
        matplotlib_renderer.plot_slice(concentration, 'z', z_mid, axs[1], 
                                     title="Middle of Room", 
                                     people_positions=people_positions, 
                                     source_position=(model.room_dimensions[0]*0.1, model.room_dimensions[1]*0.1, model.room_dimensions[2]*0.1))
        
        matplotlib_renderer.plot_slice(concentration, 'z', z_high, axs[2], 
                                     title="Head Height", 
                                     people_positions=people_positions, 
                                     source_position=(model.room_dimensions[0]*0.1, model.room_dimensions[1]*0.1, model.room_dimensions[2]*0.1))
        
        plt.tight_layout()
        st.pyplot(fig_z)
    
    with viz_tab3:
        # Create vertical slices (x and y slices)
        fig_xy = plt.figure(figsize=(12, 4))
        
        # Calculate slice indices
        x_mid = int(0.5 * model.grid_resolution[0])
        y_mid = int(0.5 * model.grid_resolution[1])
        
        # Create subplots
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        
        # Plot slices
        matplotlib_renderer.plot_slice(concentration, 'x', x_mid, ax1, 
                                     title="Vertical Slice (YZ Plane)", 
                                     people_positions=people_positions, 
                                     source_position=(model.room_dimensions[0]*0.1, model.room_dimensions[1]*0.1, model.room_dimensions[2]*0.1))
        
        matplotlib_renderer.plot_slice(concentration, 'y', y_mid, ax2, 
                                     title="Vertical Slice (XZ Plane)", 
                                     people_positions=people_positions, 
                                     source_position=(model.room_dimensions[0]*0.1, model.room_dimensions[1]*0.1, model.room_dimensions[2]*0.1))
        
        plt.tight_layout()
        st.pyplot(fig_xy)
    
    # Display detailed detection results
    st.header("Detector Reports")
    for detector_id, result in detection_results.items():
        with st.expander(f"{result['position_name']} ({result['position'][0]:.1f}m, {result['position'][1]:.1f}m, {result['position'][2]:.1f}m)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Current concentration:** {result['concentration']:.5f}")
                st.markdown(f"**Detection threshold:** {result['threshold']:.5f}")
            
            with col2:
                st.markdown(f"**Is detected:** {'Yes' if result['is_detected'] else 'No'}")
                st.markdown(f"**Detection probability:** {result['detection_probability']:.1%}")
    
    # Educational content with Farty mascot
    st.header("Science Corner with Farty")
    mascot_col, text_col = st.columns([1, 3])
    
    with mascot_col:
        st.markdown(create_farty_mascot(), unsafe_allow_html=True)
    
    with text_col:
        st.markdown("### Did you know?")
        st.markdown(f"**{gas_profile.fun_fact}**")
        
        st.markdown("### How does gas diffusion work?")
        st.markdown("""
        Gas diffusion follows Fick's laws of diffusion. The rate of diffusion depends on:
        
        1. **Concentration gradient** - Gas moves from higher to lower concentration
        2. **Temperature** - Higher temperatures increase diffusion rates
        3. **Molecular weight** - Lighter molecules diffuse faster
        4. **Pressure** - Higher pressure can slow diffusion
        
        In this simulation, we're modeling how gas molecules spread out in a room over time!
        """)


def main():
    """
    Main application entry point.
    """
    # Display header
    st.title("DiffuScent: Gas Diffusion Simulator")
    st.markdown("### *Is It Safe to Fart?* A Scientifically Accurate Simulator")
    
    # Show introduction on first run
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.markdown("""
        ## Welcome to DiffuScent!
        
        This simulator models the diffusion of gas in a room with scientifically accurate physics.
        Use the sidebar to configure your simulation, and discover if your flatulence will be detected!
        
        The simulator uses real fluid dynamics principles to model how gas spreads in an enclosed space.
        """)
    
    # Show configuration sidebar
    selected_profile, model_type, room_dims, position, temperature, simulation_time = show_sidebar()
    
    # Run simulation button
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            # Run the simulation
            concentration, detection_results, verdict, people_positions, model = run_simulation(
                model_type, selected_profile, room_dims, position, temperature, simulation_time
            )
            
            # Show results
            show_results(concentration, detection_results, verdict, people_positions, 
                       model, selected_profile, room_dims)
    else:
        # Show instructions if simulation hasn't been run yet
        if 'result_shown' not in st.session_state:
            st.info("""
            ## How to use this simulator:
            
            1. Choose a gas profile from the sidebar
            2. Select a physics model (basic or advanced)
            3. Configure the room dimensions and source position
            4. Adjust environmental conditions like temperature
            5. Set simulation duration
            6. Click "Run Simulation" to see the results
            
            The simulator will tell you if your emission would be detected and provide
            visualizations of how the gas diffuses through the room!
            """)


if __name__ == "__main__":
    main()
