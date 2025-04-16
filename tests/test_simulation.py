"""
Test script for DiffuScent physics and visualization.

This script demonstrates the basic functionality of the DiffuScent
physics models and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.physics.diffusion import DiffusionModel
from src.physics.gas_profiles import get_gas_profile
from src.physics.detection import GasDetector
from src.visualization.renderer import MatplotlibRenderer, PlotlyRenderer

def test_basic_diffusion():
    """
    Test the basic diffusion model and visualize the results.
    """
    print("Testing basic diffusion model...")
    
    # Set up simulation parameters
    room_dims = (5.0, 4.0, 2.5)  # 5m x 4m x 2.5m room
    grid_res = (50, 40, 25)      # 50x40x25 grid
    
    # Create diffusion model
    model = DiffusionModel(
        room_dimensions=room_dims,
        temperature=20.0,
        diffusion_coefficient=0.175,
        time_step=0.1,
        grid_resolution=grid_res
    )
    
    # Get gas profile (Taco Bell Banger)
    gas_profile = get_gas_profile("Taco Bell Banger")
    print(f"Using gas profile: {gas_profile.name}")
    print(f"Composition: {gas_profile.get_composition_string()}")
    print(f"Diffusion coefficient: {gas_profile.diffusion_coefficient}")
    print(f"Buoyancy factor: {gas_profile.buoyancy_factor}")
    
    # Initialize the model with the gas profile's properties
    model.base_diffusion_coefficient = gas_profile.diffusion_coefficient
    
    # Add gas source near the floor
    source_position = (0.5, 0.5, 0.3)  # Near the corner, close to floor
    model.add_gas_source(source_position, amount=1.0, radius=0.2)
    print(f"Added gas source at position {source_position}")
    
    # Set up detector positions (people in the room)
    people_positions = [
        (4.0, 3.5, 1.6),  # Person in far corner at nose height
        (2.5, 2.0, 1.6)   # Person in center at nose height
    ]
    
    # Create gas detector
    detector = GasDetector(
        detector_positions=people_positions,
        room_dimensions=room_dims,
        temperature=20.0
    )
    
    # Create visualization renderers
    matplotlib_renderer = MatplotlibRenderer(room_dims, grid_res)
    plotly_renderer = PlotlyRenderer(room_dims, grid_res)
    
    # Run simulation for a few steps
    print("\nRunning simulation for 20 steps...")
    total_steps = 20
    detection_times = []
    
    # Create figure for animation frames
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for step in range(total_steps):
        # Step the simulation
        concentration = model.step(num_steps=5)  # 5 sub-steps
        
        # Clear the axis for new frame
        ax.clear()
        
        # Plot the current state (z-slice at nose height)
        nose_height_idx = int(1.6 / room_dims[2] * grid_res[2])
        matplotlib_renderer.plot_slice(
            concentration, 
            slice_dim='z', 
            slice_idx=nose_height_idx,
            ax=ax,
            title=f"Gas Concentration at t={model.current_time:.1f}s (Nose Height)",
            people_positions=people_positions,
            source_position=source_position
        )
        
        # Check detection
        detection_results = detector.check_detection(
            concentration, gas_profile, grid_res
        )
        
        # Store detection information
        detected = False
        for detector_id, result in detection_results.items():
            if result["is_detected"]:
                detected = True
                print(f"Step {step}, Time {model.current_time:.1f}s: "
                      f"Gas detected at {result['position_name']} "
                      f"(Concentration: {result['concentration']:.4f})")
        
        detection_times.append((model.current_time, detected))
        
        # Update the display
        plt.pause(0.1)
    
    plt.close()
    
    # Create final visualizations
    print("\nCreating final visualizations...")
    
    # 1. Plot multiple z-slices
    fig_slices = matplotlib_renderer.plot_multiple_slices(
        concentration, 
        slice_dim='z', 
        num_slices=3,
        people_positions=people_positions,
        source_position=source_position
    )
    fig_slices.suptitle(f"Gas Concentration at t={model.current_time:.1f}s (Multiple Z-Slices)")
    
    # 2. Create 3D visualization
    fig_3d = plotly_renderer.create_3d_volume(
        concentration,
        threshold=0.01,
        opacity=0.5,
        people_positions=people_positions,
        source_position=source_position
    )
    
    # Display detection verdict
    verdict = detector.get_overall_detection_verdict(detection_results)
    print("\nFinal detection verdict:")
    print(f"Status: {verdict['status']}")
    print(f"Message: {verdict['message']}")
    print(f"Time info: {verdict['time_info']}")
    
    # Show the final 2D plots
    plt.show()
    
    # Show the 3D plot (if in an interactive environment)
    try:
        fig_3d.show()
    except Exception as e:
        print(f"Could not display 3D plot: {e}")
        print("3D plots require an interactive environment or Jupyter notebook.")
    
    return model, concentration, detection_results


if __name__ == "__main__":
    test_basic_diffusion()
