#!/usr/bin/env python3
"""
Basic DiffuScent Model using FiPy

This prototype demonstrates a basic 3D gas diffusion model for the DiffuScent project,
simulating the diffusion of gas (flatulence) in a room environment.

The model uses FiPy, a finite volume PDE solver developed by NIST, to simulate
how gas concentrations change over time in a 3D space.

Key scientific principles:
- Diffusion follows Fick's Second Law: ∂C/∂t = D∇²C
  where C is concentration, t is time, and D is the diffusion coefficient
- Concentration is measured in parts per million (ppm)
- Diffusion coefficient varies by gas type and temperature
"""

import os
import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Check if FiPy is installed, provide helpful message if not
try:
    from fipy import Grid3D, CellVariable, TransientTerm, DiffusionTerm, Viewer
    from fipy.tools import numerix
except ImportError:
    print("FiPy is required for this simulation.")
    print("Install it with: pip install fipy")
    print("For more information, visit: https://www.ctcms.nist.gov/fipy/")
    sys.exit(1)

class DiffuscentModel:
    """
    A 3D gas diffusion model for simulating flatulence dispersion in a room.
    
    This class handles:
    - Setting up the 3D mesh representing the room
    - Initializing gas properties and diffusion parameters
    - Solving the diffusion equation
    - Visualizing the results
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the model with either default parameters or from a config file.
        
        Args:
            config_file (str, optional): Path to YAML configuration file.
        """
        # Default configuration
        self.config = {
            # Room dimensions in meters
            'room': {
                'width': 5.0,   # x-direction (m)
                'length': 4.0,  # y-direction (m)
                'height': 2.5,  # z-direction (m)
                'mesh_cells': {
                    'nx': 50,   # Number of cells in x-direction
                    'ny': 40,   # Number of cells in y-direction
                    'nz': 25    # Number of cells in z-direction
                }
            },
            # Gas properties
            'gas': {
                'name': 'methane',  # Primary component
                # Diffusion coefficient in air (m²/s)
                'diffusion_coefficient': 1.75e-5,  
                # Initial emission
                'initial_volume_ml': 100.0,  # Gas volume in milliliters
                'initial_concentration_ppm': 50000.0,  # Concentration in ppm
                # Source location (relative coordinates 0-1)
                'source_location': [0.1, 0.1, 0.1],  
                # Source size (radius in meters)
                'source_radius': 0.1
            },
            # Simulation parameters
            'simulation': {
                'total_time': 60.0,  # Total simulation time (seconds)
                'time_step': 0.5,     # Time step (seconds)
                'detection_threshold': 0.5  # Detection threshold (ppm)
            },
            # Visualization settings
            'visualization': {
                'contour_levels': 10,
                'save_plots': True,
                'plot_interval': 5.0  # Seconds between plots
            }
        }
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Update default config with loaded values
                self._update_nested_dict(self.config, loaded_config)
        
        # Initialize model components
        self._setup_mesh()
        self._setup_variables()
        self._setup_equation()
        
        # Store for results
        self.time_points = []
        self.max_concentrations = []
        self.detection_points = []
    
    def _update_nested_dict(self, d, u):
        """
        Update a nested dictionary with values from another dictionary.
        
        Args:
            d (dict): Dictionary to update
            u (dict): Dictionary with updates
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def _setup_mesh(self):
        """
        Create the 3D mesh representing the room.
        """
        # Extract room dimensions
        width = self.config['room']['width']
        length = self.config['room']['length']
        height = self.config['room']['height']
        
        # Extract mesh resolution
        nx = self.config['room']['mesh_cells']['nx']
        ny = self.config['room']['mesh_cells']['ny']
        nz = self.config['room']['mesh_cells']['nz']
        
        # Calculate cell sizes
        self.dx = width / nx
        self.dy = length / ny
        self.dz = height / nz
        
        # Create the 3D mesh using FiPy's Grid3D
        self.mesh = Grid3D(
            dx=self.dx, nx=nx,
            dy=self.dy, ny=ny,
            dz=self.dz, nz=nz
        )
        
        print(f"Created 3D mesh: {nx}x{ny}x{nz} cells")
        print(f"Room dimensions: {width}m x {length}m x {height}m")
        print(f"Cell size: {self.dx:.3f}m x {self.dy:.3f}m x {self.dz:.3f}m")
    
    def _setup_variables(self):
        """
        Initialize the gas concentration variable and set initial conditions.
        """
        # Create a cell variable for gas concentration (in ppm)
        self.concentration = CellVariable(
            name="Gas Concentration (ppm)",
            mesh=self.mesh,
            value=0.0  # Initial concentration is zero everywhere
        )
        
        # Add the initial gas source
        self._add_gas_source()
        
        # Set the diffusion coefficient
        self.D = self.config['gas']['diffusion_coefficient']
        
        print(f"Initialized concentration field with {self.config['gas']['name']} diffusion")
        print(f"Diffusion coefficient: {self.D:.2e} m²/s")
    
    def _add_gas_source(self):
        """
        Add an initial gas source (emission) at the specified location.
        """
        # Get source parameters
        volume_ml = self.config['gas']['initial_volume_ml']
        conc_ppm = self.config['gas']['initial_concentration_ppm']
        source_loc = self.config['gas']['source_location']
        source_radius = self.config['gas']['source_radius']
        
        # Convert relative source location to absolute coordinates
        source_x = source_loc[0] * self.config['room']['width']
        source_y = source_loc[1] * self.config['room']['length']
        source_z = source_loc[2] * self.config['room']['height']
        
        # Calculate indices for the source location
        x_idx = int(source_x / self.dx)
        y_idx = int(source_y / self.dy)
        z_idx = int(source_z / self.dz)
        
        # Calculate radius in terms of cells
        x_radius = max(1, int(source_radius / self.dx))
        y_radius = max(1, int(source_radius / self.dy))
        z_radius = max(1, int(source_radius / self.dz))
        
        # Add a Gaussian distribution of concentration
        for i in range(self.mesh.nx):
            for j in range(self.mesh.ny):
                for k in range(self.mesh.nz):
                    # Calculate distance from source (in cell units)
                    dist_sq = (
                        ((i - x_idx) / x_radius) ** 2 + 
                        ((j - y_idx) / y_radius) ** 2 + 
                        ((k - z_idx) / z_radius) ** 2
                    )
                    
                    # Add concentration using a Gaussian distribution
                    if dist_sq <= 4.0:  # Within 2 radii
                        # Cell index
                        cell_idx = k * (self.mesh.nx * self.mesh.ny) + j * self.mesh.nx + i
                        
                        # Set the value
                        self.concentration.value[cell_idx] = conc_ppm * np.exp(-dist_sq)
        
        # Calculate and print the total amount of gas (volume in ml)
        total_ppm = self.concentration.cellVolumeAverage * self.mesh.numberOfCells
        print(f"Added gas source: {volume_ml:.1f}ml at ({source_x:.2f}, {source_y:.2f}, {source_z:.2f})")
        print(f"Source radius: {source_radius:.2f}m")
        print(f"Peak concentration: {conc_ppm:.1f} ppm")
    
    def _setup_equation(self):
        """
        Set up the diffusion equation.
        
        The diffusion equation is: ∂C/∂t = D∇²C
        where:
        - C is the concentration (ppm)
        - t is time (s)
        - D is the diffusion coefficient (m²/s)
        - ∇² is the Laplacian operator
        """
        # Create the diffusion equation using FiPy terms
        self.eq = TransientTerm() == DiffusionTerm(coeff=self.D)
        
        print("Created diffusion equation: ∂C/∂t = D∇²C")
    
    def run_simulation(self):
        """
        Run the simulation for the specified time duration.
        """
        # Get simulation parameters
        total_time = self.config['simulation']['total_time']
        dt = self.config['simulation']['time_step']
        plot_interval = self.config['visualization']['plot_interval']
        
        # Calculate number of steps
        steps = int(total_time / dt)
        
        # Initialize time tracking
        self.current_time = 0.0
        next_plot_time = 0.0
        
        print(f"Running simulation for {total_time:.1f} seconds with dt={dt:.2f}s")
        print(f"Total steps: {steps}")
        
        # Time stepping loop
        for step in range(steps):
            # Advance time
            self.current_time += dt
            
            # Solve the diffusion equation for this time step
            self.eq.solve(var=self.concentration, dt=dt)
            
            # Store data for plotting
            self.time_points.append(self.current_time)
            self.max_concentrations.append(self.concentration.value.max())
            
            # Plot at specified intervals
            if self.current_time >= next_plot_time:
                self._visualize_results()
                next_plot_time += plot_interval
            
            # Check for detection events
            self._check_detection()
            
            # Periodic console update
            if step % 10 == 0:
                current_max = self.concentration.value.max()
                print(f"Time: {self.current_time:.1f}s, Max concentration: {current_max:.2f} ppm")
        
        # Final visualization & plot
        self._visualize_results(is_final=True)
        self._plot_concentration_vs_time()
        
        print("Simulation complete.")


    
    def _check_detection(self):
        """
        Check if gas concentration exceeds detection threshold at specific points.
        """
        # Get detection threshold
        threshold = self.config['simulation']['detection_threshold']
        
        # Define detection points (could be loaded from config)
        # For example, points at nose height (1.6m) at different room locations
        detection_points = [
            (0.8, 0.8, 0.64),  # point at 80% room width, 80% length, nose height
            (0.5, 0.5, 0.64),  # center of room at nose height
        ]
        
        for point_idx, (x_rel, y_rel, z_rel) in enumerate(detection_points):
            # Convert to absolute coordinates
            x = x_rel * self.config['room']['width']
            y = y_rel * self.config['room']['length']
            z = z_rel * self.config['room']['height']
            
            # Find nearest cell
            i = min(max(0, int(x / self.dx)), self.mesh.nx - 1)
            j = min(max(0, int(y / self.dy)), self.mesh.ny - 1)
            k = min(max(0, int(z / self.dz)), self.mesh.nz - 1)
            
            # Get the cell index
            cell_idx = k * (self.mesh.nx * self.mesh.ny) + j * self.mesh.nx + i
            
            # Get concentration at this point
            conc = self.concentration.value[cell_idx]
            
            # Check if it exceeds the threshold
            if conc >= threshold:
                if len(self.detection_points) <= point_idx:
                    self.detection_points.append(self.current_time)
                    print(f"Detection at point {point_idx+1} at t={self.current_time:.1f}s")
    
    def _visualize_results(self, is_final=False):
        """
        Visualize the current gas concentration distribution.
        
        Args:
            is_final (bool): Whether this is the final visualization
        """
        # Create a figure for a slice at nose height
        nose_height = 1.6  # meters
        nose_idx = min(max(0, int(nose_height / self.dz)), self.mesh.nz - 1)
        
        # Extract data for the slice
        data = self._get_slice_data(nose_idx, axis='z')
        
        # Create the figure
        plt.figure(figsize=(10, 8))
        
        # Create a meshgrid for plotting
        x = np.linspace(0, self.config['room']['width'], self.mesh.nx)
        y = np.linspace(0, self.config['room']['length'], self.mesh.ny)
        X, Y = np.meshgrid(x, y)
        
        # Plot the contour
        levels = self.config['visualization']['contour_levels']
        contour = plt.contourf(X, Y, data.T, levels=levels, cmap='YlOrRd')
        
        # Add a colorbar
        cbar = plt.colorbar(contour)
        cbar.set_label('Concentration (ppm)')
        
        # Add title and labels
        plt.title(f'Gas Concentration at Nose Height ({nose_height}m), t={self.current_time:.1f}s')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        # Save the figure if requested
        if self.config['visualization']['save_plots']:
            plt.savefig(f'concentration_t{self.current_time:.0f}.png')
            print(f"Saved plot: concentration_t{self.current_time:.0f}.png")
        
        # Don't display interactively - just save to file
        plt.close()
    
    def _get_slice_data(self, idx, axis='z'):
        """
        Extract a 2D slice of concentration data.
        
        Args:
            idx (int): Index along the specified axis
            axis (str): The axis perpendicular to the slice ('x', 'y', or 'z')
            
        Returns:
            numpy.ndarray: 2D array of concentration values
        """
        # Create an empty array
        if axis == 'z':
            data = np.zeros((self.mesh.nx, self.mesh.ny))
            for i in range(self.mesh.nx):
                for j in range(self.mesh.ny):
                    cell_idx = idx * (self.mesh.nx * self.mesh.ny) + j * self.mesh.nx + i
                    data[i, j] = self.concentration.value[cell_idx]
                    
        elif axis == 'y':
            data = np.zeros((self.mesh.nx, self.mesh.nz))
            for i in range(self.mesh.nx):
                for k in range(self.mesh.nz):
                    cell_idx = k * (self.mesh.nx * self.mesh.ny) + idx * self.mesh.nx + i
                    data[i, k] = self.concentration.value[cell_idx]
                    
        elif axis == 'x':
            data = np.zeros((self.mesh.ny, self.mesh.nz))
            for j in range(self.mesh.ny):
                for k in range(self.mesh.nz):
                    cell_idx = k * (self.mesh.nx * self.mesh.ny) + j * self.mesh.nx + idx
                    data[j, k] = self.concentration.value[cell_idx]
        
        return data
    
    def _plot_concentration_vs_time(self):
        """
        Plot the maximum concentration vs. time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.max_concentrations)
        
        # Add vertical lines for detection times
        for i, t in enumerate(self.detection_points):
            plt.axvline(x=t, color='r', linestyle='--', alpha=0.7, 
                       label=f'Detection at point {i+1}' if i==0 else None)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Maximum Concentration (ppm)')
        plt.title('Maximum Gas Concentration vs. Time')
        plt.grid(True)
        
        if len(self.detection_points) > 0:
            plt.legend()
        
        # Save the figure if requested
        if self.config['visualization']['save_plots']:
            plt.savefig('concentration_vs_time.png')
        
        plt.show()


def main():
    """
    Main function to run the model.
    """
    # Check for command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        print(f"Loading configuration from: {config_file}")
        model = DiffuscentModel(config_file)
    else:
        # Look for the default config in the configs directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "configs", "config.yaml")
        if os.path.exists(config_path):
            print(f"Loading default configuration from: {config_path}")
            model = DiffuscentModel(config_path)
        else:
            print("Using hardcoded default configuration")
            model = DiffuscentModel()
    
    # Run the simulation
    model.run_simulation()


if __name__ == "__main__":
    main()
