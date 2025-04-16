"""
Basic diffusion model for gas simulation in DiffuScent.

This module implements a simple diffusion model using FluidDyn/FluidSim
to simulate gas diffusion in a room with temperature effects.
"""

import numpy as np
from fluidsim.base.params import Parameters
from fluidsim.solvers.ns2d.solver import Simul

class DiffusionModel:
    """
    A class to simulate gas diffusion in a room using FluidDyn/FluidSim.
    
    This model simulates the diffusion of gas particles in a 2D or 3D room,
    taking into account temperature effects on diffusion rates.
    
    Attributes:
        room_dimensions (tuple): The dimensions of the room (x, y, z) in meters.
        temperature (float): The ambient temperature of the room in Celsius.
        diffusion_coefficient (float): The base diffusion coefficient for the gas.
        time_step (float): The time step for simulation in seconds.
        max_time (float): The maximum simulation time in seconds.
        grid_resolution (tuple): The resolution of the simulation grid.
    """
    
    def __init__(self,
                 room_dimensions=(5.0, 4.0, 2.5),  # Default 5m x 4m x 2.5m room
                 temperature=20.0,                 # Default 20°C
                 diffusion_coefficient=0.175,      # Default value for methane at STP
                 time_step=0.1,                    # Default 0.1 second time step
                 max_time=600.0,                   # Default 10 minute simulation
                 grid_resolution=(50, 40, 25)):    # Default grid resolution
        """
        Initialize the diffusion model.
        
        Args:
            room_dimensions (tuple): The dimensions of the room (x, y, z) in meters.
            temperature (float): The ambient temperature of the room in Celsius.
            diffusion_coefficient (float): The base diffusion coefficient for the gas.
            time_step (float): The time step for simulation in seconds.
            max_time (float): The maximum simulation time in seconds.
            grid_resolution (tuple): The resolution of the simulation grid.
        """
        self.room_dimensions = room_dimensions
        self.temperature = temperature
        self.base_diffusion_coefficient = diffusion_coefficient
        self.time_step = time_step
        self.max_time = max_time
        self.grid_resolution = grid_resolution
        
        # Initialize the simulation parameters
        self.params = self._initialize_params()
        
        # Initialize the simulation
        self.sim = None
        self.concentration = np.zeros(grid_resolution)
        self.current_time = 0.0
        
        # Initialize simulation state
        self.is_initialized = False
    
    def _initialize_params(self):
        """
        Initialize FluidSim parameters.
        
        Returns:
            Parameters: FluidSim parameters object.
        """
        # Create a parameters object with a tag
        params = Parameters(tag='params')
        
        # Set up simulation parameters using _set_attrib
        params._set_attrib('short_name_type_run', "diffusion")
        params._set_attrib('NEW_DIR_RESULTS', False)
        
        # Provide a dummy path_run so FluidSim does not complain
        params._set_attrib('path_run', './tmp_simul')
        
        # Set ONLY_COARSE_OPER parameter (required by operators2d)
        params._set_attrib('ONLY_COARSE_OPER', True)
        
        # Create output parameters
        params._set_attrib("output", Parameters(tag='output'))
        params.output._set_attrib("HAS_TO_SAVE", True)
        
        # Create operator parameters
        params._set_attrib("oper", Parameters(tag='oper'))
        params.oper._set_attrib('Lx', self.room_dimensions[0])
        params.oper._set_attrib('Ly', self.room_dimensions[1])
        params.oper._set_attrib('nx', self.grid_resolution[0])
        params.oper._set_attrib('ny', self.grid_resolution[1])
        params.oper._set_attrib('type_fft', 'sequential')
        params.oper._set_attrib('coef_dealiasing', 2/3)
        
        # Create forcing parameters (required by output initialization)
        params._set_attrib("forcing", Parameters(tag='forcing'))
        params.forcing._set_attrib('enable', False)  # Disable forcing for simple diffusion
        
        # Create time stepping parameters
        params._set_attrib("time_stepping", Parameters(tag='time_stepping'))
        params.time_stepping._set_attrib('USE_CFL', True)
        params.time_stepping._set_attrib('deltat0', self.time_step)
        
        # Set diffusion coefficient (adjusted for temperature)
        params._set_attrib('nu_2', self._adjust_diffusion_for_temperature())
        
        return params
    
    def _adjust_diffusion_for_temperature(self):
        """
        Adjust the diffusion coefficient based on temperature.
        
        The diffusion coefficient increases with temperature according to
        the Chapman-Enskog theory, which can be approximated as proportional
        to T^(3/2) for gases.
        
        Returns:
            float: Temperature-adjusted diffusion coefficient.
        """
        # Convert Celsius to Kelvin
        temp_kelvin = self.temperature + 273.15
        
        # Reference temperature (20°C = 293.15K)
        ref_temp_kelvin = 293.15
        
        # Adjust diffusion coefficient using Chapman-Enskog relation (approximate)
        # D ∝ T^(3/2)
        adjusted_coefficient = self.base_diffusion_coefficient * (
            (temp_kelvin / ref_temp_kelvin) ** 1.5
        )
        
        return adjusted_coefficient
    
    def initialize_simulation(self):
        """
        Initialize the simulation without relying on FluidSim's Simul class.
        """
        # Just reset the concentration field and time
        self.concentration = np.zeros(self.grid_resolution)
        self.current_time = 0.0
        self.is_initialized = True
        
        return True
    
    def add_gas_source(self, position, amount, radius=0.5):
        """
        Add a gas source to the simulation.
        
        Args:
            position (tuple): The position (x, y, z) of the gas source in meters.
            amount (float): The amount of gas released in kg.
            radius (float): The initial radius of the gas cloud in meters.
        """
        if not self.is_initialized:
            self.initialize_simulation()
        
        # Convert position from meters to grid indices
        x_idx = int(position[0] / self.room_dimensions[0] * self.grid_resolution[0])
        y_idx = int(position[1] / self.room_dimensions[1] * self.grid_resolution[1])
        
        # Convert radius from meters to grid units
        x_radius = int(radius / self.room_dimensions[0] * self.grid_resolution[0])
        y_radius = int(radius / self.room_dimensions[1] * self.grid_resolution[1])
        
        # Create a Gaussian distribution of gas
        for i in range(max(0, x_idx - 2*x_radius), min(self.grid_resolution[0], x_idx + 2*x_radius)):
            for j in range(max(0, y_idx - 2*y_radius), min(self.grid_resolution[1], y_idx + 2*y_radius)):
                # Calculate distance from center in grid units
                dist_squared = ((i - x_idx) / x_radius)**2 + ((j - y_idx) / y_radius)**2
                
                # Add gas concentration using Gaussian distribution
                self.concentration[i, j, 0] += amount * np.exp(-dist_squared)
    
    def step(self, num_steps=1):
        """
        Step the simulation forward in time.
        
        Args:
            num_steps (int): The number of time steps to simulate.
            
        Returns:
            numpy.ndarray: The current gas concentration field.
        """
        if not self.is_initialized:
            self.initialize_simulation()
        
        for _ in range(num_steps):
            # Update current time
            self.current_time += self.time_step
            
            # Check if we've reached max time
            if self.current_time >= self.max_time:
                return self.concentration
            
            # Apply diffusion to concentration field using our own implementation
            self._apply_diffusion()
        
        return self.concentration

    def _apply_diffusion(self):
        """
        Apply diffusion to the concentration field using a finite difference method.
        """
        # Copy the current concentration field
        new_concentration = np.copy(self.concentration)
        
        # Get the adjusted diffusion coefficient
        diff_coef = self._adjust_diffusion_for_temperature()
        
        # Calculate the diffusion constant for the finite difference method
        # stability requires that D * dt / dx^2 <= 0.5
        dx = self.room_dimensions[0] / self.grid_resolution[0]
        dy = self.room_dimensions[1] / self.grid_resolution[1]
        dz = self.room_dimensions[2] / self.grid_resolution[2]
        
        # Use the smallest dimension for stability calculation
        min_dim = min(dx, dy, dz)
        diff_const = diff_coef * self.time_step / (min_dim ** 2)
        
        # Ensure stability
        if diff_const > 0.25:
            print(f"Warning: Diffusion constant {diff_const} exceeds stability threshold 0.25")
            print("Reducing time step to ensure stability")
            self.time_step = 0.25 * (min_dim ** 2) / diff_coef
            diff_const = 0.25
        
        # Apply diffusion using a 3D finite difference method
        for i in range(1, self.grid_resolution[0]-1):
            for j in range(1, self.grid_resolution[1]-1):
                for k in range(1, self.grid_resolution[2]-1):
                    # Calculate the Laplacian of the concentration
                    laplacian = (
                        self.concentration[i+1, j, k] +
                        self.concentration[i-1, j, k] +
                        self.concentration[i, j+1, k] +
                        self.concentration[i, j-1, k] +
                        self.concentration[i, j, k+1] +
                        self.concentration[i, j, k-1] -
                        6 * self.concentration[i, j, k]
                    ) / (min_dim ** 2)
                    
                    # Apply the diffusion equation: dc/dt = D * ∇²c
                    new_concentration[i, j, k] = self.concentration[i, j, k] + diff_coef * self.time_step * laplacian
        
        # Update the concentration field
        self.concentration = new_concentration
    
    def get_concentration_at_point(self, position):
        """
        Get the gas concentration at a specific point in the room.
        
        Args:
            position (tuple): The position (x, y, z) in meters.
            
        Returns:
            float: The gas concentration at the specified position.
        """
        # Convert position from meters to grid indices
        x_idx = int(position[0] / self.room_dimensions[0] * self.grid_resolution[0])
        y_idx = int(position[1] / self.room_dimensions[1] * self.grid_resolution[1])
        z_idx = int(position[2] / self.room_dimensions[2] * self.grid_resolution[2])
        
        # Ensure indices are within bounds
        x_idx = max(0, min(x_idx, self.grid_resolution[0]-1))
        y_idx = max(0, min(y_idx, self.grid_resolution[1]-1))
        z_idx = max(0, min(z_idx, self.grid_resolution[2]-1))
        
        return self.concentration[x_idx, y_idx, z_idx]
    
    def is_detectable_at_point(self, position, threshold=0.1):
        """
        Check if the gas concentration at a specific point exceeds the detection threshold.
        
        Args:
            position (tuple): The position (x, y, z) in meters.
            threshold (float): The detection threshold concentration.
            
        Returns:
            bool: True if the gas is detectable at the specified position, False otherwise.
        """
        concentration = self.get_concentration_at_point(position)
        return concentration >= threshold
    
    def reset(self):
        """
        Reset the simulation.
        """
        self.concentration = np.zeros(self.grid_resolution)
        self.current_time = 0.0
        self.is_initialized = False
