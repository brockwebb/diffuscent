"""
Advanced diffusion-advection model for gas simulation in DiffuScent.

This module implements a diffusion-advection model using FluidDyn/FluidSim
to simulate gas diffusion and movement in a room with buoyancy effects.
"""

import numpy as np
from fluidsim.base.params import Parameters
from fluidsim.solvers.ns2d.solver import Simul
from fluidsim.solvers.ns2d import solver

class AdvectionDiffusionModel:
    """
    A class to simulate gas diffusion and advection in a room using FluidDyn/FluidSim.
    
    This model simulates the diffusion and advection of gas particles in a 2D or 3D room,
    taking into account temperature effects, buoyancy, and background air currents.
    
    Attributes:
        room_dimensions (tuple): The dimensions of the room (x, y, z) in meters.
        temperature (float): The ambient temperature of the room in Celsius.
        diffusion_coefficient (float): The base diffusion coefficient for the gas.
        buoyancy_factor (float): The buoyancy factor of the gas.
        background_flow (tuple): The background flow velocity (vx, vy, vz) in m/s.
        time_step (float): The time step for simulation in seconds.
        max_time (float): The maximum simulation time in seconds.
        grid_resolution (tuple): The resolution of the simulation grid.
    """
    
    def __init__(self, 
                 room_dimensions=(5.0, 4.0, 2.5),  # Default 5m x 4m x 2.5m room
                 temperature=20.0,                 # Default 20°C
                 diffusion_coefficient=0.175,      # Default value for methane at STP
                 buoyancy_factor=1.0,              # Default neutral buoyancy
                 background_flow=(0.0, 0.0, 0.0),  # Default no background flow
                 time_step=0.1,                    # Default 0.1 second time step
                 max_time=600.0,                   # Default 10 minute simulation
                 grid_resolution=(50, 40, 25)):    # Default grid resolution
        """
        Initialize the advection-diffusion model.
        
        Args:
            room_dimensions (tuple): The dimensions of the room (x, y, z) in meters.
            temperature (float): The ambient temperature of the room in Celsius.
            diffusion_coefficient (float): The base diffusion coefficient for the gas.
            buoyancy_factor (float): The buoyancy factor of the gas.
            background_flow (tuple): The background flow velocity (vx, vy, vz) in m/s.
            time_step (float): The time step for simulation in seconds.
            max_time (float): The maximum simulation time in seconds.
            grid_resolution (tuple): The resolution of the simulation grid.
        """
        self.room_dimensions = room_dimensions
        self.temperature = temperature
        self.base_diffusion_coefficient = diffusion_coefficient
        self.buoyancy_factor = buoyancy_factor
        self.background_flow = background_flow
        self.time_step = time_step
        self.max_time = max_time
        self.grid_resolution = grid_resolution
        
        # Initialize the simulation parameters
        self.params = self._initialize_params()
        
        # Initialize the simulation
        self.sim = None
        self.concentration = np.zeros(grid_resolution)
        self.velocity_field = np.zeros((2, *grid_resolution))  # (vx, vy) components
        self.current_time = 0.0
        
        # Initialize simulation state
        self.is_initialized = False
    
    def _initialize_params(self):
        """
        Initialize FluidSim parameters for the NS2D solver.
        
        Returns:
            Parameters: FluidSim parameters object.
        """
        # Create a parameters object
        params = Parameters(tag='params')
        
        # Set dimensions and grid resolution
        params.short_name_type_run = 'advection-diffusion'
        
        # We need to use the ns2d solver
        params.family = 'fluidsim.solvers.ns2d'
        
        # Initialize output parameters
        params.output = Parameters(tag='output')
        params.output.HAS_TO_SAVE = True
        params.output.sub_directory = 'diffuscent_sim'
        
        # Set periodic boundary conditions
        params.ONLY_COARSE_OPER = True
        
        # Set dimensions and resolution
        params.oper = Parameters(tag='oper')
        params.oper.Lx = self.room_dimensions[0]
        params.oper.Ly = self.room_dimensions[1]
        params.oper.nx = self.grid_resolution[0]
        params.oper.ny = self.grid_resolution[1]
        
        # Set time stepping parameters
        params.time_stepping = Parameters(tag='time_stepping')
        params.time_stepping.USE_CFL = True
        params.time_stepping.deltat0 = self.time_step
        params.time_stepping.type_time_scheme = 'RK4'
        
        # Set forcing parameters (for buoyancy and background flow)
        params.forcing = Parameters(tag='forcing')
        params.forcing.enable = True
        params.forcing.type = 'user'
        
        # Set physical parameters
        params.nu_2 = self._adjust_diffusion_for_temperature()
        params.c2 = 0.5  # Set Courant number

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
        Initialize the FluidSim simulation.
        """
        # Create the simulation
        self.sim = Simul(self.params)
        self.is_initialized = True
        self.current_time = 0.0
        
        # Reset concentration field
        self.concentration = np.zeros(self.grid_resolution)
        
        # Initialize velocity field with background flow
        self.velocity_field = np.zeros((2, *self.grid_resolution))
        self.velocity_field[0, :, :] = self.background_flow[0]  # vx
        self.velocity_field[1, :, :] = self.background_flow[1]  # vy
        
        # Set initial fields in the simulation
        self._update_sim_fields()
        
        return self.sim
    
    def _update_sim_fields(self):
        """
        Update the fields in the FluidSim simulation.
        """
        if self.sim is None:
            return
        
        # Update velocity field in simulation
        self.sim.state.set_state_physical(self.velocity_field[0], self.velocity_field[1])
    
    def add_gas_source(self, position, amount, radius=0.5, initial_velocity=(0.0, 0.0, 0.0)):
        """
        Add a gas source to the simulation.
        
        Args:
            position (tuple): The position (x, y, z) of the gas source in meters.
            amount (float): The amount of gas released in kg.
            radius (float): The initial radius of the gas cloud in meters.
            initial_velocity (tuple): Initial velocity of the gas (vx, vy, vz) in m/s.
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
                
                # Add initial velocity (with Gaussian falloff)
                velocity_factor = np.exp(-0.5 * dist_squared)
                self.velocity_field[0, i, j, 0] += initial_velocity[0] * velocity_factor
                self.velocity_field[1, i, j, 0] += initial_velocity[1] * velocity_factor
        
        # Apply buoyancy effect (upward velocity component)
        # For gases lighter than air (buoyancy_factor > 1), add upward velocity
        # For gases heavier than air (buoyancy_factor < 1), add downward velocity
        buoyancy_velocity = 0.5 * (self.buoyancy_factor - 1.0)  # m/s, positive for upward
        for i in range(max(0, x_idx - 2*x_radius), min(self.grid_resolution[0], x_idx + 2*x_radius)):
            for j in range(max(0, y_idx - 2*y_radius), min(self.grid_resolution[1], y_idx + 2*y_radius)):
                dist_squared = ((i - x_idx) / x_radius)**2 + ((j - y_idx) / y_radius)**2
                if dist_squared <= 4:  # Within 2 radii
                    # Add vertical velocity component (in z-direction)
                    # Since we're using a 2D simulation, we can't directly set vz
                    # but we can use this to affect our advection calculations later
                    self.velocity_field[1, i, j, 0] += buoyancy_velocity * np.exp(-dist_squared)
        
        # Update simulation fields
        self._update_sim_fields()
    
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
            
            # Perform a simulation step to update velocity field
            self.sim.time_stepping.advance(self.time_step)
            
            # Get updated velocity field from simulation
            vx, vy = self.sim.state.get_state_physical()
            self.velocity_field[0, :, :, 0] = vx
            self.velocity_field[1, :, :, 0] = vy
            
            # Apply advection-diffusion to concentration field
            self._apply_advection_diffusion()
        
        return self.concentration
    
    def _apply_advection_diffusion(self):
        """
        Apply advection and diffusion to the concentration field.
        
        This uses an operator splitting approach, first applying advection
        using a semi-Lagrangian scheme, then applying diffusion using an
        explicit finite difference method.
        """
        # Copy the current concentration field
        new_concentration = np.copy(self.concentration)
        
        # Get the adjusted diffusion coefficient
        diff_coef = self._adjust_diffusion_for_temperature()
        
        # Calculate grid spacing
        dx = self.room_dimensions[0] / self.grid_resolution[0]
        dy = self.room_dimensions[1] / self.grid_resolution[1]
        
        # Apply advection using a semi-Lagrangian scheme
        for i in range(self.grid_resolution[0]):
            for j in range(self.grid_resolution[1]):
                for k in range(self.grid_resolution[2]):
                    # Get velocity at current position
                    vx = self.velocity_field[0, i, j, k]
                    vy = self.velocity_field[1, i, j, k]
                    
                    # Calculate the backtracked position
                    x_back = i - vx * self.time_step / dx
                    y_back = j - vy * self.time_step / dy
                    
                    # Ensure backtracked position is within bounds
                    x_back = max(0, min(x_back, self.grid_resolution[0] - 1))
                    y_back = max(0, min(y_back, self.grid_resolution[1] - 1))
                    
                    # Interpolate concentration at backtracked position using bilinear interpolation
                    x0 = int(x_back)
                    y0 = int(y_back)
                    x1 = min(x0 + 1, self.grid_resolution[0] - 1)
                    y1 = min(y0 + 1, self.grid_resolution[1] - 1)
                    
                    wx = x_back - x0
                    wy = y_back - y0
                    
                    c00 = self.concentration[x0, y0, k]
                    c01 = self.concentration[x0, y1, k]
                    c10 = self.concentration[x1, y0, k]
                    c11 = self.concentration[x1, y1, k]
                    
                    c0 = c00 * (1 - wy) + c01 * wy
                    c1 = c10 * (1 - wy) + c11 * wy
                    
                    new_concentration[i, j, k] = c0 * (1 - wx) + c1 * wx
        
        # Update concentration field after advection
        self.concentration = new_concentration
        
        # Apply diffusion using an explicit finite difference method
        new_concentration = np.copy(self.concentration)
        
        # Calculate the diffusion constant for the finite difference method
        diff_const = diff_coef * self.time_step / (dx * dy)
        
        # Ensure stability
        if diff_const > 0.25:
            print(f"Warning: Diffusion constant {diff_const} exceeds stability threshold 0.25")
            print("Reducing time step to ensure stability")
            self.time_step = 0.25 * (dx * dy) / diff_coef
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
                    )
                    
                    # Apply the diffusion equation: dc/dt = D * ∇²c
                    new_concentration[i, j, k] = self.concentration[i, j, k] + diff_coef * laplacian
        
        # Update the concentration field after diffusion
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
        self.velocity_field = np.zeros((2, *self.grid_resolution))
        self.current_time = 0.0
        self.is_initialized = False
        
        # Set background flow in velocity field
        self.velocity_field[0, :, :] = self.background_flow[0]  # vx
        self.velocity_field[1, :, :] = self.background_flow[1]  # vy
