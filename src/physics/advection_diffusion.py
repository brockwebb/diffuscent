import numpy as np
import fipy as fp
import time
import yaml
import os
from multiprocessing import Pool, cpu_count

class AdvectionDiffusionModel:
    """
    Enhanced gas diffusion model that includes advection, buoyancy, and 
    initial momentum for more realistic gas transport simulation.
    """
    def __init__(self, config_file='configs/default.yaml'):
        """Initialize the model with configuration parameters."""
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract key parameters
        self._setup_mesh()
        self._setup_physics_parameters()
        self._setup_time_parameters()
        self._setup_gas_properties()
        self._setup_detection_settings()
        
        # Storage for results
        self.results = []
        self.detection_times = {}
        
        print("AdvectionDiffusionModel initialized with mesh size: "
              f"{self.nx}x{self.ny}x{self.nz}")
    
    def _setup_mesh(self):
        """Create the computational mesh based on config."""
        mesh_config = self.config['simulation']['mesh']
        self.nx = mesh_config['x_cells']
        self.ny = mesh_config['y_cells']
        self.nz = mesh_config['z_cells']
        self.Lx = mesh_config['x_size']
        self.Ly = mesh_config['y_size']
        self.Lz = mesh_config['z_size']
        
        # Create 3D mesh
        self.mesh = fp.Grid3D(
            nx=self.nx, ny=self.ny, nz=self.nz,
            dx=self.Lx/self.nx, dy=self.Ly/self.ny, dz=self.Lz/self.nz
        )
        
        # Center coordinates for each cell
        self.cell_centers = self.mesh.cellCenters
        self.X, self.Y, self.Z = self.cell_centers
    
    def _setup_physics_parameters(self):
        """Set up physics-related parameters."""
        physics_config = self.config['physics']
        
        # Diffusion coefficients
        diff_config = physics_config['diffusion']
        self.D_methane = diff_config['methane']
        self.D_h2s = diff_config['hydrogen_sulfide']
        
        # Velocity components
        velocity_config = physics_config['initial_velocity']
        speed = velocity_config['speed']
        angle_h = np.radians(velocity_config['angle_horizontal'])
        angle_v = np.radians(velocity_config['angle_vertical'])
        
        # Initial velocity vector (converted to Cartesian components)
        self.initial_vx = speed * np.cos(angle_v) * np.cos(angle_h)
        self.initial_vy = speed * np.cos(angle_v) * np.sin(angle_h)
        self.initial_vz = -speed * np.sin(angle_v)  # Negative because down is negative z
        
        # Buoyancy parameters
        buoyancy_config = physics_config['buoyancy']
        self.buoyancy_enabled = buoyancy_config['enabled']
        self.room_temp = buoyancy_config['room_temperature'] + 273.15  # K
        self.gas_temp = buoyancy_config['gas_temperature'] + 273.15  # K
        
        # Air circulation
        circ_config = physics_config['air_circulation']
        self.air_circ_enabled = circ_config['enabled']
        self.air_circ_pattern = circ_config['pattern']
        self.air_circ_strength = circ_config['strength']
    
    def _setup_time_parameters(self):
        """Set up time-stepping parameters."""
        time_config = self.config['simulation']['time']
        self.dt = time_config['dt']
        self.total_time = time_config['total_time']
        self.save_interval = time_config['save_interval']
        self.steps = int(self.total_time / self.dt)
        self.save_steps = int(self.save_interval / self.dt)
    
    def _setup_gas_properties(self):
        """Set up gas release properties."""
        gas_config = self.config['gas']
        self.gas_profile = gas_config['profile']
        self.gas_volume = gas_config['volume']  # Liters
        
        # Convert to m³
        self.gas_volume_m3 = self.gas_volume / 1000.0
        
        # Gas composition (mass fractions)
        self.gas_composition = gas_config['composition']
        
        # Get molecular weights from config (no more hardcoding!)
        mw = self.config['physics']['gas_properties']['molecular_weights']
        
        # Gas constant
        R = 8.314  # J/(mol·K)
        
        # Calculate average molecular weight
        avg_mw = sum(self.gas_composition[gas] * mw[gas]
                     for gas in self.gas_composition)
        
        # Density at given temperature (kg/m³)
        self.gas_density = (avg_mw * 101325) / (R * self.gas_temp) / 1000
        
        # Air density at room temperature (simplified as 78% N₂, 21% O₂, 1% Ar)
        air_mw = 0.78 * mw['nitrogen'] + 0.21 * mw['oxygen'] + 0.01 * 40  # approx for argon
        self.air_density = (air_mw * 101325) / (R * self.room_temp) / 1000
        
        # Calculate buoyancy force
        self.buoyancy_acceleration = 9.81 * (self.air_density - self.gas_density) / self.air_density
    
    def _setup_detection_settings(self):
        """Set up detection thresholds and positions."""
        detection_config = self.config['detection']
        self.thresholds = detection_config['thresholds']
        self.detector_positions = detection_config['positions']
    
    def _create_room_airflow(self):
        """Create a basic room airflow pattern."""
        # Initialize velocity field
        vx = fp.CellVariable(name="vx", mesh=self.mesh, value=0.)
        vy = fp.CellVariable(name="vy", mesh=self.mesh, value=0.)
        vz = fp.CellVariable(name="vz", mesh=self.mesh, value=0.)
        
        if not self.air_circ_enabled:
            return vx, vy, vz
        
        # Create different patterns based on configuration
        if self.air_circ_pattern == "basic":
            # Basic circular pattern in xy-plane with slight upward flow
            strength = self.air_circ_strength
            room_center_x = self.Lx / 2
            room_center_y = self.Ly / 2
            
            # Calculate vector from center to each point
            dx = self.X - room_center_x
            dy = self.Y - room_center_y
            
            # Create circular pattern (counter-clockwise)
            distance = np.sqrt(dx**2 + dy**2)
            max_distance = np.sqrt((self.Lx/2)**2 + (self.Ly/2)**2)
            scale_factor = strength * (1 - distance/max_distance)
            
            # Set velocity components (rotate 90° for circular motion)
            vx.setValue(-dy * scale_factor)
            vy.setValue(dx * scale_factor)
            
            # Add a small upward component near walls and downward in center
            wall_influence = distance / max_distance
            vz.setValue(strength * 0.1 * (wall_influence - 0.5))
            
        elif self.air_circ_pattern == "hvac":
            # HVAC-like pattern with inflow/outflow
            # (This would be expanded in a full implementation)
            pass
        
        return vx, vy, vz
    
    def _create_initial_conditions(self):
        """Set up initial gas concentrations and velocity."""
        # Create gas concentration variables
        methane = fp.CellVariable(name="methane", mesh=self.mesh, value=0.)
        h2s = fp.CellVariable(name="hydrogen_sulfide", mesh=self.mesh, value=0.)
        
        # Release point (bottom center of room, near one end)
        release_x = self.Lx * 0.5
        release_y = self.Lx * 0.2
        release_z = self.Lz * 0.1  # Near floor
        
        # Create a small initial volume with gas
        # Calculate radius of spherical release
        release_radius = (3 * self.gas_volume_m3 / (4 * np.pi))**(1/3)
        
        # Set initial concentrations
        distance = ((self.X - release_x)**2 +
                   (self.Y - release_y)**2 +
                   (self.Z - release_z)**2)**0.5
        
        # Create release as a smooth Gaussian profile
        sigma = release_radius / 2
        gaussian = np.exp(-distance**2 / (2 * sigma**2))
        
        # Normalize to ensure total mass is conserved
        total_mass = self.gas_volume_m3 * self.gas_density
        cell_volume = (self.Lx * self.Ly * self.Lz) / (self.nx * self.ny * self.nz)
        gaussian_sum = np.sum(gaussian) * cell_volume
        normalized_gaussian = gaussian * total_mass / gaussian_sum
        
        # Set initial concentrations based on composition
        methane.setValue(normalized_gaussian * self.gas_composition['methane'])
        h2s.setValue(normalized_gaussian * self.gas_composition['hydrogen_sulfide'])
        
        # Create velocity field including initial jet and room airflow
        vx, vy, vz = self._create_room_airflow()
        
        # Add initial jet velocity in a small region around release point
        jet_radius = release_radius * 1.5
        jet_region = distance < jet_radius
        
        # Calculate proportion of jet to apply at each point (smooth transition)
        jet_factor = np.exp(-distance**2 / (2 * (jet_radius/1.5)**2))
        
        # Update velocities with initial jet
        vx.setValue(vx.value + self.initial_vx * jet_factor)
        vy.setValue(vy.value + self.initial_vy * jet_factor)
        vz.setValue(vz.value + self.initial_vz * jet_factor)
        
        return methane, h2s, vx, vy, vz
    
    def simulate(self):
        """Run the simulation."""
        start_time = time.time()
        
        # Set up initial conditions
        methane, h2s, vx, vy, vz = self._create_initial_conditions()
        
        # Save initial state
        self._save_timestep(0, methane, h2s)
        
        # Check if parallel processing is enabled
        parallel = self.config['simulation']['performance']['parallel_processing']
        
        if parallel:
            self._run_parallel_simulation(methane, h2s, vx, vy, vz)
        else:
            self._run_sequential_simulation(methane, h2s, vx, vy, vz)
        
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.2f} seconds")
        return self.results
    
    def _run_sequential_simulation(self, methane, h2s, vx, vy, vz):
        """Run simulation in single-process mode."""
        # Create equation terms
        methane_eq, h2s_eq = self._create_equations(methane, h2s, vx, vy, vz)
        
        # Time stepping loop
        for step in range(1, self.steps + 1):
            # Solve equations for this time step
            methane_eq.solve(var=methane, dt=self.dt)
            h2s_eq.solve(var=h2s, dt=self.dt)
            
            # Update buoyancy effects
            if self.buoyancy_enabled:
                self._update_buoyancy(methane, h2s, vx, vy, vz)
            
            # Save results at specified intervals
            if step % self.save_steps == 0:
                t = step * self.dt
                self._save_timestep(t, methane, h2s)
                self._check_detection(t, methane, h2s)
                print(f"Completed step {step}/{self.steps} (t={t:.1f}s)")
    
    def _run_parallel_simulation(self, methane, h2s, vx, vy, vz):
        """
        Run simulation with parallel time stepping.
        Note: This is a simplified approach. A full implementation would
        require more sophisticated domain decomposition.
        """
        # Placeholder implementation - this would need refinement
        pass
    
    def _create_equations(self, methane, h2s, vx, vy, vz):
        """Create the advection-diffusion equations."""
        # Create advection terms
        methane_advection = fp.PowerLawConvectionTerm(coeff=[-vx, -vy, -vz])
        h2s_advection = fp.PowerLawConvectionTerm(coeff=[-vx, -vy, -vz])
        
        # Create diffusion terms
        methane_diffusion = fp.DiffusionTerm(coeff=self.D_methane)
        h2s_diffusion = fp.DiffusionTerm(coeff=self.D_h2s)
        
        # Combined equations
        methane_eq = methane_advection + methane_diffusion
        h2s_eq = h2s_advection + h2s_diffusion
        
        return methane_eq, h2s_eq
    
    def _update_buoyancy(self, methane, h2s, vx, vy, vz):
        """Update velocity field based on buoyancy effects."""
        if not self.buoyancy_enabled:
            return
        
        # Calculate local gas density based on concentration
        # This is a simplified model; a full implementation would be more complex
        total_gas = methane.value + h2s.value
        
        # Only apply buoyancy where there's significant gas concentration
        significant_gas = total_gas > 1e-6
        
        # Apply vertical buoyancy force proportional to gas concentration
        # and buoyancy acceleration
        buoyancy_factor = np.where(significant_gas,
                                   total_gas * self.buoyancy_acceleration,
                                   0)
        
        # Update vertical velocity component
        current_vz = vz.value.copy()
        vz.setValue(current_vz + buoyancy_factor * self.dt)
    
    def _save_timestep(self, t, methane, h2s):
        """Save current concentrations for later visualization."""
        result = {
            'time': t,
            'methane': methane.value.copy(),
            'h2s': h2s.value.copy(),
            'methane_max': np.max(methane.value),
            'h2s_max': np.max(h2s.value),
            'h2s_ppm': np.max(h2s.value) * 1e6  # Convert to ppm
        }
        self.results.append(result)
    
    def _check_detection(self, t, methane, h2s):
        """Check if gas has reached detection threshold at detector positions."""
        for detector in self.detector_positions:
            name = detector['name']
            if name in self.detection_times:
                continue  # Already detected
                
            # Find closest cell to detector position
            dx = np.abs(self.X - detector['x'])
            dy = np.abs(self.Y - detector['y'])
            dz = np.abs(self.Z - detector['z'])
            dist = dx**2 + dy**2 + dz**2
            cell_idx = np.argmin(dist)
            
            # Get concentration at detector
            h2s_conc = h2s.value.flat[cell_idx]
            h2s_ppm = h2s_conc * 1e6  # Convert to ppm
            
            # Check if concentration exceeds detection threshold
            if h2s_ppm >= self.thresholds['hydrogen_sulfide']:
                self.detection_times[name] = t
                print(f"DETECTED by {name} at t={t:.1f}s! "
                      f"H2S: {h2s_ppm:.6f} ppm")

    def get_results(self):
        """Return simulation results and detection times."""
        return {
            'timesteps': self.results,
            'detection': self.detection_times
        }


if __name__ == "__main__":
    # Example usage
    model = AdvectionDiffusionModel('configs/default.yaml')
    results = model.simulate()
    
    # Print detection summary
    detection = model.get_results()['detection']
    if detection:
        print("\nDetection summary:")
        for name, time in detection.items():
            print(f"- {name}: Detected at t={time:.1f}s")
    else:
        print("\nNo detection occurred during simulation time.")
