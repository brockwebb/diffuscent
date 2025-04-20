import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DiffuscentVisualizer:
    """
    Enhanced visualization class for DiffuScent with improved concentration
    scaling, detection threshold markers, and multiple display options.
    """
    def __init__(self, config_file='configs/default.yaml', results=None):
        """Initialize visualizer with configuration and results."""
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get visualization settings
        vis_config = self.config['visualization']
        self.scale_type = vis_config['concentration_scale']
        self.show_detection = vis_config['detection_markers']
        self.colormap = vis_config['color_map']
        self.slice_views = vis_config['slice_views']
        
        # Get mesh dimensions
        mesh_config = self.config['simulation']['mesh']
        self.nx = mesh_config['x_cells']
        self.ny = mesh_config['y_cells']
        self.nz = mesh_config['z_cells']
        self.Lx = mesh_config['x_size']
        self.Ly = mesh_config['y_size']
        self.Lz = mesh_config['z_size']
        
        # Create coordinate grids
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.z = np.linspace(0, self.Lz, self.nz)
        
        # Set detection thresholds
        self.thresholds = self.config['detection']['thresholds']
        self.detector_positions = self.config['detection']['positions']
        
        # Store results if provided
        self.results = results
        
    def set_results(self, results):
        """Set simulation results for visualization."""
        self.results = results
        
    def _prepare_concentration_data(self, data, gas_type):
        """
        Prepare concentration data for visualization with appropriate scaling.
        """
        # Get 3D array of concentration values
        if gas_type == 'h2s':
            # Convert to ppm for H2S
            concentrations = data[gas_type] * 1e6
            threshold = self.thresholds['hydrogen_sulfide']
        else:
            concentrations = data[gas_type]
            threshold = self.thresholds.get(gas_type, 0)
        
        # Reshape to 3D grid
        conc_3d = concentrations.reshape((self.nx, self.ny, self.nz))
        
        # Create mask for very low values (improved from previous version)
        # This helps focus on the interesting concentration range
        if self.scale_type == 'logarithmic':
            # Avoid log(0) issues by setting a minimum value
            min_val = max(np.min(concentrations[concentrations > 0]) / 10, 1e-20)
            mask = concentrations <= min_val
            
            # Apply logarithmic scaling
            concentrations = np.log10(np.maximum(concentrations, min_val))
            
            # Also log-scale the threshold
            log_threshold = np.log10(max(threshold, min_val))
        else:
            # Linear scaling
            mask = concentrations < threshold / 100
            log_threshold = threshold
        
        return conc_3d, mask, threshold, log_threshold
        
    def create_slice_plots(self, timestep_idx, gas_type='h2s', output_dir='visualizations'):
        """
        Create 2D slice visualizations of gas concentration for given timestep.
        """
        if self.results is None or timestep_idx >= len(self.results):
            print("No results available for visualization")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get data for this timestep
        data = self.results[timestep_idx]
        time = data['time']
        
        # Prepare concentration data
        conc_3d, mask, threshold, log_threshold = self._prepare_concentration_data(data, gas_type)
        
        # Create figure with subplots for each slice view
        fig, axes = plt.subplots(1, len(self.slice_views),
                                figsize=(5*len(self.slice_views), 5))
        
        # Handle case of single slice
        if len(self.slice_views) == 1:
            axes = [axes]
        
        for i, slice_config in enumerate(self.slice_views):
            ax = axes[i]
            plane = slice_config['plane']
            
            if plane == 'xy':
                # XY plane at given Z
                z_idx = int(slice_config['z'] / self.Lz * self.nz)
                z_pos = self.z[min(z_idx, self.nz-1)]
                slice_data = conc_3d[:, :, z_idx].T
                
                # Create plot
                im = ax.imshow(slice_data, origin='lower',
                              extent=[0, self.Lx, 0, self.Ly],
                              cmap=self.colormap)
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title(f'XY Plane at Z={z_pos:.2f}m')
                
            elif plane == 'xz':
                # XZ plane at given Y
                y_idx = int(slice_config['y'] / self.Ly * self.ny)
                y_pos = self.y[min(y_idx, self.ny-1)]
                slice_data = conc_3d[:, y_idx, :].T
                
                # Create plot
                im = ax.imshow(slice_data, origin='lower',
                              extent=[0, self.Lx, 0, self.Lz],
                              cmap=self.colormap)
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Z (m)')
                ax.set_title(f'XZ Plane at Y={y_pos:.2f}m')
                
            elif plane == 'yz':
                # YZ plane at given X
                x_idx = int(slice_config['x'] / self.Lx * self.nx)
                x_pos = self.x[min(x_idx, self.nx-1)]
                slice_data = conc_3d[x_idx, :, :].T
                
                # Create plot
                im = ax.imshow(slice_data, origin='lower',
                              extent=[0, self.Ly, 0, self.Lz],
                              cmap=self.colormap)
                
                ax.set_xlabel('Y (m)')
                ax.set_ylabel('Z (m)')
                ax.set_title(f'YZ Plane at X={x_pos:.2f}m')
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            
            # Set colorbar label based on gas type and scale
            if gas_type == 'h2s':
                if self.scale_type == 'logarithmic':
                    cbar.set_label('H₂S Concentration (log₁₀ ppm)')
                else:
                    cbar.set_label('H₂S Concentration (ppm)')
            else:
                if self.scale_type == 'logarithmic':
                    cbar.set_label(f'{gas_type.capitalize()} Concentration (log₁₀)')
                else:
                    cbar.set_label(f'{gas_type.capitalize()} Concentration')
            
            # Add threshold line in colorbar if enabled
            if self.show_detection and gas_type == 'h2s':
                if self.scale_type == 'logarithmic':
                    cbar.ax.axhline(y=log_threshold, color='r', linestyle='--', linewidth=2)
                    cbar.ax.text(0.5, log_threshold, 'Detection', color='r',
                                ha='center', va='bottom', transform=cbar.ax.transAxes)
                else:
                    norm = plt.Normalize(vmin=np.min(slice_data), vmax=np.max(slice_data))
                    y_pos = norm(threshold)
                    cbar.ax.axhline(y=y_pos, color='r', linestyle='--', linewidth=2)
                    cbar.ax.text(0.5, y_pos, 'Detection', color='r',
                                ha='center', va='bottom', transform=cbar.ax.transAxes)
            
            # Add detector positions if in this plane
            for detector in self.detector_positions:
                name = detector['name']
                dx, dy, dz = detector['x'], detector['y'], detector['z']
                
                if plane == 'xy' and abs(dz - z_pos) < 0.1:
                    ax.plot(dx, dy, 'ro', markersize=10)
                    ax.text(dx, dy, name, color='r', fontsize=10,
                            ha='center', va='bottom')
                elif plane == 'xz' and abs(dy - y_pos) < 0.1:
                    ax.plot(dx, dz, 'ro', markersize=10)
                    ax.text(dx, dz, name, color='r', fontsize=10,
                            ha='center', va='bottom')
                elif plane == 'yz' and abs(dx - x_pos) < 0.1:
                    ax.plot(dy, dz, 'ro', markersize=10)
                    ax.text(dy, dz, name, color='r', fontsize=10,
                            ha='center', va='bottom')
        
        # Set overall title with time information
        title = f"{gas_type.upper()} Concentration at t={time:.1f}s"
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        filename = f"{output_dir}/{gas_type}_{int(time)}s_slices.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        
        return filename
    
    def create_3d_visualization(self, timestep_idx, gas_type='h2s', output_dir='visualizations'):
        """
        Create 3D interactive visualization of gas concentration using Plotly.
        """
        if self.results is None or timestep_idx >= len(self.results):
            print("No results available for visualization")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get data for this timestep
        data = self.results[timestep_idx]
        time = data['time']
        
        # Prepare concentration data
        conc_3d, mask, threshold, log_threshold = self._prepare_concentration_data(data, gas_type)
        
        # Create meshgrid for 3D visualization
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Apply mask to focus on relevant concentrations
        masked_conc = np.ma.array(conc_3d, mask=mask.reshape(conc_3d.shape))
        
        # Create figure
        fig = go.Figure()
        
        # Create 3D volume visualization
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=masked_conc.flatten(),
            isomin=log_threshold if self.scale_type == 'logarithmic' else threshold,
            isomax=np.max(masked_conc),
            opacity=0.3,  # Makes the isosurfaces transparent
            surface_count=10,  # Number of isosurfaces
            colorscale=self.colormap,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        
        # Add detection boundary isosurface
        if self.show_detection:
            # Create an isosurface at detection threshold
            iso_value = log_threshold if self.scale_type == 'logarithmic' else threshold
            
            # Only add if there are values above threshold
            if np.any(masked_conc >= iso_value):
                fig.add_trace(go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=masked_conc.flatten(),
                    isomin=iso_value,
                    isomax=iso_value,
                    surface=dict(count=1),  # Fixed: removed invalid 'color' property
                    colorscale='Reds',
                    opacity=0.3,  # Moved opacity to the main level
                    hoverinfo='skip',
                    name='Detection Threshold'
                ))
        
        # Add detector positions
        detector_x = [d['x'] for d in self.detector_positions]
        detector_y = [d['y'] for d in self.detector_positions]
        detector_z = [d['z'] for d in self.detector_positions]
        detector_names = [d['name'] for d in self.detector_positions]
        
        fig.add_trace(go.Scatter3d(
            x=detector_x,
            y=detector_y,
            z=detector_z,
            mode='markers+text',
            marker=dict(
                size=8,
                color='red',
            ),
            text=detector_names,
            textposition="top center",
            name='Detectors'
        ))
        
        # Set up layout
        fig.update_layout(
            title=f"{gas_type.upper()} Concentration at t={time:.1f}s",
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectratio=dict(x=self.Lx/max(self.Lx, self.Ly, self.Lz),
                                 y=self.Ly/max(self.Lx, self.Ly, self.Lz),
                                 z=self.Lz/max(self.Lx, self.Ly, self.Lz))
            ),
            width=900,
            height=700,
            margin=dict(t=100, b=0, l=0, r=0)
        )
        
        # Add colorbar title based on gas type and scaling
        if gas_type == 'h2s':
            colorbar_title = 'H₂S Concentration (log₁₀ ppm)' if self.scale_type == 'logarithmic' else 'H₂S Concentration (ppm)'
        else:
            colorbar_title = f'{gas_type.capitalize()} Concentration (log₁₀)' if self.scale_type == 'logarithmic' else f'{gas_type.capitalize()} Concentration'
        
        fig.update_coloraxes(colorbar_title_text=colorbar_title)
        
        # Save as HTML file for interactivity
        filename = f"{output_dir}/{gas_type}_{int(time)}s_3d.html"
        fig.write_html(filename)
        
        return filename
    
    def create_detection_summary(self, detection_times, output_dir='visualizations'):
        """
        Create a summary visualization of detection times.
        """
        if not detection_times:
            print("No detection data available")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort detectors by detection time
        sorted_detections = sorted(detection_times.items(), key=lambda x: x[1])
        names = [d[0] for d in sorted_detections]
        times = [d[1] for d in sorted_detections]
        
        # Create horizontal bar chart
        bars = ax.barh(names, times, color='skyblue')
        
        # Add detection time labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 5, bar.get_y() + bar.get_height()/2,
                   f'{times[i]:.1f}s',
                   ha='left', va='center')
        
        # Set titles and labels
        ax.set_title('Detection Times by Observer', fontsize=14)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Observer', fontsize=12)
        
        # Add grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save figure
        filename = f"{output_dir}/detection_summary.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
        return filename
    
    def create_time_series_animation(self, gas_type='h2s', output_dir='visualizations'):
        """
        Create an animation of gas concentration over time using slice plots.
        """
        if self.results is None or len(self.results) == 0:
            print("No results available for animation")
            return
        
        # This is a placeholder for a full animation implementation
        # In a complete implementation, this would create a video file or animated GIF
        # showing the evolution of gas concentration over time
        
        print("Time series animation feature is a placeholder for future implementation")
        
        # For now, just create individual frames
        filenames = []
        for i in range(len(self.results)):
            filename = self.create_slice_plots(i, gas_type, output_dir)
            filenames.append(filename)
            
        return filenames
    
    def visualize_all(self, gas_type='h2s', output_dir='visualizations'):
        """
        Create all visualizations for all timesteps.
        """
        if self.results is None or len(self.results) == 0:
            print("No results available for visualization")
            return
        
        print(f"Creating visualizations for {len(self.results)} timesteps...")
        
        # Create 2D slice plots for each timestep
        for i in range(len(self.results)):
            self.create_slice_plots(i, gas_type, output_dir)
            
            # Create 3D visualization for selected timesteps (e.g., every 5th)
            if i % 5 == 0:
                self.create_3d_visualization(i, gas_type, output_dir)
        
        print("Visualizations created successfully!")


if __name__ == "__main__":
    # Example usage
    import pickle
    
    # Load simulation results
    try:
        with open('simulation_results.pkl', 'rb') as f:
            results = pickle.load(f)
            
        # Create visualizer
        visualizer = DiffuscentVisualizer('config.yaml', results['timesteps'])
        
        # Create visualizations
        visualizer.visualize_all()
        
        # Create detection summary if available
        if 'detection' in results and results['detection']:
            visualizer.create_detection_summary(results['detection'])
            
    except Exception as e:
        print(f"Error creating visualizations: {e}")
