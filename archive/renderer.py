"""
Visualization module for the DiffuScent simulator.

This module provides classes and functions for rendering
gas diffusion simulations in both 2D and 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go


class ColorMap:
    """
    Defines color mapping for gas concentration visualization.
    """
    
    @staticmethod
    def get_concentration_colormap():
        """
        Get a custom colormap for gas concentration visualization.
        
        Returns:
            matplotlib.colors.LinearSegmentedColormap: A colormap for gas concentration.
        """
        # Define colors for the colormap (transparent blue to dark blue)
        colors = [
            (1.0, 1.0, 1.0, 0.0),    # Transparent
            (0.8, 0.8, 1.0, 0.1),    # Very light blue, slightly visible
            (0.6, 0.6, 0.9, 0.3),    # Light blue
            (0.4, 0.4, 0.8, 0.5),    # Medium blue
            (0.2, 0.2, 0.7, 0.7),    # Darker blue
            (0.0, 0.0, 0.6, 0.9)     # Dark blue
        ]
        
        # Create a new colormap
        cmap_name = 'gas_concentration'
        return LinearSegmentedColormap.from_list(cmap_name, colors)
    
    @staticmethod
    def get_plotly_colorscale():
        """
        Get a Plotly colorscale for gas concentration visualization.
        
        Returns:
            list: A Plotly colorscale for gas concentration.
        """
        return [
            [0.0, 'rgba(255,255,255,0)'],    # Transparent
            [0.2, 'rgba(204,204,255,0.1)'],  # Very light blue, slightly visible
            [0.4, 'rgba(153,153,230,0.3)'],  # Light blue
            [0.6, 'rgba(102,102,204,0.5)'],  # Medium blue
            [0.8, 'rgba(51,51,179,0.7)'],    # Darker blue
            [1.0, 'rgba(0,0,153,0.9)']       # Dark blue
        ]


class MatplotlibRenderer:
    """
    A renderer for 2D visualization using Matplotlib.
    
    This renderer provides methods for visualizing gas concentration
    in 2D slices through the room.
    """
    
    def __init__(self, room_dimensions, grid_resolution):
        """
        Initialize the Matplotlib renderer.
        
        Args:
            room_dimensions (tuple): The dimensions of the room (x, y, z) in meters.
            grid_resolution (tuple): The resolution of the simulation grid.
        """
        self.room_dimensions = room_dimensions
        self.grid_resolution = grid_resolution
        self.colormap = ColorMap.get_concentration_colormap()
    
    def plot_slice(self, concentration, slice_dim='z', slice_idx=None,
                   ax=None, title=None, people_positions=None, source_position=None):
        """
        Plot a 2D slice of the gas concentration.
        
        Args:
            concentration (numpy.ndarray): The 3D gas concentration field.
            slice_dim (str): The dimension to slice along ('x', 'y', or 'z').
            slice_idx (int, optional): The index of the slice. If None, the middle slice is used.
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure is created.
            title (str, optional): The title of the plot.
            people_positions (list, optional): A list of (x, y, z) positions of people in the room.
            source_position (tuple, optional): The (x, y, z) position of the gas source.
            
        Returns:
            matplotlib.axes.Axes: The axis containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set default slice index to the middle of the dimension
        if slice_idx is None:
            if slice_dim == 'x':
                slice_idx = self.grid_resolution[0] // 2
            elif slice_dim == 'y':
                slice_idx = self.grid_resolution[1] // 2
            else:  # z
                slice_idx = self.grid_resolution[2] // 2
        
        # Calculate the actual position in meters
        if slice_dim == 'x':
            slice_pos = slice_idx / self.grid_resolution[0] * self.room_dimensions[0]
        elif slice_dim == 'y':
            slice_pos = slice_idx / self.grid_resolution[1] * self.room_dimensions[1]
        else:  # z
            slice_pos = slice_idx / self.grid_resolution[2] * self.room_dimensions[2]
        
        # Extract the 2D slice
        if slice_dim == 'x':
            slice_data = concentration[slice_idx, :, :]
            xlabel, ylabel = 'Y (m)', 'Z (m)'
            extent = [0, self.room_dimensions[1], 0, self.room_dimensions[2]]
        elif slice_dim == 'y':
            slice_data = concentration[:, slice_idx, :]
            xlabel, ylabel = 'X (m)', 'Z (m)'
            extent = [0, self.room_dimensions[0], 0, self.room_dimensions[2]]
        else:  # z
            slice_data = concentration[:, :, slice_idx]
            xlabel, ylabel = 'X (m)', 'Y (m)'
            extent = [0, self.room_dimensions[0], 0, self.room_dimensions[1]]
        
        # Plot the slice
        im = ax.imshow(slice_data.T, origin='lower', extent=extent,
                      cmap=self.colormap, interpolation='bilinear')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Concentration')
        
        # Add labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{slice_dim.upper()}-slice at {slice_pos:.2f} m")
        
        # Add people positions if provided
        if people_positions:
            for i, pos in enumerate(people_positions):
                x_pos, y_pos, z_pos = pos
                if (slice_dim == 'x' and abs(x_pos - slice_pos) < 0.5 or
                    slice_dim == 'y' and abs(y_pos - slice_pos) < 0.5 or
                    slice_dim == 'z' and abs(z_pos - slice_pos) < 0.5):
                    
                    # Determine the 2D coordinates based on the slice dimension
                    if slice_dim == 'x':
                        plot_x, plot_y = y_pos, z_pos
                    elif slice_dim == 'y':
                        plot_x, plot_y = x_pos, z_pos
                    else:  # z
                        plot_x, plot_y = x_pos, y_pos
                    
                    # Plot person position
                    ax.plot(plot_x, plot_y, 'ro', markersize=10, label=f"Person {i+1}")
        
        # Add source position if provided
        if source_position:
            x_pos, y_pos, z_pos = source_position
            if (slice_dim == 'x' and abs(x_pos - slice_pos) < 0.5 or
                slice_dim == 'y' and abs(y_pos - slice_pos) < 0.5 or
                slice_dim == 'z' and abs(z_pos - slice_pos) < 0.5):
                
                # Determine the 2D coordinates based on the slice dimension
                if slice_dim == 'x':
                    plot_x, plot_y = y_pos, z_pos
                elif slice_dim == 'y':
                    plot_x, plot_y = x_pos, z_pos
                else:  # z
                    plot_x, plot_y = x_pos, y_pos
                
                # Plot source position
                ax.plot(plot_x, plot_y, 'gx', markersize=10, label="Source")
        
        # Add legend if necessary
        if people_positions or source_position:
            ax.legend()
        
        return ax
    
    def plot_multiple_slices(self, concentration, slice_dim='z', num_slices=3, 
                            people_positions=None, source_position=None):
        """
        Plot multiple slices of the gas concentration.
        
        Args:
            concentration (numpy.ndarray): The 3D gas concentration field.
            slice_dim (str): The dimension to slice along ('x', 'y', or 'z').
            num_slices (int): The number of slices to plot.
            people_positions (list, optional): A list of (x, y, z) positions of people in the room.
            source_position (tuple, optional): The (x, y, z) position of the gas source.
            
        Returns:
            matplotlib.figure.Figure: The figure containing the plots.
        """
        # Create figure and axes
        fig, axes = plt.subplots(1, num_slices, figsize=(5 * num_slices, 5))
        
        # Get the dimension size
        if slice_dim == 'x':
            dim_size = self.grid_resolution[0]
        elif slice_dim == 'y':
            dim_size = self.grid_resolution[1]
        else:  # z
            dim_size = self.grid_resolution[2]
        
        # Plot slices at different positions
        for i in range(num_slices):
            slice_idx = int(dim_size * (i + 1) / (num_slices + 1))
            self.plot_slice(concentration, slice_dim, slice_idx, axes[i], 
                          people_positions=people_positions, 
                          source_position=source_position)
        
        plt.tight_layout()
        return fig


class PlotlyRenderer:
    """
    A renderer for 3D visualization using Plotly.
    
    This renderer provides methods for visualizing gas concentration
    in 3D using isosurfaces and volume rendering.
    """
    
    def __init__(self, room_dimensions, grid_resolution):
        """
        Initialize the Plotly renderer.
        
        Args:
            room_dimensions (tuple): The dimensions of the room (x, y, z) in meters.
            grid_resolution (tuple): The resolution of the simulation grid.
        """
        self.room_dimensions = room_dimensions
        self.grid_resolution = grid_resolution
        self.colorscale = ColorMap.get_plotly_colorscale()
    
    def create_3d_volume(self, concentration, threshold=0.01, opacity=0.5,
                        people_positions=None, source_position=None):
        """
        Create a 3D volume visualization of the gas concentration.
        
        Args:
            concentration (numpy.ndarray): The 3D gas concentration field.
            threshold (float): The minimum concentration to display.
            opacity (float): The opacity of the volume.
            people_positions (list, optional): A list of (x, y, z) positions of people in the room.
            source_position (tuple, optional): The (x, y, z) position of the gas source.
            
        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the 3D visualization.
        """
        # Create a new figure
        fig = go.Figure()
        
        # Prepare coordinate arrays
        x = np.linspace(0, self.room_dimensions[0], self.grid_resolution[0])
        y = np.linspace(0, self.room_dimensions[1], self.grid_resolution[1])
        z = np.linspace(0, self.room_dimensions[2], self.grid_resolution[2])
        
        # Create a copy of the concentration data and apply threshold
        conc_data = concentration.copy()
        conc_data[conc_data < threshold] = 0
        
        # Add volume trace
        fig.add_trace(go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=conc_data.flatten(),
            isomin=threshold,
            isomax=np.max(conc_data),
            opacity=opacity,
            surface_count=17,  # Number of isosurfaces
            colorscale=self.colorscale,
            colorbar=dict(title="Concentration"),
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        
        # Add room boundaries (wireframe box)
        fig.add_trace(go.Scatter3d(
            x=[0, self.room_dimensions[0], self.room_dimensions[0], 0, 0, 
               0, self.room_dimensions[0], self.room_dimensions[0], 0, 0,
               0, self.room_dimensions[0], self.room_dimensions[0], 0, 0],
            y=[0, 0, self.room_dimensions[1], self.room_dimensions[1], 0,
               0, 0, self.room_dimensions[1], self.room_dimensions[1], 0,
               0, 0, self.room_dimensions[1], self.room_dimensions[1], 0],
            z=[0, 0, 0, 0, 0,
               self.room_dimensions[2], self.room_dimensions[2], self.room_dimensions[2], self.room_dimensions[2], self.room_dimensions[2],
               0, 0, 0, 0, self.room_dimensions[2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))
        
        # Add people positions if provided
        if people_positions:
            for i, pos in enumerate(people_positions):
                x_pos, y_pos, z_pos = pos
                
                # Create a simple person representation (sphere)
                fig.add_trace(go.Scatter3d(
                    x=[x_pos],
                    y=[y_pos],
                    z=[z_pos],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle'
                    ),
                    name=f"Person {i+1}"
                ))
        
        # Add source position if provided
        if source_position:
            x_pos, y_pos, z_pos = source_position
            
            # Create a source representation (star)
            fig.add_trace(go.Scatter3d(
                x=[x_pos],
                y=[y_pos],
                z=[z_pos],
                mode='markers',
                marker=dict(
                    size=10,
                    color='green',
                    symbol='cross'
                ),
                name="Source"
            ))
        
        # Update layout
        fig.update_layout(
            title="DiffuScent 3D Gas Concentration Visualization",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode='manual',
                aspectratio=dict(
                    x=self.room_dimensions[0] / max(self.room_dimensions),
                    y=self.room_dimensions[1] / max(self.room_dimensions),
                    z=self.room_dimensions[2] / max(self.room_dimensions)
                )
            ),
            margin=dict(r=10, l=10, b=10, t=50)
        )
        
        return fig
    
    def create_3d_isosurface(self, concentration, isovalue=0.05, 
                           people_positions=None, source_position=None):
        """
        Create a 3D isosurface visualization of the gas concentration.
        
        Args:
            concentration (numpy.ndarray): The 3D gas concentration field.
            isovalue (float): The concentration value for the isosurface.
            people_positions (list, optional): A list of (x, y, z) positions of people in the room.
            source_position (tuple, optional): The (x, y, z) position of the gas source.
            
        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the 3D visualization.
        """
        # Create a new figure
        fig = go.Figure()
        
        # Prepare coordinate arrays
        x = np.linspace(0, self.room_dimensions[0], self.grid_resolution[0])
        y = np.linspace(0, self.room_dimensions[1], self.grid_resolution[1])
        z = np.linspace(0, self.room_dimensions[2], self.grid_resolution[2])
        
        # Add isosurface trace
        fig.add_trace(go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=concentration.flatten(),
            isomin=isovalue,
            isomax=np.max(concentration),
            surface=dict(count=5, fill=0.8, pattern='odd'),
            colorscale=self.colorscale,
            colorbar=dict(title="Concentration"),
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        
        # Add room boundaries (wireframe box)
        fig.add_trace(go.Scatter3d(
            x=[0, self.room_dimensions[0], self.room_dimensions[0], 0, 0, 
               0, self.room_dimensions[0], self.room_dimensions[0], 0, 0,
               0, self.room_dimensions[0], self.room_dimensions[0], 0, 0],
            y=[0, 0, self.room_dimensions[1], self.room_dimensions[1], 0,
               0, 0, self.room_dimensions[1], self.room_dimensions[1], 0,
               0, 0, self.room_dimensions[1], self.room_dimensions[1], 0],
            z=[0, 0, 0, 0, 0,
               self.room_dimensions[2], self.room_dimensions[2], self.room_dimensions[2], self.room_dimensions[2], self.room_dimensions[2],
               0, 0, 0, 0, self.room_dimensions[2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))
        
        # Add people positions if provided
        if people_positions:
            for i, pos in enumerate(people_positions):
                x_pos, y_pos, z_pos = pos
                
                # Create a simple person representation (sphere)
                fig.add_trace(go.Scatter3d(
                    x=[x_pos],
                    y=[y_pos],
                    z=[z_pos],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle'
                    ),
                    name=f"Person {i+1}"
                ))
        
        # Add source position if provided
        if source_position:
            x_pos, y_pos, z_pos = source_position
            
            # Create a source representation (star)
            fig.add_trace(go.Scatter3d(
                x=[x_pos],
                y=[y_pos],
                z=[z_pos],
                mode='markers',
                marker=dict(
                    size=10,
                    color='green',
                    symbol='cross'
                ),
                name="Source"
            ))
        
        # Update layout
        fig.update_layout(
            title="DiffuScent 3D Isosurface Visualization",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode='manual',
                aspectratio=dict(
                    x=self.room_dimensions[0] / max(self.room_dimensions),
                    y=self.room_dimensions[1] / max(self.room_dimensions),
                    z=self.room_dimensions[2] / max(self.room_dimensions)
                )
            ),
            margin=dict(r=10, l=10, b=10, t=50)
        )
        
        return fig
