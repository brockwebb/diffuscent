#!/usr/bin/env python3
"""
DiffuScent - Gas Diffusion Simulator
Main application file that integrates the physics model and visualization.
"""

import os
import sys
import time
import argparse
import yaml
import pickle
import numpy as np
from datetime import datetime

# Import local modules directly (since we're in the src directory)
from physics.advection_diffusion import AdvectionDiffusionModel
from visualization.visualization import DiffuscentVisualizer

def load_config(config_file='configs/default.yaml'):
    """Load configuration from YAML file."""
    try:
        # Check if the config directory exists, create if not
        config_dir = os.path.dirname(config_file)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
            print(f"Created directory: {config_dir}")
        
        # Check if the config file exists
        if not os.path.exists(config_file):
            # Try to copy from root directory if it exists there
            root_config = 'config.yaml'
            if os.path.exists(root_config):
                print(f"Config file not found at {config_file}, copying from {root_config}")
                with open(root_config, 'r') as src, open(config_file, 'w') as dst:
                    dst.write(src.read())
            else:
                # Create a default config
                print(f"Creating default config at {config_file}")
                create_default_config(config_file)
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded configuration from {config_file}")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def create_default_config(config_file):
    """Create a default configuration file."""
    from physics.gas_profiles import load_gas_profiles
    
    # Load gas profiles from config
    gas_profiles_data = load_gas_profiles('configs/gas_profiles.yaml')
    if not gas_profiles_data:
        print("Warning: Could not load gas profiles. Using basic default configuration.")
        gas_profiles_data = {
            'gas_properties': {
                'molecular_weights': {
                    'methane': 16.04,
                    'hydrogen_sulfide': 34.08,
                    'nitrogen': 28.01,
                    'oxygen': 32.0,
                    'carbon_dioxide': 44.01
                },
                'odor_thresholds': {
                    'hydrogen_sulfide': 0.00047,
                    'methane': 5000.0
                }
            },
            'profiles': {
                'standard': {
                    'description': 'Standard profile',
                    'volume': 0.5,
                    'composition': {
                        'methane': 0.60,
                        'hydrogen_sulfide': 0.00001,
                        'carbon_dioxide': 0.20,
                        'nitrogen': 0.15,
                        'oxygen': 0.05
                    }
                }
            }
        }
    
    # Get standard profile
    standard_profile = gas_profiles_data['profiles']['standard']
    
    default_config = {
        'simulation': {
            'mesh': {
                'x_cells': 25,
                'y_cells': 20,
                'z_cells': 12,
                'x_size': 5.0,
                'y_size': 4.0,
                'z_size': 2.5
            },
            'time': {
                'dt': 1.0,
                'total_time': 300.0,
                'save_interval': 10
            },
            'performance': {
                'parallel_processing': True,
                'parallel_threads': 4,
                'visualization_during_computation': False
            }
        },
        'physics': {
            'model': 'advection_diffusion',
            'gas_properties': gas_profiles_data['gas_properties'],
            'diffusion': {
                'methane': 0.000022,
                'hydrogen_sulfide': 0.000018,
                'nitrogen': 0.000020,
                'oxygen': 0.000021,
                'carbon_dioxide': 0.000016
            },
            'initial_velocity': {
                'speed': 3.0,
                'angle_horizontal': 0.0,
                'angle_vertical': 30.0
            },
            'buoyancy': {
                'enabled': True,
                'room_temperature': 20.0,
                'gas_temperature': 37.0
            },
            'air_circulation': {
                'enabled': True,
                'pattern': 'basic',
                'strength': 0.2
            }
        },
        'gas': {
            'profile': 'standard',
            'volume': standard_profile['volume'],
            'composition': standard_profile['composition']
        },
        'detection': {
            'thresholds': gas_profiles_data['gas_properties']['odor_thresholds'],
            'positions': [
                {
                    'name': 'colleague',
                    'x': 2.5,
                    'y': 3.0,
                    'z': 1.5
                },
                {
                    'name': 'boss',
                    'x': 4.0,
                    'y': 3.5,
                    'z': 1.5
                }
            ]
        },
        'visualization': {
            'concentration_scale': 'logarithmic',
            'detection_markers': True,
            'color_map': 'plasma',
            'slice_views': [
                {'plane': 'xy', 'z': 1.5},
                {'plane': 'xz', 'y': 2.0},
                {'plane': 'yz', 'x': 2.5}
            ]
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created default configuration file at {config_file}")

def save_results(results, filename='../results/simulation_results.pkl'):
    """Save simulation results to file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

def load_results(filename='../results/simulation_results.pkl'):
    """Load simulation results from file."""
    try:
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"Results loaded from {filename}")
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def run_simulation(config_file='configs/default.yaml', output_dir='../results'):
    """Run the diffusion simulation and save results."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load configuration
    config = load_config(config_file)
    
    # Print simulation parameters
    print("\n=== DiffuScent Simulation ===")
    print(f"Room size: {config['simulation']['mesh']['x_size']}m x "
          f"{config['simulation']['mesh']['y_size']}m x "
          f"{config['simulation']['mesh']['z_size']}m")
    print(f"Total simulation time: {config['simulation']['time']['total_time']}s")
    print(f"Time step: {config['simulation']['time']['dt']}s")
    
    # Create and run model
    print("\nInitializing model...")
    model = AdvectionDiffusionModel(config_file)
    
    print("\nRunning simulation...")
    start_time = time.time()
    model.simulate()
    elapsed = time.time() - start_time
    
    print(f"\nSimulation completed in {elapsed:.2f} seconds")
    
    # Get results
    results = model.get_results()
    
    # Print detection summary
    if results['detection']:
        print("\nDetection Summary:")
        for name, detect_time in results['detection'].items():
            print(f"- {name}: Detected at t={detect_time:.1f}s")
    else:
        print("\nNo detection occurred during simulation time.")
    
    # Save results
    results_file = f"{output_dir}/simulation_{timestamp}.pkl"
    save_results(results, results_file)
    
    return results, results_file

def create_visualizations(results, config_file='configs/default.yaml', output_dir='../results/visualizations'):
    """Create visualizations from simulation results."""
    # Ensure visualization directory exists
    vis_dir = f"{output_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(vis_dir, exist_ok=True)
    
    print("\nCreating visualizations...")
    
    # Create visualizer
    visualizer = DiffuscentVisualizer(config_file, results['timesteps'])
    
    # Create all visualizations
    visualizer.visualize_all(gas_type='h2s', output_dir=vis_dir)
    
    # Create detection summary if available
    if results['detection']:
        visualizer.create_detection_summary(results['detection'], output_dir=vis_dir)
    
    print(f"\nVisualizations saved to {vis_dir}")
    return vis_dir

def print_gas_profile_info(config):
    """Print information about the gas profile used."""
    gas_config = config['gas']
    profile = gas_config['profile']
    volume = gas_config['volume']
    
    # Define descriptive names for profiles
    profile_names = {
        'standard': 'Standard',
        'veggie': 'Veggie Burger',
        'taco_bell': 'Taco Bell Banger',
        'egg_revenge': 'Egg\'s Revenge',
        'sbd': 'Silent But Deadly'
    }
    
    # Print profile info
    print("\nGas Profile:")
    print(f"- Profile: {profile_names.get(profile, profile)}")
    print(f"- Volume: {volume} liters")
    
    # Print composition
    print("- Composition:")
    for gas, fraction in gas_config['composition'].items():
        if fraction > 0:
            print(f"  - {gas.capitalize()}: {fraction:.1%}")

def print_detection_info(config, results):
    """Print information about detection settings and results."""
    detection_config = config['detection']
    
    print("\nDetection Settings:")
    print(f"- H₂S detection threshold: {detection_config['thresholds']['hydrogen_sulfide']} ppm")
    
    print("\nDetectors:")
    for detector in detection_config['positions']:
        name = detector['name']
        position = f"({detector['x']:.1f}m, {detector['y']:.1f}m, {detector['z']:.1f}m)"
        
        detect_time = results['detection'].get(name, None)
        if detect_time is not None:
            status = f"DETECTED at {detect_time:.1f}s"
        else:
            status = "Not detected"
        
        print(f"- {name} at {position}: {status}")

def print_simulation_summary(config, results):
    """Print a summary of the simulation results."""
    print("\n=== Simulation Summary ===")
    
    # Print gas profile info
    print_gas_profile_info(config)
    
    # Print detection information
    print_detection_info(config, results)
    
    # Print physics model info
    physics_config = config['physics']
    model_type = physics_config['model']
    print("\nPhysics Model:")
    print(f"- Type: {model_type}")
    
    if physics_config['buoyancy']['enabled']:
        print(f"- Buoyancy enabled: Room temp = {physics_config['buoyancy']['room_temperature']}°C, "
              f"Gas temp = {physics_config['buoyancy']['gas_temperature']}°C")
    
    if physics_config['air_circulation']['enabled']:
        print(f"- Room air circulation: {physics_config['air_circulation']['pattern']} "
              f"(strength: {physics_config['air_circulation']['strength']} m/s)")
    
    # Get maximum concentrations
    max_h2s = max(step['h2s_ppm'] for step in results['timesteps'])
    max_methane = max(step['methane_max'] for step in results['timesteps'])
    
    print("\nSimulation Results:")
    print(f"- Maximum H₂S concentration: {max_h2s:.6f} ppm")
    print(f"- Maximum methane concentration: {max_methane:.6f}")
    
    # Overall verdict
    if results['detection']:
        first_detection = min(results['detection'].values())
        print(f"\nVerdict: BUSTED! First detection at {first_detection:.1f}s")
    else:
        print("\nVerdict: SAFE! No detection occurred during simulation time.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DiffuScent Gas Diffusion Simulator')
    
    parser.add_argument('-c', '--config',
                        default='configs/default.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('-o', '--output',
                        default='../results',
                        help='Output directory for results')
    
    parser.add_argument('-v', '--visualize-only',
                        action='store_true',
                        help='Only create visualizations from existing results')
    
    parser.add_argument('-r', '--results-file',
                        help='Path to existing results file for visualization')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print banner
    print("\n" + "="*60)
    print("                   DiffuScent Simulator")
    print("            Gas Diffusion Physics Simulator")
    print("="*60)
    
    # Load configuration
    config = load_config(args.config)
    
    if args.visualize_only:
        # Only create visualizations from existing results
        if args.results_file:
            results = load_results(args.results_file)
        else:
            results = load_results()
        
        if results:
            vis_dir = create_visualizations(results, args.config, args.output)
            print_simulation_summary(config, results)
    else:
        # Run simulation and create visualizations
        results, results_file = run_simulation(args.config, args.output)
        vis_dir = create_visualizations(results, args.config,
                                       f"{args.output}/visualizations")
        print_simulation_summary(config, results)
    
    print("\nDiffuScent simulation completed!")

if __name__ == "__main__":
    main()
