#!/usr/bin/env python3
"""
DiffuScent - Gas Profiles Module
Provides utility functions for managing gas profiles from configuration.
"""

import yaml
import os
import numpy as np

def load_gas_profiles(config_file='configs/gas_profiles.yaml'):
    """
    Load gas profiles and properties from the configuration file.
    
    Args:
        config_file: Path to the gas profiles configuration file
    
    Returns:
        Dictionary containing gas properties and profiles
    """
    try:
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"Error loading gas profiles: {e}")
        return None

def get_profile(profile_name, config_file='configs/gas_profiles.yaml'):
    """
    Get a specific gas profile by name.
    
    Args:
        profile_name: Name of the gas profile to retrieve
        config_file: Path to the gas profiles configuration file
    
    Returns:
        Dictionary with the profile data or None if not found
    """
    data = load_gas_profiles(config_file)
    if not data:
        return None
    
    profiles = data.get('profiles', {})
    if profile_name in profiles:
        return profiles[profile_name]
    else:
        print(f"Warning: Profile '{profile_name}' not found. Using 'standard' profile.")
        return profiles.get('standard')

def list_profiles(config_file='configs/gas_profiles.yaml'):
    """
    List all available gas profiles with descriptions.
    
    Args:
        config_file: Path to the gas profiles configuration file
    """
    data = load_gas_profiles(config_file)
    if not data:
        print("No profiles found.")
        return
    
    profiles = data.get('profiles', {})
    gas_props = data.get('gas_properties', {})
    
    print("\nAvailable Gas Profiles:")
    print("=" * 50)
    
    for name, profile in profiles.items():
        print(f"\n{name.upper()}")
        print(f"Description: {profile['description']}")
        print(f"Volume: {profile['volume']} liters")
        
        print("Composition:")
        for gas, fraction in profile['composition'].items():
            if fraction >= 0.01:
                print(f"  - {gas.capitalize()}: {fraction:.1%}")
            else:
                print(f"  - {gas.capitalize()}: {fraction:.6f}")
        
        # Calculate some basic properties
        density = get_density(profile['composition'], 37.0, gas_props.get('molecular_weights', {}))
        print(f"Density at 37°C: {density:.3f} kg/m³")
        
        # Estimate "potency" based on H2S content
        h2s = profile['composition'].get('hydrogen_sulfide', 0)
        potency = h2s * profile['volume'] / 0.00001
        
        if potency < 0.5:
            potency_desc = "Very mild"
        elif potency < 1:
            potency_desc = "Mild"
        elif potency < 2:
            potency_desc = "Moderate"
        elif potency < 5:
            potency_desc = "Strong"
        else:
            potency_desc = "Extreme"
            
        print(f"Odor potency: {potency_desc}")
    
    print("\n" + "=" * 50)

def get_density(composition, temperature, molecular_weights):
    """
    Calculate the density of a gas mixture based on composition and temperature.
    
    Args:
        composition: Dictionary of gas components and their mass fractions
        temperature: Temperature in Celsius
        molecular_weights: Dictionary of molecular weights for gases
    
    Returns:
        Density in kg/m³
    """
    # Convert to Kelvin
    T = temperature + 273.15
    
    # Gas constant
    R = 8.314  # J/(mol·K)
    
    # Standard pressure
    P = 101325  # Pa
    
    # Calculate average molecular weight
    avg_mw = sum(composition[gas] * molecular_weights.get(gas, 0)
                 for gas in composition if gas in molecular_weights)
    
    # Calculate density using ideal gas law
    density = (avg_mw * P) / (R * T) / 1000
    
    return density

def get_diffusion_coefficients(temperature, config_file='configs/gas_profiles.yaml'):
    """
    Get temperature-adjusted diffusion coefficients.
    
    Args:
        temperature: Temperature in Celsius
        config_file: Path to the gas profiles configuration file
    
    Returns:
        Dictionary of diffusion coefficients
    """
    # Load gas properties
    data = load_gas_profiles(config_file)
    if not data:
        return {}
    
    gas_props = data.get('gas_properties', {})
    diffusion_coefs = gas_props.get('diffusion_coefficients', {})
    
    # Convert to Kelvin
    T = temperature + 273.15
    
    # Reference temperature (20°C in Kelvin)
    T_ref = 293.15
    
    # Adjust coefficients based on temperature (simplified model)
    # D(T) = D_ref * (T/T_ref)^1.75
    adjusted = {}
    for gas, D_ref in diffusion_coefs.items():
        adjusted[gas] = D_ref * ((T / T_ref) ** 1.75)
    
    return adjusted

def calculate_detection_time(profile, distance, threshold_factor=1.0, config_file='configs/gas_profiles.yaml'):
    """
    Estimate detection time for a gas profile.
    This is a simplified model for educational purposes.
    
    Args:
        profile: Gas profile dictionary
        distance: Distance from source to detector in meters
        threshold_factor: Multiplier for detection threshold
        config_file: Path to the gas profiles configuration file
        
    Returns:
        Estimated time in seconds
    """
    # Load gas properties
    data = load_gas_profiles(config_file)
    if not data:
        return float('inf')
    
    gas_props = data.get('gas_properties', {})
    diffusion_coefs = gas_props.get('diffusion_coefficients', {})
    thresholds = gas_props.get('odor_thresholds', {})
    
    # Get H2S concentration and diffusion coefficient
    h2s_fraction = profile['composition'].get('hydrogen_sulfide', 0)
    D_h2s = diffusion_coefs.get('hydrogen_sulfide', 0.000018)
    
    # Get detection threshold
    threshold = thresholds.get('hydrogen_sulfide', 0.00047) * threshold_factor
    
    # Volume in m³
    volume = profile['volume'] / 1000.0
    
    # Initial concentration (mass/m³) if released in a 0.5m³ volume
    initial_concentration = h2s_fraction / 0.5
    
    # Check if enough H2S to be detected
    if h2s_fraction == 0:
        return float('inf')  # Never detected
    
    # Calculate time to reach threshold at given distance
    # This is derived by solving the diffusion equation for time
    # when C(r,t) = threshold
    # t = (r² / (4D)) * (1 / ln(M / ((4π)^(3/2) * threshold * r³)))
    
    mass = h2s_fraction * volume * 1000  # Convert to grams for calculation
    
    # Apply safety factor to avoid division by zero or negative values
    ln_arg = max(1e-10, (mass / ((4 * np.pi)**(3/2) * threshold * distance**3)))
    
    # Calculate time
    time = (distance**2 / (4 * D_h2s)) * (1 / np.log(ln_arg))
    
    # Apply some practical constraints
    if time < 0 or np.isnan(time):
        return float('inf')  # Invalid calculation, assume never detected
    
    return max(1.0, time)  # Minimum 1 second

def generate_config_for_profile(profile_name, output_file='configs/default.yaml',
                                profile_config='configs/gas_profiles.yaml'):
    """
    Generate a configuration file with the specified gas profile.
    
    Args:
        profile_name: Name of the gas profile to use
        output_file: Path to output configuration file
        profile_config: Path to the gas profiles configuration file
    """
    # Get the profile data
    profile_data = load_gas_profiles(profile_config)
    if not profile_data:
        print("Error: Could not load gas profiles.")
        return
    
    profile = profile_data['profiles'].get(profile_name)
    if not profile:
        print(f"Warning: Profile '{profile_name}' not found. Using 'standard' profile.")
        profile = profile_data['profiles'].get('standard')
    
    # Load base configuration
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
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
                'gas_properties': profile_data['gas_properties'],
                'diffusion': get_diffusion_coefficients(20, profile_config),
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
            'detection': {
                'thresholds': profile_data['gas_properties']['odor_thresholds'],
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
    
    # Update config with profile information
    config['gas'] = {
        'profile': profile_name,
        'volume': profile['volume'],
        'composition': profile['composition']
    }
    
    # Ensure diffusion coefficients exist for all gases in composition
    for gas in profile['composition']:
        if gas not in config['physics']['diffusion']:
            diffusion_coefs = profile_data['gas_properties']['diffusion_coefficients']
            config['physics']['diffusion'][gas] = diffusion_coefs.get(gas, 0.00002)
    
    # Write updated config
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created configuration with '{profile_name}' profile at {output_file}")
    print(f"Description: {profile['description']}")
    
    # Estimate detection times
    colleague_dist = np.sqrt((2.5 - 2.5)**2 + (3.0 - 0.8)**2 + (1.5 - 0.25)**2)
    boss_dist = np.sqrt((4.0 - 2.5)**2 + (3.5 - 0.8)**2 + (1.5 - 0.25)**2)
    
    colleague_time = calculate_detection_time(profile, colleague_dist, config_file=profile_config)
    boss_time = calculate_detection_time(profile, boss_dist, config_file=profile_config)
    
    print(f"\nEstimated detection times (simplified model):")
    if colleague_time == float('inf'):
        print("- Colleague: Not detected")
    else:
        print(f"- Colleague: ~{colleague_time:.1f}s")
        
    if boss_time == float('inf'):
        print("- Boss: Not detected")
    else:
        print(f"- Boss: ~{boss_time:.1f}s")
    
    return config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DiffuScent Gas Profile Manager')
    
    parser.add_argument('-l', '--list', action='store_true',
                       help='List all available gas profiles')
    
    parser.add_argument('-p', '--profile',
                       help='Generate config file for specified profile')
    
    parser.add_argument('-o', '--output', default='configs/default.yaml',
                       help='Output configuration file path')
                       
    parser.add_argument('-c', '--config', default='configs/gas_profiles.yaml',
                       help='Path to gas profiles configuration file')
    
    args = parser.parse_args()
    
    if args.list:
        list_profiles(args.config)
    elif args.profile:
        generate_config_for_profile(args.profile, args.output, args.config)
    else:
        parser.print_help()
