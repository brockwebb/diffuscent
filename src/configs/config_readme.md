# DiffuScent Configuration

This directory contains all configuration files for the DiffuScent simulation.

## Configuration Files

### default.yaml

The main configuration file for the simulation. Contains settings for:

- Mesh resolution and dimensions
- Time stepping parameters
- Physics model settings
- Gas release properties
- Detection settings
- Visualization options

### gas_profiles.yaml

Contains definitions for different gas profiles and properties:

- Gas properties (molecular weights, diffusion coefficients, odor thresholds)
- Predefined gas profiles with different compositions

## Working with Gas Profiles

The `gas_profiles.yaml` file contains several predefined gas profiles that can be selected when running the simulation:

1. **standard** - Balanced profile with moderate volume
2. **veggie** - Vegetarian diet profile with higher hydrogen content
3. **taco_bell** - High volume with extra methane ("Taco Bell Banger")
4. **egg_revenge** - Sulfur-rich profile
5. **sbd** - Small volume but high sulfur content ("Silent But Deadly")

## Adding New Gas Profiles

To add a new gas profile:

1. Edit the `gas_profiles.yaml` file
2. Add a new entry under the `profiles` section with the following structure:

```yaml
my_new_profile:
  description: "Description of the new profile"
  volume: 0.5  # Gas volume in liters
  composition:
    methane: 0.60  # Gas component fractions (must sum to 1.0)
    hydrogen_sulfide: 0.00001
    carbon_dioxide: 0.20
    nitrogen: 0.15
    oxygen: 0.05
```

3. Use the profile in the simulation:

```bash
python physics/gas_profiles.py --profile my_new_profile
python app.py
```

## Adding New Gases

To add a new gas type to the simulation:

1. Edit the `gas_profiles.yaml` file
2. Add the gas properties under each section in `gas_properties`:

```yaml
gas_properties:
  molecular_weights:
    my_new_gas: 30.0  # g/mol
  
  diffusion_coefficients:
    my_new_gas: 0.00002  # m²/s
  
  odor_thresholds:
    my_new_gas: 1.0  # ppm
```

3. Add the gas to profile compositions as needed

All parameters in the configuration files can be modified to adjust the simulation behavior.
