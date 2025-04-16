"""
Gas profiles for the DiffuScent simulator.

This module defines various gas profiles with different compositions,
diffusion properties, and detection thresholds.
"""

class GasProfile:
    """
    A class representing a specific gas profile for simulation.
    
    This class defines the composition, physical properties, and detection
    thresholds for a specific type of gas emission.
    
    Attributes:
        name (str): The name of the gas profile.
        description (str): A brief description of the gas profile.
        composition (dict): The composition of the gas by percentage.
        diffusion_coefficient (float): The effective diffusion coefficient.
        density (float): The density of the gas mixture in kg/m³.
        detection_threshold (float): The concentration threshold for human detection.
        buoyancy_factor (float): Factor affecting the buoyancy of the gas.
        fun_fact (str): An educational fact about the gas composition.
    """
    
    def __init__(self, 
                 name, 
                 description,
                 composition, 
                 diffusion_coefficient,
                 density,
                 detection_threshold,
                 buoyancy_factor=1.0,
                 fun_fact=None):
        """
        Initialize a gas profile.
        
        Args:
            name (str): The name of the gas profile.
            description (str): A brief description of the gas profile.
            composition (dict): The composition of the gas by percentage.
            diffusion_coefficient (float): The effective diffusion coefficient.
            density (float): The density of the gas mixture in kg/m³.
            detection_threshold (float): The concentration threshold for human detection.
            buoyancy_factor (float): Factor affecting the buoyancy of the gas.
            fun_fact (str): An educational fact about the gas composition.
        """
        self.name = name
        self.description = description
        self.composition = composition
        self.diffusion_coefficient = diffusion_coefficient
        self.density = density
        self.detection_threshold = detection_threshold
        self.buoyancy_factor = buoyancy_factor
        self.fun_fact = fun_fact if fun_fact else "Gas diffusion is affected by temperature and pressure!"
    
    def get_composition_string(self):
        """
        Get a formatted string representation of the gas composition.
        
        Returns:
            str: A formatted string showing the gas composition.
        """
        components = []
        for gas, percentage in self.composition.items():
            components.append(f"{gas}: {percentage:.1f}%")
        return ", ".join(components)
    
    def __str__(self):
        """
        Get a string representation of the gas profile.
        
        Returns:
            str: A string representation of the gas profile.
        """
        return f"{self.name}: {self.description}"


# Define common gas components and their properties
# Values based on scientific literature at standard temperature and pressure (STP)
GAS_COMPONENTS = {
    # Gas name: (diffusion coefficient in air (m²/s), density (kg/m³))
    "Methane": (0.196, 0.668),
    "Hydrogen Sulfide": (0.176, 1.363),
    "Carbon Dioxide": (0.142, 1.842),
    "Nitrogen": (0.137, 1.165),
    "Dimethyl Sulfide": (0.126, 2.110),
    "Methanethiol": (0.135, 1.992),
    "Hydrogen": (0.611, 0.082),
    "Ammonia": (0.198, 0.696),
}


# Define specific gas profiles
TACO_BELL_BANGER = GasProfile(
    name="Taco Bell Banger",
    description="High volume, medium odor emission with distinctive spicy notes",
    composition={
        "Methane": 55.0,
        "Hydrogen Sulfide": 3.2,
        "Carbon Dioxide": 25.0,
        "Nitrogen": 15.0,
        "Dimethyl Sulfide": 1.8,
    },
    diffusion_coefficient=0.175,  # Effective average
    density=1.05,                 # kg/m³, slightly less than air
    detection_threshold=0.08,     # Relatively easy to detect
    buoyancy_factor=1.1,          # Slightly buoyant
    fun_fact="The hydrogen sulfide in flatulence gives it that distinctive rotten egg smell, and can be detected by the human nose at concentrations as low as 0.5 parts per billion!"
)

VEGGIE_BURGER = GasProfile(
    name="Veggie Burger",
    description="Plant-based emission with high fiber aftermath",
    composition={
        "Methane": 35.0,
        "Hydrogen": 5.0,
        "Carbon Dioxide": 40.0,
        "Nitrogen": 18.0,
        "Hydrogen Sulfide": 2.0,
    },
    diffusion_coefficient=0.192,  # Higher due to hydrogen content
    density=0.95,                 # kg/m³, more buoyant than air
    detection_threshold=0.06,     # Somewhat easy to detect
    buoyancy_factor=1.25,         # More buoyant
    fun_fact="Plant-based diets can increase hydrogen production in the gut, leading to more buoyant gas that rises quickly!"
)

EGGS_REVENGE = GasProfile(
    name="Egg's Revenge",
    description="Sulfur-rich emission with high potency",
    composition={
        "Methane": 45.0,
        "Hydrogen Sulfide": 8.5,
        "Carbon Dioxide": 20.0,
        "Nitrogen": 15.0,
        "Methanethiol": 4.5,
        "Dimethyl Sulfide": 7.0,
    },
    diffusion_coefficient=0.162,  # Lower due to heavier sulfur compounds
    density=1.15,                 # kg/m³, heavier than air
    detection_threshold=0.03,     # Very easy to detect due to high sulfur content
    buoyancy_factor=0.9,          # Slightly less buoyant
    fun_fact="Eggs are high in sulfur-containing amino acids, which when broken down by gut bacteria, produce highly odorous sulfur compounds that can be detected at extremely low concentrations!"
)

SILENT_BUT_DEADLY = GasProfile(
    name="Silent But Deadly",
    description="Low volume, high concentration stealth emission",
    composition={
        "Methane": 40.0,
        "Hydrogen Sulfide": 7.0,
        "Carbon Dioxide": 15.0,
        "Nitrogen": 20.0,
        "Dimethyl Sulfide": 5.0,
        "Methanethiol": 6.0,
        "Ammonia": 7.0,
    },
    diffusion_coefficient=0.158,  # Lower diffusion rate
    density=1.20,                 # kg/m³, heavier than air
    detection_threshold=0.02,     # Extremely easy to detect
    buoyancy_factor=0.8,          # Less buoyant, tends to stay low
    fun_fact="'Silent but deadly' emissions often contain higher concentrations of sulfur compounds and tend to linger near the ground due to their higher density!"
)


# Dictionary of all available gas profiles
GAS_PROFILES = {
    "taco_bell_banger": TACO_BELL_BANGER,
    "veggie_burger": VEGGIE_BURGER,
    "eggs_revenge": EGGS_REVENGE,
    "silent_but_deadly": SILENT_BUT_DEADLY,
}


def get_gas_profile(profile_name):
    """
    Get a gas profile by name.
    
    Args:
        profile_name (str): The name of the profile to retrieve.
        
    Returns:
        GasProfile: The requested gas profile.
        
    Raises:
        ValueError: If the requested profile does not exist.
    """
    profile_key = profile_name.lower().replace(" ", "_")
    
    if profile_key in GAS_PROFILES:
        return GAS_PROFILES[profile_key]
    else:
        valid_profiles = ", ".join(GAS_PROFILES.keys())
        raise ValueError(f"Gas profile '{profile_name}' not found. Valid profiles are: {valid_profiles}")


def calculate_mixture_properties(composition):
    """
    Calculate the properties of a gas mixture based on its composition.
    
    Args:
        composition (dict): The composition of the gas mixture as percentages.
        
    Returns:
        tuple: (diffusion_coefficient, density, buoyancy_factor)
    """
    diffusion_coefficient = 0.0
    density = 0.0
    total_percentage = 0.0
    
    for gas, percentage in composition.items():
        if gas in GAS_COMPONENTS:
            gas_diff_coef, gas_density = GAS_COMPONENTS[gas]
            weight = percentage / 100.0
            
            diffusion_coefficient += gas_diff_coef * weight
            density += gas_density * weight
            total_percentage += percentage
    
    # Normalize in case percentages don't add up to 100%
    if total_percentage > 0:
        diffusion_coefficient = diffusion_coefficient * (100.0 / total_percentage)
        density = density * (100.0 / total_percentage)
    
    # Calculate buoyancy factor (relative to air density of ~1.225 kg/m³)
    buoyancy_factor = 1.225 / density if density > 0 else 1.0
    
    return diffusion_coefficient, density, buoyancy_factor


def create_custom_profile(name, description, composition, detection_threshold=None, fun_fact=None):
    """
    Create a custom gas profile based on the provided composition.
    
    Args:
        name (str): The name of the custom profile.
        description (str): A description of the custom profile.
        composition (dict): The gas composition as percentages.
        detection_threshold (float, optional): The detection threshold. If None,
            it will be calculated based on the composition.
        fun_fact (str, optional): A fun fact about the gas profile.
        
    Returns:
        GasProfile: A custom gas profile with calculated properties.
    """
    # Calculate properties based on composition
    diff_coef, density, buoyancy = calculate_mixture_properties(composition)
    
    # Calculate detection threshold based on sulfur content if not provided
    if detection_threshold is None:
        sulfur_content = composition.get("Hydrogen Sulfide", 0) + \
                         composition.get("Methanethiol", 0) + \
                         composition.get("Dimethyl Sulfide", 0)
        
        # Higher sulfur content means lower detection threshold (easier to detect)
        detection_threshold = 0.1 * (1.0 - min(sulfur_content / 30.0, 0.95))
    
    # Create and return the custom profile
    return GasProfile(
        name=name,
        description=description,
        composition=composition,
        diffusion_coefficient=diff_coef,
        density=density,
        detection_threshold=detection_threshold,
        buoyancy_factor=buoyancy,
        fun_fact=fun_fact
    )
