"""
Detection module for the DiffuScent simulator.

This module handles the detection of gas concentrations,
calculating detection probabilities, and determining when
gas emissions would be detected by human observers.
"""

import math
import numpy as np


class GasDetector:
    """
    A class that handles the detection of gas concentrations.
    
    This class evaluates whether gas concentrations would be
    detected by human observers based on position, gas profile,
    and environmental factors.
    
    Attributes:
        detector_positions (list): List of positions where detection is evaluated.
        room_dimensions (tuple): The dimensions of the room (x, y, z) in meters.
        temperature (float): The ambient temperature of the room in Celsius.
        airflow_rate (float): The rate of air exchange in the room in m³/s.
    """
    
    def __init__(self, 
                 detector_positions=None,
                 room_dimensions=(5.0, 4.0, 2.5),
                 temperature=20.0,
                 airflow_rate=0.1):
        """
        Initialize the gas detector.
        
        Args:
            detector_positions (list, optional): List of positions where detection
                is evaluated. If None, default positions will be used.
            room_dimensions (tuple): The dimensions of the room (x, y, z) in meters.
            temperature (float): The ambient temperature of the room in Celsius.
            airflow_rate (float): The rate of air exchange in the room in m³/s.
        """
        self.room_dimensions = room_dimensions
        self.temperature = temperature
        self.airflow_rate = airflow_rate
        
        # Set default detector positions if none provided
        if detector_positions is None:
            # Default positions at four corners and center of the room at human nose height
            nose_height = 1.6  # Average human nose height in meters
            self.detector_positions = [
                (0.5, 0.5, nose_height),                                  # Near source
                (room_dimensions[0] - 0.5, 0.5, nose_height),             # Far corner 1
                (0.5, room_dimensions[1] - 0.5, nose_height),             # Far corner 2
                (room_dimensions[0] - 0.5, room_dimensions[1] - 0.5, nose_height),  # Far corner 3
                (room_dimensions[0] / 2, room_dimensions[1] / 2, nose_height)       # Center
            ]
        else:
            self.detector_positions = detector_positions
    
    def check_detection(self, concentration_field, gas_profile, grid_resolution):
        """
        Check if gas is detectable at any detector position.
        
        Args:
            concentration_field (numpy.ndarray): The gas concentration field.
            gas_profile (GasProfile): The gas profile being used.
            grid_resolution (tuple): The resolution of the simulation grid.
            
        Returns:
            dict: A dictionary with detection status for each detector position.
        """
        results = {}
        
        for i, position in enumerate(self.detector_positions):
            # Convert position from meters to grid indices
            x_idx = int(position[0] / self.room_dimensions[0] * grid_resolution[0])
            y_idx = int(position[1] / self.room_dimensions[1] * grid_resolution[1])
            z_idx = int(position[2] / self.room_dimensions[2] * grid_resolution[2])
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, grid_resolution[0]-1))
            y_idx = max(0, min(y_idx, grid_resolution[1]-1))
            z_idx = max(0, min(z_idx, grid_resolution[2]-1))
            
            # Get concentration at the detector position
            concentration = concentration_field[x_idx, y_idx, z_idx]
            
            # Adjust detection threshold based on temperature
            adjusted_threshold = self._adjust_threshold_for_temperature(gas_profile.detection_threshold)
            
            # Check if concentration exceeds the adjusted threshold
            is_detected = concentration >= adjusted_threshold
            
            # Calculate detection probability
            if concentration > 0:
                detection_prob = self._calculate_detection_probability(
                    concentration, adjusted_threshold
                )
            else:
                detection_prob = 0.0
            
            # Store results
            results[f"detector_{i+1}"] = {
                "position": position,
                "position_name": self._get_position_name(i),
                "concentration": concentration,
                "threshold": adjusted_threshold,
                "is_detected": is_detected,
                "detection_probability": detection_prob
            }
        
        return results
    
    def _get_position_name(self, detector_index):
        """
        Get a human-readable name for a detector position.
        
        Args:
            detector_index (int): The index of the detector.
            
        Returns:
            str: A human-readable name for the detector position.
        """
        position_names = [
            "Near Source",
            "Far Corner 1",
            "Far Corner 2",
            "Far Corner 3", 
            "Room Center"
        ]
        
        if detector_index < len(position_names):
            return position_names[detector_index]
        else:
            return f"Position {detector_index + 1}"
    
    def _adjust_threshold_for_temperature(self, base_threshold):
        """
        Adjust detection threshold based on temperature.
        
        Higher temperatures generally increase the volatility of odorous
        compounds, making them easier to detect.
        
        Args:
            base_threshold (float): The base detection threshold.
            
        Returns:
            float: The temperature-adjusted detection threshold.
        """
        # Reference temperature (20°C)
        ref_temp = 20.0
        
        # Calculate temperature factor
        # Higher temperatures lower the detection threshold (making detection easier)
        temp_factor = 1.0 - 0.01 * (self.temperature - ref_temp)
        
        # Ensure the factor stays within reasonable bounds
        temp_factor = max(0.5, min(temp_factor, 1.5))
        
        return base_threshold * temp_factor
    
    def _calculate_detection_probability(self, concentration, threshold):
        """
        Calculate the probability of gas detection based on concentration.
        
        This uses a sigmoid function to model the probability of detection,
        which increases as concentration exceeds the threshold.
        
        Args:
            concentration (float): The gas concentration.
            threshold (float): The detection threshold.
            
        Returns:
            float: The probability of detection (0.0 to 1.0).
        """
        # Calculate ratio of concentration to threshold
        ratio = concentration / max(threshold, 1e-10)
        
        # Use sigmoid function to calculate probability
        # At threshold, probability is 0.5
        # Below threshold, probability drops toward 0
        # Above threshold, probability approaches 1
        steepness = 5.0  # Controls how quickly probability changes around threshold
        probability = 1.0 / (1.0 + math.exp(-steepness * (ratio - 1.0)))
        
        return probability
    
    def estimate_time_to_detection(self, concentration_field, gas_profile, 
                                  grid_resolution, diffusion_model):
        """
        Estimate the time until gas is detected at each detector position.
        
        Args:
            concentration_field (numpy.ndarray): Current gas concentration field.
            gas_profile (GasProfile): The gas profile being used.
            grid_resolution (tuple): The resolution of the simulation grid.
            diffusion_model: The diffusion model being used.
            
        Returns:
            dict: A dictionary with estimated detection times for each detector position.
        """
        results = {}
        threshold = self._adjust_threshold_for_temperature(gas_profile.detection_threshold)
        
        for i, position in enumerate(self.detector_positions):
            # Convert position from meters to grid indices
            x_idx = int(position[0] / self.room_dimensions[0] * grid_resolution[0])
            y_idx = int(position[1] / self.room_dimensions[1] * grid_resolution[1])
            z_idx = int(position[2] / self.room_dimensions[2] * grid_resolution[2])
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, grid_resolution[0]-1))
            y_idx = max(0, min(y_idx, grid_resolution[1]-1))
            z_idx = max(0, min(z_idx, grid_resolution[2]-1))
            
            # Get current concentration
            current_concentration = concentration_field[x_idx, y_idx, z_idx]
            
            # If already detected, time is 0
            if current_concentration >= threshold:
                estimated_time = 0.0
            else:
                # Estimate time based on distance and diffusion rate
                # This is an approximation based on the diffusion equation
                source_position = (0.5, 0.5, 0.5)  # Assuming source near floor
                distance = math.sqrt(
                    (position[0] - source_position[0])**2 +
                    (position[1] - source_position[1])**2 +
                    (position[2] - source_position[2])**2
                )
                
                # Time estimate based on diffusion equation approximation
                # t ≈ x²/(2D) where x is distance and D is diffusion coefficient
                diffusion_coef = gas_profile.diffusion_coefficient
                estimated_time = (distance**2) / (2 * diffusion_coef)
                
                # Adjust for temperature and airflow
                # Higher temperature increases diffusion rate
                temp_factor = (self.temperature + 273.15) / 293.15  # Convert to Kelvin ratio
                
                # Airflow accelerates transport
                airflow_factor = 1.0 / (1.0 + self.airflow_rate)
                
                estimated_time = estimated_time / (temp_factor * airflow_factor)
            
            results[f"detector_{i+1}"] = {
                "position": position,
                "position_name": self._get_position_name(i),
                "current_concentration": current_concentration,
                "threshold": threshold,
                "estimated_time_to_detection": estimated_time,
                "will_be_detected": current_concentration > 0  # If concentration is positive, it will eventually be detected
            }
        
        return results
    
    def get_overall_detection_verdict(self, detection_results):
        """
        Get an overall detection verdict based on all detector results.
        
        Args:
            detection_results (dict): The results from check_detection.
            
        Returns:
            dict: An overall verdict with detection status and time.
        """
        any_detected = False
        min_detection_prob = 1.0
        min_detection_time = float('inf')
        detected_positions = []
        
        # Check each detector
        for detector_id, result in detection_results.items():
            if result.get("is_detected", False):
                any_detected = True
                detected_positions.append(result["position_name"])
            
            # Track lowest detection probability
            prob = result.get("detection_probability", 0.0)
            if prob < min_detection_prob:
                min_detection_prob = prob
            
            # Track earliest detection time
            if "estimated_time_to_detection" in result:
                time_to_detection = result["estimated_time_to_detection"]
                if time_to_detection < min_detection_time:
                    min_detection_time = time_to_detection
        
        # Format the verdict
        if any_detected:
            status = "BUSTED"
            message = f"Gas detected at: {', '.join(detected_positions)}"
            time_text = "Detection is immediate"
        else:
            status = "SAFE" if min_detection_prob < 0.1 else "RISKY"
            
            if min_detection_time < float('inf'):
                # Format time nicely
                if min_detection_time < 60:
                    time_text = f"Estimated time to detection: {min_detection_time:.1f} seconds"
                else:
                    time_text = f"Estimated time to detection: {min_detection_time/60:.1f} minutes"
            else:
                time_text = "No detection expected"
                
            message = f"Current detection probability: {min_detection_prob:.1%}"
        
        return {
            "status": status,
            "message": message,
            "time_info": time_text,
            "any_detected": any_detected,
            "min_detection_probability": min_detection_prob,
            "min_detection_time": min_detection_time
        }
