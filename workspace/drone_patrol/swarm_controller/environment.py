"""
OrchardEnvironment class for managing simulation boundaries and constraints.

This module handles:
- Boundary enforcement
- Distance calculations
- Collision detection
"""

import numpy as np


class OrchardEnvironment:
    """
    Represents the orchard environment with boundaries and physical constraints.
    """
    
    def __init__(self, xmin=0, xmax=50, ymin=0, ymax=50, zmin=1.0, zmax=10.0):
        """
        Initialize orchard environment.
        
        Args:
            xmin, xmax: X-axis boundaries (meters)
            ymin, ymax: Y-axis boundaries (meters)
            zmin, zmax: Z-axis (altitude) boundaries (meters)
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        
    def keep_inside_bounds(self, drone):
        """
        Enforce boundary constraints on a drone.
        Reflects velocity if drone crosses boundary.
        
        Args:
            drone: DroneAgent object to constrain
        """
        # Check X boundaries
        if drone.position[0] < self.xmin:
            drone.position[0] = self.xmin
            drone.velocity[0] = abs(drone.velocity[0])  # Reflect
        elif drone.position[0] > self.xmax:
            drone.position[0] = self.xmax
            drone.velocity[0] = -abs(drone.velocity[0])  # Reflect
            
        # Check Y boundaries
        if drone.position[1] < self.ymin:
            drone.position[1] = self.ymin
            drone.velocity[1] = abs(drone.velocity[1])  # Reflect
        elif drone.position[1] > self.ymax:
            drone.position[1] = self.ymax
            drone.velocity[1] = -abs(drone.velocity[1])  # Reflect
            
        # Check Z boundaries
        if drone.position[2] < self.zmin:
            drone.position[2] = self.zmin
            drone.velocity[2] = abs(drone.velocity[2])  # Reflect
        elif drone.position[2] > self.zmax:
            drone.position[2] = self.zmax
            drone.velocity[2] = -abs(drone.velocity[2])  # Reflect
    
    def distance_between(self, drone1, drone2):
        """
        Calculate Euclidean distance between two drones.
        
        Args:
            drone1: First DroneAgent object
            drone2: Second DroneAgent object
            
        Returns:
            Distance in meters (float)
        """
        diff = drone1.position - drone2.position
        return np.linalg.norm(diff)
    
    def is_inside_bounds(self, position):
        """
        Check if a position is within the environment boundaries.
        
        Args:
            position: Position array [x, y, z]
            
        Returns:
            True if inside bounds, False otherwise
        """
        x, y, z = position
        return (self.xmin <= x <= self.xmax and 
                self.ymin <= y <= self.ymax and 
                self.zmin <= z <= self.zmax)
    
    def get_bounds(self):
        """
        Get environment boundaries as a dictionary.
        
        Returns:
            Dictionary with boundary values
        """
        return {
            'xmin': self.xmin,
            'xmax': self.xmax,
            'ymin': self.ymin,
            'ymax': self.ymax,
            'zmin': self.zmin,
            'zmax': self.zmax
        }
    
    def __repr__(self):
        return (f"OrchardEnvironment(x=[{self.xmin}, {self.xmax}], "
                f"y=[{self.ymin}, {self.ymax}], z=[{self.zmin}, {self.zmax}])")
