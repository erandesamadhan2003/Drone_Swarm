"""
PatrolPlanner class for dividing orchard into sectors and generating patrol paths.

This module handles:
- Sector assignment for each drone
- Zigzag patrol path generation
- Waypoint management
"""

import numpy as np


class PatrolPlanner:
    """
    Plans patrol routes by dividing the orchard into sectors.
    """
    
    def __init__(self, xmin=0, xmax=50, ymin=0, ymax=50, altitude=3.0):
        """
        Initialize patrol planner with orchard boundaries.
        
        Args:
            xmin, xmax: X-axis boundaries (meters)
            ymin, ymax: Y-axis boundaries (meters)
            altitude: Default patrol altitude (meters)
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.altitude = altitude
        
    def assign_sector(self, drone_id, total_drones):
        """
        Assign a vertical sector to a drone.
        
        Args:
            drone_id: Drone identifier (0-indexed)
            total_drones: Total number of drones in swarm
            
        Returns:
            Dictionary with sector bounds: {'xmin', 'xmax', 'ymin', 'ymax'}
        """
        sector_width = (self.xmax - self.xmin) / total_drones
        
        sector_xmin = self.xmin + (drone_id * sector_width)
        sector_xmax = self.xmin + ((drone_id + 1) * sector_width)
        
        return {
            'xmin': sector_xmin,
            'xmax': sector_xmax,
            'ymin': self.ymin,
            'ymax': self.ymax
        }
    
    def generate_zigzag_path(self, sector_bounds, step_size=5.0):
        """
        Generate vertical zigzag sweep path within a sector.
        Alternates up/down movement while shifting right across sector.
        
        Args:
            sector_bounds: Dictionary with 'xmin', 'xmax', 'ymin', 'ymax'
            step_size: Horizontal lane width (meters)
            
        Returns:
            List of waypoints [(x, y), ...]
        """
        waypoints = []
        
        xmin = sector_bounds['xmin']
        xmax = sector_bounds['xmax']
        ymin = sector_bounds['ymin']
        ymax = sector_bounds['ymax']
        
        # Add small margin to stay within bounds
        margin = 1.0
        xmin += margin
        xmax -= margin
        ymin += margin
        ymax -= margin
        
        # Generate VERTICAL sweeps (up/down) while moving horizontally
        current_x = xmin
        going_up = True
        
        while current_x <= xmax:
            if going_up:
                # Sweep upward
                waypoints.append((current_x, ymin))
                waypoints.append((current_x, ymax))
            else:
                # Sweep downward
                waypoints.append((current_x, ymax))
                waypoints.append((current_x, ymin))
            
            current_x += step_size
            going_up = not going_up
        
        return waypoints
    
    def get_patrol_path(self, drone_id, total_drones, step_size=2.0):
        """
        Get complete patrol path for a specific drone.
        
        Args:
            drone_id: Drone identifier (0-indexed)
            total_drones: Total number of drones
            step_size: Horizontal lane width for vertical sweeps (meters)
                      Default 2.0m for full coverage with grid resolution
            
        Returns:
            List of 3D waypoints [(x, y, z), ...]
        """
        sector = self.assign_sector(drone_id, total_drones)
        waypoints_2d = self.generate_zigzag_path(sector, step_size)
        
        # Add altitude to waypoints
        waypoints_3d = [(x, y, self.altitude) for x, y in waypoints_2d]
        
        return waypoints_3d
    
    def __repr__(self):
        return (f"PatrolPlanner(x=[{self.xmin}, {self.xmax}], "
                f"y=[{self.ymin}, {self.ymax}], alt={self.altitude})")

