"""
DroneAgent class for multi-drone swarm patrol system.

This module implements individual drone behavior including:
- Position and velocity management
- Swarm rules (separation, alignment, cohesion)
- Patrol target following
"""

import numpy as np


class DroneAgent:
    """
    Represents a single drone in the swarm with autonomous behavior.
    """
    
    def __init__(self, drone_id, position, max_speed=3.0):
        """
        Initialize a drone agent.
        
        Args:
            drone_id: Unique identifier for the drone
            position: Initial position as [x, y, z]
            max_speed: Maximum velocity magnitude (m/s)
        """
        self.id = drone_id
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.max_speed = max_speed
        self.patrol_target = None
        self.neighbors = []
        self.sector_bounds = None  # Set via set_sector()
        
        # Waypoint tracking for patrol mode
        self.waypoints = []
        self.current_waypoint_index = 0
        self.patrol_mode = True  # True = follow waypoints, False = frontier mode
        
    def update_position(self, dt):
        """
        Update drone position based on current velocity.
        Enforces sector boundary constraints.
        Clears target when reached.
        
        Args:
            dt: Time step in seconds
        """
        self.position += self.velocity * dt
        
        # Clear target if reached (within 1.0 meter)
        if self.patrol_target is not None:
            distance_to_target = np.linalg.norm(self.position[:2] - self.patrol_target[:2])
            if distance_to_target < 1.0:
                self.patrol_target = None  # Force new target selection
        
        # === HARD SECTOR BOUNDARY ENFORCEMENT ===
        # Prevent drones from EVER leaving their assigned sector
        if self.sector_bounds is not None:
            xmin, xmax = self.sector_bounds
            
            # Hard clip with safety margin
            self.position[0] = np.clip(self.position[0], xmin + 0.5, xmax - 0.5)
            
            # Stop velocity at boundaries
            if self.position[0] <= xmin + 0.5:
                self.velocity[0] = max(0, self.velocity[0])  # Only allow rightward
            if self.position[0] >= xmax - 0.5:
                self.velocity[0] = min(0, self.velocity[0])  # Only allow leftward
        
    def set_patrol_target(self, target):
        """
        Set the current patrol waypoint target.
        
        Args:
            target: Target position as [x, y, z] or [x, y]
        """
        if len(target) == 2:
            # Add default altitude if only x, y provided
            self.patrol_target = np.array([target[0], target[1], self.position[2]])
        else:
            self.patrol_target = np.array(target, dtype=float)
    
    def set_sector(self, xmin, xmax):
        """
        Set the patrol sector boundaries for this drone.
        
        Args:
            xmin: Minimum x-coordinate (meters)
            xmax: Maximum x-coordinate (meters)
        """
        self.sector_bounds = (xmin, xmax)
    
    def set_waypoints(self, waypoints):
        """
        Set the patrol waypoint list.
        
        Args:
            waypoints: List of (x, y, z) waypoints
        """
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.patrol_mode = True
    
    def get_current_waypoint(self):
        """
        Get the current patrol waypoint.
        
        Returns:
            Current waypoint (x, y, z) or None if no waypoints
        """
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return None
        return self.waypoints[self.current_waypoint_index]
    
    def advance_waypoint(self):
        """
        Move to next waypoint. Loops back to start for continuous patrol.
        """
        self.current_waypoint_index += 1
        if self.current_waypoint_index >= len(self.waypoints):
            # Loop back to start - continuous systematic patrol
            self.current_waypoint_index = 0
            # Keep patrol_mode = True (never switch to frontier)
    
    def compute_separation(self, neighbors, min_distance=3.0):
        """
        Compute separation vector to avoid crowding neighbors.
        Uses strong repulsion for very close encounters.
        
        Args:
            neighbors: List of nearby DroneAgent objects
            min_distance: Minimum safe distance from neighbors (meters)
            
        Returns:
            Separation vector [vx, vy, vz]
        """
        separation = np.zeros(3)
        
        for neighbor in neighbors:
            diff = self.position - neighbor.position
            distance = np.linalg.norm(diff)
            
            if distance > 0:
                if distance < 1.5:  # Hard collision zone
                    # Apply very strong repulsion
                    separation += (diff / distance) * 5.0
                elif distance < min_distance:
                    # Normal repulsion
                    separation += diff / (distance ** 2)
                
        return separation
    
    def compute_alignment(self, neighbors):
        """
        Compute alignment vector to match average velocity of neighbors.
        
        Args:
            neighbors: List of nearby DroneAgent objects
            
        Returns:
            Alignment vector [vx, vy, vz]
        """
        if not neighbors:
            return np.zeros(3)
        
        avg_velocity = np.zeros(3)
        for neighbor in neighbors:
            avg_velocity += neighbor.velocity
            
        avg_velocity /= len(neighbors)
        
        return avg_velocity - self.velocity
    
    def compute_cohesion(self, neighbors):
        """
        Compute cohesion vector to move toward average position of neighbors.
        
        Args:
            neighbors: List of nearby DroneAgent objects
            
        Returns:
            Cohesion vector [vx, vy, vz]
        """
        if not neighbors:
            return np.zeros(3)
        
        center_of_mass = np.zeros(3)
        for neighbor in neighbors:
            center_of_mass += neighbor.position
            
        center_of_mass /= len(neighbors)
        
        return center_of_mass - self.position
    
    def apply_swarm_rules(self, neighbors, weights=(1.2, 0.2, 0.0)):
        """
        WAYPOINT-PRIORITY MOTION CONTROL
        1. If waypoints exist → follow waypoint
        2. Else → use swarm behavior
        """
        # === PRIORITY 1: WAYPOINT FOLLOWING ===
        if self.patrol_mode and self.waypoints and self.current_waypoint_index < len(self.waypoints):
            target = np.array(self.waypoints[self.current_waypoint_index])
            direction = target - self.position
            distance = np.linalg.norm(direction)
            
            if distance < 0.8:
                self.advance_waypoint()
            else:
                direction_norm = direction / (distance + 1e-6)
                separation = self.compute_separation(neighbors, min_distance=2.0)
                self.velocity = direction_norm * self.max_speed + 0.5 * separation
            
            speed = np.linalg.norm(self.velocity)
            if speed > self.max_speed:
                self.velocity = (self.velocity / speed) * self.max_speed
            return  # EXIT - waypoint mode complete
        
        # === PRIORITY 2: SWARM BEHAVIOR ===
        w_sep, w_align, w_coh = weights
        separation = self.compute_separation(neighbors)
        alignment = self.compute_alignment(neighbors)
        cohesion = self.compute_cohesion(neighbors)
        swarm_force = (w_sep * separation + w_align * alignment + w_coh * cohesion)
        
        if self.patrol_target is not None:
            target_direction = self.patrol_target - self.position
            distance_to_target = np.linalg.norm(target_direction)
            if distance_to_target > 0.5:
                target_force = target_direction / distance_to_target
                swarm_force += 2.0 * target_force
        
        self.velocity = 0.9 * self.velocity + 0.1 * swarm_force
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
            
    def __repr__(self):
        return f"Drone({self.id}, pos={self.position}, vel={self.velocity})"

