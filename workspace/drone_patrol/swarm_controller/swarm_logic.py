"""
SwarmController class for managing multi-drone coordination.

This module handles:
- Swarm coordination
- Neighbor detection
- Collision avoidance
- Patrol assignment
"""

import numpy as np
from drone_agent import DroneAgent


class SwarmController:
    """
    Manages the entire drone swarm and coordinates their behavior.
    """
    
    def __init__(self, environment, neighbor_distance=8.0, coverage_manager=None):
        """
        Initialize swarm controller.
        
        Args:
            environment: OrchardEnvironment object
            neighbor_distance: Distance threshold for neighbor detection (meters)
            coverage_manager: CoverageManager object for adaptive patrol (optional)
        """
        self.environment = environment
        self.drones = []
        self.neighbor_distance = neighbor_distance
        self.coverage_manager = coverage_manager
        self.patrol_planner = None  # Set during assign_patrols
        self.patrol_paths = {}  # drone_id -> list of waypoints
        self.current_waypoint_index = {}  # drone_id -> current waypoint index
        self.target_update_interval = 5.0  # Update targets every 5 seconds
        self.time_since_target_update = {}
        
    def add_drone(self, drone):
        """
        Add a drone to the swarm.
        
        Args:
            drone: DroneAgent object
        """
        self.drones.append(drone)
        self.current_waypoint_index[drone.id] = 0
        
    def update_neighbors(self):
        """
        Update neighbor lists for all drones based on distance threshold.
        """
        for drone in self.drones:
            drone.neighbors = []
            
            for other in self.drones:
                if drone.id != other.id:
                    distance = self.environment.distance_between(drone, other)
                    if distance < self.neighbor_distance:
                        drone.neighbors.append(other)
    
    def update_swarm(self, dt, current_time=0):
        """
        Update all drones for one time step.
        
        Args:
            dt: Time step in seconds
            current_time: Current simulation time (seconds)
        """
        # Update neighbors
        self.update_neighbors()
        
        # Update patrol targets (adaptive or fixed)
        self._update_patrol_targets(current_time, dt)
        
        # Apply swarm rules and update positions
        for drone in self.drones:
            drone.apply_swarm_rules(drone.neighbors)
            drone.update_position(dt)
            self.environment.keep_inside_bounds(drone)
            
            # Update coverage tracking
            if self.coverage_manager is not None:
                self.coverage_manager.update_coverage(drone.position, current_time)
    
    def _update_patrol_targets(self, current_time, dt):
        """
        Update patrol targets for each drone.
        PATROL-FIRST ARCHITECTURE:
        1. Follow patrol waypoints until complete
        2. Switch to adaptive frontier mode only after patrol done
        
        Args:
            current_time: Current simulation time (seconds)
            dt: Time step (seconds)
        """
        for drone in self.drones:
            # Initialize timer if needed
            if drone.id not in self.time_since_target_update:
                self.time_since_target_update[drone.id] = 0
            
            self.time_since_target_update[drone.id] += dt
            
            # MODE 1: PATROL WAYPOINT MODE (Priority)
            if drone.patrol_mode and drone.waypoints:
                current_wp = drone.get_current_waypoint()
                
                if current_wp is not None:
                    # Set current waypoint as target
                    drone.set_patrol_target(current_wp)
                    
                    # Check if waypoint reached
                    wp_pos = np.array(current_wp)
                    distance = np.linalg.norm(drone.position - wp_pos)
                    
                    if distance < 0.8:  # Waypoint reached threshold
                        drone.advance_waypoint()
                else:
                    # All waypoints completed - switch to frontier mode
                    drone.patrol_mode = False
            
            # MODE 2: ADAPTIVE FRONTIER MODE (After patrol complete)
            elif self.coverage_manager is not None and self.patrol_planner is not None:
                # Check if target reached or time to update
                needs_update = False
                
                # Always update if no target
                if drone.patrol_target is None:
                    needs_update = True
                elif drone.patrol_target is not None:
                    distance = np.linalg.norm(drone.position[:2] - drone.patrol_target[:2])
                    if distance < 2.0:  # Reached target
                        needs_update = True
                
                # Also update periodically to adapt to coverage changes
                if self.time_since_target_update[drone.id] >= self.target_update_interval:
                    needs_update = True
                    self.time_since_target_update[drone.id] = 0
                
                if needs_update and drone.sector_bounds is not None:
                    # Get best frontier target in drone's sector
                    xmin, xmax = drone.sector_bounds
                    target_x, target_y = self.coverage_manager.get_best_frontier_target(
                        drone.position, xmin, xmax
                    )
                    drone.set_patrol_target([target_x, target_y, 3.0])
            
            # MODE 3: FALLBACK - Fixed zigzag patrol path (if no coverage manager)
            elif drone.id in self.patrol_paths:
                path = self.patrol_paths[drone.id]
                if path:
                    current_idx = self.current_waypoint_index[drone.id]
                    current_target = path[current_idx]
                    
                    # Set target
                    drone.set_patrol_target(current_target)
                    
                    # Check if reached current waypoint
                    target_pos = np.array([current_target[0], current_target[1], current_target[2]])
                    distance = np.linalg.norm(drone.position - target_pos)
                    
                    if distance < 2.0:  # Reached waypoint threshold
                        # Move to next waypoint (loop around)
                        self.current_waypoint_index[drone.id] = (current_idx + 1) % len(path)
    
    def check_collisions(self, min_distance=2.0):
        """
        Check for potential collisions between drones.
        
        Args:
            min_distance: Minimum safe distance (meters)
            
        Returns:
            List of tuples (drone1_id, drone2_id, distance) for close pairs
        """
        collisions = []
        
        for i, drone1 in enumerate(self.drones):
            for drone2 in self.drones[i+1:]:
                distance = self.environment.distance_between(drone1, drone2)
                if distance < min_distance:
                    collisions.append((drone1.id, drone2.id, distance))
                    
        return collisions
    
    def assign_patrols(self, patrol_planner):
        """
        Assign patrol paths to all drones using the patrol planner.
        Sets waypoints on each drone for patrol-first mode.
        
        Args:
            patrol_planner: PatrolPlanner object
        """
        self.patrol_planner = patrol_planner
        total_drones = len(self.drones)
        
        print("\n[DEBUG] Assigning patrol waypoints...")
        
        for i, drone in enumerate(self.drones):
            path = patrol_planner.get_patrol_path(i, total_drones)
            self.patrol_paths[drone.id] = path
            self.current_waypoint_index[drone.id] = 0
            
            # CRITICAL: Set waypoints on drone for patrol mode
            drone.set_waypoints(path)
            
            print(f"[DEBUG] Drone {drone.id}: Assigned {len(path)} waypoints, patrol_mode={drone.patrol_mode}")
            if path and len(path) > 0:
                print(f"[DEBUG]   First waypoint: {path[0]}")
                print(f"[DEBUG]   Last waypoint: {path[-1]}")
            
            # Set initial target
            if path:
                drone.set_patrol_target(path[0])
    
    def get_swarm_stats(self):
        """
        Get statistics about the swarm.
        
        Returns:
            Dictionary with swarm statistics
        """
        if not self.drones:
            return {}
            
        positions = np.array([drone.position for drone in self.drones])
        velocities = np.array([drone.velocity for drone in self.drones])
        
        return {
            'num_drones': len(self.drones),
            'avg_position': np.mean(positions, axis=0),
            'avg_speed': np.mean([np.linalg.norm(v) for v in velocities]),
            'max_speed': np.max([np.linalg.norm(v) for v in velocities]),
            'spread': np.std(positions, axis=0)
        }
    
    def __repr__(self):
        return f"SwarmController({len(self.drones)} drones, neighbor_dist={self.neighbor_distance}m)"

