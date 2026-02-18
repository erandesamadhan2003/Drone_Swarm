import numpy as np

class DroneAgent:
    def __init__(self, agent_id, initial_position):
        self.id = agent_id
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.zeros(3)
        self.yaw = 0.0
        self.altitude = initial_position[2]
        
        # Control parameters
        self.speed = 10.0  # m/s
        self.influence_radius = 5.0  # meters
        
    def update_position(self, dt):
        """Update position based on velocity"""
        self.position += self.velocity * dt
        
    def compute_interaction(self, neighbors, obstacles):
        """
        Compute interaction forces based on:
        - Cohesion (move toward neighbors)
        - Separation (avoid collisions)
        - Alignment (match neighbor velocities)
        """
        if len(neighbors) == 0:
            return np.zeros(3)
        
        # Cohesion: move toward center of neighbors
        center = np.mean([n.position for n in neighbors], axis=0)
        cohesion_force = (center - self.position) * 0.01
        
        # Separation: avoid close neighbors
        separation_force = np.zeros(3)
        for neighbor in neighbors:
            diff = self.position - neighbor.position
            distance = np.linalg.norm(diff)
            if distance < 2.0 and distance > 0:  # minimum distance
                separation_force += diff / (distance ** 2)
        
        # Alignment: match neighbor velocities
        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        alignment_force = (avg_velocity - self.velocity) * 0.1
        
        # Obstacle avoidance
        obstacle_force = np.zeros(3)
        for obs_pos in obstacles:
            diff = self.position - obs_pos
            distance = np.linalg.norm(diff)
            if distance < 5.0 and distance > 0:
                obstacle_force += diff / (distance ** 2) * 2.0
        
        total_force = cohesion_force + separation_force + alignment_force + obstacle_force
        return total_force
