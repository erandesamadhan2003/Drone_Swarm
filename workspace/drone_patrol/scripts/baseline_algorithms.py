import numpy as np
from agent import DroneAgent

class BasicMASCA:
    """Basic Multi-Agent Swarm Control (no learning)"""
    def __init__(self, num_agents, start_positions):
        self.agents = [DroneAgent(i, pos) for i, pos in enumerate(start_positions)]
        self.obstacles = []
        self.target_position = None
        
    def add_obstacle(self, position):
        self.obstacles.append(np.array(position))
        
    def set_target(self, position):
        self.target_position = np.array(position)
        
    def get_neighbors(self, agent):
        neighbors = []
        for other in self.agents:
            if other.id != agent.id:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < agent.influence_radius:
                    neighbors.append(other)
        return neighbors
    
    def update(self, dt=0.1):
        forces = []
        
        for agent in self.agents:
            neighbors = self.get_neighbors(agent)
            force = agent.compute_interaction(neighbors, self.obstacles)
            
            # Simple target attraction (no learning)
            if self.target_position is not None:
                target_force = (self.target_position - agent.position) * 0.03
                force += target_force
            
            forces.append(force)
        
        for agent, force in zip(self.agents, forces):
            agent.velocity += force * dt
            speed = np.linalg.norm(agent.velocity)
            if speed > agent.speed:
                agent.velocity = agent.velocity / speed * agent.speed
            agent.update_position(dt)
    
    def get_swarm_state(self):
        return {
            'positions': [agent.position.tolist() for agent in self.agents],
            'velocities': [agent.velocity.tolist() for agent in self.agents],
            'center': np.mean([agent.position for agent in self.agents], axis=0).tolist()
        }

class RandomWalk:
    """Random walk baseline"""
    def __init__(self, num_agents, start_positions):
        self.agents = [DroneAgent(i, pos) for i, pos in enumerate(start_positions)]
        self.obstacles = []
        self.target_position = None
        
    def add_obstacle(self, position):
        self.obstacles.append(np.array(position))
        
    def set_target(self, position):
        self.target_position = np.array(position)
    
    def update(self, dt=0.1):
        for agent in self.agents:
            # Random walk with slight bias toward target
            random_force = np.random.randn(3) * 0.5
            if self.target_position is not None:
                target_force = (self.target_position - agent.position) * 0.02
                random_force += target_force
            
            agent.velocity += random_force * dt
            speed = np.linalg.norm(agent.velocity)
            if speed > agent.speed:
                agent.velocity = agent.velocity / speed * agent.speed
            agent.update_position(dt)
    
    def get_swarm_state(self):
        return {
            'positions': [agent.position.tolist() for agent in self.agents],
            'velocities': [agent.velocity.tolist() for agent in self.agents],
            'center': np.mean([agent.position for agent in self.agents], axis=0).tolist()
        }
