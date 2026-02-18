import numpy as np
import matplotlib.pyplot as plt
from swarm_controller import SwarmController
from virtual_navigator import VirtualNavigator

class DroneSwarmSimulation:
    def __init__(self):
        # Initialize 6 drones in formation
        start_positions = [
            [0, 0, 50],
            [2, 0, 50],
            [4, 0, 50],
            [0, 2, 50],
            [2, 2, 50],
            [4, 2, 50]
        ]
        
        self.swarm = SwarmController(6, start_positions)
        
        # Add obstacles (durian trees)
        obstacles = [
            [10, 10, 0],
            [20, 15, 0],
            [30, 10, 0],
            [15, 25, 0],
            [35, 20, 0]
        ]
        
        for obs in obstacles:
            self.swarm.add_obstacle(obs)
        
        # Set target
        self.target = [50, 40, 50]
        self.swarm.set_target(self.target)
        
        # Create virtual navigator
        self.navigator = VirtualNavigator(obstacles, self.target)
        
        # Simulation parameters
        self.dt = 0.1
        self.history = []
        
    def train_navigator(self):
        """Train the virtual navigator"""
        print("Training virtual navigator...")
        self.navigator.train(total_timesteps=5000)
        self.navigator.save_model()
    
    def run_simulation(self, steps=500):
        """Run the simulation"""
        print("Running simulation...")
        
        for step in range(steps):
            # Update swarm
            self.swarm.update(self.dt)
            
            # Record state
            state = self.swarm.get_swarm_state()
            self.history.append(state)
            
            if step % 50 == 0:
                center = state['center']
                print(f"Step {step}/{steps} - Center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")
        
        print("Simulation complete!")
    
    def visualize_2d(self):
        """Visualize the flight paths in 2D"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: X-Y view
        for obs in self.swarm.obstacles:
            circle = plt.Circle((obs[0], obs[1]), 2.5, color='brown', alpha=0.3)
            ax1.add_patch(circle)
            ax1.plot(obs[0], obs[1], 'o', color='brown', markersize=10)
        
        ax1.plot(self.target[0], self.target[1], '*', color='red', markersize=20, label='Target')
        
        # Plot agent paths
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
        for i in range(len(self.swarm.agents)):
            positions = [state['positions'][i] for state in self.history]
            positions = np.array(positions)
            ax1.plot(positions[:, 0], positions[:, 1], 
                    color=colors[i], label=f'Drone {i+1}', alpha=0.7, linewidth=2)
            # Mark start and end
            ax1.plot(positions[0, 0], positions[0, 1], 'o', color=colors[i], markersize=8)
            ax1.plot(positions[-1, 0], positions[-1, 1], 's', color=colors[i], markersize=8)
        
        ax1.set_xlabel('X (m)', fontsize=12)
        ax1.set_ylabel('Y (m)', fontsize=12)
        ax1.set_title('Top View (X-Y Plane)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: Altitude over time
        for i in range(len(self.swarm.agents)):
            positions = [state['positions'][i] for state in self.history]
            positions = np.array(positions)
            time = np.arange(len(positions)) * self.dt
            ax2.plot(time, positions[:, 2], color=colors[i], label=f'Drone {i+1}', linewidth=2)
        
        ax2.axhline(y=50, color='red', linestyle='--', label='Target Altitude', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Altitude (m)', fontsize=12)
        ax2.set_title('Altitude Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logs/flight_paths_2d.png', dpi=150)
        print("Saved visualization to logs/flight_paths_2d.png")
        plt.close()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        positions = np.array([state['positions'] for state in self.history])
        
        # Calculate altitude statistics
        altitudes = positions[:, :, 2]
        
        # Calculate inter-drone distances
        distances = []
        for step_pos in positions:
            for i in range(len(step_pos)):
                for j in range(i+1, len(step_pos)):
                    dist = np.linalg.norm(step_pos[i] - step_pos[j])
                    distances.append(dist)
        
        # Calculate deviation from swarm center
        deviations = []
        for state in self.history:
            center = np.array(state['center'])
            for pos in state['positions']:
                dev = np.linalg.norm(np.array(pos) - center)
                deviations.append(dev)
        
        metrics = {
            'avg_altitude': np.mean(altitudes),
            'altitude_std': np.std(altitudes),
            'avg_inter_drone_distance': np.mean(distances),
            'min_inter_drone_distance': np.min(distances),
            'avg_deviation_from_center': np.mean(deviations),
            'max_deviation_from_center': np.max(deviations),
            'final_distance_to_target': np.linalg.norm(
                np.array(self.history[-1]['center']) - np.array(self.target)
            )
        }
        
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key:.<40} {value:.2f}")
        print("="*50 + "\n")
        
        return metrics

# Main execution
if __name__ == "__main__":
    print("="*50)
    print("DRONE SWARM PATROL SIMULATION - EN-MASCA")
    print("="*50 + "\n")
    
    sim = DroneSwarmSimulation()
    
    # Option 1: Train navigator (uncomment for first run)
    print("PHASE 1: Training Virtual Navigator")
    sim.train_navigator()
    
    # Option 2: Run simulation
    print("PHASE 2: Running Swarm Simulation")
    sim.run_simulation(steps=500)
    
    # Visualize results
    print("\nPHASE 3: Generating Visualizations")
    sim.visualize_2d()
    
    # Calculate metrics
    print("\nPHASE 4: Calculating Performance Metrics")
    sim.calculate_metrics()
    
    print("\n✓ Simulation Complete!")
