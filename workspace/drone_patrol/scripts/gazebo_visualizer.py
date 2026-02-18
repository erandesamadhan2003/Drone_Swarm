#!/usr/bin/env python3
"""
Simplified Gazebo visualizer for drone swarm
Uses Gazebo's model spawning to show drone positions
"""
import subprocess
import time
import numpy as np
from swarm_controller import SwarmController

class GazeboVisualizer:
    def __init__(self):
        print("Starting Gazebo...")
        # Start Gazebo in background
        self.gazebo_process = subprocess.Popen(
            ['gazebo', '--verbose', 'worlds/drone_swarm.world'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        time.sleep(5)  # Wait for Gazebo to start
        
        # Initialize swarm
        start_positions = [
            [0, 0, 2], [2, 0, 2], [4, 0, 2],
            [0, 2, 2], [2, 2, 2], [4, 2, 2]
        ]
        
        self.swarm = SwarmController(6, start_positions)
        
        obstacles = [[10, 10, 0], [20, 15, 0], [30, 10, 0]]
        for obs in obstacles:
            self.swarm.add_obstacle(obs)
        
        self.swarm.set_target([50, 40, 2])
        
        # Spawn drone models
        self.spawn_drones()
    
    def spawn_drones(self):
        """Spawn simple sphere models to represent drones"""
        drone_sdf = """<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='drone_{id}'>
    <pose>{x} {y} {z} 0 0 0</pose>
    <static>false</static>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          <sphere>
            <radius>0.3</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
      <collision name='collision'>
        <geometry>
          <sphere>
            <radius>0.3</radius>
          </sphere>
        </geometry>
      </collision>
      <inertial>
        <mass>1.0</mass>
      </inertial>
    </link>
  </model>
</sdf>
"""
        colors = [
            (0, 0, 1),    # Blue
            (0, 1, 0),    # Green
            (1, 0.5, 0),  # Orange
            (0.5, 0, 1),  # Purple
            (0, 1, 1),    # Cyan
            (1, 0, 1)     # Magenta
        ]
        
        for i, agent in enumerate(self.swarm.agents):
            r, g, b = colors[i]
            sdf_content = drone_sdf.format(
                id=i,
                x=agent.position[0],
                y=agent.position[1],
                z=agent.position[2],
                r=r, g=g, b=b
            )
            
            # Write SDF to temp file
            with open(f'/tmp/drone_{i}.sdf', 'w') as f:
                f.write(sdf_content)
            
            # Spawn in Gazebo
            subprocess.run([
                'gz', 'model', '--spawn-file', f'/tmp/drone_{i}.sdf',
                '--model-name', f'drone_{i}'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Drones spawned in Gazebo")
    
    def update_positions(self):
        """Update drone positions in Gazebo"""
        for i, agent in enumerate(self.swarm.agents):
            cmd = f"gz model -m drone_{i} -x {agent.position[0]} -y {agent.position[1]} -z {agent.position[2]}"
            subprocess.run(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def run_simulation(self, steps=500):
        """Run the simulation and update Gazebo"""
        print("Running simulation...")
        
        for step in range(steps):
            # Update swarm
            self.swarm.update(dt=0.1)
            
            # Update Gazebo visualization
            if step % 5 == 0:  # Update every 5 steps to reduce overhead
                self.update_positions()
            
            if step % 50 == 0:
                state = self.swarm.get_swarm_state()
                center = state['center']
                print(f"Step {step}/{steps} - Center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")
            
            time.sleep(0.1)  # Real-time simulation
        
        print("Simulation complete!")
    
    def cleanup(self):
        """Stop Gazebo"""
        self.gazebo_process.terminate()
        self.gazebo_process.wait()

if __name__ == "__main__":
    viz = GazeboVisualizer()
    try:
        viz.run_simulation(steps=500)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        viz.cleanup()
