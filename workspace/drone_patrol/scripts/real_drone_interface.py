#!/usr/bin/env python3
"""
Interface for controlling real drones using MAVLink/MAVSDK
"""
import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)
import numpy as np
from swarm_controller import SwarmController

class RealDroneSwarm:
    def __init__(self, num_drones=6):
        self.num_drones = num_drones
        self.drones = []
        self.swarm = None
        
        # Connection strings (update with your actual drone IPs/ports)
        self.connection_strings = [
            "udp://:14540",  # Drone 0
            "udp://:14541",  # Drone 1
            "udp://:14542",  # Drone 2
            "udp://:14543",  # Drone 3
            "udp://:14544",  # Drone 4
            "udp://:14545",  # Drone 5
        ]
    
    async def connect_drones(self):
        """Connect to all drones"""
        print("Connecting to drones...")
        
        for i, conn_str in enumerate(self.connection_strings[:self.num_drones]):
            drone = System()
            await drone.connect(system_address=conn_str)
            
            print(f"Waiting for drone {i} to connect...")
            async for state in drone.core.connection_state():
                if state.is_connected:
                    print(f"Drone {i} connected!")
                    break
            
            self.drones.append(drone)
        
        print(f"All {self.num_drones} drones connected!")
    
    async def get_positions(self):
        """Get current positions of all drones"""
        positions = []
        
        for drone in self.drones:
            async for position in drone.telemetry.position():
                # Convert GPS to local NED coordinates
                # (You'll need to implement GPS-to-local conversion)
                pos = [
                    position.latitude_deg,   # Convert to meters
                    position.longitude_deg,  # Convert to meters
                    -position.absolute_altitude_m  # NED z is down
                ]
                positions.append(pos)
                break  # Get one reading
        
        return positions
    
    async def arm_and_takeoff(self, altitude=2.0):
        """Arm all drones and takeoff"""
        print("Arming and taking off...")
        
        tasks = []
        for i, drone in enumerate(self.drones):
            tasks.append(self._arm_and_takeoff_single(drone, i, altitude))
        
        await asyncio.gather(*tasks)
        print("All drones airborne!")
    
    async def _arm_and_takeoff_single(self, drone, drone_id, altitude):
        """Arm and takeoff a single drone"""
        # Wait for drone to be ready
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                break
        
        print(f"Arming drone {drone_id}...")
        await drone.action.arm()
        
        print(f"Taking off drone {drone_id}...")
        await drone.action.set_takeoff_altitude(altitude)
        await drone.action.takeoff()
        
        await asyncio.sleep(10)  # Wait for takeoff
    
    async def run_swarm_mission(self):
        """Execute swarm patrol mission"""
        # Get initial positions
        positions = await self.get_positions()
        
        # Initialize swarm controller
        self.swarm = SwarmController(self.num_drones, positions)
        
        # Add obstacles (from pre-mapped data)
        obstacles = [[10, 10, 0], [20, 15, 0], [30, 10, 0]]
        for obs in obstacles:
            self.swarm.add_obstacle(obs)
        
        # Set target
        self.swarm.set_target([50, 40, 2])
        
        print("Starting swarm mission...")
        
        # Start offboard mode for all drones
        for drone in self.drones:
            await drone.offboard.set_position_ned(PositionNedYaw(0, 0, 0, 0))
            await drone.offboard.start()
        
        # Mission loop
        for step in range(500):
            # Update swarm algorithm
            self.swarm.update(dt=0.1)
            
            # Send commands to each drone
            tasks = []
            for i, (drone, agent) in enumerate(zip(self.drones, self.swarm.agents)):
                # Convert to NED coordinates
                north = agent.position[0]
                east = agent.position[1]
                down = -agent.position[2]  # NED convention
                yaw = agent.yaw
                
                position = PositionNedYaw(north, east, down, yaw)
                tasks.append(drone.offboard.set_position_ned(position))
            
            await asyncio.gather(*tasks)
            
            if step % 50 == 0:
                state = self.swarm.get_swarm_state()
                print(f"Step {step}: Center at {state['center']}")
            
            await asyncio.sleep(0.1)
        
        print("Mission complete!")
    
    async def land_all(self):
        """Land all drones"""
        print("Landing all drones...")
        
        tasks = []
        for i, drone in enumerate(self.drones):
            print(f"Landing drone {i}...")
            tasks.append(drone.action.land())
        
        await asyncio.gather(*tasks)
        print("All drones landed!")

async def main():
    swarm = RealDroneSwarm(num_drones=6)
    
    try:
        # Connect to drones
        await swarm.connect_drones()
        
        # Arm and takeoff
        await swarm.arm_and_takeoff(altitude=2.0)
        
        # Run swarm mission
        await swarm.run_swarm_mission()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always land
        await swarm.land_all()

if __name__ == "__main__":
    asyncio.run(main())
