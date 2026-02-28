"""
Main simulation script for multi-drone swarm patrol system.

This script demonstrates:
- Environment setup
- Drone initialization
- Patrol planning
- Swarm coordination
- Simulation loop
"""

import numpy as np
from environment import OrchardEnvironment
from drone_agent import DroneAgent
from patrol_planner import PatrolPlanner
from swarm_logic import SwarmController
from coverage_manager import CoverageManager


def main():
    """
    Run the drone swarm patrol simulation.
    """
    print("=" * 60)
    print("DRONE SWARM PATROL SIMULATION")
    print("=" * 60)
    
    # Simulation parameters
    dt = 0.1  # Time step (seconds)
    num_steps = 1500  # Total simulation steps (150 seconds)
    print_interval = 50  # Print every 10 steps for detailed tracking
    
    # Step 1: Create OrchardEnvironment
    print("\n[1] Creating Orchard Environment...")
    environment = OrchardEnvironment(
        xmin=0, xmax=50,
        ymin=0, ymax=50,
        zmin=1.0, zmax=10.0
    )
    print(f"    {environment}")
    
    # Step 2: Create 3 DroneAgent objects
    print("\n[2] Initializing Drones...")
    drones = [
        DroneAgent(drone_id=0, position=[2, 2, 3], max_speed=3.0),
        DroneAgent(drone_id=1, position=[2, 10, 3], max_speed=3.0),
        DroneAgent(drone_id=2, position=[2, 20, 3], max_speed=3.0)
    ]
    
    for drone in drones:
        print(f"    Drone {drone.id}: pos={drone.position}")
    
    # Step 3: Create PatrolPlanner
    print("\n[3] Creating Patrol Planner...")
    planner = PatrolPlanner(
        xmin=0, xmax=50,
        ymin=0, ymax=50,
        altitude=3.0
    )
    print(f"    {planner}")
    
    # Step 3.5: Create CoverageManager
    print("\n[3.5] Creating Coverage Manager...")
    coverage = CoverageManager(
        xmin=0, xmax=50,
        ymin=0, ymax=50,
        grid_resolution=2.0  # 2 meter grid cells
    )
    print(f"    {coverage}")
    
    # Step 4: Create SwarmController
    print("\n[4] Creating Swarm Controller...")
    controller = SwarmController(
        environment=environment,
        neighbor_distance=8.0,
        coverage_manager=coverage
    )
    
    for drone in drones:
        controller.add_drone(drone)
    print(f"    {controller}")
    
    # Step 5: Assign patrol sectors
    print("\n[5] Assigning Patrol Sectors...")
    controller.assign_patrols(planner)
    
    for drone in drones:
        sector = planner.assign_sector(drone.id, len(drones))
        drone.set_sector(sector['xmin'], sector['xmax'])  # Enforce hard sector boundaries
        print(f"    Drone {drone.id}: Sector x=[{sector['xmin']:.1f}, {sector['xmax']:.1f}]")
    
    # Step 6: Run simulation loop
    print("\n[6] Starting Simulation...")
    print(f"    Time step: {dt}s, Total steps: {num_steps}")
    print("-" * 60)
    
    # Track coverage growth
    previous_coverage = 0.0
    
    for step in range(num_steps):
        current_time = step * dt
        
        # Update swarm
        controller.update_swarm(dt, current_time)
        
        # Print drone positions at intervals
        if step % print_interval == 0:
            print(f"\nStep {step:4d} (t={current_time:6.1f}s):")
            
            for drone in drones:
                pos = drone.position
                vel = drone.velocity
                speed = np.linalg.norm(vel)
                wp_info = f"WP:{drone.current_waypoint_index}/{len(drone.waypoints)}" if drone.waypoints else "No WP"
                mode = "PATROL" if drone.patrol_mode else "FRONTIER"
                print(f"  Drone {drone.id}: "
                      f"pos=({pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}) "
                      f"speed={speed:4.2f} m/s | {mode} | {wp_info}")
            
            # Check for collisions
            collisions = controller.check_collisions(min_distance=2.0)
            if collisions:
                print(f"  ⚠️  WARNING: {len(collisions)} potential collision(s)!")
                for d1, d2, dist in collisions:
                    print(f"      Drone {d1} ↔ Drone {d2}: {dist:.2f}m apart")
            
            # Show swarm stats
            stats = controller.get_swarm_stats()
            print(f"  Swarm: avg_speed={stats['avg_speed']:.2f} m/s, "
                  f"max_speed={stats['max_speed']:.2f} m/s")
            
            # Show coverage stats with frontier and growth rate
            coverage_pct = coverage.get_coverage_percentage()
            overlap_pct = coverage.get_overlap_ratio()
            frontier_count = len(coverage.get_frontier_cells())
            coverage_delta = coverage_pct - previous_coverage
            previous_coverage = coverage_pct
            
            print(f"  Coverage: {coverage_pct:.1f}% | Overlap: {overlap_pct:.1f}% | "
                  f"Frontiers: {frontier_count} | Growth: +{coverage_delta:.1f}%")
    
    # Final summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print("\nFinal Positions:")
    for drone in drones:
        pos = drone.position
        print(f"  Drone {drone.id}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    final_stats = controller.get_swarm_stats()
    print(f"\nFinal Statistics:")
    print(f"  Average Speed: {final_stats['avg_speed']:.2f} m/s")
    print(f"  Maximum Speed: {final_stats['max_speed']:.2f} m/s")
    print(f"  Average Position: ({final_stats['avg_position'][0]:.2f}, "
          f"{final_stats['avg_position'][1]:.2f}, {final_stats['avg_position'][2]:.2f})")
    
    print(f"\nCoverage Statistics:")
    print(f"  Total Coverage: {coverage.get_coverage_percentage():.1f}%")
    print(f"  Overlap Ratio: {coverage.get_overlap_ratio():.1f}%")
    for drone in drones:
        if drone.sector_bounds:
            sector_cov = coverage.get_sector_coverage(*drone.sector_bounds)
            print(f"  Drone {drone.id} Sector: {sector_cov:.1f}% covered")


if __name__ == "__main__":
    main()

