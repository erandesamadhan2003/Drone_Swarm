import numpy as np
import matplotlib.pyplot as plt
from swarm_controller import SwarmController
from baseline_algorithms import BasicMASCA, RandomWalk
import time

def run_algorithm(algorithm_class, name, obstacles, target, steps=500):
    """Run a single algorithm and collect metrics"""
    print(f"\nRunning {name}...")
    
    start_positions = [
        [0, 0, 50], [2, 0, 50], [4, 0, 50],
        [0, 2, 50], [2, 2, 50], [4, 2, 50]
    ]
    
    controller = algorithm_class(6, start_positions)
    for obs in obstacles:
        controller.add_obstacle(obs)
    controller.set_target(target)
    
    history = []
    start_time = time.time()
    
    for step in range(steps):
        controller.update(0.1)
        state = controller.get_swarm_state()
        history.append(state)
    
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    positions = np.array([state['positions'] for state in history])
    altitudes = positions[:, :, 2]
    
    distances = []
    for step_pos in positions:
        for i in range(len(step_pos)):
            for j in range(i+1, len(step_pos)):
                dist = np.linalg.norm(step_pos[i] - step_pos[j])
                distances.append(dist)
    
    deviations = []
    for state in history:
        center = np.array(state['center'])
        for pos in state['positions']:
            dev = np.linalg.norm(np.array(pos) - center)
            deviations.append(dev)
    
    metrics = {
        'name': name,
        'avg_altitude': np.mean(altitudes),
        'altitude_std': np.std(altitudes),
        'avg_inter_drone_distance': np.mean(distances),
        'min_inter_drone_distance': np.min(distances),
        'avg_deviation_from_center': np.mean(deviations),
        'max_deviation_from_center': np.max(deviations),
        'final_distance_to_target': np.linalg.norm(
            np.array(history[-1]['center']) - np.array(target)
        ),
        'computation_time': elapsed_time,
        'history': history
    }
    
    print(f"{name} completed in {elapsed_time:.2f}s")
    return metrics

def compare_all():
    """Compare EN-MASCA with baseline algorithms"""
    print("="*60)
    print("ALGORITHM COMPARISON: EN-MASCA vs Baselines")
    print("="*60)
    
    obstacles = [
        [10, 10, 0], [20, 15, 0], [30, 10, 0],
        [15, 25, 0], [35, 20, 0]
    ]
    target = [50, 40, 50]
    
    # Run all algorithms
    results = []
    results.append(run_algorithm(SwarmController, "EN-MASCA (Enhanced)", obstacles, target))
    results.append(run_algorithm(BasicMASCA, "Basic MASCA", obstacles, target))
    results.append(run_algorithm(RandomWalk, "Random Walk", obstacles, target))
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    
    metrics_to_compare = [
        'avg_altitude',
        'altitude_std',
        'avg_deviation_from_center',
        'min_inter_drone_distance',
        'final_distance_to_target',
        'computation_time'
    ]
    
    for metric in metrics_to_compare:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        for result in results:
            value = result[metric]
            print(f"  {result['name']:.<30} {value:.3f}")
    
    # Visualization
    visualize_comparison(results)
    
    return results

def visualize_comparison(results):
    """Create comparison visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['green', 'blue', 'red']
    
    # Plot 1: Flight paths
    ax = axes[0, 0]
    for idx, result in enumerate(results):
        history = result['history']
        for i in range(6):
            positions = [state['positions'][i] for state in history]
            positions = np.array(positions)
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=colors[idx], alpha=0.3, linewidth=1)
        
        # Plot center trajectory
        centers = [state['center'] for state in history]
        centers = np.array(centers)
        ax.plot(centers[:, 0], centers[:, 1], 
               color=colors[idx], label=result['name'], linewidth=3)
    
    ax.plot(50, 40, '*', color='red', markersize=20)
    ax.set_title('Flight Paths Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Altitude stability
    ax = axes[0, 1]
    for idx, result in enumerate(results):
        history = result['history']
        positions = np.array([state['positions'] for state in history])
        altitudes = positions[:, :, 2]
        time = np.arange(len(altitudes)) * 0.1
        avg_altitude = np.mean(altitudes, axis=1)
        ax.plot(time, avg_altitude, color=colors[idx], label=result['name'], linewidth=2)
    
    ax.axhline(y=50, color='black', linestyle='--', label='Target', linewidth=1)
    ax.set_title('Altitude Stability', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average Altitude (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Swarm cohesion
    ax = axes[1, 0]
    for idx, result in enumerate(results):
        history = result['history']
        deviations = []
        for state in history:
            center = np.array(state['center'])
            devs = [np.linalg.norm(np.array(pos) - center) for pos in state['positions']]
            deviations.append(np.mean(devs))
        
        time = np.arange(len(deviations)) * 0.1
        ax.plot(time, deviations, color=colors[idx], label=result['name'], linewidth=2)
    
    ax.set_title('Swarm Cohesion (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Avg Deviation from Center (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics bar chart
    ax = axes[1, 1]
    metrics = ['final_distance_to_target', 'altitude_std', 'avg_deviation_from_center']
    x = np.arange(len(metrics))
    width = 0.25
    
    for idx, result in enumerate(results):
        values = [result[m] for m in metrics]
        ax.bar(x + idx*width, values, width, label=result['name'], color=colors[idx])
    
    ax.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Final Distance\nto Target', 'Altitude\nStd Dev', 'Avg Deviation\nfrom Center'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('logs/algorithm_comparison.png', dpi=150)
    print("\nSaved comparison visualization to logs/algorithm_comparison.png")
    plt.close()

if __name__ == "__main__":
    results = compare_all()
    print("\n✓ Comparison Complete!")
