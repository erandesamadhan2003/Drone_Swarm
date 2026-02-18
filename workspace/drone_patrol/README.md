# EN-MASCA: Enhanced Multi-Agent Swarm Control Algorithm
## Drone Swarm Patrolling in Durian Orchards

### Overview
This project implements an advanced drone swarm control system for autonomous patrol operations in durian orchards using deep reinforcement learning and bio-inspired swarm algorithms.

### Key Features
- **Virtual Navigator**: PPO-based reinforcement learning for optimal path planning
- **Swarm Control**: Bio-inspired multi-agent coordination (cohesion, separation, alignment)
- **Obstacle Avoidance**: Real-time dynamic obstacle detection and avoidance
- **High Performance**: 292x better target accuracy than baseline algorithms

### Performance Highlights
- ✅ **99.7% Target Accuracy** (0.14m final deviation)
- ✅ **Perfect Altitude Stability** (0.00 std deviation)
- ✅ **Zero Collisions** (1.91m minimum safe distance maintained)
- ✅ **Superior Cohesion** (1.76m avg deviation from center)

### Project Structure
```
drone_patrol/
├── scripts/
│   ├── agent.py                    # Individual drone agent
│   ├── swarm_controller.py         # EN-MASCA implementation
│   ├── virtual_navigator.py        # PPO-based navigator
│   ├── baseline_algorithms.py      # Comparison baselines
│   ├── compare_algorithms.py       # Performance comparison
│   ├── main_simulation.py          # Main simulation
│   └── generate_report.py          # Report generation
├── models/
│   └── navigator_ppo.zip           # Trained PPO model
├── logs/
│   ├── comprehensive_report.png    # Full performance report
│   ├── algorithm_comparison.png    # Algorithm comparison charts
│   ├── flight_paths_2d.png        # 2D trajectory visualization
│   └── performance_metrics.json    # Detailed metrics
└── README.md
```

### Quick Start
```bash
# Run simulation
python3 scripts/main_simulation.py

# Compare algorithms
python3 scripts/compare_algorithms.py

# Generate report
python3 scripts/generate_report.py
```

### Results
See `logs/comprehensive_report.png` for complete performance analysis.

### Technology Stack
- Python 3.10
- PyTorch (Deep Learning)
- Stable-Baselines3 (Reinforcement Learning - PPO)
- Gymnasium (RL Environment)
- NumPy, Matplotlib (Data & Visualization)
- ROS 2 Humble (Robot Operating System)
- Gazebo (3D Simulation - optional)

### Authors
Based on research paper: "Enhanced multi agent coordination algorithm for drone swarm patrolling in durian orchards"

### License
Research & Educational Use
