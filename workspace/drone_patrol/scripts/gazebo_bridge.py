#!/usr/bin/env python3
"""
Bridge between EN-MASCA algorithm and Gazebo simulation
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String
import numpy as np
from swarm_controller import SwarmController

class GazeboBridge(Node):
    def __init__(self):
        super().__init__('gazebo_bridge')
        
        # Initialize swarm controller
        start_positions = [
            [0, 0, 2],  # Starting at 2m altitude
            [2, 0, 2],
            [4, 0, 2],
            [0, 2, 2],
            [2, 2, 2],
            [4, 2, 2]
        ]
        
        self.swarm = SwarmController(6, start_positions)
        
        # Add obstacles
        obstacles = [[10, 10, 0], [20, 15, 0], [30, 10, 0]]
        for obs in obstacles:
            self.swarm.add_obstacle(obs)
        
        self.swarm.set_target([50, 40, 2])
        
        # Publishers for each drone
        self.publishers = []
        for i in range(6):
            pub = self.create_publisher(
                PoseStamped,
                f'/drone_{i}/command/pose',
                10
            )
            self.publishers.append(pub)
        
        # Timer for control loop (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Gazebo Bridge initialized')
    
    def control_loop(self):
        """Update swarm and publish commands to Gazebo"""
        # Update swarm state
        self.swarm.update(dt=0.1)
        
        # Get current state
        state = self.swarm.get_swarm_state()
        
        # Publish pose commands to each drone
        for i, (pub, agent) in enumerate(zip(self.publishers, self.swarm.agents)):
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            
            # Position
            msg.pose.position.x = float(agent.position[0])
            msg.pose.position.y = float(agent.position[1])
            msg.pose.position.z = float(agent.position[2])
            
            # Orientation (simplified - facing forward)
            msg.pose.orientation.w = 1.0
            
            pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    bridge = GazeboBridge()
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
