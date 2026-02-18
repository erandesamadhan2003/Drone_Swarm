import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces

class NavigatorEnv(gym.Env):
    def __init__(self, obstacles, target):
        super(NavigatorEnv, self).__init__()
        
        self.obstacles = obstacles
        self.target = np.array(target)
        self.navigator_pos = np.array([0.0, 0.0, 50.0])
        self.navigator_vel = np.zeros(3)
        
        # Action space: velocity vector (vx, vy, vz)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation space: navigator pos (3) + target pos (3) + 3 closest obstacles (3x3=9) = 15
        # FIXED: Changed from 12 to 15
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        self.max_steps = 1000
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.navigator_pos = np.array([0.0, 0.0, 50.0])
        self.navigator_vel = np.zeros(3)
        self.current_step = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Find 3 closest obstacles
        if len(self.obstacles) > 0:
            distances = [np.linalg.norm(self.navigator_pos - obs) for obs in self.obstacles]
            closest_indices = np.argsort(distances)[:3]
            closest_obstacles = [self.obstacles[i] for i in closest_indices]
        else:
            closest_obstacles = [np.zeros(3)] * 3
        
        # Ensure we always have exactly 3 obstacles (pad with zeros if needed)
        while len(closest_obstacles) < 3:
            closest_obstacles.append(np.zeros(3))
        
        obs = np.concatenate([
            self.navigator_pos,      # 3 values
            self.target,             # 3 values
            closest_obstacles[0],    # 3 values
            closest_obstacles[1],    # 3 values
            closest_obstacles[2]     # 3 values
        ])  # Total: 15 values
        
        return obs.astype(np.float32)
    
    def step(self, action):
        # Update velocity
        max_speed = 15.0
        self.navigator_vel = action * max_speed
        
        # Update position
        dt = 0.1
        self.navigator_pos += self.navigator_vel * dt
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        self.current_step += 1
        terminated = self._is_at_target()
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _calculate_reward(self):
        # Distance to target
        target_dist = np.linalg.norm(self.navigator_pos - self.target)
        reward_target = -target_dist * 0.1
        
        # Obstacle avoidance
        reward_obstacle = 0
        for obs in self.obstacles:
            dist = np.linalg.norm(self.navigator_pos - obs)
            if dist < 5.0:
                reward_obstacle -= (5.0 - dist) * 2.0
        
        # Altitude penalty
        altitude_penalty = 0
        if self.navigator_pos[2] < 30 or self.navigator_pos[2] > 70:
            altitude_penalty = -abs(50 - self.navigator_pos[2]) * 0.1
        
        return reward_target + reward_obstacle + altitude_penalty
    
    def _is_at_target(self):
        return np.linalg.norm(self.navigator_pos - self.target) < 2.0

class VirtualNavigator:
    def __init__(self, obstacles, target):
        self.env = NavigatorEnv(obstacles, target)
        self.model = None
        
    def train(self, total_timesteps=10000):
        """Train the navigator using PPO"""
        print(f"Training virtual navigator for {total_timesteps} timesteps...")
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
        print("Training complete!")
        
    def save_model(self, path="models/navigator_ppo.zip"):
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}")
    
    def load_model(self, path="models/navigator_ppo.zip"):
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    def get_action(self, obs):
        """Get action from trained model"""
        if self.model is None:
            return np.zeros(3)
        action, _ = self.model.predict(obs, deterministic=True)
        return action
