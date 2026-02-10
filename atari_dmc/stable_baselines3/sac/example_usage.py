#!/usr/bin/env python3
"""
Example usage of SAC with metrics tracking.

This script demonstrates how to use the enhanced SAC implementation
with weight norm, rank, and dormant unit tracking.
"""

import gym
import numpy as np
from sac import SAC

def main():
    # Create environment
    env = gym.make("Pendulum-v1")
    
    # Initialize SAC with metrics tracking enabled
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        # Metrics tracking parameters
        enable_metrics_tracking=True,
        metrics_log_frequencies={
            'weight_norm': 1000,    # Log weight norms every 1000 steps
            'rank': 5000,           # Log rank metrics every 5000 steps
            'dormant_units': 1000   # Log dormant units every 1000 steps
        }
    )
    
    # Train the model
    print("Training SAC with metrics tracking...")
    model.learn(total_timesteps=10000)
    
    # Get tracked metrics
    metrics = model.get_tracked_metrics()
    if metrics:
        print("\n=== Tracked Metrics ===")
        
        # Weight norms
        print("Weight Norms:")
        for network_name, weight_data in metrics['weight_norms'].items():
            print(f"  {network_name}: {weight_data.shape}")
            # Show some sample values
            non_zero_rows = np.any(weight_data != 0, axis=1)
            if np.any(non_zero_rows):
                last_logged = weight_data[non_zero_rows][-1]
                print(f"    Last logged values: {last_logged}")
        
        # Rank metrics
        print("\nRank Metrics:")
        for rank_type, rank_data in metrics['ranks'].items():
            non_zero_values = rank_data[rank_data != 0]
            if len(non_zero_values) > 0:
                print(f"  {rank_type}: {non_zero_values[-1]:.2f}")
        
        # Feature activity (dormant units)
        print("\nFeature Activity (Dormant Units):")
        for network_name, activity_data in metrics['feature_activity'].items():
            non_zero_rows = np.any(activity_data != 0, axis=1)
            if np.any(non_zero_rows):
                last_logged = activity_data[non_zero_rows][-1]
                print(f"  {network_name}: {last_logged}")
                # Calculate percentage of dormant units (low activity)
                dormant_percentage = (last_logged < 0.1).mean() * 100
                print(f"    Dormant units (%): {dormant_percentage:.1f}%")
    else:
        print("No metrics available. Make sure tracking is enabled.")
    
    # Test the trained model
    print("\nTesting trained model...")
    obs = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    
    env.close()
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
