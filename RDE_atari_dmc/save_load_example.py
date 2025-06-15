#!/usr/bin/env python
"""
Example script demonstrating how to save and load a SAC model and its replay buffer mid-training.
"""

import os
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_dmc_env
from continuous_control.utils import make_env as dmc_make_env

def parse_args():
    parser = argparse.ArgumentParser(description="Save and load SAC model and replay buffer")
    parser.add_argument("--env", default="hopper-hop", help="Environment to use")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--save_path", default="./saved_models", help="Path to save models and buffers")
    parser.add_argument("--load_path", default=None, help="Path to load models and buffers from")
    parser.add_argument("--total_timesteps", default=1e6, type=int, help="Total timesteps to train for")
    parser.add_argument("--save_interval", default=1e5, type=int, help="Interval to save model and buffer")
    parser.add_argument("--mode", default="save", choices=["save", "load", "save_and_load"], 
                        help="Mode: save, load, or save_and_load")
    return parser.parse_args()

def save_model_and_buffer(model, save_path, timesteps):
    """Save the model and replay buffer to the specified path."""
    os.makedirs(save_path, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(save_path, f"model_{timesteps}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save the replay buffer
    buffer_path = os.path.join(save_path, f"replay_buffer_{timesteps}.pkl")
    model.save_replay_buffer(buffer_path)
    print(f"Replay buffer saved to {buffer_path}")
    
    return model_path, buffer_path

def load_model_and_buffer(load_path, env, timesteps=None):
    """Load the model and replay buffer from the specified path."""
    if timesteps is None:
        # Find the latest model and buffer
        model_files = [f for f in os.listdir(load_path) if f.startswith("model_") and f.endswith(".zip")]
        buffer_files = [f for f in os.listdir(load_path) if f.startswith("replay_buffer_") and f.endswith(".pkl")]
        
        if not model_files or not buffer_files:
            raise FileNotFoundError(f"No model or buffer files found in {load_path}")
        
        # Extract timesteps from the latest file
        latest_model = max(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        timesteps = int(latest_model.split("_")[-1].split(".")[0])
    
    model_path = os.path.join(load_path, f"model_{timesteps}.zip")
    buffer_path = os.path.join(load_path, f"replay_buffer_{timesteps}.pkl")
    
    # Load the model
    model = SAC.load(model_path, env=env, verbose=1)
    print(f"Model loaded from {model_path}")
    
    # Load the replay buffer
    model.load_replay_buffer(buffer_path)
    print(f"Replay buffer loaded from {buffer_path}")
    
    return model, timesteps

def main():
    args = parse_args()
    
    # Create the environment
    env = make_dmc_env(args.env, seed=args.seed)
    
    if args.mode == "save":
        # Create a new model
        model = SAC("MlpPolicy", env, verbose=1, seed=args.seed)
        
        # Train for a while
        print(f"Training for {args.save_interval} timesteps...")
        model.learn(total_timesteps=args.save_interval)
        
        # Save the model and replay buffer
        save_model_and_buffer(model, args.save_path, model.num_timesteps)
        print("Training stopped and model saved.")
        
    elif args.mode == "load":
        if args.load_path is None:
            print("Error: --load_path must be specified when using 'load' mode")
            return
        
        # Load the model and replay buffer
        model, timesteps = load_model_and_buffer(args.load_path, env)
        
        # Continue training
        print(f"Continuing training for {args.total_timesteps - timesteps} more timesteps...")
        model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=False)
        print("Training completed.")
        
    elif args.mode == "save_and_load":
        # Create a new model
        model = SAC("MlpPolicy", env, verbose=1, seed=args.seed)
        
        # Train for a while
        print(f"Training for {args.save_interval} timesteps...")
        model.learn(total_timesteps=args.save_interval)
        
        # Save the model and replay buffer
        save_model_and_buffer(model, args.save_path, model.num_timesteps)
        print("Training stopped and model saved.")
        
        # Load the model and replay buffer
        model, timesteps = load_model_and_buffer(args.save_path, env)
        
        # Continue training
        print(f"Continuing training for {args.total_timesteps - timesteps} more timesteps...")
        model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=False)
        print("Training completed.")
    
    env.close()

if __name__ == "__main__":
    main() 