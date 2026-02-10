#!/usr/bin/env python3
"""
Simple parameter counting script that calculates parameter counts
without requiring the full SAC environment setup.
"""

import torch as th
import torch.nn as nn


def calculate_mlp_parameters(input_dim, hidden_layers, output_dim):
    """
    Calculate the number of parameters in an MLP network.
    
    Args:
        input_dim: Input dimension
        hidden_layers: List of hidden layer sizes
        output_dim: Output dimension
        
    Returns:
        Total number of parameters
    """
    total_params = 0
    prev_dim = input_dim
    
    for hidden_dim in hidden_layers:
        # Linear layer: (input_dim * output_dim) + output_dim (bias)
        layer_params = (prev_dim * hidden_dim) + hidden_dim
        total_params += layer_params
        prev_dim = hidden_dim
    
    return total_params


def calculate_sac_parameters(observation_dim, action_dim, net_arch, num_agent=1, n_critics=2):
    """
    Calculate SAC parameters for given architecture.
    
    Args:
        observation_dim: Observation space dimension
        action_dim: Action space dimension  
        net_arch: Network architecture (e.g., [1024, 1024])
        num_agent: Number of agents
        n_critics: Number of critic networks per agent
        
    Returns:
        Dictionary with parameter breakdown
    """
    # Actor network parameters
    # Actor has: features -> hidden_layers -> mu + log_std
    actor_hidden_params = calculate_mlp_parameters(observation_dim, net_arch, net_arch[-1])
    actor_output_params = (net_arch[-1] * action_dim) * 2  # mu and log_std layers
    actor_params_per_agent = actor_hidden_params + actor_output_params
    
    # Critic network parameters  
    # Critic has: (obs + action) -> hidden_layers -> q_value
    critic_input_dim = observation_dim + action_dim
    critic_hidden_params = calculate_mlp_parameters(critic_input_dim, net_arch, net_arch[-1])
    critic_output_params = net_arch[-1] * 1  # single Q-value output
    critic_params_per_agent = (critic_hidden_params + critic_output_params) * n_critics
    
    # Entropy coefficient (1 parameter per agent)
    entropy_coef_params = 1
    
    # Per agent total
    params_per_agent = actor_params_per_agent + critic_params_per_agent + entropy_coef_params
    
    # Total for all agents
    total_params = params_per_agent * num_agent
    
    return {
        'observation_dim': observation_dim,
        'action_dim': action_dim,
        'net_arch': net_arch,
        'num_agent': num_agent,
        'n_critics': n_critics,
        'actor_params_per_agent': actor_params_per_agent,
        'critic_params_per_agent': critic_params_per_agent,
        'entropy_coef_params_per_agent': entropy_coef_params,
        'total_params_per_agent': params_per_agent,
        'total_params': total_params
    }


def print_parameter_analysis(analysis):
    """Print formatted parameter analysis."""
    print("=" * 70)
    print("SAC PARAMETER ANALYSIS")
    print("=" * 70)
    print(f"Observation Dimension: {analysis['observation_dim']}")
    print(f"Action Dimension: {analysis['action_dim']}")
    print(f"Network Architecture: {analysis['net_arch']}")
    print(f"Number of Agents: {analysis['num_agent']}")
    print(f"Critics per Agent: {analysis['n_critics']}")
    print()
    print(f"Per Agent Breakdown:")
    print(f"  Actor: {analysis['actor_params_per_agent']:,} parameters")
    print(f"  Critic: {analysis['critic_params_per_agent']:,} parameters")
    print(f"  Entropy Coef: {analysis['entropy_coef_params_per_agent']:,} parameters")
    print(f"  Total per Agent: {analysis['total_params_per_agent']:,} parameters")
    print()
    print(f"TOTAL PARAMETERS: {analysis['total_params']:,}")
    print("=" * 70)


def compare_architectures():
    """Compare different network architectures."""
    # Common environment dimensions
    pendulum_obs_dim = 3  # Pendulum-v1
    pendulum_action_dim = 1
    
    # Different architectures to compare
    architectures = [
        ([1024, 1024], 1),
        ([2048, 2048], 1), 
        ([1024, 1024], 2),
        ([2048, 2048], 2),
    ]
    
    results = {}
    
    for net_arch, num_agent in architectures:
        print(f"\n{'='*80}")
        print(f"ARCHITECTURE: {net_arch}, AGENTS: {num_agent}")
        print(f"{'='*80}")
        
        analysis = calculate_sac_parameters(
            pendulum_obs_dim, 
            pendulum_action_dim, 
            net_arch, 
            num_agent
        )
        print_parameter_analysis(analysis)
        results[(tuple(net_arch), num_agent)] = analysis['total_params']
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    for (net_arch, num_agent), total_params in results.items():
        print(f"Architecture {list(net_arch)} with {num_agent} agent(s): {total_params:,} parameters")
    
    # Calculate ratios
    single_agent_1024 = results[((1024, 1024), 1)]
    single_agent_2048 = results[((2048, 2048), 1)]
    two_agent_1024 = results[((1024, 1024), 2)]
    two_agent_2048 = results[((2048, 2048), 2)]
    
    print(f"\nRatios:")
    print(f"2048 vs 1024 (single agent): {single_agent_2048 / single_agent_1024:.2f}x")
    print(f"2 agents vs 1 agent (1024): {two_agent_1024 / single_agent_1024:.2f}x")
    print(f"2 agents vs 1 agent (2048): {two_agent_2048 / single_agent_2048:.2f}x")
    print(f"2 agents 2048 vs 1 agent 1024: {two_agent_2048 / single_agent_1024:.2f}x")


if __name__ == "__main__":
    print("SAC Parameter Counter")
    print("This script calculates parameter counts for different SAC architectures")
    print()
    
    compare_architectures()
    
    print(f"\n{'='*80}")
    print("USAGE EXAMPLES")
    print(f"{'='*80}")
    print("To calculate parameters for specific architecture:")
    print("  analysis = calculate_sac_parameters(obs_dim, action_dim, [1024, 1024], num_agent=2)")
    print("  print_parameter_analysis(analysis)")
    print()
    print("For your current setup with net_arch=[2048, 2048] and num_agent=2:")
    
    # Example for your current setup
    analysis = calculate_sac_parameters(3, 1, [2048, 2048], num_agent=2)
    print_parameter_analysis(analysis)


