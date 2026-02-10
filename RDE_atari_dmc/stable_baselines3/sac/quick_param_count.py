#!/usr/bin/env python3
"""
Quick parameter counting script for SAC models.
Run this to get exact parameter counts for your current setup.
"""

import sys
import os
import gym
import torch as th

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sac import SAC
from parameter_counter import analyze_sac_parameters, print_parameter_analysis, get_detailed_layer_analysis, print_layer_analysis


def count_params_for_architecture(net_arch, num_agent=1, env_name="Pendulum-v1"):
    """
    Count parameters for a specific architecture.
    
    Args:
        net_arch: Network architecture (e.g., [1024, 1024] or [2048, 2048])
        num_agent: Number of agents
        env_name: Environment name
    """
    print(f"Analyzing SAC with net_arch={net_arch}, num_agent={num_agent}")
    print("=" * 60)
    
    # Create environment
    env = gym.make(env_name)
    
    # Create SAC model
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        num_agent=num_agent,
        policy_kwargs={"net_arch": net_arch}
    )
    
    # Analyze parameters
    analysis = analyze_sac_parameters(model)
    print_parameter_analysis(analysis)
    
    # Detailed layer analysis for first agent
    if model.num_agent > 0:
        layer_analysis = get_detailed_layer_analysis(model, agent_id=0)
        print_layer_analysis(layer_analysis)
    
    env.close()
    return analysis


def compare_architectures():
    """Compare different network architectures."""
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
        
        analysis = count_params_for_architecture(net_arch, num_agent)
        results[(tuple(net_arch), num_agent)] = analysis['total_parameters']
    
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
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            compare_architectures()
        else:
            # Custom analysis
            net_arch = eval(sys.argv[1]) if len(sys.argv) > 1 else [1024, 1024]
            num_agent = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            count_params_for_architecture(net_arch, num_agent)
    else:
        # Default analysis
        print("Running default parameter analysis...")
        print("Usage: python quick_param_count.py [net_arch] [num_agent]")
        print("       python quick_param_count.py compare  # Compare different architectures")
        print("Examples:")
        print("  python quick_param_count.py [1024, 1024] 1")
        print("  python quick_param_count.py [2048, 2048] 2")
        print("  python quick_param_count.py compare")
        print()
        
        count_params_for_architecture([1024, 1024], 1)


