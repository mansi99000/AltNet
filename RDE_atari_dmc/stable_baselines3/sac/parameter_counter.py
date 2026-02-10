#!/usr/bin/env python3
"""
Parameter counting utility for SAC implementation.
This script provides functions to count trainable parameters in the multi-agent SAC setup.
"""

import torch as th
from typing import Dict, List, Tuple, Any
from sac import SAC
from policies import SACPolicy


def count_parameters(model: th.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model: th.nn.Module) -> Dict[str, int]:
    """
    Count parameters by layer name.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    param_counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_counts[name] = param.numel()
    return param_counts


def analyze_sac_parameters(sac_model: SAC) -> Dict[str, Any]:
    """
    Analyze parameters for the entire SAC model including all agents.
    
    Args:
        sac_model: Trained SAC model
        
    Returns:
        Dictionary containing detailed parameter analysis
    """
    analysis = {
        'total_parameters': 0,
        'per_agent_parameters': [],
        'network_breakdown': {},
        'agent_details': []
    }
    
    total_params = 0
    
    for i in range(sac_model.num_agent):
        agent_info = {
            'agent_id': i,
            'actor_parameters': 0,
            'critic_parameters': 0,
            'entropy_coef_parameters': 0,
            'total_agent_parameters': 0
        }
        
        # Count actor parameters
        actor_params = count_parameters(sac_model.actor[i])
        agent_info['actor_parameters'] = actor_params
        
        # Count critic parameters (both Q-networks)
        critic_params = count_parameters(sac_model.critic[i])
        agent_info['critic_parameters'] = critic_params
        
        # Count entropy coefficient parameters
        if hasattr(sac_model, 'log_ent_coef') and sac_model.log_ent_coef[i] is not None:
            ent_coef_params = count_parameters(sac_model.log_ent_coef[i])
            agent_info['entropy_coef_parameters'] = ent_coef_params
        
        agent_info['total_agent_parameters'] = (
            agent_info['actor_parameters'] + 
            agent_info['critic_parameters'] + 
            agent_info['entropy_coef_parameters']
        )
        
        total_params += agent_info['total_agent_parameters']
        analysis['per_agent_parameters'].append(agent_info['total_agent_parameters'])
        analysis['agent_details'].append(agent_info)
    
    analysis['total_parameters'] = total_params
    
    # Network breakdown
    if analysis['agent_details']:
        first_agent = analysis['agent_details'][0]
        analysis['network_breakdown'] = {
            'actor_per_agent': first_agent['actor_parameters'],
            'critic_per_agent': first_agent['critic_parameters'],
            'entropy_coef_per_agent': first_agent['entropy_coef_parameters'],
            'total_per_agent': first_agent['total_agent_parameters']
        }
    
    return analysis


def print_parameter_analysis(analysis: Dict[str, Any]) -> None:
    """
    Print a formatted parameter analysis.
    
    Args:
        analysis: Analysis dictionary from analyze_sac_parameters
    """
    print("=" * 60)
    print("SAC PARAMETER ANALYSIS")
    print("=" * 60)
    
    print(f"Total Parameters: {analysis['total_parameters']:,}")
    print(f"Number of Agents: {len(analysis['agent_details'])}")
    
    if analysis['network_breakdown']:
        breakdown = analysis['network_breakdown']
        print(f"\nPer Agent Breakdown:")
        print(f"  Actor: {breakdown['actor_per_agent']:,} parameters")
        print(f"  Critic: {breakdown['critic_per_agent']:,} parameters")
        print(f"  Entropy Coef: {breakdown['entropy_coef_per_agent']:,} parameters")
        print(f"  Total per agent: {breakdown['total_per_agent']:,} parameters")
    
    print(f"\nAgent Details:")
    for agent in analysis['agent_details']:
        print(f"  Agent {agent['agent_id']}: {agent['total_agent_parameters']:,} parameters")
        print(f"    - Actor: {agent['actor_parameters']:,}")
        print(f"    - Critic: {agent['critic_parameters']:,}")
        print(f"    - Entropy Coef: {agent['entropy_coef_parameters']:,}")
    
    print("=" * 60)


def get_detailed_layer_analysis(sac_model: SAC, agent_id: int = 0) -> Dict[str, Dict[str, int]]:
    """
    Get detailed parameter counts for each layer of a specific agent.
    
    Args:
        sac_model: Trained SAC model
        agent_id: Which agent to analyze (default: 0)
        
    Returns:
        Dictionary with layer-wise parameter counts
    """
    if agent_id >= sac_model.num_agent:
        raise ValueError(f"Agent ID {agent_id} out of range. Model has {sac_model.num_agent} agents.")
    
    analysis = {
        'actor_layers': count_parameters_by_layer(sac_model.actor[agent_id]),
        'critic_layers': count_parameters_by_layer(sac_model.critic[agent_id])
    }
    
    return analysis


def print_layer_analysis(layer_analysis: Dict[str, Dict[str, int]]) -> None:
    """
    Print detailed layer analysis.
    
    Args:
        layer_analysis: Layer analysis from get_detailed_layer_analysis
    """
    print("\n" + "=" * 60)
    print("DETAILED LAYER ANALYSIS")
    print("=" * 60)
    
    print("\nActor Layers:")
    actor_total = 0
    for layer_name, param_count in layer_analysis['actor_layers'].items():
        print(f"  {layer_name}: {param_count:,} parameters")
        actor_total += param_count
    print(f"  Actor Total: {actor_total:,} parameters")
    
    print("\nCritic Layers:")
    critic_total = 0
    for layer_name, param_count in layer_analysis['critic_layers'].items():
        print(f"  {layer_name}: {param_count:,} parameters")
        critic_total += param_count
    print(f"  Critic Total: {critic_total:,} parameters")
    
    print(f"\nCombined Total: {actor_total + critic_total:,} parameters")
    print("=" * 60)


# Example usage function
def analyze_trained_model(model_path: str = None, env_name: str = "Pendulum-v1") -> None:
    """
    Analyze parameters of a trained SAC model or create a new one for analysis.
    
    Args:
        model_path: Path to saved model (optional)
        env_name: Environment name for new model
    """
    import gym
    
    if model_path:
        # Load existing model
        print(f"Loading model from {model_path}...")
        model = SAC.load(model_path)
    else:
        # Create new model for analysis
        print(f"Creating new model for environment: {env_name}")
        env = gym.make(env_name)
        model = SAC("MlpPolicy", env, verbose=1)
        env.close()
    
    # Perform analysis
    analysis = analyze_sac_parameters(model)
    print_parameter_analysis(analysis)
    
    # Detailed layer analysis for first agent
    if model.num_agent > 0:
        layer_analysis = get_detailed_layer_analysis(model, agent_id=0)
        print_layer_analysis(layer_analysis)


if __name__ == "__main__":
    # Example usage
    analyze_trained_model()


