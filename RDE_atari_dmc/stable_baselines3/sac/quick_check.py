#!/usr/bin/env python3
"""
Quick parameter check - add this to your SAC training script to get exact counts.
"""

def get_sac_parameter_count(observation_dim, action_dim, net_arch, num_agent=1, n_critics=2):
    """
    Get exact parameter count for SAC model.
    
    Args:
        observation_dim: Observation space dimension
        action_dim: Action space dimension
        net_arch: Network architecture (e.g., [1024, 1024])
        num_agent: Number of agents
        n_critics: Number of critic networks per agent
        
    Returns:
        Dictionary with parameter counts
    """
    def calc_mlp_params(input_dim, hidden_layers, output_dim):
        total = 0
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            total += (prev_dim * hidden_dim) + hidden_dim  # weights + bias
            prev_dim = hidden_dim
        return total
    
    # Actor: obs -> hidden -> (mu + log_std)
    actor_hidden = calc_mlp_params(observation_dim, net_arch, net_arch[-1])
    actor_output = (net_arch[-1] * action_dim) * 2  # mu and log_std
    actor_per_agent = actor_hidden + actor_output
    
    # Critic: (obs + action) -> hidden -> q_value
    critic_input = observation_dim + action_dim
    critic_hidden = calc_mlp_params(critic_input, net_arch, net_arch[-1])
    critic_output = net_arch[-1] * 1  # single Q-value
    critic_per_agent = (critic_hidden + critic_output) * n_critics
    
    # Entropy coefficient
    ent_coef_per_agent = 1
    
    # Per agent total
    per_agent = actor_per_agent + critic_per_agent + ent_coef_per_agent
    
    # Total for all agents
    total = per_agent * num_agent
    
    return {
        'total_parameters': total,
        'per_agent_parameters': per_agent,
        'actor_per_agent': actor_per_agent,
        'critic_per_agent': critic_per_agent,
        'entropy_coef_per_agent': ent_coef_per_agent,
        'num_agents': num_agent
    }


# Example usage for your current setup
if __name__ == "__main__":
    # Your current setup based on the code
    obs_dim = 3  # Pendulum-v1
    action_dim = 1
    net_arch = [2048, 2048]  # From line 244 in policies.py
    num_agent = 2  # Assuming 2 agents based on the multi-agent setup
    
    result = get_sac_parameter_count(obs_dim, action_dim, net_arch, num_agent)
    
    print("Current SAC Parameter Count:")
    print(f"Total Parameters: {result['total_parameters']:,}")
    print(f"Per Agent: {result['per_agent_parameters']:,}")
    print(f"Actor per Agent: {result['actor_per_agent']:,}")
    print(f"Critic per Agent: {result['critic_per_agent']:,}")
    print(f"Number of Agents: {result['num_agents']}")
    
    # Compare with smaller architecture
    small_result = get_sac_parameter_count(obs_dim, action_dim, [1024, 1024], num_agent)
    print(f"\nComparison with [1024, 1024]:")
    print(f"Small architecture: {small_result['total_parameters']:,} parameters")
    print(f"Your architecture: {result['total_parameters']:,} parameters")
    print(f"Ratio: {result['total_parameters'] / small_result['total_parameters']:.2f}x more parameters")


