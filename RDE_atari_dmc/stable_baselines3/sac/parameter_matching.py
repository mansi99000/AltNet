#!/usr/bin/env python3
"""
Parameter matching calculator to find equivalent architectures.
"""

def calculate_mlp_parameters(input_dim, hidden_layers, output_dim):
    """Calculate parameters in MLP."""
    total = 0
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        total += (prev_dim * hidden_dim) + hidden_dim
        prev_dim = hidden_dim
    return total

def calculate_sac_parameters(observation_dim, action_dim, net_arch, num_agent=1, n_critics=2):
    """Calculate SAC parameters."""
    # Actor
    actor_hidden = calculate_mlp_parameters(observation_dim, net_arch, net_arch[-1])
    actor_output = (net_arch[-1] * action_dim) * 2
    actor_per_agent = actor_hidden + actor_output
    
    # Critic
    critic_input = observation_dim + action_dim
    critic_hidden = calculate_mlp_parameters(critic_input, net_arch, net_arch[-1])
    critic_output = net_arch[-1] * 1
    critic_per_agent = (critic_hidden + critic_output) * n_critics
    
    # Entropy coefficient
    ent_coef_per_agent = 1
    
    per_agent = actor_per_agent + critic_per_agent + ent_coef_per_agent
    total = per_agent * num_agent
    
    return total, per_agent

def find_equivalent_architectures(target_params, obs_dim=3, action_dim=1):
    """Find architectures that match target parameter count."""
    print(f"Target parameters: {target_params:,}")
    print("=" * 60)
    
    # Option 1: Deeper networks with 1024 width
    print("Option 1: Deeper networks (1024 width)")
    for depth in range(2, 8):
        arch = [1024] * depth
        total, per_agent = calculate_sac_parameters(obs_dim, action_dim, arch, 1)
        ratio = total / target_params
        print(f"  {arch}: {total:,} params (ratio: {ratio:.3f})")
        if 0.95 <= ratio <= 1.05:  # Within 5%
            print(f"    *** CLOSE MATCH ***")
    
    print("\nOption 2: Wider networks (2 layers)")
    print("Option 2: Wider networks (2 layers)")
    for width in [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]:
        arch = [width, width]
        total, per_agent = calculate_sac_parameters(obs_dim, action_dim, arch, 1)
        ratio = total / target_params
        print(f"  {arch}: {total:,} params (ratio: {ratio:.3f})")
        if 0.95 <= ratio <= 1.05:
            print(f"    *** CLOSE MATCH ***")
    
    print("\nOption 3: Mixed architectures")
    print("Option 3: Mixed architectures")
    mixed_archs = [
        [1024, 1024, 512],
        [1024, 1024, 1024, 256],
        [1200, 1200],
        [1400, 1400],
        [1600, 1600],
        [1800, 1800],
        [2000, 2000],
    ]
    
    for arch in mixed_archs:
        total, per_agent = calculate_sac_parameters(obs_dim, action_dim, arch, 1)
        ratio = total / target_params
        print(f"  {arch}: {total:,} params (ratio: {ratio:.3f})")
        if 0.95 <= ratio <= 1.05:
            print(f"    *** CLOSE MATCH ***")

def find_exact_match(target_params, obs_dim=3, action_dim=1, max_width=3000, max_depth=6):
    """Find the closest architecture to target parameters."""
    best_arch = None
    best_diff = float('inf')
    best_params = 0
    
    # Try different depths and widths
    for depth in range(2, max_depth + 1):
        for width in range(800, max_width + 1, 50):
            arch = [width] * depth
            total, per_agent = calculate_sac_parameters(obs_dim, action_dim, arch, 1)
            diff = abs(total - target_params)
            
            if diff < best_diff:
                best_diff = diff
                best_arch = arch
                best_params = total
    
    return best_arch, best_params, best_diff

if __name__ == "__main__":
    # Target: 2 agents with [1024, 1024] = 6,334,466 parameters
    target_params = 6334466
    
    print("FINDING EQUIVALENT SINGLE-AGENT ARCHITECTURES")
    print("=" * 60)
    print(f"Target: 2 agents [1024, 1024] = {target_params:,} parameters")
    print(f"Goal: Find single-agent architecture with ~{target_params:,} parameters")
    print()
    
    find_equivalent_architectures(target_params)
    
    print("\n" + "=" * 60)
    print("BEST EXACT MATCH")
    print("=" * 60)
    
    best_arch, best_params, best_diff = find_exact_match(target_params)
    print(f"Best architecture: {best_arch}")
    print(f"Parameters: {best_params:,}")
    print(f"Difference from target: {best_diff:,} ({best_diff/target_params*100:.2f}%)")
    
    # Show some close alternatives
    print(f"\nClose alternatives:")
    alternatives = [
        [1024, 1024, 1024],  # 3 layers
        [1200, 1200],         # 2 wider layers
        [1400, 1400],         # 2 even wider layers
        [1024, 1024, 1024, 256],  # 4 layers with tapering
    ]
    
    for arch in alternatives:
        total, per_agent = calculate_sac_parameters(3, 1, arch, 1)
        diff = abs(total - target_params)
        print(f"  {arch}: {total:,} params (diff: {diff:,}, {diff/target_params*100:.2f}%)")


