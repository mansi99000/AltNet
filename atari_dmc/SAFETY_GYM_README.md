# Running Algorithms on Safety Gym Environments

This guide explains how to run algorithms (SAC, RDE+SAC, etc.) on Safety Gym environments.

## Installation

First, install Safety Gym:

```bash
# Option 1: Original safety-gym (may have compatibility issues with newer gym versions)
pip install safety-gym

# Option 2: Recommended - safety-gymnasium (maintained fork, compatible with gymnasium)
pip install safety-gymnasium
```

If using `safety-gymnasium`, you may need to update the environment names (e.g., `SafetyPointGoal1-v0` instead of `Safexp-PointGoal1-v0`).

## Available Environments

### Safety Gym (original)
- `Safexp-PointGoal1-v0` - Point robot, goal task, level 1
- `Safexp-PointGoal2-v0` - Point robot, goal task, level 2
- `Safexp-PointButton1-v0` - Point robot, button task, level 1
- `Safexp-CarGoal1-v0` - Car robot, goal task, level 1
- `Safexp-CarButton1-v0` - Car robot, button task, level 1
- `Safexp-DoggoGoal1-v0` - Doggo robot, goal task, level 1

### Safety Gymnasium (newer)
- `SafetyPointGoal1-v0`
- `SafetyCarGoal1-v0`
- `SafetyPointButton1-v0`
- etc.

## Running SAC on Safety Gym

### Basic SAC
```bash
python train_safety_gym.py --env Safexp-PointGoal1-v0 --wandb --entity_name YOUR_WANDB_ENTITY
```

### RDE+SAC (Reset Deep Ensemble)
```bash
python train_safety_gym.py --env Safexp-PointGoal1-v0 --RDE --reset_freq 4e5 --replay_ratio 1 --wandb --entity_name YOUR_WANDB_ENTITY
```

### SR+SAC (Self-Reset)
```bash
python train_safety_gym.py --env Safexp-PointGoal1-v0 --SR --reset_freq 4e5 --replay_ratio 1 --wandb --entity_name YOUR_WANDB_ENTITY
```

### PS+SAC (Population-based Search)
```bash
python train_safety_gym.py --env Safexp-PointGoal1-v0 --PS --reset_freq 4e5 --replay_ratio 1 --wandb --entity_name YOUR_WANDB_ENTITY
```

## About WCSAC (Worst-Case Soft Actor-Critic)

**WCSAC** is a safety-constrained variant of SAC that explicitly handles cost constraints from Safety Gym environments. The current implementation uses standard SAC, which doesn't explicitly optimize for safety constraints.

### Key Differences:
- **Standard SAC**: Maximizes reward, doesn't consider cost constraints
- **WCSAC**: Maximizes reward while ensuring cost constraints are satisfied (typically using Lagrangian methods or constrained optimization)

### Should you use WCSAC?

**Yes, if:**
- You need to ensure the agent respects safety constraints (cost limits)
- You're doing safety-critical applications
- You want to explicitly optimize the safety-return tradeoff

**No, if:**
- You're just exploring the environment
- Standard SAC performance is sufficient
- You're using RDE/SR/PS which may help with exploration

### Implementing WCSAC

To implement WCSAC, you would need to:

1. **Modify the SAC algorithm** to include a cost critic (similar to Q-critic but for costs)
2. **Add Lagrangian multiplier** to handle the constraint: `L = reward - λ * cost`
3. **Update the policy** to minimize cost while maximizing reward
4. **Track cost violations** and adjust the Lagrangian multiplier

The current codebase doesn't include WCSAC, but you can:
- Use standard SAC and monitor costs manually
- Implement WCSAC by extending the SAC class in `stable_baselines3/sac/sac.py`
- Use a library like [safe-rl](https://github.com/liuzuxin/safe-rl) that includes WCSAC

## Monitoring Safety Metrics

Safety Gym environments provide a `cost` signal in the `info` dict. You can monitor this:

```python
obs, reward, done, info = env.step(action)
cost = info.get('cost', 0)  # Cost for this step
```

The `--use_cost` flag in `train_safety_gym.py` will detect if cost signals are available, but standard SAC doesn't use them in training.

## Example: Running on Different Safety Gym Tasks

```bash
# Easy task
python train_safety_gym.py --env Safexp-PointGoal1-v0 --RDE --wandb

# Harder task
python train_safety_gym.py --env Safexp-PointGoal2-v0 --RDE --wandb

# Different robot
python train_safety_gym.py --env Safexp-CarGoal1-v0 --RDE --wandb
```

## Troubleshooting

1. **ImportError for safety-gym**: Install with `pip install safety-gym` or `pip install safety-gymnasium`
2. **Environment not found**: Check the exact environment name. Safety Gymnasium uses different names.
3. **Cost signal not available**: Some environments may not provide cost in info dict. Check the environment documentation.


