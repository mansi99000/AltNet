# AltNet: Addressing the Plasticity-Stability Dilemma in Reinforcement Learning

**Accepted at [AAMAS 2026](https://www.aamas2026-conference.auckland.ac.nz/)** (25th International Conference on Autonomous Agents and Multi-Agent Systems)

**Paper:** [arXiv:2512.01034](https://arxiv.org/abs/2512.01034)  
**Authors:** Mansi Maheshwari, John C. Raisbeck, Bruno Castro da Silva  
**Affiliation:** University of Massachusetts Amherst

## Overview

Neural networks in reinforcement learning progressively lose their ability to learn from new experiences over time -- a phenomenon known as **plasticity loss**. Prior work has shown that periodically resetting network parameters can restore plasticity, but resets cause temporary **performance collapses** that are dangerous in real-world settings.

**AltNet** resolves this plasticity-stability dilemma with a dual-network architecture:

```
                     Shared Replay Buffer
                    /                     \
              [Network A1]           [Network A2]
               (active)               (passive)
             acts in env          trains off-policy
                    \                     /
                     --- after ResetFreq ---
                    /                     \
              [Network A1]           [Network A2]
            reset -> passive      trained -> active
```

- Two networks alternate roles at fixed intervals (**ResetFreq** steps).
- The **active** network interacts with the environment.
- The **passive** network trains off-policy from the shared replay buffer.
- At each reset interval, the active network is **fully reset** and becomes passive, while the trained passive network becomes active.
- A freshly reset network **never acts in the environment**, preventing post-reset performance drops.

### Key Results

- AltNet outperforms SAC by ~38%, Standard Resets by ~12%, and RDE by ~6% (normalized AUC) across DeepMind Control Suite environments.
- AltNet achieves superior sample efficiency even at low replay ratios (RR=1), where Standard Resets fail entirely and RDE still exhibits sharp performance drops.
- Benefits are not due to increased model capacity -- reducing AltNet's parameters to match SAC yields nearly identical performance.

## Codebase Structure

This codebase builds on the implementation released by [Kim et al. (2024)](https://openreview.net/forum?id=bTidcHIK2t) as supplementary material for their NeurIPS 2023 paper *"Sample-Efficient and Safe Deep Reinforcement Learning via Reset Deep Ensemble Agents"* (RDE). Their code extends [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) (v1.7.0) with multi-agent ensemble and reset support, and uses DeepMind Control Suite wrappers from [rl_with_resets](https://github.com/evgenii-nikishin/rl_with_resets) (Nikishin et al., 2022). We modified their codebase to implement AltNet; our specific changes are listed in the table below.

```
atari_dmc/
├── train_dmc.py                    # Training script for DMC environments (SAC-based)
├── train_atari.py                  # Training script for Atari 100k (DQN-based)
├── train_safety_gym.py             # Training script for Safety Gym environments
├── continuous_control/             # DMC environment wrappers
│   ├── utils.py                    # Environment factory
│   └── wrappers/                   # Gym-compatible DMC wrappers
├── stable_baselines3/              # Modified SB3 with multi-agent reset support
│   ├── sac/
│   │   ├── sac.py                  # SAC with AltNet/RDE reset mechanism
│   │   └── policies.py            # Multi-agent SAC policies with AltNet action selection
│   ├── dqn/
│   │   ├── dqn.py                  # DQN with multi-agent reset mechanism
│   │   └── policies.py            # Multi-agent DQN policies
│   └── common/                     # Shared SB3 utilities (buffers, callbacks, etc.)
├── Atari100k_results/              # Baseline result CSVs for Atari 100k benchmark
├── requirements.txt
└── README.md
```

### Key Modified Files (our contributions)

The following files contain the core AltNet implementation, built on top of the original stable-baselines3 and RDE codebases:

| File | What we modified |
|------|-----------------|
| `stable_baselines3/sac/policies.py` | **AltNet action selection** (`_predict` method): always selects the trained (non-reset) network for environment interaction. Also supports RDE's Q-value weighted ensemble selection. |
| `stable_baselines3/sac/sac.py` | **Periodic reset mechanism** (`train` method): full network resets (actor, critic, entropy coef, optimizers) with round-robin agent cycling. Added dynamic replay ratio and dynamic reset frequency scheduling. |
| `stable_baselines3/dqn/dqn.py` | DQN variant of the reset mechanism with configurable layer reset scope. |
| `stable_baselines3/dqn/policies.py` | Multi-agent DQN policy with ensemble action selection. |
| `train_dmc.py` | DMC training script with mode selection (AltNet/SR/RDE/SAC). |
| `train_atari.py` | Atari training script with mode selection. |
| `train_safety_gym.py` | Safety Gym training script with mode selection. |

## Setup

```bash
pip install -r requirements.txt
```

For DMC environments, you also need MuJoCo. See the [dm_control installation guide](https://github.com/google-deepmind/dm_control#requirements-and-installation).

## Usage

### AltNet (our method) on DeepMind Control Suite

```bash
# AltNet with replay ratio 1 (default settings from the paper)
python train_dmc.py --env hopper-hop --PS --reset_freq 4e5 --replay_ratio 1 --seed 0

# AltNet with replay ratio 4
python train_dmc.py --env walker-run --PS --reset_freq 4e5 --replay_ratio 4 --seed 0

# With Weights & Biases logging
python train_dmc.py --env hopper-hop --PS --reset_freq 4e5 --replay_ratio 1 --wandb --entity_name YOUR_ENTITY
```

### Baseline comparisons

```bash
# Vanilla SAC (no resets)
python train_dmc.py --env hopper-hop --seed 0

# Standard Resets (Nikishin et al., 2022)
python train_dmc.py --env hopper-hop --SR --reset_freq 4e5 --replay_ratio 1 --seed 0

# RDE (Kim et al., 2024) - 10-agent ensemble
python train_dmc.py --env hopper-hop --RDE --reset_freq 4e5 --replay_ratio 1 --seed 0
```

### Atari 100k

```bash
# AltNet+DQN
python train_atari.py --env AlienNoFrameskip-v4 --PS --reset_freq 2e5 --replay_ratio 1 --seed 0

# RDE+DQN
python train_atari.py --env AlienNoFrameskip-v4 --RDE --reset_freq 2e5 --replay_ratio 1 --seed 0
```

## Supported Modes

| Flag | Mode | Agents | Action Selection | Description |
|------|------|--------|-----------------|-------------|
| `--PS` | **AltNet** | 2 | Always use trained (non-reset) network | Our method |
| `--SR` | Standard Resets | 1 | Single agent | Nikishin et al., 2022 |
| `--RDE` | Reset Deep Ensembles | 10 (SAC) / 3 (DQN) | Q-value weighted voting | Kim et al., 2024 |
| (none) | Baseline | 1 | Single agent, no resets | Vanilla SAC/DQN |

## Citation

```bibtex
@inproceedings{maheshwari2026altnet,
  title={Addressing the Plasticity-Stability Dilemma in Reinforcement Learning},
  author={Maheshwari, Mansi and Raisbeck, John C. and Castro da Silva, Bruno},
  booktitle={Proceedings of the 25th International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
