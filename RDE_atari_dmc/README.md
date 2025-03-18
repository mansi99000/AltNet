# Sample-Efficient and Safe Deep Reinforcement Learning via Reset Deep Ensemble Agents (RDE), submission to NeurIPS 2023



This codebase is implemented based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) githubs.
Our code for DeepMind Control Suite environments is based on the [rl_with_resets](https://github.com/evgenii-nikishin/rl_with_resets).

## Setup instructions

Set up Stable-baselines3:
```
pip install -r requirements.txt
```

## Run an experiment 

To run the RDE+DQN on Atari 100k:
```
python train_atari.py --env AlienNoFrameskip-v4 --RDE --reset_freq 8e4 --replay_ratio 1 --entity
_name [wandb_name] --wandb
```

To run the RDE+SAC on DeepMind Control Suite:
```
python train_dmc.py --env hopper-hop --RDE --reset_freq 4e5 --replay_ratio 1 --entity
_name [wandb_name] --wandb
```

