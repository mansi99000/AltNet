"""
Train SAC-based agents on DeepMind Control Suite environments.

Supports the following modes:
  --PS    : AltNet (Maheshwari et al., 2025) - dual-network alternating resets
  --SR    : Standard Resets (Nikishin et al., 2022) - single agent with periodic resets
  --RDE   : Reset Deep Ensembles (Kim et al., 2024) - 10-agent ensemble with Q-value gating
  default : Vanilla SAC baseline (no resets)

Reference: "Addressing the Plasticity-Stability Dilemma in Reinforcement Learning"
           Maheshwari, Raisbeck, & Castro da Silva (2025). arXiv:2512.01034
"""

from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_dmc_env
from continuous_control.utils import make_env as dmc_make_env
import argparse
import wandb
import csv
import os
import subprocess
wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="hopper-hop")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--total_timesteps", default=1e6, type=int)
parser.add_argument("--eval_freq", default=1e4, type=int)
parser.add_argument("--SR", action='store_true')
parser.add_argument("--RDE", action='store_true')
parser.add_argument("--PS", action='store_true')
parser.add_argument("--reset_freq", default=4e5, type=float)
parser.add_argument("--replay_ratio", default=1, type=int)
parser.add_argument("--learning_rate", default=3e-4, type=float)
parser.add_argument("--learning_starts", default=5000, type=int)
parser.add_argument("--action_select_coef", default=50, type=int)
parser.add_argument("--wandb", action='store_true')
parser.add_argument("--entity_name", type=str)
parser.add_argument("--job_id", type=str, default=os.getenv("SLURM_JOB_ID", "unknown"))
parser.add_argument("--dynamic_rr", action='store_true', help="Enable dynamic replay ratio changes")
parser.add_argument("--rr_change_timestep", default=400000, type=int, help="Timestep at which to change replay ratio")
parser.add_argument("--rr_after_change", default=8, type=int, help="Replay ratio after the change point")
parser.add_argument("--buffer_size", default=1000000, type=int, help="Size of the replay buffer")
parser.add_argument("--reset_stop_timestep", default=1e6, type=int, help="Timestep at which resets stop")
parser.add_argument("--dynamic_resets", action='store_true', help="Enable dynamic reset frequency: 50k steps until 200k, then 100k steps")

args = parser.parse_args()

set_random_seed(args.seed)

policy_kwargs = dict()

# --- Ablation: reduced network capacity (Section 4.2, RQ1) ---
# Halves the network size to match single-SAC parameter count.
# policy_kwargs.update(net_arch=[512, 512])

branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('utf-8')

# ---- Mode selection ----
# AltNet (PS): 2 agents alternating roles (active/passive) with periodic resets
# SR: 1 agent with standard periodic resets (Nikishin et al., 2022)
# RDE: 10-agent ensemble with Q-value weighted action selection (Kim et al., 2024)
# SAC: Vanilla SAC baseline (no resets)
if args.RDE:
    mode = 'RDE'
    num_agent = 10
    reset = True
    ps = False
elif args.SR:
    mode = 'SR'
    num_agent = 1
    reset = True
    ps = False
elif args.PS:
    mode = 'PS'
    num_agent = 2
    reset = True
    ps = True
else:
    mode = 'SAC'
    num_agent = 1
    reset = False
    ps = False

policy_kwargs.update(num_agent=num_agent)

if args.action_select_coef != 50:
    policy_kwargs.update(action_select_coef=args.action_select_coef)

print(f'env:{args.env}, mode:{mode}')

env = make_dmc_env(args.env, seed=args.seed)
eval_env = dmc_make_env(args.env, args.seed+42)

# Reset frequency normalization (Kim et al., 2024):
# ResetFreq (env steps) = U / (RR * N)
# This ensures each agent is reset after the same number of gradient updates
# regardless of the replay ratio or ensemble size.
reset_freq = int((args.reset_freq / num_agent) / args.replay_ratio)

log_path = f"./logs/{args.env}/{args.replay_ratio}/{mode}"
os.makedirs(log_path, exist_ok=True)

filename = f'{log_path}/result.csv'
f = open(filename, 'a', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow([args])
args.filename = filename
f.close()

eval_callback = EvalCallback(eval_env, best_model_save_path=log_path, log_path=log_path,
                             eval_freq=args.eval_freq, deterministic=True,
                             render=False, wandb=args.wandb)

if args.wandb:
    policy_kwargs.update(wandb=args.wandb)
    wandb.init(project=f"rebuttal_AAMAS",
               name=f"{args.job_id}_{mode}_rr_{args.replay_ratio}_seed_{args.seed}_num_{num_agent}_{branch}_{reset_freq}",
               group=f"{args.env}",
               job_type=f"{args.env}_{mode}_{num_agent}agents_{args.replay_ratio}_{reset_freq}",
               dir="/work/pi_bsilva_umass_edu/mmaheshwari_umass_edu/wandb",
               reinit=True)

model = SAC("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, reset=reset,
            reset_frequency=reset_freq, reset_stop_timestep=args.reset_stop_timestep,
            gradient_steps=args.replay_ratio, learning_rate=args.learning_rate,
            learning_starts=args.learning_starts, seed=args.seed, num_agent=num_agent,
            wandb=args.wandb, dynamic_rr=args.dynamic_rr, rr_change_timestep=args.rr_change_timestep,
            rr_after_change=args.rr_after_change, buffer_size=args.buffer_size,
            dynamic_resets=args.dynamic_resets)

model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

env.close()
eval_env.close()
wandb.finish()
