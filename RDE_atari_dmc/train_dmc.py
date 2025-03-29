from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_dmc_env
from continuous_control.utils import make_env as dmc_make_env
import argparse
import wandb
import csv
import os

wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="hopper-hop")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--total_timesteps", default=1e6, type=int)
parser.add_argument("--eval_freq", default=1e4, type=int)
parser.add_argument("--SR", action='store_true')
parser.add_argument("--RDE", action='store_true')
parser.add_argument("--DISTILL", action='store_true')
parser.add_argument("--reset_freq", default=4e5, type=float)
parser.add_argument("--distill_freq", default=4e5, type=float)
parser.add_argument("--replay_ratio", default=1, type=int)
parser.add_argument("--learning_rate", default=3e-4, type=float)
parser.add_argument("--learning_starts", default=5000, type=int)
parser.add_argument("--action_select_coef", default=50, type=int)
parser.add_argument("--wandb", action='store_true')
parser.add_argument("--entity_name", type=str)
parser.add_argument("--job_id", os.getenv("SLURM_JOB_ID", "unknown"))

args = parser.parse_args()

set_random_seed(args.seed)

policy_kwargs = dict()

if args.RDE:
    mode = 'RDE+SAC'
    num_agent = 4
    reset = True
    distill = False
elif args.SR:
    mode = 'SR+SAC'
    num_agent = 1
    reset = True
    distill = False
elif args.DISTILL:
    mode = 'DISTILL+SAC' # check where does mode make a difference? 
    num_agent = 1
    reset = False
    distill = True
else:
    mode = 'SAC'
    num_agent = 1
    reset = False
    distill = False

policy_kwargs.update(num_agent=num_agent)

if args.action_select_coef != 50:
    policy_kwargs.update(action_select_coef=args.action_select_coef)

print(f'env:{args.env}, mode:{mode}')

env = make_dmc_env(args.env, seed=args.seed)
eval_env = dmc_make_env(args.env, args.seed+42)

reset_freq = int((args.reset_freq/num_agent)/args.replay_ratio)
distill_freq = int(args.distill_freq)

log_path = f"./logs/{args.env}/{args.replay_ratio}/{mode}"

# M
# Ensure the directory exists
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
    wandb.init(project="RDE_SAC", 
               name=f"{mode}_rr_{args.replay_ratio}_seed_{args.seed}_num_{num_agent}",
               group=f"{args.env}",
               job_type=f"{mode}_{args.replay_ratio}_{args.action_select_coef}",
               reinit=True)
    # wandb.init(project="RDE_SAC",
    #            name=f"{mode}_{args.replay_ratio}_{args.seed}"
    #            )

model = SAC("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, reset=reset,
            reset_frequency=reset_freq, distill_frequency=distill_freq, gradient_steps=args.replay_ratio,
            learning_rate=args.learning_rate, learning_starts=args.learning_starts,
            seed=args.seed, num_agent=num_agent, wandb=args.wandb)

model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

env.close()
eval_env.close()
wandb.finish()
