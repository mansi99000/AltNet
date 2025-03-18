from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import argparse
import wandb
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="AlienNoFrameskip-v4")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--n_envs", default=4, type=int)
parser.add_argument("--total_timesteps", default=1e5, type=float)
parser.add_argument("--eval_freq", default=5e3, type=float)
parser.add_argument("--SR", action='store_true')
parser.add_argument("--RDE", action='store_true')
parser.add_argument("--reset_freq", default=4e4, type=float)
parser.add_argument("--replay_ratio", default=1, type=int)
parser.add_argument("--learning_starts", default=2000, type=int)
parser.add_argument("--action_select_coef", default=50, type=int)
parser.add_argument("--wandb", action='store_true')
parser.add_argument("--all_reset", action='store_true')
parser.add_argument("--entity_name", type=str)

args = parser.parse_args()

set_random_seed(args.seed)

policy_kwargs = dict()

if args.RDE:
    mode = 'RDE+DQN'
    num_agent = 2
    reset = True
elif args.SR:
    mode = 'SR+DQN'
    num_agent = 1
    reset = True
else:
    mode = 'DQN'
    num_agent = 1
    reset = False

policy_kwargs.update(num_agent=num_agent)

if args.action_select_coef != 50:
    policy_kwargs.update(action_select_coef=args.action_select_coef)

if args.wandb:
    policy_kwargs.update(wandb=args.wandb)

print(f'env:{args.env}, mode:{mode}')

env = make_atari_env(args.env, n_envs=args.n_envs, seed=args.seed)
env = VecFrameStack(env, n_stack=4)
eval_env = make_atari_env(args.env, n_envs=args.n_envs, seed=args.seed+42)
eval_env = VecFrameStack(eval_env, n_stack=4)

reset_freq = int((args.reset_freq/num_agent)/args.replay_ratio)

log_path = f"./logs/{args.env}/{args.replay_ratio}/{mode}"

filename = f'{log_path}/result.csv'
f = open(filename, 'a', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow([args])
args.filename = filename
f.close()

eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path,
                             eval_freq=max(args.eval_freq // args.n_envs, 1), deterministic=True,
                             render=False, wandb=args.wandb)

if args.wandb:
    policy_kwargs.update(wandb=args.wandb)
    wandb.init(project="RDE+DQN", entity=args.entity_name,
               name=f"{mode}_{args.replay_ratio}_{args.seed}",
               group=f"{args.env}",
               job_type=f"{mode}_{args.replay_ratio}_{args.action_select_coef}",
               reinit=True)

model = DQN('CnnPolicy', env, verbose=1, buffer_size=int(args.total_timesteps),
            learning_starts=args.learning_starts, tau=0.005,
            train_freq=(1, "step"), gradient_steps=args.replay_ratio,
            target_update_interval=1, policy_kwargs=policy_kwargs,
            seed=args.seed, reset=reset, reset_freqency=args.reset_freq,
            num_agent=num_agent, all_reset=args.all_reset, wandb=args.wandb)

model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

env.close()
eval_env.close()
wandb.finish()
