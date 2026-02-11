from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import argparse
import wandb
import csv
import os
import subprocess

# Try to import safety-gym or safety-gymnasium, but handle gracefully if not installed
SAFETY_GYM_AVAILABLE = False
try:
    import safety_gym
    SAFETY_GYM_AVAILABLE = True
except ImportError:
    try:
        import safety_gymnasium
        SAFETY_GYM_AVAILABLE = True
    except ImportError:
        SAFETY_GYM_AVAILABLE = False
        print("Warning: safety-gym or safety-gymnasium not installed.")
        print("Install with: pip install safety-gym")
        print("Or: pip install safety-gymnasium (recommended)")

wandb.login()

parser = argparse.ArgumentParser(description='Train SAC on Safety Gym environments')
parser.add_argument("--env", default="Safexp-PointGoal1-v0", 
                    help="Safety Gym environment name (e.g., Safexp-PointGoal1-v0, Safexp-CarGoal1-v0)")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--total_timesteps", default=1e6, type=int)
parser.add_argument("--eval_freq", default=1e4, type=int)
parser.add_argument("--SR", action='store_true', help="Use Self-Reset mode")
parser.add_argument("--RDE", action='store_true', help="Use Reset Deep Ensemble mode")
parser.add_argument("--PS", action='store_true', help="Use Population-based Search mode")
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
parser.add_argument("--dynamic_resets", action='store_true', help="Enable dynamic reset frequency")
parser.add_argument("--use_cost", action='store_true', help="Use cost signal from Safety Gym (for WCSAC-like behavior)")

args = parser.parse_args()

if not SAFETY_GYM_AVAILABLE:
    print("ERROR: safety-gym or safety-gymnasium is not installed.")
    print("Install with: pip install safety-gym")
    print("Or (recommended): pip install safety-gymnasium")
    exit(1)

set_random_seed(args.seed)

policy_kwargs = dict()

branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('utf-8')

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

# Create Safety Gym environment
# Safety Gym environments are standard gym environments, so we can use make_vec_env
env = make_vec_env(args.env, n_envs=1, seed=args.seed)
eval_env = make_vec_env(args.env, n_envs=1, seed=args.seed+42)

# Check if environment has cost signal (Safety Gym feature)
# Safety Gym environments return info dict with 'cost' key
test_obs = env.reset()
test_action = env.action_space.sample()
test_obs, test_reward, test_done, test_info = env.step(test_action)
has_cost_signal = 'cost' in test_info[0] if isinstance(test_info, list) else 'cost' in test_info

if has_cost_signal and args.use_cost:
    print("Note: Environment provides cost signal. For WCSAC, you would need to modify the SAC algorithm.")
    print("Current implementation uses standard SAC. Cost signal is available in info dict but not used in training.")

# Reset the environment after testing
env.reset()

reset_freq = int((args.reset_freq/num_agent)/args.replay_ratio)

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
    wandb.init(project=f"SafetyGym_{args.env}",
               name=f"{args.job_id}_{mode}_rr_{args.replay_ratio}_seed_{args.seed}_num_{num_agent}_{branch}_{reset_freq}",
               group=f"{args.env}",
               job_type=f"{mode}_{num_agent}agents_{args.replay_ratio}_{reset_freq}",
               dir=os.environ.get("WANDB_DIR", "./wandb"),
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

