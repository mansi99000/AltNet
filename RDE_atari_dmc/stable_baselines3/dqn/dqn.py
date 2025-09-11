import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th
from numpy import ndarray
from torch.nn import functional as F
import wandb

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.1,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        reset: bool = False,
        reset_freqency: float = 8e4,
        num_agent: int = 1,
        wandb: bool = False,
        all_reset: bool = False,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None
        self.reset = reset
        self.reset_freqency = reset_freqency
        self.num_reset = 0
        self.num_agent = num_agent
        self.wandb = wandb
        self.all_reset = all_reset

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = []
        self.batch_norm_stats_target = []
        for i in range(self.num_agent):
            self.batch_norm_stats.append(get_parameters_by_name(self.q_net[i], ["running_"]))
            self.batch_norm_stats_target.append(get_parameters_by_name(self.q_net_target[i], ["running_"]))
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)

    def _create_aliases(self) -> None:
        self.q_net = []
        self.q_net_target = []
        for i in range(self.num_agent):
            self.q_net.append(getattr(self.policy, f"q_net{i}"))
            self.q_net_target.append(getattr(self.policy, f"q_net_target{i}"))
        self.num_agent = self.policy.num_agent

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            for i in range(self.num_agent):
                polyak_update(self.q_net[i].parameters(), self.q_net_target[i].parameters(), self.tau)
                polyak_update(self.batch_norm_stats[i], self.batch_norm_stats_target[i], 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        for i in range(self.num_agent):
            self._update_learning_rate(self.q_net[i].optimizer)

        total_losses = []
        for _ in range(gradient_steps):
            losses = []
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            for i in range(self.num_agent):
                with th.no_grad():
                    # Compute the next Q-values using the target network
                    next_q_values = self.q_net_target[i](replay_data.next_observations)
                    # Follow greedy policy: use the one with the highest value
                    next_q_values, _ = next_q_values.max(dim=1)
                    # Avoid potential broadcast issue
                    next_q_values = next_q_values.reshape(-1, 1)
                    # 1-step TD target
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates
                current_q_values = self.q_net[i](replay_data.observations)

                # Retrieve the q-values for the actions from the replay buffer
                current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

                # Compute Huber loss (less sensitive to outliers)
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                losses.append(loss.item())

                # Optimize the policy
                self.q_net[i].optimizer.zero_grad()
                loss.backward()
                # Clip gradient norm
                th.nn.utils.clip_grad_norm_(self.q_net[i].parameters(), self.max_grad_norm)
                self.q_net[i].optimizer.step()

            total_losses.append(losses)

        if self.reset and self.num_timesteps % self.reset_freqency == 0:
            # Check if it's time to reset (every reset_frequency steps)

            # Select which agent to reset (cycles through agents: 0, 1, 2, 0, 1, 2, ...)
            q_num = int(self.num_reset % self.num_agent)

            # Reset the OUTPUT LAYER of the Q-network (the layer that produces Q-values for each action)
            self.policy.init_weights(self.q_net[q_num].q_net[0])
            # Also reset the corresponding target network output layer
            self.policy.init_weights(self.q_net_target[q_num].q_net[0])

            if self.all_reset:
                # If all_reset flag is set, ALSO reset the FIRST LINEAR LAYER of the feature extractor
                # This is a different layer from the output layer above
                self.policy.init_weights(self.q_net[q_num].features_extractor.linear[0])
                self.policy.init_weights(self.q_net_target[q_num].features_extractor.linear[0])

            # Create a fresh optimizer for this agent (resets optimizer state like momentum)
            self.q_net[q_num].optimizer = th.optim.Adam(self.q_net[q_num].parameters(), lr=self.lr_schedule(1))

            # Track how many resets we've done
            self.num_reset += 1
            self.policy.num_reset += 1

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for i in range(self.num_agent):
            self.logger.record(f"train/loss_{i+1}", np.mean(total_losses, 0)[i])

        if self.wandb:
            for i in range(self.num_agent):
                wandb.log({f"loss{i}": float(np.mean(total_losses, 0)[i])}, step=self.num_timesteps)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        evaluation: bool = False,
    ) -> Tuple[Union[ndarray, Any], Union[Optional[Tuple[ndarray, ...]], Any], Union[float, Any]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
            action_index = 0.5
        else:
            action, state, action_index = self.policy.predict(observation, state, episode_start, deterministic, evaluation)
        return action, state, action_index

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
