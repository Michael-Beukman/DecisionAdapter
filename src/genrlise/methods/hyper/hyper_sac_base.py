from typing import Any, Dict, List, Optional, Union
from gym import spaces
import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
import torch
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

from genrlise.contexts.context_encoder import ContextEncoder


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def get_obs_context(obs, context_encoder: ContextEncoder):
    if obs.shape[-1] == 1:
        context = context_encoder.get_context()
    else:
        D = context_encoder.context_dimension
        obs, context = obs[:, :-D], obs[:, -D:]
    if len(obs.shape) == 3:
        obs = torch.squeeze(obs, -1)
        context = torch.squeeze(context, -1)
    return obs, context

class MyContextReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        context_encoder: ContextEncoder = None
    ):
        super(MyContextReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.context_encoder = context_encoder
        self.contexts = np.zeros((self.buffer_size, self.n_envs, self.context_encoder.context_dimension), dtype=observation_space.dtype)
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        context = self.context_encoder.get_context(return_prev_ctx=done[0])
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.contexts[self.pos] = np.array(context.detach().cpu()).reshape(1, -1).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        contexts = self.contexts[batch_inds, env_indices, :]
        if len(obs.shape) == 3:
            obs = obs.squeeze(-1)
        
        if len(next_obs.shape) == 3:
            next_obs = next_obs.squeeze(-1)
        
        obs = np.concatenate((obs, contexts), axis=-1)
        next_obs = np.concatenate((next_obs, contexts), axis=-1)
        data = (
            obs,
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
