import gym
import torch
import numpy as np


class RewardScaler(gym.RewardWrapper):
    """Bring rewards to a reasonable scale."""

    def __init__(self, env, scale=0.01):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        action = np.nan_to_num(action)  # less error prone but your optimization won't benefit from this
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TorchTensorWrapper(gym.Wrapper):
    """Takes care of torch Tensors in step and reset modules."""

    def step(self, action):
        action = action.detach().numpy()
        state, reward, done, info = self.env.step(action)
        return torch.Tensor(state), reward, done, info

    def reset(self, **kwargs):
        return torch.Tensor(self.env.reset(**kwargs))

    def set_from_torch_state(self, state):
        qpos, qvel = np.split(state.detach().numpy(), 2)
        self.env.set_state(qpos, qvel)

