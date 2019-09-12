import gym
import torch
import numpy as np


class AccumulateWrapper(gym.Wrapper):

    def __init__(self, env, gamma):
        super(AccumulateWrapper, self).__init__(env)
        self.gamma = gamma
        self.accumulated_return = 0.
        self.accumulated_observations = []
        self.accumulated_rewards = []

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.accumulated_observations.append(observation)
        self.accumulated_rewards.append(reward)
        self.accumulated_return = self.gamma * self.accumulated_return + reward

    def reset(self, **kwargs):
        self.accumulated_return = 0.
        self.accumulated_observations = []
        self.accumulated_rewards = []
        return torch.Tensor(self.env.reset(**kwargs))


class RewardScaleWrapper(gym.RewardWrapper):
    """Bring rewards to a reasonable scale."""

    def __init__(self, env, scale=0.01):
        super(RewardScaleWrapper, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return self.scale * reward

    def torch_reward(self, state, action, next_state):
        return self.scale * self.env.torch_reward(state, action, next_state)


class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        action = np.nan_to_num(action)  # less error prone but your optimization won't benefit from this
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FixedStateWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FixedStateWrapper, self).__init__(env)
        self.fixed_state = self.env.reset_model()
        self.fixed_qpos = self.env.sim.data.qpos
        self.fixed_qvel = self.env.sim.data.qvel

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.env.set_state(self.fixed_qpos, self.fixed_qvel)
        return self.env._get_obs()


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

    def is_done(self, state):
        state = state.detach().numpy()
        return self.env.is_done(state)
