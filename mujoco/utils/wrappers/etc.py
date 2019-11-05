import gym
import torch
import numpy as np
from copy import deepcopy


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
        return self.env.env._get_obs()


class TorchTensorWrapper(gym.Wrapper):
    """Takes care of torch Tensors in step and reset modules."""

    #def step(self, action):
    #    action = action.detach().numpy()
    #    state, reward, done, info = self.env.step(action)
    #    return torch.Tensor(state), reward, done, info

    def reset(self, **kwargs):
        return torch.Tensor(self.env.reset(**kwargs))

    #def set_from_torch_state(self, state):
    #    qpos, qvel = np.split(state.detach().numpy(), 2)
    #    self.env.set_state(qpos, qvel)

    #def is_done(self, state):
    #    state = state.detach().numpy()
    #    return self.env.is_done(state)


class SnapshotWrapper(gym.Wrapper):
    """Handles all stateful stuff, like getting and setting snapshots of states, and resetting"""
    def get_snapshot(self):

        class DataSnapshot:
            # Note: You should not modify these parameters after creation

            def __init__(self, d_source):
                self.time = deepcopy(d_source.time)
                self.qpos = deepcopy(d_source.qpos)
                self.qvel = deepcopy(d_source.qvel)
                self.qacc_warmstart = deepcopy(d_source.qacc_warmstart)
                self.ctrl = deepcopy(d_source.ctrl)
                self.act = deepcopy(d_source.act)

                # These probably aren't necessary, but they should fix the body in the same position with
                # respect to worldbody frame?
                self.body_xpos = deepcopy(d_source.body_xpos)
                self.body_xquat = deepcopy(d_source.body_xquat)

        return DataSnapshot(self.env.sim.data)

    def set_snapshot(self, snapshot_data):
        self.env.sim.data.time = deepcopy(snapshot_data.time)
        self.env.sim.data.qpos[:] = deepcopy(snapshot_data.qpos)
        self.env.sim.data.qvel[:] = deepcopy(snapshot_data.qvel)
        self.env.sim.data.qacc_warmstart[:] = deepcopy(snapshot_data.qacc_warmstart)
        self.env.sim.data.ctrl[:] = deepcopy(snapshot_data.ctrl)
        if snapshot_data.act is not None:
            self.env.sim.data.act[:] = deepcopy(snapshot_data.act)

        self.env.sim.data.body_xpos[:] = deepcopy(snapshot_data.body_xpos)
        self.env.sim.data.body_xquat[:] = deepcopy(snapshot_data.body_xquat)
